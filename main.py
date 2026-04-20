from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
import google.generativeai as genai
import assemblyai as aai
import requests
import tempfile
import os
import uuid
import logging
from dotenv import load_dotenv
from typing import Optional

# ── Load environment variables ────────────────────────
load_dotenv()

# ── Logging setup ─────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Voice Agent")

# ── API Keys (environment variables se load karo) ────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_KEY", "").strip()
MURF_API_KEY   = os.getenv("MURF_API_KEY", "").strip()

# ── Validate API Keys ─────────────────────────────────
def validate_api_keys():
    """API keys check karo startup mein"""
    missing_keys = []
    if not GEMINI_API_KEY:
        missing_keys.append("GEMINI_API_KEY")
    if not ASSEMBLYAI_KEY:
        missing_keys.append("ASSEMBLYAI_KEY")
    
    if missing_keys:
        error_msg = f"❌ Missing API keys: {', '.join(missing_keys)}. Check .env file!"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("✅ All API keys loaded successfully")

validate_api_keys()

# ── Setup APIs ────────────────────────────────────────
try:
    genai.configure(api_key=GEMINI_API_KEY)
    aai.settings.api_key = ASSEMBLYAI_KEY
    model = genai.GenerativeModel("gemini-pro")
    logger.info("✅ Gemini & AssemblyAI configured")
except Exception as e:
    logger.error(f"❌ API configuration failed: {str(e)}")
    raise

# ── Session Memory (Day 2) ────────────────────────────
chat_history = {}  # { session_id: [ {role, text}, ... ] }

# ── Custom Error Response ─────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exceptions ko proper JSON format mein return karo"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error": True}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Validation errors ko handle karo"""
    logger.warning(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request format", "error": True}
    )

# ── Serve index.html ──────────────────────────────────
@app.get("/")
def root():
    try:
        return FileResponse("index.html")
    except FileNotFoundError:
        logger.error("index.html not found")
        raise HTTPException(status_code=404, detail="Frontend file not found")

# ── Health Check ──────────────────────────────────────
@app.get("/health")
def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "gemini": "configured" if GEMINI_API_KEY else "missing",
        "assemblyai": "configured" if ASSEMBLYAI_KEY else "missing"
    }

# ── /chat/{session_id} Route ──────────────────────────
@app.post("/chat/{session_id}")
async def chat(session_id: str, audio: UploadFile = File(...)):
    """
    Complete voice chat flow:
    1. Save audio file
    2. STT (AssemblyAI)
    3. Session history update
    4. LLM response (Gemini)
    5. TTS (Murf AI) — optional
    """
    
    logger.info(f"🎤 Chat request started - session: {session_id[:8]}...")
    tmp_path = None
    
    try:
        # ── Step 1: Save uploaded audio ───────────────────
        logger.info("Step 1: Saving audio file...")
        if not audio.filename:
            raise HTTPException(status_code=400, detail="Audio file required")
        
        if not audio.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg')):
            raise HTTPException(
                status_code=400, 
                detail="Invalid audio format. Supported: WAV, MP3, M4A, OGG"
            )
        
        try:
            audio_data = await audio.read()
            if len(audio_data) == 0:
                raise HTTPException(status_code=400, detail="Audio file is empty")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            logger.info(f"✅ Audio saved: {len(audio_data)} bytes")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Audio save error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Audio save failed: {str(e)}")

        # ── Step 2: STT — AssemblyAI ─────────────────────
        logger.info("Step 2: Running speech-to-text...")
        user_text = None
        try:
            config = aai.TranscriptionConfig(
                speech_model=aai.SpeechModel.universal,
                language_code="en"
            )
            transcriber = aai.Transcriber(config=config)
            transcript = transcriber.transcribe(tmp_path)
            
            if not transcript.text or transcript.text.strip() == "":
                logger.warning("Empty transcription")
                raise HTTPException(
                    status_code=400, 
                    detail="Could not understand audio. Dobara bolo thoda clear!"
                )
            
            user_text = transcript.text.strip()
            logger.info(f"✅ Transcribed: '{user_text[:50]}...'")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"STT error: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Speech recognition failed: {str(e)}"
            )
        finally:
            # Always cleanup temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.warning(f"Could not delete temp file: {str(e)}")

        # ── Step 3: Session history update ────────────────
        logger.info("Step 3: Updating session history...")
        if session_id not in chat_history:
            chat_history[session_id] = []
        chat_history[session_id].append({"role": "user", "text": user_text})
        logger.info(f"✅ Session history updated ({len(chat_history[session_id])} messages)")

        # ── Step 4: Gemini AI ─────────────────────────────
        logger.info("Step 4: Generating AI response...")
        ai_text = None
        try:
            # Build context from last 6 messages
            history_messages = chat_history[session_id][-6:]
            history_text = "\n".join(
                [f"{m['role'].upper()}: {m['text']}" for m in history_messages]
            )
            
            prompt = f"Conversation so far:\n{history_text}\n\nRespond naturally and briefly (1-2 sentences):\nAI:"
            
            gemini_resp = model.generate_content(prompt)
            ai_text = gemini_resp.text
            
            if not ai_text or ai_text.strip() == "":
                logger.error("Gemini returned empty response")
                raise HTTPException(
                    status_code=500, 
                    detail="AI could not generate response. Try again!"
                )
            
            ai_text = ai_text.strip()
            logger.info(f"✅ AI response: '{ai_text[:50]}...'")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Gemini error: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"AI service error: {str(e)}"
            )

        # ── Step 5: Add AI response to history ───────────
        chat_history[session_id].append({"role": "ai", "text": ai_text})

        # ── Step 6: TTS — Murf AI ────────────────────────
        audio_url = None
        if MURF_API_KEY:
            logger.info("Step 6: Generating voice audio...")
            try:
                murf_resp = requests.post(
                    "https://api.murf.ai/v1/speech/generate",
                    headers={
                        "api-key": MURF_API_KEY,
                        "Content-Type": "application/json"
                    },
                    json={
                        "voiceId": "en-US-natalie",
                        "text": ai_text,
                        "audioFormat": "MP3"
                    },
                    timeout=30  # 30 second timeout
                )
                
                if murf_resp.status_code == 200:
                    audio_url = murf_resp.json().get("audioFile")
                    logger.info(f"✅ Voice audio generated")
                else:
                    logger.warning(f"Murf API error: {murf_resp.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning("Murf AI timeout - voice skipped")
            except Exception as e:
                # TTS fail ho toh bhi response do, sirf audio nahi aayega
                logger.warning(f"TTS error (non-fatal): {str(e)}")
        else:
            logger.warning("Murf API key not configured - skipping TTS")

        # ── Step 7: Success response return karo ─────────
        logger.info("✅ Chat complete - returning response")
        return {
            "success": True,
            "transcription": user_text,
            "response": ai_text,
            "audio_url": audio_url
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

# ── 404 Handler ───────────────────────────────────────
@app.exception_handler(404)
async def not_found_handler(request, exc):
    logger.warning(f"404 Not found: {request.url.path}")
    return JSONResponse(
        status_code=404,
        content={"detail": "Not found", "error": True}
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting AI Voice Agent...")
    uvicorn.run(app, host="0.0.0.0", port=8000)