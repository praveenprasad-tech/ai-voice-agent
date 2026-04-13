from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import google.generativeai as genai
import assemblyai as aai
import requests
import tempfile
import os
import uuid

app = FastAPI()

# ── API Keys ──────────────────────────────────────────
GEMINI_API_KEY   = "your_gemini_api_key"
ASSEMBLYAI_KEY   = "your_assemblyai_api_key"
MURF_API_KEY     = "your_murf_api_key"

# ── Setup ─────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
aai.settings.api_key = ASSEMBLYAI_KEY
model = genai.GenerativeModel("gemini-pro")

# ── Session Memory (Day 2) ────────────────────────────
chat_history = {}  # { session_id: [ {role, text}, ... ] }

# ── Serve index.html ──────────────────────────────────
@app.get("/")
def root():
    return FileResponse("index.html")

# ── /chat/{session_id} Route ──────────────────────────
@app.post("/chat/{session_id}")
async def chat(session_id: str, audio: UploadFile = File(...)):

    # ── Step 1: Save uploaded audio ───────────────────
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio save failed: {str(e)}")

    # ── Step 2: STT — AssemblyAI ──────────────────────
    try:
        transcriber = aai.Transcriber()
        transcript  = transcriber.transcribe(tmp_path)
        user_text   = transcript.text
        if not user_text:
            raise HTTPException(status_code=400, detail="Transcription empty. Dobara bolo!")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT error: {str(e)}")
    finally:
        os.unlink(tmp_path)  # temp file delete karo

    # ── Step 3: Session history update ────────────────
    if session_id not in chat_history:
        chat_history[session_id] = []
    chat_history[session_id].append({"role": "user", "text": user_text})

    # ── Step 4: Gemini AI ─────────────────────────────
    try:
        history_text = "\n".join(
            [f"{m['role'].upper()}: {m['text']}" for m in chat_history[session_id][-6:]]
        )
        prompt = f"Conversation so far:\n{history_text}\nAI:"
        gemini_resp = model.generate_content(prompt)
        ai_text     = gemini_resp.text
        if not ai_text:
            raise HTTPException(status_code=500, detail="Gemini ne kuch return nahi kiya")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

    chat_history[session_id].append({"role": "ai", "text": ai_text})

    # ── Step 5: TTS — Murf AI ─────────────────────────
    audio_url = None
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
            }
        )
        if murf_resp.status_code == 200:
            audio_url = murf_resp.json().get("audioFile")
    except Exception as e:
        # TTS fail ho toh bhi response do, sirf audio nahi aayega
        print(f"TTS error (non-fatal): {str(e)}")

    # ── Step 6: Response return karo ──────────────────
    return {
        "transcription": user_text,
        "response":      ai_text,
        "audio_url":     audio_url
    }