from fastapi import FastAPI, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from dotenv import load_dotenv
from services.stt import transcribe_audio
from services.tts import synthesize_speech
from services.ai_brain import get_ai_response
import os

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/chat")
async def chat(audio: UploadFile = File(...)):
    # Save audio file
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{audio.filename}"
    with open(path, "wb") as f:
        f.write(await audio.read())

    # Step 1 - Transcribe voice to text
    transcription = transcribe_audio(path)

    # Step 2 - Get AI response
    ai_reply = get_ai_response(transcription)

    # Step 3 - Convert reply to voice
    audio_url = await synthesize_speech(ai_reply)

    return {
        "transcription": transcription,
        "response": ai_reply,
        "audio_url": audio_url
    }