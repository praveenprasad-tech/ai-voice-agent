import os
import aiohttp

MURF_API_KEY = os.getenv("MURF_API_KEY")

async def synthesize_speech(text: str) -> str:
    url = "https://api.murf.ai/v1/speech/generate"
    headers = {"api-key": MURF_API_KEY, "Content-Type": "application/json"}
    payload = {"voiceId": "en-US-cooper", "text": text, "format": "MP3"}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as res:
            data = await res.json()
            return data.get("audioFile")