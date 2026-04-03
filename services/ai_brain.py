import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

def get_ai_response(text: str) -> str:
    response = model.generate_content(text)
    return response.text