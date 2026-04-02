from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>AI Voice Agent</title>
        </head>
        <body style="text-align:center; font-family:Arial; margin-top:100px;">
            <h1>🎙️ AI Voice Agent</h1>
            <p style="color:green; font-size:20px;">✅ Day 1 Complete — Server is Running!</p>
            <p>Built by Praveen 🚀</p>
        </body>
    </html>
    """
