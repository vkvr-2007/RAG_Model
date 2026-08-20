import os
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from speech_to_text import transcribe_audio

app = FastAPI(title="HH Goa Voice RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Backend is running"
    }


@app.post("/api/query")
async def query(audio: UploadFile = File(...)):
    file_extension = os.path.splitext(audio.filename or "")[1] or ".webm"

    audio_data = await audio.read()

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name

        transcript = transcribe_audio(temp_path)

        return {
            "status": "success",
            "transcript": transcript
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error)
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)