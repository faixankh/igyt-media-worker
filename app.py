import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="IGYT Media Worker")

WORKER_API_KEY = os.getenv("WORKER_API_KEY", "")


class TranscribeRequest(BaseModel):
    job_id: str
    source_media_id: str
    source_username: str = ""
    video_url: str
    caption: str = ""
    callback_url: str


def auth(key: str | None):
    if WORKER_API_KEY and key != WORKER_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid worker key",
        )


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "igyt-media-worker",
        "role": "render-gateway",
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "igyt-media-worker",
        "role": "render-gateway",
        "whisper": "camber-gpu",
    }


@app.post("/transcribe")
def transcribe(
    payload: TranscribeRequest,
    x_worker_key: str | None = Header(default=None),
):
    auth(x_worker_key)

    return {
        "ok": True,
        "accepted": True,
        "status": "QUEUED",
        "job_id": payload.job_id,
        "source_media_id": payload.source_media_id,
        "message": "Transcription job accepted for Camber GPU processing.",
        "video_url": payload.video_url,
        "callback_url": payload.callback_url,
    }
