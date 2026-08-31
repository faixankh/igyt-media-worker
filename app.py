import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, Header, HTTPException
from faster_whisper import WhisperModel
from pydantic import BaseModel

app = FastAPI(title="IGYT Media Worker")

WORKER_API_KEY = os.getenv("WORKER_API_KEY", "")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")

BASE_DIR = Path("/tmp/igyt")
BASE_DIR.mkdir(parents=True, exist_ok=True)

_model = None


class TranscribeRequest(BaseModel):
    job_id: str = ""
    source_media_id: str = ""
    video_url: str
    caption: str = ""


class RenderRequest(BaseModel):
    job_id: str = ""
    source_media_id: str = ""
    source_username: str = ""
    video_url: str
    thumbnail_url: str = ""
    caption: str = ""
    transcript: str = ""
    segments: list[Any] = []
    title: str = ""
    description: str = ""
    hashtags: list[str] = []
    tags: list[str] = []
    hook: str = ""
    transformation_plan: str = ""
    subtitle_style: str = "clean"


def auth(key: str | None):
    if WORKER_API_KEY and key != WORKER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid worker key")


def get_model():
    global _model

    if _model is None:
        _model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8",
            cpu_threads=1,
            num_workers=1,
        )

    return _model


def download_file(url: str, target: Path):
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid video URL")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=180,
        ) as r:
            r.raise_for_status()

            with target.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Video download failed: {e}",
        )


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "igyt-media-worker",
        "whisper_model": WHISPER_MODEL_NAME,
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }


@app.post("/transcribe")
def transcribe(
    payload: TranscribeRequest,
    x_worker_key: str | None = Header(default=None),
):
    auth(x_worker_key)

    if not payload.video_url:
        raise HTTPException(
            status_code=400,
            detail="video_url is required",
        )

    work = Path(
        tempfile.mkdtemp(
            prefix="igyt_",
            dir=BASE_DIR,
        )
    )

    try:
        input_file = work / "input.mp4"

        download_file(
            payload.video_url,
            input_file,
        )

        model = get_model()

        segments_iter, info = model.transcribe(
            str(input_file),
            beam_size=1,
            vad_filter=True,
            word_timestamps=True,
        )

        transcript = []
        segments = []
        words = []

        for segment in segments_iter:
            text = (segment.text or "").strip()

            if text:
                transcript.append(text)

            segment_words = []

            if segment.words:
                for word in segment.words:
                    data = {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                    }

                    segment_words.append(data)
                    words.append(data)

            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": text,
                "words": segment_words,
            })

        return {
            "ok": True,
            "job_id": payload.job_id,
            "source_media_id": payload.source_media_id,
            "language": info.language,
            "transcript": " ".join(transcript),
            "segments": segments,
            "words": words,
        }

    finally:
        shutil.rmtree(
            work,
            ignore_errors=True,
        )


@app.post("/render")
def render(
    payload: RenderRequest,
    x_worker_key: str | None = Header(default=None),
):
    auth(x_worker_key)

    if not payload.video_url:
        raise HTTPException(
            status_code=400,
            detail="video_url is required",
        )

    work = Path(
        tempfile.mkdtemp(
            prefix="render_",
            dir=BASE_DIR,
        )
    )

    try:
        input_file = work / "input.mp4"
        output_file = work / "output.mp4"

        download_file(
            payload.video_url,
            input_file,
        )

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file),
            "-t",
            "60",
            "-vf",
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output_file),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=900,
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=result.stderr[-3000:],
            )

        if not output_file.exists():
            raise HTTPException(
                status_code=500,
                detail="Render produced no output",
            )

        from fastapi.responses import FileResponse

        return FileResponse(
            str(output_file),
            media_type="video/mp4",
            filename=f"{payload.job_id or 'render'}.mp4",
        )

    finally:
        # Do not delete immediately before FileResponse finishes.
        pass
