import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel

app = FastAPI(title="IGYT Media Worker", version="1.0.0")


# ============================================================
# CONFIG
# ============================================================

WORKER_API_KEY = os.getenv("WORKER_API_KEY", "").strip()
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny").strip()

DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "180"))
MAX_VIDEO_MB = int(os.getenv("MAX_VIDEO_MB", "250"))

BASE_DIR = Path("/tmp/igyt")
BASE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# WHISPER
# ============================================================

_model: WhisperModel | None = None


def get_whisper_model() -> WhisperModel:
    global _model

    if _model is None:
        _model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8",
        )

    return _model


# ============================================================
# MODELS
# ============================================================

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


# ============================================================
# AUTH
# ============================================================

def check_worker_key(x_worker_key: str | None) -> None:
    # During initial setup we allow the health endpoint to work
    # without authentication.
    if not WORKER_API_KEY:
        return

    if x_worker_key != WORKER_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid worker key",
        )


# ============================================================
# HELPERS
# ============================================================

def safe_filename(value: str, fallback: str) -> str:
    value = "".join(
        c for c in value
        if c.isalnum() or c in ("-", "_", ".")
    )

    return value[:100] or fallback


def download_video(url: str, destination: Path) -> None:
    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="video_url must be an HTTP/HTTPS URL",
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
        )
    }

    try:
        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=DOWNLOAD_TIMEOUT,
            allow_redirects=True,
        ) as response:

            response.raise_for_status()

            content_length = response.headers.get("content-length")

            if content_length:
                size_mb = int(content_length) / (1024 * 1024)

                if size_mb > MAX_VIDEO_MB:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video exceeds {MAX_VIDEO_MB} MB limit",
                    )

            total = 0
            max_bytes = MAX_VIDEO_MB * 1024 * 1024

            with destination.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue

                    total += len(chunk)

                    if total > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Video exceeds {MAX_VIDEO_MB} MB limit",
                        )

                    output.write(chunk)

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unable to download video: {exc}",
        ) from exc


def run_ffmpeg(command: list[str]) -> None:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail="FFmpeg processing timed out",
        ) from exc

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"FFmpeg failed: {result.stderr[-4000:]}",
        )


def escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("%", r"\%")
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "igyt-media-worker",
        "whisper_model": WHISPER_MODEL_NAME,
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }


# ============================================================
# TRANSCRIBE
# ============================================================

@app.post("/transcribe")
def transcribe(
    payload: TranscribeRequest,
    x_worker_key: str | None = Header(default=None),
) -> dict[str, Any]:

    check_worker_key(x_worker_key)

    if not payload.video_url:
        raise HTTPException(
            status_code=400,
            detail="video_url is required",
        )

    job_name = safe_filename(
        payload.job_id,
        "transcription",
    )

    work_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{job_name}_",
            dir=BASE_DIR,
        )
    )

    try:
        input_file = work_dir / "input.mp4"

        download_video(
            payload.video_url,
            input_file,
        )

        model = get_whisper_model()

        segments_iter, info = model.transcribe(
            str(input_file),
            beam_size=1,
            vad_filter=True,
            word_timestamps=True,
        )

        transcript_parts: list[str] = []
        segments: list[dict[str, Any]] = []
        words: list[dict[str, Any]] = []

        for segment in segments_iter:
            text = (segment.text or "").strip()

            if text:
                transcript_parts.append(text)

            segment_words: list[dict[str, Any]] = []

            if segment.words:
                for word in segment.words:
                    word_data = {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }

                    segment_words.append(word_data)
                    words.append(word_data)

            segments.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": text,
                    "words": segment_words,
                }
            )

        return {
            "ok": True,
            "job_id": payload.job_id,
            "source_media_id": payload.source_media_id,
            "language": info.language,
            "language_probability": info.language_probability,
            "transcript": " ".join(transcript_parts).strip(),
            "segments": segments,
            "words": words,
        }

    finally:
        shutil.rmtree(
            work_dir,
            ignore_errors=True,
        )


# ============================================================
# RENDER
# ============================================================

@app.post("/render")
def render(
    payload: RenderRequest,
    x_worker_key: str | None = Header(default=None),
):
    check_worker_key(x_worker_key)

    if not payload.video_url:
        raise HTTPException(
            status_code=400,
            detail="video_url is required",
        )

    job_name = safe_filename(
        payload.job_id,
        "render",
    )

    work_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{job_name}_",
            dir=BASE_DIR,
        )
    )

    try:
        input_file = work_dir / "input.mp4"
        output_file = work_dir / "output.mp4"

        download_video(
            payload.video_url,
            input_file,
        )

        # Use the first 60 seconds for the initial worker version.
        # This prevents accidentally rendering an extremely long
        # source video.
        duration_limit = 60

        # Optional title overlay.
        title = escape_drawtext(
            payload.title[:80]
        )

        if title:
            draw_filter = (
                "drawtext="
                "fontfile=/usr/share/fonts/truetype/dejavu/"
                "DejaVuSans-Bold.ttf:"
                f"text='{title}':"
                "x=(w-text_w)/2:"
                "y=80:"
                "fontsize=42:"
                "fontcolor=white:"
                "box=1:"
                "boxcolor=black@0.55:"
                "boxborderw=18"
            )
        else:
            draw_filter = "null"

        # Convert to vertical 1080x1920.
        #
        # The source is scaled to fill the vertical frame and cropped
        # in the center. This is intentionally a simple first renderer.
        filter_complex = (
            "[0:v]"
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            f"{draw_filter}"
            "[v]"
        )

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file),
            "-t",
            str(duration_limit),

            "-filter_complex",
            filter_complex,

            "-map",
            "[v]",

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

        run_ffmpeg(command)

        if not output_file.exists():
            raise HTTPException(
                status_code=500,
                detail="Render completed but output file was not created",
            )

        return FileResponse(
            path=str(output_file),
            media_type="video/mp4",
            filename=f"{job_name}.mp4",
            headers={
                "X-IGYT-Job-ID": payload.job_id,
                "X-IGYT-Source-Media-ID": payload.source_media_id,
            },
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
