import os
import shutil
import tempfile
from pathlib import Path

import requests
from faster_whisper import WhisperModel


MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")

BASE_DIR = Path("/tmp/igyt")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, target: Path):
    if not url.startswith(("http://", "https://")):
        raise ValueError("Invalid video URL")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=180,
    ) as response:
        response.raise_for_status()

        with target.open("wb") as output:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    output.write(chunk)


def transcribe_video(video_url: str):
    work = Path(
        tempfile.mkdtemp(
            prefix="igyt_",
            dir=BASE_DIR,
        )
    )

    try:
        video_file = work / "input.mp4"

        download_file(
            video_url,
            video_file,
        )

        model = WhisperModel(
            MODEL_NAME,
            device="cuda",
            compute_type="float16",
        )

        segments_iter, info = model.transcribe(
            str(video_file),
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
        )

        transcript_parts = []
        segments = []
        words = []

        for segment in segments_iter:
            text = (segment.text or "").strip()

            if text:
                transcript_parts.append(text)

            segment_words = []

            if segment.words:
                for word in segment.words:
                    word_data = {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                    }

                    segment_words.append(word_data)
                    words.append(word_data)

            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": text,
                "words": segment_words,
            })

        return {
            "ok": True,
            "language": info.language,
            "language_probability": getattr(
                info,
                "language_probability",
                None,
            ),
            "transcript": " ".join(transcript_parts),
            "segments": segments,
            "words": words,
        }

    finally:
        shutil.rmtree(
            work,
            ignore_errors=True,
        )


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python transcribe_worker.py VIDEO_URL"
        )

    result = transcribe_video(sys.argv[1])

    print(
        json.dumps(
            result,
            ensure_ascii=False
        )
    )
