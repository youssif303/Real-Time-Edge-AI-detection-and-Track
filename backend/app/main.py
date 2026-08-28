from datetime import datetime, timedelta, timezone
import os
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from backend.app.pipeline import (
    ProcessingResponse,
    ProcessingSettings,
    process_video,
)


APP_VERSION = "0.1.0"

app = FastAPI(
    title="EdgeTrack API",
    description="Real-time object detection and tracking service.",
    version=APP_VERSION,
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_RETENTION = timedelta(hours=2)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=OUTPUT_DIR), name="media")
configured_origins = [origin.strip() for origin in os.getenv("EDGETRACK_CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
cors_origins = [*configured_origins, "https://edgetrack-nu.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        return False

def cleanup_old_outputs() -> None:
    cutoff = datetime.now(timezone.utc) - OUTPUT_RETENTION
    for output_path in OUTPUT_DIR.glob("*.mp4"):
        modified = datetime.fromtimestamp(output_path.stat().st_mtime, tz=timezone.utc)
        if modified < cutoff:
            output_path.unlink(missing_ok=True)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, object]:
    """Report whether the API process is available."""
    return {
        "status": "ok",
        "service": "edgetrack-api",
        "version": APP_VERSION,
        "capabilities": {"cpu": True, "cuda": cuda_available()},
    }


@app.post("/api/v1/process", response_model=ProcessingResponse, tags=["perception"])
async def process_video_endpoint(
    video: UploadFile = File(...),
    confidence_threshold: float = Form(0.25, ge=0.0, le=1.0),
    model_name: str = Form("yolov8n.pt"),
    device: str = Form("cpu"),
    max_fps: int | None = Form(None, ge=1),
    image_size: int = Form(640, ge=320, le=1280),
) -> ProcessingResponse:
    """Process an uploaded video with YOLO detection and tracking."""
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    temp_path: Path | None = None
    output_path = OUTPUT_DIR / f"{uuid4().hex}.mp4"

    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await video.read())
            temp_path = Path(temp_file.name)

        if temp_path.stat().st_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded video file is empty.",
            )

        settings = ProcessingSettings(
            confidence_threshold=confidence_threshold,
            model_name=model_name,
            device=device,
            max_fps=max_fps,
            image_size=image_size,
        )

        try:
            result = await run_in_threadpool(process_video, temp_path, settings, output_path=output_path)
            return result.model_copy(update={
                "video_name": video.filename or temp_path.name,
                "annotated_video_url": f"/media/{output_path.name}",
            })
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - protects the API boundary.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Video processing failed: {exc}",
            ) from exc
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
