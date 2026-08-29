from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from backend.app.pipeline import (
    ProcessingSettings,
    load_yolo_model,
    process_video,
)


APP_VERSION = "0.1.0"

# In-memory job store: job_id -> {status, result, error, created_at}
_jobs: dict[str, dict] = {}
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_RETENTION = timedelta(hours=2)
JOB_RETENTION = timedelta(hours=2)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Pre-load the default YOLO model at startup so the first request is fast."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        await run_in_threadpool(load_yolo_model, "yolov8n.pt")
    except Exception:
        pass  # Non-fatal: model will download on first use if this fails.
    yield


app = FastAPI(
    title="EdgeTrack API",
    description="Real-time object detection and tracking service.",
    version=APP_VERSION,
    lifespan=lifespan,
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=OUTPUT_DIR), name="media")
configured_origins = [origin.strip() for origin in os.getenv("EDGETRACK_CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
cors_origins = [*configured_origins, "https://edgetrack-dashboard.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cleanup_jobs() -> None:
    cutoff = datetime.now(timezone.utc) - JOB_RETENTION
    stale = [jid for jid, job in _jobs.items() if job.get("created_at", datetime.now(timezone.utc)) < cutoff]
    for jid in stale:
        del _jobs[jid]
        (OUTPUT_DIR / f"{jid}.json").unlink(missing_ok=True)

def cleanup_old_outputs() -> None:
    cutoff = datetime.now(timezone.utc) - OUTPUT_RETENTION
    for output_path in list(OUTPUT_DIR.glob("*.mp4")) + list(OUTPUT_DIR.glob("*.jpg")) + list(OUTPUT_DIR.glob("*.json")):
        modified = datetime.fromtimestamp(output_path.stat().st_mtime, tz=timezone.utc)
        if modified < cutoff:
            output_path.unlink(missing_ok=True)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, object]:
    """Report whether the API process is available."""
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        cuda = False
    return {
        "status": "ok",
        "service": "edgetrack-api",
        "version": APP_VERSION,
        "capabilities": {"cpu": True, "cuda": cuda},
    }


async def _run_job(
    job_id: str,
    temp_path: Path,
    settings: ProcessingSettings,
    output_path: Path,
    original_filename: str,
) -> None:
    """Background task: run video processing and update job store."""
    job_file = OUTPUT_DIR / f"{job_id}.json"
    _jobs[job_id]["status"] = "processing"
    try:
        result = await run_in_threadpool(process_video, temp_path, settings, output_path=output_path)
        payload = result.model_copy(update={"video_name": original_filename})
        finished = {"status": "complete", "result": payload.model_dump(), "error": None}
        _jobs[job_id].update(finished)
        # Persist to disk so result survives a server restart during polling.
        import json as _json
        job_file.write_text(_json.dumps(finished))
    except ValueError as exc:
        err = {"status": "error", "result": None, "error": str(exc)}
        _jobs[job_id].update(err)
        job_file.write_text(__import__("json").dumps(err))
    except Exception as exc:
        err = {"status": "error", "result": None, "error": f"Video processing failed: {exc}"}
        _jobs[job_id].update(err)
        job_file.write_text(__import__("json").dumps(err))
    finally:
        temp_path.unlink(missing_ok=True)
        _cleanup_jobs()


@app.post("/api/v1/process", status_code=status.HTTP_202_ACCEPTED, tags=["perception"])
async def submit_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    confidence_threshold: float = Form(0.25, ge=0.0, le=1.0),
    model_name: str = Form("yolov8n.pt"),
    device: str = Form("cpu"),
    max_fps: int | None = Form(None, ge=1),
    image_size: int = Form(320, ge=320, le=1280),
) -> dict:
    """Accept a video upload and start async processing. Returns a job_id to poll."""
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    output_path = OUTPUT_DIR / f"{uuid4().hex}.mp4"

    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await video.read())
        temp_path = Path(temp_file.name)

    if temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
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

    job_id = uuid4().hex
    _jobs[job_id] = {
        "status": "queued",
        "created_at": datetime.now(timezone.utc),
        "result": None,
        "error": None,
    }
    background_tasks.add_task(
        _run_job, job_id, temp_path, settings, output_path, video.filename or temp_path.name
    )
    return {"job_id": job_id}


@app.get("/api/v1/jobs/{job_id}", tags=["perception"])
async def get_job_status(job_id: str) -> dict:
    """Poll the status of a submitted processing job."""
    job = _jobs.get(job_id)
    if job is None:
        # Check disk — covers the case where the server restarted after processing finished.
        import json as _json
        job_file = OUTPUT_DIR / f"{job_id}.json"
        if job_file.exists():
            try:
                job = _json.loads(job_file.read_text())
                _jobs[job_id] = job  # Reload into memory
            except Exception:
                pass
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job.get("result"),
        "error": job.get("error"),
    }

