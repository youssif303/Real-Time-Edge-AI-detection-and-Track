# Development Setup

## Environment

Use Windows for VS Code and the browser, and WSL2 Ubuntu for project commands. Keep the project in the WSL filesystem rather than under `/mnt/c` for better performance.

Recommended location:

```text
/home/youss/projects/real-time-edge-ai
```

## Prerequisites

- Ubuntu 24.04 on WSL2
- Python 3.12+
- Git
- Node.js and npm for the frontend
- A modern browser

No NVIDIA GPU is required for the first milestone. CPU inference is the baseline; GPU acceleration can be added after correctness and measurement are established.

## First-time backend setup

```bash
cd ~/projects/real-time-edge-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## Run the backend

```bash
source .venv/bin/activate
uvicorn backend.app.main:app --reload --reload-dir backend
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

## Run checks

```bash
source .venv/bin/activate
pytest
```

## Frontend workflow

The React application will be initialized during Milestone 2. It will use the WSL Node runtime and connect to the FastAPI service through `http://127.0.0.1:8000` during local development.

## Development rules

- Run commands from WSL, not from mixed Windows and WSL shells.
- Keep secrets in `.env`, never in Git.
- Prefer small, reviewable changes.
- Add or update documentation when behavior changes.
- Record measured performance instead of estimating it.


## Benchmark a video

Run the real CPU pipeline and save an annotated result:

```bash
python -m backend.benchmark "/path/to/video.mp4" --max-fps 5
```

The command prints JSON containing source FPS, processed frames, wall-clock time, measured throughput, and average inference latency. The output MP4 is written to `backend/outputs/` unless `--output` is provided.
