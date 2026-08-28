# EdgeTrack

Real-time object detection and tracking for video and webcam input, with an interactive performance dashboard.

## Project status

The backend now exposes a YOLO video processing endpoint at `/api/v1/process` for uploaded video files. The next milestone focuses on connecting that pipeline to the dashboard and polishing the user experience around it.

## Product goal

EdgeTrack makes edge AI perception understandable and measurable. A user should be able to provide video, see tracked objects with stable IDs, and understand the system's accuracy and performance at a glance.

## Planned MVP

- Upload a video or use a webcam.
- Detect common objects such as people, cars, bicycles, and trucks.
- Track objects across frames.
- Render boxes, labels, confidence, and track IDs.
- Display object counts, FPS, latency, and resolution.
- Adjust the confidence threshold.
- Pause, reset, and replay a sample video.

## Documentation

- [Project specification](docs/PROJECT_SPEC.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Development setup](docs/DEVELOPMENT.md)
- [API contract](docs/API.md)
- [Roadmap and milestones](docs/ROADMAP.md)
- [Engineering decisions](docs/DECISIONS.md)
- [Contribution workflow](docs/CONTRIBUTING.md)

## Repository layout

```text
backend/       FastAPI service and perception pipeline
frontend/      React dashboard
docs/          Project documentation
data/          Small public samples and local-only datasets
```

## Current local run

The backend shell can be started from WSL after installing dependencies:

```bash
cd ~/projects/real-time-edge-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --reload-dir backend
```

The API health check is available at `http://127.0.0.1:8000/health`, and the video processing endpoint is available at `POST /api/v1/process`.

Frontend setup instructions will be added with the dashboard implementation in Milestone 2. See [Development setup](docs/DEVELOPMENT.md) for the complete workflow.

## License

This project is currently unpublished. Add a license before making the repository public.

To capture repeatable CPU performance evidence from a local clip:

```bash
python -m backend.benchmark "/path/to/video.mp4" --max-fps 5
```
