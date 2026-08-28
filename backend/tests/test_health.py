from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app import main as main_module
from backend.app import pipeline
from backend.app.main import app
from backend.app.pipeline import (
    BoundingBox,
    Detection,
    FrameSummary,
    ProcessingResponse,
    ProcessingSettings,
)


client = TestClient(app)


class FakeArray:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def cpu(self) -> "FakeArray":
        return self

    def tolist(self) -> list[object]:
        return list(self._values)


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = FakeArray([[10.0, 20.0, 30.0, 40.0]])
        self.conf = FakeArray([0.91])
        self.cls = FakeArray([0])
        self.id = FakeArray([7])


class FakeResult:
    def __init__(self, frame=None) -> None:
        self.boxes = FakeBoxes()
        self.names = {0: "person"}
        self.frame = frame

    def plot(self):
        return self.frame


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def track(self, frame, **kwargs):
        self.calls.append(str(frame))
        return [FakeResult(frame)]


class FakeCapture:
    def __init__(self, frames: list[str]) -> None:
        self._frames = frames
        self._index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def get(self, prop: int) -> float:
        if prop == pipeline.cv2.CAP_PROP_FPS:
            return 30.0
        if prop == pipeline.cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        return 0.0

    def release(self) -> None:
        self.released = True


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "edgetrack-api"
    assert payload["version"] == "0.1.0"
    assert payload["capabilities"]["cpu"] is True
    assert isinstance(payload["capabilities"]["cuda"], bool)


def test_process_video_extracts_detections(monkeypatch) -> None:
    fake_capture = FakeCapture(["frame-1", "frame-2"])
    fake_model = FakeModel()

    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda _: fake_capture)
    monkeypatch.setattr(pipeline, "load_yolo_model", lambda model_name: fake_model)

    response = pipeline.process_video(
        Path("sample.mp4"),
        ProcessingSettings(
            confidence_threshold=0.4,
            model_name="yolov8n.pt",
            device="cpu",
            max_fps=None,
        ),
    )

    assert response.video_name == "sample.mp4"
    assert response.source_fps == 30.0
    assert response.source_frame_count == 2
    assert response.processed_frame_count == 2
    assert fake_capture.released is True
    assert fake_model.calls == ["frame-1", "frame-2"]
    assert response.frames[0].frame_index == 0
    assert response.frames[0].active_track_count == 1
    assert response.frames[0].detections == [
        Detection(
            track_id=7,
            class_id=0,
            class_name="person",
            confidence=0.91,
            box=BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0),
        )
    ]


def test_process_endpoint_returns_pipeline_summary(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_process_video(video_path: Path, settings: ProcessingSettings, output_path: Path | None = None) -> ProcessingResponse:
        captured["video_path"] = video_path
        captured["settings"] = settings
        return ProcessingResponse(
            video_name=video_path.name,
            model_name=settings.model_name,
            device=settings.device,
            confidence_threshold=settings.confidence_threshold,
            source_fps=24.0,
            source_frame_count=1,
            processed_frame_count=1,
            frames=[
                FrameSummary(
                    frame_index=0,
                    timestamp_ms=0.0,
                    detections=[
                        Detection(
                            track_id=7,
                            class_id=0,
                            class_name="person",
                            confidence=0.91,
                            box=BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0),
                        )
                    ],
                    active_track_count=1,
                    fps=58.0,
                    latency_ms=17.2,
                )
            ],
        )

    monkeypatch.setattr(main_module, "process_video", fake_process_video)

    response = client.post(
        "/api/v1/process",
        data={
            "confidence_threshold": "0.4",
            "model_name": "yolov8s.pt",
            "device": "cpu",
            "max_fps": "12",
        },
        files={"video": ("sample.mp4", b"fake-bytes", "video/mp4")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["video_name"] == "sample.mp4"
    assert payload["model_name"] == "yolov8s.pt"
    assert payload["processed_frame_count"] == 1
    assert payload["frames"][0]["detections"][0]["track_id"] == 7
    assert isinstance(captured["video_path"], Path)
    assert captured["settings"] == ProcessingSettings(
        confidence_threshold=0.4,
        model_name="yolov8s.pt",
        device="cpu",
        max_fps=12,
    )


def test_process_endpoint_rejects_empty_upload() -> None:
    response = client.post(
        "/api/v1/process",
        files={"video": ("empty.mp4", b"", "video/mp4")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded video file is empty."

def test_process_video_writes_playable_annotated_video(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "annotated.mp4"
    writer = cv2.VideoWriter(str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (48, 32))
    for value in (40, 80):
        writer.write(np.full((32, 48, 3), value, dtype=np.uint8))
    writer.release()

    monkeypatch.setattr(pipeline, "load_yolo_model", lambda model_name: FakeModel())
    response = pipeline.process_video(input_path, ProcessingSettings(max_fps=None), output_path=output_path)

    assert response.processed_frame_count == 2
    assert response.processing_time_ms is not None
    assert response.measured_fps is not None
    assert output_path.is_file()

    output_capture = cv2.VideoCapture(str(output_path))
    assert output_capture.isOpened()
    assert int(output_capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    output_capture.release()
