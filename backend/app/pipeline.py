from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import gc

import cv2
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    track_id: int | None = None
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox


class FrameSummary(BaseModel):
    frame_index: int
    timestamp_ms: float
    detections: list[Detection]
    active_track_count: int
    fps: float
    latency_ms: float


class ProcessingResponse(BaseModel):
    video_name: str
    model_name: str
    device: str
    confidence_threshold: float
    source_fps: float
    source_frame_count: int
    processed_frame_count: int
    processing_time_ms: float | None = None
    measured_fps: float | None = None
    annotated_video_url: str | None = None
    preview_image_url: str | None = None
    frame_image_urls: list[str] | None = None
    frames: list[FrameSummary]


@dataclass(slots=True)
class ProcessingSettings:
    confidence_threshold: float = 0.25
    model_name: str = "yolov8n.pt"
    device: str = "cpu"
    max_fps: int | None = None
    image_size: int = 640


@lru_cache(maxsize=1)
def load_yolo_model(model_name: str) -> Any:
    from ultralytics import YOLO

    return YOLO(model_name)


def _extract_detections(result: Any) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = getattr(result, "names", None) or {}
    xyxy = getattr(boxes, "xyxy", None)
    conf = getattr(boxes, "conf", None)
    cls = getattr(boxes, "cls", None)
    ids = getattr(boxes, "id", None)

    if xyxy is None or conf is None or cls is None:
        return []

    try:
        xyxy_values = xyxy.cpu().tolist()
        conf_values = conf.cpu().tolist()
        cls_values = cls.cpu().tolist()
        id_values = ids.cpu().tolist() if ids is not None else [None] * len(xyxy_values)
    except AttributeError:
        xyxy_values = list(xyxy)
        conf_values = list(conf)
        cls_values = list(cls)
        id_values = list(ids) if ids is not None else [None] * len(xyxy_values)

    detections: list[Detection] = []
    for index, box_values in enumerate(xyxy_values):
        class_id = int(cls_values[index])
        detections.append(
            Detection(
                track_id=int(id_values[index]) if id_values[index] is not None else None,
                class_id=class_id,
                class_name=str(names.get(class_id, class_id)),
                confidence=float(conf_values[index]),
                box=BoundingBox(
                    x1=float(box_values[0]),
                    y1=float(box_values[1]),
                    x2=float(box_values[2]),
                    y2=float(box_values[3]),
                ),
            )
        )

    return detections



def process_video(
    video_path: Path,
    settings: ProcessingSettings,
    model: Any | None = None,
    output_path: Path | None = None,
) -> ProcessingResponse:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened.")

    try:
        processing_started = perf_counter()
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        effective_max_fps = settings.max_fps if settings.max_fps and settings.max_fps > 0 else None
        frame_stride = 1
        if source_fps > 0.0 and effective_max_fps is not None and effective_max_fps < source_fps:
            frame_stride = max(1, round(source_fps / effective_max_fps))

        if model is None:
            model = load_yolo_model(settings.model_name)

        frames: list[FrameSummary] = []
        raw_frame_index = 0
        last_annotated: bytes | None = None
        frame_jpegs: list[bytes] = []  # In-memory JPEG bytes per frame (tiny at ~320px)
        MAX_FRAMES = 5

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if len(frames) >= MAX_FRAMES:
                break

            if raw_frame_index % frame_stride != 0:
                raw_frame_index += 1
                del frame
                continue

            # Pre-resize to YOLO's input size before inference so the large original
            # frame buffer is released immediately and result.plot() returns a small image.
            h_f, w_f = frame.shape[:2]
            if max(h_f, w_f) > settings.image_size:
                scale = settings.image_size / max(h_f, w_f)
                frame = cv2.resize(frame, (int(w_f * scale), int(h_f * scale)))

            frame_start = perf_counter()
            results = model.track(
                frame,
                persist=True,
                conf=settings.confidence_threshold,
                device=settings.device,
                verbose=False,
                tracker="bytetrack.yaml",
                imgsz=settings.image_size
            )
            inference_ms = (perf_counter() - frame_start) * 1000.0
            del frame  # Release frame buffer before annotation/GC

            result = results[0] if isinstance(results, list) and results else results
            detections = _extract_detections(result)
            if output_path is not None:
                annotated_frame = result.plot() if hasattr(result, "plot") else None
                if annotated_frame is not None:
                    ok_jpg, jpg_buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok_jpg:
                        jpg_bytes = jpg_buf.tobytes()
                        frame_jpegs.append(jpg_bytes)
                        last_annotated = jpg_bytes
                    del jpg_buf
                del annotated_frame
            del results, result
            gc.collect()
            active_track_count = len({det.track_id for det in detections if det.track_id is not None})
            processed_frame_index = len(frames)
            timestamp_ms = (
                (raw_frame_index / source_fps) * 1000.0 if source_fps > 0.0 else float(raw_frame_index)
            )

            frames.append(
                FrameSummary(
                    frame_index=processed_frame_index,
                    timestamp_ms=timestamp_ms,
                    detections=detections,
                    active_track_count=active_track_count,
                    fps=1000.0 / inference_ms if inference_ms > 0.0 else 0.0,
                    latency_ms=inference_ms,
                )
            )
            raw_frame_index += 1

        processing_time_ms = (perf_counter() - processing_started) * 1000.0
        measured_fps = len(frames) / (processing_time_ms / 1000.0) if processing_time_ms > 0.0 else 0.0

        # Write per-frame JPEG files to disk (no codec, no subprocess, browser-displayable).
        frame_image_urls: list[str] | None = None
        preview_path: Path | None = None
        if output_path is not None and frame_jpegs:
            stem = output_path.stem
            frame_image_urls = []
            for idx, jpg_bytes in enumerate(frame_jpegs):
                fpath = output_path.parent / f"{stem}_f{idx:02d}.jpg"
                fpath.write_bytes(jpg_bytes)
                frame_image_urls.append(f"/media/{fpath.name}")
            # Write last frame as thumbnail preview
            preview_path = output_path.with_suffix(".jpg")
            preview_path.write_bytes(frame_jpegs[-1])
        frame_jpegs.clear()
        last_annotated = None

        return ProcessingResponse(
            video_name=video_path.name,
            model_name=settings.model_name,
            device=settings.device,
            confidence_threshold=settings.confidence_threshold,
            source_fps=source_fps,
            source_frame_count=source_frame_count,
            processed_frame_count=len(frames),
            processing_time_ms=processing_time_ms,
            measured_fps=measured_fps,
            annotated_video_url=None,
            preview_image_url=f"/media/{preview_path.name}" if preview_path else None,
            frame_image_urls=frame_image_urls,
            frames=frames,
        )
    finally:
        capture.release()
