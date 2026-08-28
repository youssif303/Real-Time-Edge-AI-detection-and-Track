from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from backend.app.pipeline import ProcessingSettings, process_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark EdgeTrack on a local video.")
    parser.add_argument("video", type=Path, help="Path to an input video")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--max-fps", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=640, choices=(416, 640, 960))
    parser.add_argument("--output", type=Path, default=None, help="Annotated MP4 output path")
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")

    output_path = args.output or Path("backend/outputs") / f"benchmark-{uuid4().hex}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    settings = ProcessingSettings(
        confidence_threshold=args.confidence,
        model_name=args.model,
        device=args.device,
        max_fps=args.max_fps,
        image_size=args.image_size,
    )

    started = perf_counter()
    result = process_video(args.video, settings, output_path=output_path)
    wall_time_s = perf_counter() - started
    measured_fps = result.processed_frame_count / wall_time_s if wall_time_s > 0 else 0.0
    average_latency_ms = (
        sum(frame.latency_ms for frame in result.frames) / len(result.frames)
        if result.frames
        else 0.0
    )

    print(json.dumps({
        "video": str(args.video),
        "model": args.model,
        "device": args.device,
        "confidence": args.confidence,
        "image_size": args.image_size,
        "source_fps": result.source_fps,
        "source_frame_count": result.source_frame_count,
        "processed_frame_count": result.processed_frame_count,
        "wall_time_seconds": round(wall_time_s, 3),
        "measured_fps": round(measured_fps, 3),
        "average_inference_latency_ms": round(average_latency_ms, 3),
        "annotated_output": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()