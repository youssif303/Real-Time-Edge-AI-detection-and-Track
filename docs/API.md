# API Contract

## Health check

`GET /health`

Response:

```json
{
  "status": "ok",
  "service": "edgetrack-api",
  "version": "0.1.0"
}
```

## Video processing

`POST /api/v1/process`

Accepts a video upload plus YOLO and tracking settings as multipart form fields.

Form fields:

- `video` - uploaded video file
- `confidence_threshold`
- `confidence_threshold` - float between 0 and 1, default `0.25`
- `model_name` - Ultralytics model name, default `yolov8n.pt`
- `device` - runtime device string such as `cpu` or `0`
- `max_fps` - optional processing cap for sampling frames
- `image_size` - YOLO inference size, default `640`; lower values are faster

Response fields:

- `video_name`
- `model_name`
- `device`
- `confidence_threshold`
- `image_size`
- `source_fps`
- `source_frame_count`
- `processed_frame_count`
- `processing_time_ms` - total wall-clock processing time
- `measured_fps` - processed frames per second
- `annotated_video_url` - local URL for the rendered video with boxes and track IDs
- `frames`

Per-frame fields:

- `frame_index`
- `timestamp_ms`
- `detections`
- `active_track_count`
- `fps`
- `latency_ms`

Each detection includes:

- `track_id`
- `class_id`
- `class_name`
- `confidence`
- `box`

## API principles

- Use stable, explicit field names.
- Return useful validation errors.
- Keep model internals out of the public response unless they help the user.
- Version endpoints under `/api/v1` once processing is exposed.


Generated annotated videos older than two hours are removed when a new processing request starts.
