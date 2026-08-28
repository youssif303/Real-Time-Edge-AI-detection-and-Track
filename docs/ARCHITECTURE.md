# Architecture

## System flow

```text
Video file or webcam
        |
        v
Input decoder (OpenCV)
        |
        v
Object detector (YOLO)
        |
        v
Object tracker (track IDs)
        |
        +--> Annotated frames
        +--> Per-frame events
        +--> Performance metrics
                         |
                         v
                    FastAPI service
                         |
                         v
                    React dashboard
```

## Backend responsibilities

- Validate input and runtime settings.
- Decode frames.
- Run detection and tracking.
- Calculate latency, FPS, counts, and confidence summaries.
- Return structured results to the UI.
- Fail with useful, safe error messages.

## Frontend responsibilities

- Make the primary workflow obvious.
- Display annotated media and current metrics.
- Expose only useful controls.
- Represent loading, empty, error, paused, and completed states.
- Remain usable on desktop and narrow screens.

## Data flow decisions

The first implementation will process short uploaded videos locally or through a simple API request. A streaming protocol can be added after the core pipeline is measured and stable.

## Performance measurements

- End-to-end latency per frame
- Inference latency per frame
- Frames per second
- Input resolution
- Model size and device
- Number of active tracks

