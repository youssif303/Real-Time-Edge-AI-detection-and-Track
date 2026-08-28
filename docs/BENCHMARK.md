# Benchmark Evidence

Captured 2026-08-28 with the local WSL CPU runtime.

| Input | Model | Device | Confidence | Max FPS | Source | Processed | Wall time | Measured FPS | Avg inference latency |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2026-03-24_10-19-52_mission.mp4` | `yolov8n.pt` | `cpu` | `0.50` | `5` | 605 frames at 30 FPS | 101 frames | 108.982 s | 0.927 | 675.883 ms |

The annotated result was written to `backend/outputs/benchmark-mission.mp4`. This is a baseline measurement, not a general performance guarantee; results vary with CPU, video resolution, model, and confidence settings.

Repeat it with:

```bash
python -m backend.benchmark "2026-03-24_10-19-52_mission.mp4" --max-fps 5 --confidence 0.5 --output backend/outputs/benchmark-mission.mp4
```