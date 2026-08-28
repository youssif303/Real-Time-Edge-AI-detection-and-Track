# Project Specification

## Working name

EdgeTrack

## Problem

Object detection demos often show only a prediction image. They do not make real-time behavior, tracking continuity, or edge-device performance easy to understand.

## Solution

EdgeTrack combines detection, multi-object tracking, and performance telemetry in one focused interface. The UI connects the visual result to measurable system behavior.

## Target audience

- Engineers evaluating an edge AI pipeline
- Hiring managers reviewing a practical computer vision project
- Developers learning how perception systems are structured

## Primary user story

As a visitor, I can upload a short video, start processing, and understand what the model sees, how objects move, and how quickly the system runs.

## MVP scope

### Included

- Uploaded video processing
- Optional webcam input after the upload flow is stable
- Common object detection
- Stable object IDs across frames
- Bounding boxes and labels
- Counts, confidence, FPS, and latency
- Configurable confidence threshold
- Health and processing APIs
- Reproducible sample input

### Not included initially

- Custom model training UI
- User accounts
- Cloud-scale streaming
- Safety-critical decisions
- Custom annotation tools
- Mobile-native applications

## Success criteria

- A clean setup can run the demo locally.
- A first-time visitor understands the main workflow without assistance.
- The project reports measured latency and FPS on documented hardware.
- The README explains the architecture and limitations.
- The public demo and repository communicate one coherent story.

## Known limitations

Model performance depends on video quality, lighting, camera angle, hardware, and object classes. The system is a demonstration and must not be used as the sole basis for safety-critical decisions.

