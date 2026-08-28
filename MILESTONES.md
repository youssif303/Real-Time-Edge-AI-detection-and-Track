# Real-Time Edge AI Perception Platform

## Project outcome

Build a software-only, portfolio-ready edge AI perception platform with an interactive UI, reproducible demos, clear documentation, and a public launch story.

## Milestones

### M0 - Product Definition

- Write the one-sentence problem statement.
- Choose the first perception task: object detection, tracking, segmentation, or anomaly detection.
- Define the demo input and output.
- Record success metrics and the target device/runtime.

**Exit criteria:** someone can understand the product and its first demo in under one minute.

### M1 - Working Perception Pipeline

- Create the repository and development setup.
- Implement input ingestion and inference.
- Add a small, reproducible sample dataset or video.
- Expose latency, throughput, and confidence results.
- Add basic tests and a reproducible run command.

**Exit criteria:** the pipeline runs locally from a clean setup and produces a trustworthy result.

### M2 - Interactive UI

- Build a focused dashboard around the main demo.
- Add live or uploaded input.
- Show detections/tracks/alerts on the visual output.
- Show performance metrics and model settings.
- Add clear loading, empty, error, and reset states.

**Exit criteria:** a first-time visitor can operate the demo without instructions.

### M3 - Evidence and Reliability

- Add benchmark results and a comparison baseline.
- Document hardware, model, dataset, and limitations.
- Add logging and graceful failure handling.
- Add tests for the main user flow.
- Capture screenshots and a short screen recording.

**Exit criteria:** claims are supported by visible evidence and the demo fails safely.

### M4 - Deployment and Public Repo

- Deploy the UI and demo endpoint where practical.
- Add environment configuration and setup instructions.
- Write a concise README with architecture, quick start, and demo link.
- Add screenshots, a system diagram, and known limitations.
- Clean secrets, generated files, and unnecessary large assets from Git.

**Exit criteria:** a reviewer can clone, run, understand, and try the project.

### M5 - Launch and LinkedIn

- Prepare a 30-60 second demo video or GIF.
- Write a LinkedIn post with the problem, result, technical choices, and links.
- Publish the repository and deployment together.
- Ask for targeted feedback from engineers or hiring managers.
- Track follow-up improvements as versioned issues.

**Exit criteria:** the project is publicly discoverable and communicates both technical depth and product value.

## Recommended order of work

1. Finish one narrow end-to-end demo before adding features.
2. Measure performance before polishing the UI.
3. Make the UI tell the story of the pipeline, not just display charts.
4. Document limitations honestly; this increases credibility.
5. Launch with one strong workflow instead of several unfinished ones.

## Definition of done

- The demo works from a clean setup.
- The UI is usable on desktop and mobile widths.
- README and architecture documentation are complete.
- Performance and limitations are stated with evidence.
- The repository and live demo links are ready for the LinkedIn post.
