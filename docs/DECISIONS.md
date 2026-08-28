# Engineering Decisions

## 001 - WSL2 as the primary development environment

Decision: use Ubuntu on WSL2 for Python, AI dependencies, backend commands, and project files.

Reason: the computer vision ecosystem is strongly Linux-friendly, and keeping one command environment reduces Windows/Linux dependency drift.

## 002 - CPU-first inference

Decision: establish correctness and measurements on CPU before adding GPU support.

Reason: the project should be reproducible on more machines and does not yet have confirmed NVIDIA tooling.

## 003 - Uploaded video before webcam

Decision: implement uploaded/sample video first.

Reason: files are reproducible, easier to test, and easier for reviewers to use. Webcam support follows after the pipeline is stable.

## 004 - Separate API and UI

Decision: use FastAPI for the perception service and React for the dashboard.

Reason: this makes the pipeline independently testable and gives the public demo a clear system boundary.

