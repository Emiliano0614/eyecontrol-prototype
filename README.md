# EyeControl Prototype — Eye-Tracking Fundamentals

A from-scratch computer vision pipeline for hands-free computer control using only a webcam: face/iris tracking, gaze-based zone classification, blink detection, and a binary drill-tree for text input — all built and validated stage by stage.

This is the proving ground for [**EyeControl**](https://github.com/Emiliano0614/EyeControl), a full rewrite that turns this into a real hands-free browser-control tool (Chrome extension + Python/WebSocket backend). That project is currently in progress.

## What it does

Using nothing but a laptop webcam, this pipeline can:
- Track a user's iris position in real time via MediaPipe FaceLandmarker
- Classify where on the screen the user is looking (left / right zone) from gaze position alone
- Detect intentional blinks (filtering out natural blinking noise)
- Drive a binary decision tree via gaze + blink to select menu options or type characters — no keyboard, mouse, or touch required

## Why this exists

Most "eye tracking" demos rely on a pretrained model or a commercial SDK. This project builds the gaze-to-screen-position mapping from raw iris landmarks, derives the math by hand, and validates every design decision (zone count, thresholds, timing) against real recorded data rather than guesswork.

## Stages

| Stage | What it proves |
|---|---|
| `01-webcam-basics/` | Webcam capture pipeline works end-to-end |
| `02-face-mesh/` | MediaPipe FaceLandmarker (Tasks API) extracts facial/iris landmarks live |
| `03-calibration-math/` | Derives the gaze-ratio formula `t = (P - A) / (B - A)` from iris position, builds a calibration sweep + tkinter target UI, and classifies gaze into screen zones via nearest-centroid matching |
| `04-blink-detection/` | Detects intentional blinks using an eye-openness threshold with a reopen-gate to prevent false triggers |
| `05-cursor-control/` | Combines gaze zone + blink into a working binary drill-tree — walks a decision tree via blinks to select menu items or type characters |

## Key technical results

- **Gaze-ratio formula derived from scratch**: iris position is normalized against known screen-edge references, not black-boxed by a library.
- **Nearest-centroid zone classification, not regression.** Linear/polynomial regression (`lstsq`) was tried first and abandoned after root-causing the failure: within-target measurement noise (~0.04) is nearly as large as the full-screen signal range (~0.06–0.15), making any regression fundamentally unstable regardless of model complexity. Nearest-centroid matching against calibration-derived zone averages proved far more robust.
- **Validated on held-out data, not training data.** `solve_mapping.py` trains zone centroids on 4 calibration sweeps and evaluates against a separate held-out sweep collected in its own session — a genuine generalization test, achieving a 60% hit rate on 2-zone classification.
- **Blink threshold measured, not assumed.** `BLINK_THRESHOLD` was set from the user's actual closed-eye measurement floor (~0.12–0.13) rather than a guessed constant, with a reopen-gate (`REOPEN_FRAMES_REQUIRED`) added after live testing revealed noisy semi-closed-eye stretches were triggering multiple false blinks per intended one.

## Running it

Each stage is self-contained. Example — running the calibration stage:

```bash
cd 03-calibration-math
python calibration_test.py
```

This launches a tkinter window that walks through calibration targets, records gaze data, and (via `solve_mapping.py`) evaluates zone-classification accuracy against held-out test data.

**Requirements:** see `requirements.txt`. Uses MediaPipe (Tasks API), OpenCV, tkinter, and numpy.

## What's next

This prototype's core logic — gaze math, zone classification, blink detection, drill-tree navigation — is being carried into [**EyeControl**](https://github.com/Emiliano0614/EyeControl), a full rewrite as a two-component system:
- **Python backend**: MediaPipe eye-tracking + all decision logic, communicating over a local WebSocket
- **Chrome extension**: scans the active page, numbers links, and executes navigation/scroll commands from the backend

Head-pitch will drive continuous scrolling; gaze + blink will drive link selection and menu navigation — bringing this from a proof-of-concept into a real hands-free browsing tool.