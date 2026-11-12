# Tennis Pose Extraction, Stroke Classification & Shot Recommendation

An end-to-end computer vision and deep learning pipeline for automated **tennis pose extraction**, **temporal stroke detection**, **multi-task stroke & posture analysis**, and **shot selection recommendation** using **MediaPipe Pose Landmarker** and a **PyTorch Temporal Transformer**.

---

## Overview

This repository contains the research and implementation notebook (`improved-tennis-pose-extraction-research.ipynb`) covering the full workflow from raw court-level practice videos of professional and amateur players to annotated rally analysis videos:

1. **Phase 1 — 3D Pose & Keypoint Extraction**
   - Uses **MediaPipe Pose Landmarker (Full model, `RunningMode.VIDEO`)** with temporal smoothing across frames.
   - Extracts 13 key anatomical landmarks (`nose`, `left/right shoulder`, `left/right elbow`, `left/right wrist`, `left/right hip`, `left/right knee`, `left/right ankle`) in both normalized image coordinates (`x, y, z, visibility`) and metric 3D hip-centered world coordinates (`wx, wy, wz`).
   - Processes 11 practice videos across multiple players (*Alcaraz, Federer, Djokovic, Nadal, Sinner, Wawrinka, Zverev*, and amateur recordings), producing **172,611 frame records** (`keypoints.csv`).

2. **Phase 2 — Velocity-Based Stroke Detection**
   - Computes frame-to-frame 3D world velocity across wrists, elbows, and shoulders to calculate a weighted, rolling-smoothed `stroke_score`:
     $$\text{Stroke Score} = 0.5 \cdot v_{\text{wrist}} + 0.3 \cdot v_{\text{elbow}} + 0.2 \cdot v_{\text{shoulder}}$$
   - Detects local maxima above player-calibrated thresholds with minimum frame gap constraints, isolating **1,281 stroke windows** (`stroke_windows.csv`).

3. **Phase 3 & 4 — Biomechanical Feature Engineering & Sequence Extraction**
   - Clips or resamples each stroke window to a fixed 30-frame temporal sequence (`seq_len = 30`), focusing on the preparation/pre-contact phase.
   - Applies per-window z-score normalization and computes **11 derived biomechanical features** per frame alongside the **39 base 3D world coordinates** (total `input_dim = 50`):
     - Left & right elbow angles (shoulder–elbow–wrist)
     - Left & right shoulder angles (hip–shoulder–elbow)
     - Left & right knee bend angles (hip–knee–ankle)
     - Shoulder–hip trunk rotation (X-Factor yaw in the XZ plane)
     - Forward trunk lean angle
     - Lateral stance width (hip spread)
     - Left & right wrist height relative to shoulder midpoint

4. **Phase 5 — Multi-Task Temporal Transformer (`TennisTransformerGPU`)**
   - Architecture: Linear input projection $\rightarrow$ learnable positional embedding $\rightarrow$ 4-layer `TransformerEncoder` (`d_model=128`, `nhead=4`, `dim_feedforward=256`, `dropout=0.3`) $\rightarrow$ global average pooling $\rightarrow$ three task-specific classification heads:
     - **Shot Direction Head** (`center`, `cross_court`, `down_the_line`)
     - **Stroke Type Head** (`forehand`, `backhand`)
     - **Posture Quality Head** (`good`, `bad` — trained with class-weighted cross-entropy loss)
   - Evaluated under two protocols:
     - **Random 80/20 Stratified Split** (`model_random_split.pt`) — ~83.3% stroke type accuracy, ~63.8% posture accuracy, ~61.1% direction accuracy.
     - **Cross-Player Generalization Split** (`model_cross_player.pt`, trained on *Alcaraz + Federer*, tested on unseen amateur player) — ~84.1% stroke type accuracy, ~68.3% direction accuracy, ~64.6% posture accuracy.

5. **Phase 6 — Unseen Video Inference & Annotated Dashboard Rendering**
   - End-to-end rally analysis on unseen court-level videos (e.g., *Andy Murray*, *Dominic Thiem*).
   - Renders side-by-side biomechanical telemetry dashboards (knee angle, elbow angle, X-Factor, shot direction, technique quality, and rule-based shot selection recommendation) as well as full annotated `.mp4` video overlays with real-time skeletal tracking.
   - Supports frame-by-frame visualization of detected poses and predicted tennis stroke categories.

---

## Repository Contents

| File | Description |
| :--- | :--- |
| [`improved-tennis-pose-extraction-research.ipynb`](./improved-tennis-pose-extraction-research.ipynb) | Complete Jupyter/Kaggle research notebook covering pose extraction, stroke segmentation, biomechanical feature extraction, Temporal Transformer training, and video inference. |
| [`README.md`](./README.md) | Project documentation and usage guide. |

---

## Requirements & Installation

The notebook is designed to run on **Kaggle Notebooks (GPU T4)**, **Google Colab**, or a local Python 3.10+ environment with CUDA support.

```bash
pip install mediapipe==0.10.13 opencv-python numpy pandas scipy scikit-learn matplotlib torch torchvision
```

Download the MediaPipe Pose Landmarker Full model asset required for Phase 1 & video inference:

```bash
wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

---

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/Prince1941/tennis.git
   cd tennis
   ```
2. Open `improved-tennis-pose-extraction-research.ipynb` in Jupyter Lab, VS Code, Google Colab, or Kaggle.
3. Update `VIDEO_DIR` and `OUTPUT_DIR` in **Phase 1** to point to your local or Kaggle dataset directory containing `.mp4` tennis videos.
4. Run the cells sequentially to:
   - Extract 3D pose landmarks (`keypoints.csv`)
   - Segment strokes (`stroke_windows.csv`)
   - Generate normalized 30-frame sequences (`X_sequences_new.npy`)
   - Train the multi-task Temporal Transformer (`model_random_split.pt`, `model_cross_player.pt`)
   - Generate annotated rally inference dashboards and output videos (`output_annotated.mp4`).
