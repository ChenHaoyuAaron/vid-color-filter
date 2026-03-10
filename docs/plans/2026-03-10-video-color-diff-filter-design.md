# Video Editing Pair Color Difference Filter - Design

## Problem

Video editing processes can contaminate unedited regions, introducing color shifts between the original and edited video. We need to automatically detect and filter out video editing pairs where unedited regions have perceptible color differences.

## Scope

- **In scope**: Local edits (object add/remove/modify/replace) that pollute unedited regions
- **Out of scope**: Global color grading, background replacement (already pre-filtered)

## Input / Output

- **Input**: Original video + edited video pairs, with editing instruction text (no spatial masks)
- **Output**: Per-pair score (mean CIEDE2000) + pass/fail classification
- **Scale**: 100K+ pairs, <10s duration, <720p resolution
- **Hardware**: GPU cluster available

## Pipeline

```
Input video pair (src, edited)
        |
        v
  1. Frame Sampling
     (uniform, 8-16 frames per video)
        |
        v
  2. Auto Mask Generation
     (frame diff -> binarize -> dilate)
        |
        v
  3. Color Difference on Unedited Regions
     (RGB -> Lab -> CIEDE2000, mean per frame)
        |
        v
  4. Aggregation -> Video-level Score
     (max of per-frame means)
        |
        v
  5. Threshold Filter -> pass/fail
```

## Step Details

### 1. Frame Sampling

- Uniform sampling, 8-16 frames per video
- Align by frame index (editing should not change frame count)
- If frame counts differ, align by timestamp proportion
- No spatial alignment needed; spatial misalignment itself indicates poor edit quality

### 2. Auto Mask Generation

1. Compute per-pixel absolute difference in Lab color space (Euclidean distance across L/a/b)
2. Binarize: pixels with diff > tau_1 (CIEDE2000 ~ 2.0) marked as "changed"
3. Morphological dilation: kernel ~15-25 pixels, to buffer edit boundaries
4. Connected component filtering: remove noise regions < 100 pixels
5. If mask covers > 80% of frame, flag for manual review (likely global contamination)

### 3. Color Difference Calculation

- Metric: CIEDE2000 (industry standard perceptual color difference)
  - dE00 < 1.0: indistinguishable
  - dE00 1.0-2.0: barely noticeable
  - dE00 > 2.0: clearly visible
- For each sampled frame pair, compute mean CIEDE2000 over all unmasked pixels
- Primary indicator: **mean dE00 per frame**

### 4. Aggregation

- Video-level score = **max** of all per-frame mean dE00 values
- Rationale: ensures even the worst frame is within acceptable range

### 5. Filtering

- Threshold: mean dE00 < 2.0 = pass
- Threshold can be tuned based on manual spot-check results

## Tech Stack

- Python + NumPy + OpenCV (frame IO, mask generation)
- scikit-image `color.deltaE_ciede2000` (vectorized color diff)

## Parallelization

- Video-pair level parallelism via multiprocessing / job queue
- Pipeline is CPU-bound (no deep learning models); GPU for video decoding acceleration
- Estimated throughput: ~1-2s per pair, ~60K pairs/hour on 32-core machine
- 100K pairs completable in ~2 hours on single machine

## Output Format

CSV/JSON per video pair:
- `video_pair_id`
- `mean_delta_e_per_frame` (list of floats)
- `max_mean_delta_e` (video-level score)
- `pass` (bool)
- `mask_coverage_ratio` (metadata for analysis)
