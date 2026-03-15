# Temporal-Aware S-CIELAB Video Color Difference Analysis

## Problem Statement

The current video color difference analysis method applies traditional image-level color difference analysis on sampled frames independently:

1. Sample N frames from each video pair
2. Compute per-pixel CIEDE2000 ΔE on unedited regions per frame
3. Take mean ΔE per frame, then max across frames as the video score
4. Pass/fail based on a single threshold

This approach has critical shortcomings for video:

- **Mean aggregation loses spatial distribution**: Large areas of low ΔE mask small regions of high ΔE. Conversely, uniform codec-induced micro-shifts inflate the mean.
- **Max-of-means is outlier-sensitive**: A single anomalous frame (dark scene, scene transition, high-compression area) can fail an otherwise good video.
- **No temporal signal utilization**: True color differences (e.g., grading shifts) are temporally stable. Codec noise fluctuates randomly across frames. This distinction is completely ignored.
- **Fixed mask threshold**: The edit region mask uses a hardcoded Lab distance threshold (5.0), which may incorrectly exclude color-shifted regions.

Experimental results confirm that many video pairs with no perceptible color difference fail under the current method.

## Design

### Pipeline Overview

```
Frame Sampling (configurable frame count)
  → RGB → Lab conversion
  → Edit Region Mask (adaptive threshold)
  → S-CIELAB Spatial Filtering (CSF convolution, simulates human spatial perception)
  → CIEDE2000 ΔE (on filtered Lab images)
  → Temporal Aggregation (median/IQR per pixel, separates true color diff from codec noise)
  → Spatial Distribution Analysis (global shift vs local anomaly)
  → Multi-dimensional Scoring + Pass/Fail
```

### Module 1: S-CIELAB Spatial Filtering

**Purpose**: Filter out spatially isolated pixel-level color differences that are imperceptible to the human visual system, before computing ΔE.

**How it works**:

1. Convert Lab image to opponent color channels: L (luminance), A (red-green), B (blue-yellow)
2. Apply Contrast Sensitivity Function (CSF) kernels via convolution to each channel independently:
   - **L channel**: Highest spatial frequency cutoff (human vision is most sensitive to luminance detail)
   - **A channel (red-green)**: Lower cutoff
   - **B channel (blue-yellow)**: Lowest cutoff (human vision is least sensitive to blue-yellow spatial variation)
3. Convert filtered opponent channels back to Lab space
4. Compute CIEDE2000 ΔE on the filtered Lab images

**Key parameter**: `--pixels-per-degree` (default: 60, corresponding to desktop monitor at ~60cm viewing distance). This controls the CSF kernel sizes — higher values mean smaller kernels (finer perception).

**GPU implementation**: Pre-compute CSF kernels once, apply via `F.conv2d`. Three separable convolutions per frame, minimal overhead.

**Impact**: Codec-induced random per-pixel color noise is smoothed away. Only spatially coherent color differences (perceptible to the human eye) remain in the ΔE map.

### Module 2: Adaptive Edit Region Mask

**Current approach**: Fixed Lab distance threshold of 5.0 + morphological dilation with fixed kernel.

**Problems**: A fixed threshold may exclude legitimate color-shifted regions (marking them as "edits") or miss subtle edits.

**New approach**:

1. Compute per-pixel Lab Euclidean distance between source and edited frames (same as current)
2. **Otsu adaptive thresholding**: Automatically determine the binarization threshold from the distance histogram, adapting to each frame's content
3. **Hysteresis thresholding** (dual threshold):
   - High threshold (from Otsu): seed regions — definitely edited
   - Low threshold (Otsu × 0.5): expansion — connected regions above this are also marked as edited
   - Similar to Canny edge detection logic
4. Morphological dilation with existing kernel as safety margin

**GPU implementation**: `torch.histc` for histogram computation, Otsu threshold via inter-class variance maximization. Hysteresis via iterative `F.max_pool2d` on seed mask with low-threshold gate.

### Module 3: Temporal Aggregation

**Purpose**: Distinguish temporally stable color differences (real) from frame-to-frame fluctuations (codec noise).

For each unmasked pixel position (x, y) across all N sampled frames:

- **Temporal median ΔE(x, y)**: The stable color difference signal at this position. Robust to codec-induced frame-to-frame fluctuation.
- **Temporal IQR(x, y)**: Inter-quartile range of ΔE across frames. Measures how much the color difference fluctuates.

Interpretation:

| Median | IQR | Interpretation |
|--------|-----|----------------|
| High | Low | True color difference (stable across frames) |
| Low | High | Codec noise (random fluctuation) |
| High | High | Ambiguous — flagged for review |
| Low | Low | No color difference |

**Output**: Two spatial maps — `median_map(H, W)` and `iqr_map(H, W)` — representing the stable color difference signal and its reliability.

**GPU implementation**: `torch.median` and `torch.quantile` along the frame dimension. Requires all N frames' ΔE maps in memory simultaneously (shape: `(N, H, W)`).

### Module 4: Spatial Distribution Analysis & Multi-dimensional Scoring

From the temporal median ΔE map (unmasked pixels only), compute three scores:

**1. Global Shift Score**
- `median(temporal_median_map)` — the median of the stable ΔE values across all unmasked pixels
- Represents uniform color shift across the entire frame (e.g., codec-induced overall warm/cool shift)
- Threshold: `--global-threshold` (default: 2.0 ΔE)

**2. Local Difference Score**
- `P95(temporal_median_map) - median(temporal_median_map)` — how much the worst 5% deviates from the global level
- Represents localized color anomalies above the global baseline
- Threshold: `--local-threshold` (default: 3.0 ΔE)

**3. Temporal Instability Score**
- `mean(temporal_iqr_map)` — average frame-to-frame fluctuation
- High values indicate the color differences are mostly codec noise rather than true color shifts
- Reported as metadata/confidence indicator, not used in pass/fail by default

**Pass/Fail Logic**:
```
pass_global = global_shift_score < global_threshold
pass_local = local_diff_score < local_threshold
pass = pass_global AND pass_local
```

### Output Format

```json
{
  "video_pair_id": "clip_001",
  "global_shift_score": 0.8,
  "local_diff_score": 0.3,
  "temporal_instability": 0.15,
  "pass_global": true,
  "pass_local": true,
  "pass": true,
  "mask_coverage_ratio": 0.23,
  "per_frame_mean_delta_e": [0.12, 0.15, 0.11, ...]
}
```

- `per_frame_mean_delta_e` retained for backward compatibility (computed on S-CIELAB-filtered ΔE)
- `mask_coverage_ratio` retained (max across frames)

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--num-frames` | 32 | Frames sampled per video. Increased from 16 to support temporal statistics. |
| `--pixels-per-degree` | 60 | Viewing condition for S-CIELAB. Desktop monitor at ~60cm. |
| `--global-threshold` | 2.0 | ΔE threshold for global color shift score |
| `--local-threshold` | 3.0 | ΔE threshold for local color difference score |
| `--threshold` | — | Alias for `--global-threshold` (backward compatibility) |
| `--metric` | `ciede2000` | Default changed from `cie94` to `ciede2000`. CIE76/CIE94 still available. |
| `--diff-threshold` | (auto) | Edit mask threshold. Now defaults to Otsu adaptive. Legacy fixed value still accepted. |
| `--dilate-kernel` | 21 | Mask dilation kernel size (unchanged) |

### GPU Compatibility

All new computations have native PyTorch implementations:

| Operation | Implementation |
|---|---|
| S-CIELAB CSF convolution | `F.conv2d` with pre-computed kernels |
| Otsu thresholding | `torch.histc` + argmax on inter-class variance |
| Hysteresis expansion | Iterative `F.max_pool2d` with mask gating |
| Temporal median | `torch.median` along dim=0 |
| Temporal IQR | `torch.quantile` (Q75 - Q25) along dim=0 |
| Spatial percentiles | `torch.quantile` on flattened map |

Memory requirement: `N × H × W` float32 tensor for ΔE maps (32 frames × 1080 × 1920 ≈ 250 MB per video pair). Well within GPU memory for typical cases.

### Backward Compatibility

- Old CIE76/CIE94 metric options remain available via `--metric`
- `--threshold` still works as alias for `--global-threshold`
- `per_frame_mean_delta_e` field retained in output
- CPU pipeline (`cli.py`) unchanged; new features in GPU pipeline only

### Files to Modify/Create

| File | Action | Description |
|---|---|---|
| `src/vid_color_filter/gpu/scielab.py` | Create | S-CIELAB spatial filtering (CSF kernels, opponent color conversion) |
| `src/vid_color_filter/gpu/adaptive_mask.py` | Create | Otsu + hysteresis adaptive mask generation |
| `src/vid_color_filter/gpu/temporal_aggregator.py` | Create | Temporal median/IQR aggregation and multi-dimensional scoring |
| `src/vid_color_filter/gpu/batch_scorer.py` | Modify | Integrate new modules into scoring pipeline |
| `run.py` | Modify | Add new CLI parameters |
| `tests/test_scielab.py` | Create | S-CIELAB correctness tests |
| `tests/test_adaptive_mask.py` | Create | Adaptive mask tests |
| `tests/test_temporal_aggregator.py` | Create | Temporal aggregation tests |
| `tests/test_gpu_scorer.py` | Modify | Update end-to-end tests for new output format |
