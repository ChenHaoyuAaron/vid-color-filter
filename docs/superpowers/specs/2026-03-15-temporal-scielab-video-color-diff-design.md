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

**Algorithm** (following Zhang & Wandell, 1997):

1. Convert RGB to XYZ (requires adding `rgb_to_xyz` to `color_space.py`)
2. Convert XYZ to Poirson-Wandell opponent color channels (O1, O2, O3) using the transformation matrix:
   ```
   O1 =  0.9795 X + 1.5318 Y + 0.1225 Z   (achromatic)
   O2 = -0.1071 X + 0.3122 Y + 0.0215 Z   (red-green)
   O3 =  0.0383 X + 0.0023 Y + 0.5765 Z   (blue-yellow)
   ```
3. Apply Contrast Sensitivity Function (CSF) kernels via convolution to each opponent channel independently. Each CSF is a weighted sum of 2-3 Gaussians (spatial domain):
   - **O1 (achromatic)**: `CSF(r) = 0.921 * G(r, σ₁) + 0.105 * G(r, σ₂) - 0.026 * G(r, σ₃)` where σ₁=0.0283°, σ₂=0.133°, σ₃=4.336° (in visual degrees, converted to pixels via `pixels_per_degree`)
   - **O2 (red-green)**: `CSF(r) = 0.531 * G(r, σ₁) + 0.330 * G(r, σ₂)` where σ₁=0.0392°, σ₂=0.494°
   - **O3 (blue-yellow)**: `CSF(r) = 0.488 * G(r, σ₁) + 0.371 * G(r, σ₂)` where σ₁=0.0536°, σ₂=0.386°
   - Kernel size: truncate at 3σ of the largest Gaussian component, must be odd
   - All kernels are normalized to sum to 1.0
4. Convert filtered opponent channels back to XYZ (inverse of the matrix above)
5. Convert XYZ to Lab
6. Compute CIEDE2000 ΔE on the filtered Lab images

**Key parameter**: `--pixels-per-degree` (default: 60, corresponding to desktop monitor at ~60cm viewing distance). Visual degree σ values are converted to pixel σ values by multiplying with this parameter.

**GPU implementation**: Pre-compute CSF kernels once per resolution/viewing-distance combination, cache as tensors. Apply via `F.conv2d` with padding='same'. Three 2D convolutions per frame, minimal overhead.

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

**GPU implementation**: `torch.histc` for histogram computation, Otsu threshold via inter-class variance maximization. Hysteresis via iterative `F.max_pool2d` on seed mask with low-threshold gate — iterate until convergence (no new pixels added) with a maximum of 50 iterations to bound computation.

### Module 3: Temporal Aggregation

**Purpose**: Distinguish temporally stable color differences (real) from frame-to-frame fluctuations (codec noise).

For each pixel position (x, y) across all N sampled frames:

- **Pixel masking policy**: A pixel is included in temporal aggregation only if it is unmasked in at least 50% of sampled frames. The temporal median/IQR are computed only over the unmasked subset of frames for that pixel. Pixels masked in >50% of frames are excluded from scoring.
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

**GPU implementation**: `torch.median` and `torch.quantile` along the frame dimension. Requires all N frames' ΔE maps in memory simultaneously (shape: `(N, H, W)`). For pixels with varying mask status across frames, use `torch.nan_median`/`torch.nanquantile` with masked values set to NaN.

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
| `--diff-threshold` | `None` | Edit mask threshold. `None` = Otsu adaptive (new default). If a float is provided, use as fixed threshold (legacy behavior). |
| `--dilate-kernel` | 21 | Mask dilation kernel size (unchanged) |
| `--chunk-size` | 8 | Frames per GPU processing chunk. Lower values reduce memory usage for high-resolution video. |

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

**Memory budget** (1080p, 32 frames, float32):

| Tensor | Shape | Size |
|---|---|---|
| Source + Edited Lab frames | 2 × 32 × 1080 × 1920 × 3 | ~1.5 GB |
| S-CIELAB filtered Lab (both) | 2 × 32 × 1080 × 1920 × 3 | ~1.5 GB |
| Per-pixel ΔE maps | 32 × 1080 × 1920 | ~250 MB |
| Masks (float during dilation) | 32 × 1080 × 1920 | ~250 MB |
| **Peak total** | | **~3.5 GB** |

For 4K (2160×3840), peak memory quadruples to ~14 GB. To support large resolutions on limited GPU memory, the pipeline should support **frame-chunked processing**: process frames in chunks of `--chunk-size` (default: 8), accumulating per-pixel ΔE maps incrementally. This trades throughput for memory. Maximum supported resolution without chunking: 1080p on 8 GB GPU, 4K on 24 GB GPU.

### Backward Compatibility

**Breaking changes** (documented):
- `--num-frames` default changes from 16 → 32 (doubles decode time and memory). Users can pass `--num-frames 16` for old behavior.
- `--metric` default changes from `cie94` → `ciede2000` (more accurate but slower). Users can pass `--metric cie94` for old behavior.
- `--diff-threshold` default changes from `5.0` → `None` (Otsu adaptive). Users can pass `--diff-threshold 5.0` for old behavior.
- Output field `max_mean_delta_e` is replaced by `global_shift_score`. The old field name is retained as an alias pointing to `global_shift_score` for one release cycle.

**Preserved**:
- Old CIE76/CIE94 metric options remain available via `--metric`
- `--threshold` still works as alias for `--global-threshold`
- `per_frame_mean_delta_e` and `mean_delta_e_per_frame` both retained in output (same data, two field names)
- `mask_coverage_ratio` retained
- CPU pipeline (`cli.py`) unchanged; new features in GPU pipeline only

### Files to Modify/Create

| File | Action | Description |
|---|---|---|
| `src/vid_color_filter/gpu/color_space.py` | Modify | Add `rgb_to_xyz` function needed for S-CIELAB opponent conversion |
| `src/vid_color_filter/gpu/color_metrics.py` | Modify | Add per-pixel ΔE mode (return `(B, H, W)` maps instead of `(B,)` means) for temporal aggregation |
| `src/vid_color_filter/gpu/scielab.py` | Create | S-CIELAB spatial filtering (CSF kernels, Poirson-Wandell opponent color conversion) |
| `src/vid_color_filter/gpu/adaptive_mask.py` | Create | Otsu + hysteresis adaptive mask generation |
| `src/vid_color_filter/gpu/temporal_aggregator.py` | Create | Temporal median/IQR aggregation and multi-dimensional scoring |
| `src/vid_color_filter/gpu/batch_scorer.py` | Modify | Integrate new modules into scoring pipeline |
| `run.py` | Modify | Add new CLI parameters |
| `tests/test_scielab.py` | Create | S-CIELAB correctness tests |
| `tests/test_adaptive_mask.py` | Create | Adaptive mask tests |
| `tests/test_temporal_aggregator.py` | Create | Temporal aggregation tests |
| `tests/test_gpu_scorer.py` | Modify | Update end-to-end tests for new output format |
