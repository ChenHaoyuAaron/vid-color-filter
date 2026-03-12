# vid-color-filter

Filter video editing pairs by detecting color contamination in unedited regions.

When video editing tools perform local edits (object add/remove/modify/replace), they sometimes introduce unwanted color shifts in regions that should remain unchanged. This tool scores and filters video editing pairs based on perceptual color difference in unedited regions, ensuring high-quality training data for video editing models.

Supports both CPU (multiprocessing) and **GPU-accelerated** (PyTorch + torchrun) modes.

## How It Works

```
Original video + Edited video
        |
        v
  1. Frame Sampling (uniform, 8-16 frames)
        |
        v
  2. Auto Edit Mask (frame diff in Lab space -> binarize -> dilate)
        |
        v
  3. Color Diff on Unedited Regions (CIE76 / CIE94 / CIEDE2000)
        |
        v
  4. Video-level Score (max of per-frame means)
        |
        v
  5. Pass/Fail (threshold = 2.0 dE)
```

**Key idea:** Automatically detect edited regions via pixel-level differences, then measure color contamination only in the remaining unedited areas using perceptual color difference metrics.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+ and PyTorch 2.0+.

## Usage

### GPU Mode (recommended for large-scale processing)

GPU mode uses PyTorch for batched color computation and `torchrun` for multi-GPU distribution.

**Single GPU:**

```bash
python run.py \
  --csv pairs.csv \
  --output results.jsonl \
  --metric cie94
```

**Multi-GPU on one node:**

```bash
torchrun --nproc_per_node=4 run.py \
  --csv pairs.csv \
  --output results.jsonl \
  --metric cie94
```

**Multi-node:**

```bash
torchrun --nnodes=2 --nproc_per_node=4 \
  --rdzv_backend=c10d --rdzv_endpoint=HOST:PORT \
  run.py \
  --csv pairs.csv \
  --output results.jsonl
```

**From directories (GPU):**

```bash
torchrun --nproc_per_node=4 run.py \
  --src-dir /path/to/original/videos \
  --edited-dir /path/to/edited/videos \
  --output results.jsonl
```

**GPU mode arguments:**

| Argument | Default | Description |
|---|---|---|
| `--csv` | | CSV file with `video1_path` and `video2_path` columns |
| `--src-dir` | | Directory containing original videos (alternative to --csv) |
| `--edited-dir` | | Directory containing edited videos (use with --src-dir) |
| `--root-dir` | `""` | Root directory to prepend to relative paths in CSV |
| `--output` | (required) | Output JSONL file path |
| `--pattern` | `*.mp4` | Glob pattern for video files (used with --src-dir) |
| `--num-frames` | `16` | Number of frames to sample per video |
| `--threshold` | `2.0` | Delta E threshold for pass/fail |
| `--metric` | `cie94` | Color metric: `cie76`, `cie94`, or `ciede2000` |
| `--diff-threshold` | `5.0` | Lab distance threshold for mask binarization |
| `--dilate-kernel` | `21` | Dilation kernel size for edit mask |

### CPU Mode (legacy)

The original CPU-based CLI using multiprocessing:

```bash
vid-color-filter \
  --csv pairs.csv \
  --output results.jsonl \
  --workers 32
```

Or from directories:

```bash
vid-color-filter \
  --src-dir /path/to/original/videos \
  --edited-dir /path/to/edited/videos \
  --output results.jsonl \
  --workers 32
```

Where `pairs.csv` has columns `video1_path` (original) and `video2_path` (edited):

```csv
video1_path,video2_path
/data/original/clip_001.mp4,/data/edited/clip_001.mp4
/data/original/clip_002.mp4,/data/edited/clip_002.mp4
```

Videos are matched by filename -- `clip_001.mp4` in `--src-dir` pairs with `clip_001.mp4` in `--edited-dir`.

### Python API

**GPU (batched):**

```python
import torch
from vid_color_filter.frame_sampler import sample_frames_as_tensors
from vid_color_filter.gpu.batch_scorer import score_video_pair_gpu

src_t, edited_t = sample_frames_as_tensors("original.mp4", "edited.mp4", device="cuda")

with torch.no_grad():
    result = score_video_pair_gpu(src_t, edited_t, src_path="original.mp4", metric="cie94")

print(result["max_mean_delta_e"])     # video-level score
print(result["pass"])                 # True if below threshold
print(result["mask_coverage_ratio"])  # fraction of frame detected as edited
```

**CPU (original):**

```python
from vid_color_filter.scorer import score_video_pair

result = score_video_pair("original.mp4", "edited.mp4", threshold=2.0)
```

## Color Metrics

Three perceptual color difference metrics are available via `--metric`:

| Metric | Speed | Accuracy | Description |
|---|---|---|---|
| `cie76` | Fastest | Good | Lab Euclidean distance. Sufficient for detecting large color shifts. |
| `cie94` | Fast | Better | Weighted CIE76 with chroma/hue corrections. **Recommended default.** |
| `ciede2000` | Slowest | Best | Full CIEDE2000 formula. Industry gold standard for perceptual difference. |

For detecting color contamination with thresholds around dE 2.0, CIE94 provides the best balance of speed and accuracy. Use CIE76 for maximum throughput or CIEDE2000 when precision matters.

### Delta E Reference

| dE | Interpretation |
|---|---|
| < 1.0 | Indistinguishable to human eye |
| 1.0 - 2.0 | Barely noticeable under close inspection |
| > 2.0 | Clearly visible difference |

The default threshold of 2.0 targets "not noticeable during normal viewing."

## Output Format

Each line in the JSONL output contains:

```json
{
  "video_pair_id": "clip_001",
  "mean_delta_e_per_frame": [0.12, 0.15, 0.11, ...],
  "max_mean_delta_e": 0.15,
  "pass": true,
  "mask_coverage_ratio": 0.23
}
```

| Field | Description |
|---|---|
| `video_pair_id` | Derived from source video filename |
| `mean_delta_e_per_frame` | Mean delta E of unedited pixels per sampled frame |
| `max_mean_delta_e` | Worst-case frame score (used for pass/fail) |
| `pass` | `true` if `max_mean_delta_e < threshold` |
| `mask_coverage_ratio` | Max fraction of frame area detected as edited |

## Performance

**GPU mode (PyTorch + torchrun):**
- Batched frame processing: all sampled frames processed in one GPU pass
- RGB-to-Lab conversion, mask generation, and color diff all run on GPU
- Multi-GPU linear scaling via `torchrun --nproc_per_node=N`
- Estimated ~100-500x throughput improvement over CPU mode per GPU

**CPU mode (multiprocessing):**
- ~1-2 seconds per video pair (short videos, <720p)
- ~60,000 pairs/hour on a 32-core machine
- 100K+ pairs in ~2 hours on a single machine

## Project Structure

```
run.py                                 # GPU torchrun entry point
src/vid_color_filter/
    cli.py                             # CPU CLI entry point
    scorer.py                          # CPU scoring pipeline
    frame_sampler.py                   # Frame sampling (numpy + tensor output)
    color_diff.py                      # CPU CIEDE2000 (skimage)
    mask_generator.py                  # CPU mask generation (skimage + OpenCV)
    distributed.py                     # torch.distributed utilities
    gpu/
        color_space.py                 # GPU RGB-to-Lab (PyTorch tensor ops)
        color_metrics.py               # GPU CIE76 / CIE94 / CIEDE2000
        mask_generator.py              # GPU mask generation (max_pool2d dilation)
        batch_scorer.py                # GPU batched scoring pipeline
```

## Tests

```bash
pytest tests/ -v
```

33 tests covering both CPU and GPU paths, including cross-validation of GPU color metrics against skimage reference implementations.

## License

MIT
