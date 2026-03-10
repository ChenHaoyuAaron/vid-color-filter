# vid-color-filter

Filter video editing pairs by detecting color contamination in unedited regions.

When video editing tools perform local edits (object add/remove/modify/replace), they sometimes introduce unwanted color shifts in regions that should remain unchanged. This tool scores and filters video editing pairs based on perceptual color difference (CIEDE2000) in unedited regions, ensuring high-quality training data for video editing models.

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
  3. CIEDE2000 on Unedited Regions (mean per frame)
        |
        v
  4. Video-level Score (max of per-frame means)
        |
        v
  5. Pass/Fail (threshold = 2.0 dE00)
```

**Key idea:** Automatically detect edited regions via pixel-level differences, then measure color contamination only in the remaining unedited areas using CIEDE2000 perceptual color difference.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Usage

### CLI

**From CSV (recommended):**

```bash
vid-color-filter \
  --csv pairs.csv \
  --output results.jsonl \
  --workers 32
```

Where `pairs.csv` has columns `video1_path` (original) and `video2_path` (edited):

```csv
video1_path,video2_path
/data/original/clip_001.mp4,/data/edited/clip_001.mp4
/data/original/clip_002.mp4,/data/edited/clip_002.mp4
```

**From directories:**

```bash
vid-color-filter \
  --src-dir /path/to/original/videos \
  --edited-dir /path/to/edited/videos \
  --output results.jsonl \
  --workers 32
```

Videos are matched by filename — `clip_001.mp4` in `--src-dir` pairs with `clip_001.mp4` in `--edited-dir`.

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--csv` | | CSV file with `video1_path` and `video2_path` columns |
| `--src-dir` | | Directory containing original videos (alternative to --csv) |
| `--edited-dir` | | Directory containing edited videos (use with --src-dir) |
| `--output` | (required) | Output JSONL file path |
| `--pattern` | `*.mp4` | Glob pattern for video files (used with --src-dir) |
| `--num-frames` | `16` | Number of frames to sample per video |
| `--threshold` | `2.0` | CIEDE2000 threshold for pass/fail |
| `--workers` | `8` | Number of parallel workers |

### Python API

```python
from vid_color_filter.scorer import score_video_pair

result = score_video_pair("original.mp4", "edited.mp4", threshold=2.0)

print(result["max_mean_delta_e"])  # video-level score
print(result["pass"])              # True if below threshold
print(result["mask_coverage_ratio"])  # fraction of frame detected as edited
```

### Batch Processing

```python
from vid_color_filter.cli import run_batch

pairs = [("src_0.mp4", "edited_0.mp4"), ("src_1.mp4", "edited_1.mp4")]
run_batch(pairs, "results.jsonl", num_workers=8)
```

## Output Format

Each line in the JSONL output contains:

```json
{
  "video_pair_id": "src_0",
  "mean_delta_e_per_frame": [0.12, 0.15, 0.11, ...],
  "max_mean_delta_e": 0.15,
  "pass": true,
  "mask_coverage_ratio": 0.23
}
```

| Field | Description |
|---|---|
| `video_pair_id` | Derived from source video filename |
| `mean_delta_e_per_frame` | Mean CIEDE2000 of unedited pixels per sampled frame |
| `max_mean_delta_e` | Worst-case frame score (used for pass/fail) |
| `pass` | `true` if `max_mean_delta_e < threshold` |
| `mask_coverage_ratio` | Max fraction of frame area detected as edited |

## CIEDE2000 Reference

| dE00 | Interpretation |
|---|---|
| < 1.0 | Indistinguishable to human eye |
| 1.0 - 2.0 | Barely noticeable under close inspection |
| > 2.0 | Clearly visible difference |

The default threshold of 2.0 targets "not noticeable during normal viewing."

## Performance

- ~1-2 seconds per video pair (short videos, <720p)
- ~60,000 pairs/hour on a 32-core machine
- 100K+ pairs in ~2 hours on a single machine
- CPU-bound (no GPU required for scoring)

## Tests

```bash
pytest tests/ -v
```

## License

MIT
