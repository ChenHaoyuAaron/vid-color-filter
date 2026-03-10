# Video Color Difference Filter - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a pipeline that scores and filters video editing pairs based on perceptible color differences in unedited regions.

**Architecture:** A 5-stage pipeline (frame sampling → auto mask → CIEDE2000 color diff → aggregation → filtering) processing video pairs independently. No deep learning models — pure OpenCV + NumPy + scikit-image.

**Tech Stack:** Python 3.10+, NumPy, OpenCV (cv2), scikit-image, multiprocessing, argparse

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/vid_color_filter/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "vid-color-filter"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "opencv-python>=4.8",
    "scikit-image>=0.21",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[project.scripts]
vid-color-filter = "vid_color_filter.cli:main"
```

**Step 2: Create package and test directories**

```bash
mkdir -p src/vid_color_filter tests
touch src/vid_color_filter/__init__.py tests/__init__.py
```

**Step 3: Install in dev mode and verify**

```bash
pip install -e ".[dev]"
python -c "import vid_color_filter; print('OK')"
```
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "feat: scaffold project structure with dependencies"
```

---

### Task 2: Frame Sampling

**Files:**
- Create: `src/vid_color_filter/frame_sampler.py`
- Create: `tests/test_frame_sampler.py`

**Step 1: Write the failing test**

```python
# tests/test_frame_sampler.py
import numpy as np
import cv2
import tempfile
import os
import pytest
from vid_color_filter.frame_sampler import sample_frame_pairs


def _make_test_video(path: str, num_frames: int = 30, h: int = 64, w: int = 64):
    """Create a minimal test video with solid color frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
    for i in range(num_frames):
        frame = np.full((h, w, 3), fill_value=i * 8 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestSampleFramePairs:
    def test_returns_correct_number_of_pairs(self, tmp_path):
        src = str(tmp_path / "src.mp4")
        edited = str(tmp_path / "edited.mp4")
        _make_test_video(src, num_frames=30)
        _make_test_video(edited, num_frames=30)

        pairs = sample_frame_pairs(src, edited, num_frames=8)

        assert len(pairs) == 8
        for src_frame, edited_frame in pairs:
            assert src_frame.shape == (64, 64, 3)
            assert edited_frame.shape == (64, 64, 3)

    def test_handles_different_frame_counts(self, tmp_path):
        src = str(tmp_path / "src.mp4")
        edited = str(tmp_path / "edited.mp4")
        _make_test_video(src, num_frames=30)
        _make_test_video(edited, num_frames=25)

        pairs = sample_frame_pairs(src, edited, num_frames=8)

        assert len(pairs) == 8

    def test_fewer_frames_than_requested(self, tmp_path):
        src = str(tmp_path / "src.mp4")
        edited = str(tmp_path / "edited.mp4")
        _make_test_video(src, num_frames=5)
        _make_test_video(edited, num_frames=5)

        pairs = sample_frame_pairs(src, edited, num_frames=16)

        assert len(pairs) == 5
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_frame_sampler.py -v
```
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

**Step 3: Write minimal implementation**

```python
# src/vid_color_filter/frame_sampler.py
import cv2
import numpy as np


def sample_frame_pairs(
    src_path: str,
    edited_path: str,
    num_frames: int = 16,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sample aligned frame pairs from source and edited videos.

    Returns list of (src_frame, edited_frame) tuples in BGR format.
    Frames are aligned by proportional index when frame counts differ.
    """
    src_cap = cv2.VideoCapture(src_path)
    edited_cap = cv2.VideoCapture(edited_path)

    src_total = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    edited_total = int(edited_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    actual_samples = min(num_frames, src_total, edited_total)
    src_indices = np.linspace(0, src_total - 1, actual_samples, dtype=int)
    edited_indices = np.linspace(0, edited_total - 1, actual_samples, dtype=int)

    pairs = []
    for src_idx, edited_idx in zip(src_indices, edited_indices):
        src_cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_idx))
        ret_s, src_frame = src_cap.read()

        edited_cap.set(cv2.CAP_PROP_POS_FRAMES, int(edited_idx))
        ret_e, edited_frame = edited_cap.read()

        if ret_s and ret_e:
            pairs.append((src_frame, edited_frame))

    src_cap.release()
    edited_cap.release()
    return pairs
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_frame_sampler.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/vid_color_filter/frame_sampler.py tests/test_frame_sampler.py
git commit -m "feat: add frame sampling with proportional alignment"
```

---

### Task 3: Auto Mask Generation

**Files:**
- Create: `src/vid_color_filter/mask_generator.py`
- Create: `tests/test_mask_generator.py`

**Step 1: Write the failing test**

```python
# tests/test_mask_generator.py
import numpy as np
import pytest
from vid_color_filter.mask_generator import generate_edit_mask


class TestGenerateEditMask:
    def test_identical_frames_produce_empty_mask(self):
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask, coverage = generate_edit_mask(frame, frame.copy())

        assert mask.shape == (64, 64)
        assert mask.dtype == bool
        assert not mask.any()  # no edited region
        assert coverage == 0.0

    def test_detects_edited_region(self):
        src = np.full((64, 64, 3), 128, dtype=np.uint8)
        edited = src.copy()
        # Paint a 20x20 block with a very different color
        edited[20:40, 20:40] = [255, 0, 0]

        mask, coverage = generate_edit_mask(src, edited)

        # The edited block should be masked (True = edited)
        assert mask[30, 30] == True  # center of edited block
        assert coverage > 0.05  # at least some coverage

    def test_dilation_expands_mask(self):
        src = np.full((100, 100, 3), 128, dtype=np.uint8)
        edited = src.copy()
        edited[50, 50] = [255, 0, 0]  # single pixel edit

        mask, coverage = generate_edit_mask(src, edited, dilate_kernel=21)

        # Dilation should expand beyond the single pixel
        assert mask.sum() > 1

    def test_high_coverage_flagged(self):
        src = np.full((64, 64, 3), 128, dtype=np.uint8)
        edited = np.full((64, 64, 3), 200, dtype=np.uint8)  # everything changed

        mask, coverage = generate_edit_mask(src, edited)

        assert coverage > 0.8
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_mask_generator.py -v
```
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# src/vid_color_filter/mask_generator.py
import cv2
import numpy as np
from skimage.color import rgb2lab


def generate_edit_mask(
    src_frame: np.ndarray,
    edited_frame: np.ndarray,
    diff_threshold: float = 5.0,
    dilate_kernel: int = 21,
    min_component_size: int = 100,
) -> tuple[np.ndarray, float]:
    """Generate a boolean mask of edited regions by frame differencing.

    Args:
        src_frame: BGR source frame.
        edited_frame: BGR edited frame.
        diff_threshold: Lab Euclidean distance threshold for binarization.
        dilate_kernel: Dilation kernel size in pixels.
        min_component_size: Minimum connected component area to keep.

    Returns:
        (mask, coverage_ratio) where mask is bool array (True = edited region).
    """
    # BGR -> RGB -> Lab
    src_lab = rgb2lab(cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB))
    edited_lab = rgb2lab(cv2.cvtColor(edited_frame, cv2.COLOR_BGR2RGB))

    # Per-pixel Euclidean distance in Lab space
    diff = np.sqrt(np.sum((src_lab - edited_lab) ** 2, axis=2))

    # Binarize
    binary = (diff > diff_threshold).astype(np.uint8)

    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_component_size:
            binary[labels == i] = 0

    # Dilate to expand edit boundary
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel)
    )
    dilated = cv2.dilate(binary, kernel)

    mask = dilated.astype(bool)
    coverage = mask.sum() / mask.size

    return mask, coverage
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_mask_generator.py -v
```
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/vid_color_filter/mask_generator.py tests/test_mask_generator.py
git commit -m "feat: add auto edit mask generation with dilation and filtering"
```

---

### Task 4: CIEDE2000 Color Difference Calculation

**Files:**
- Create: `src/vid_color_filter/color_diff.py`
- Create: `tests/test_color_diff.py`

**Step 1: Write the failing test**

```python
# tests/test_color_diff.py
import numpy as np
import pytest
from vid_color_filter.color_diff import compute_mean_ciede2000


class TestComputeMeanCIEDE2000:
    def test_identical_frames_zero_diff(self):
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=bool)

        result = compute_mean_ciede2000(frame, frame.copy(), mask)

        assert result == pytest.approx(0.0, abs=0.01)

    def test_different_frames_nonzero_diff(self):
        src = np.full((64, 64, 3), 128, dtype=np.uint8)
        edited = np.full((64, 64, 3), 140, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=bool)

        result = compute_mean_ciede2000(src, edited, mask)

        assert result > 0.0

    def test_masked_region_excluded(self):
        src = np.full((64, 64, 3), 128, dtype=np.uint8)
        edited = src.copy()
        # Large diff only in top half
        edited[:32, :] = [255, 0, 0]
        # Mask out top half (the edited part)
        mask = np.zeros((64, 64), dtype=bool)
        mask[:32, :] = True

        result = compute_mean_ciede2000(src, edited, mask)

        # Only bottom half compared, which is identical
        assert result == pytest.approx(0.0, abs=0.01)

    def test_all_masked_returns_nan(self):
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=bool)  # everything masked

        result = compute_mean_ciede2000(frame, frame.copy(), mask)

        assert np.isnan(result)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_color_diff.py -v
```
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# src/vid_color_filter/color_diff.py
import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000


def compute_mean_ciede2000(
    src_frame: np.ndarray,
    edited_frame: np.ndarray,
    edit_mask: np.ndarray,
) -> float:
    """Compute mean CIEDE2000 over unmasked (unedited) pixels.

    Args:
        src_frame: BGR source frame.
        edited_frame: BGR edited frame.
        edit_mask: Boolean mask where True = edited region (excluded).

    Returns:
        Mean CIEDE2000 of unmasked pixels. NaN if all pixels are masked.
    """
    unmasked = ~edit_mask
    if not unmasked.any():
        return float("nan")

    src_lab = rgb2lab(cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB))
    edited_lab = rgb2lab(cv2.cvtColor(edited_frame, cv2.COLOR_BGR2RGB))

    de = deltaE_ciede2000(src_lab[unmasked], edited_lab[unmasked])

    return float(np.mean(de))
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_color_diff.py -v
```
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/vid_color_filter/color_diff.py tests/test_color_diff.py
git commit -m "feat: add CIEDE2000 color diff on unmasked regions"
```

---

### Task 5: Video-Level Scoring Pipeline

**Files:**
- Create: `src/vid_color_filter/scorer.py`
- Create: `tests/test_scorer.py`

**Step 1: Write the failing test**

```python
# tests/test_scorer.py
import numpy as np
import cv2
import pytest
from vid_color_filter.scorer import score_video_pair


class TestScoreVideoPair:
    def test_identical_videos_pass(self, tmp_path):
        """Identical videos should score ~0 and pass."""
        src = str(tmp_path / "src.mp4")
        edited = str(tmp_path / "edited.mp4")
        self._make_video(src, color=(128, 128, 128))
        self._make_video(edited, color=(128, 128, 128))

        result = score_video_pair(src, edited)

        assert result["max_mean_delta_e"] == pytest.approx(0.0, abs=0.1)
        assert result["pass"] is True

    def test_globally_shifted_video_fails(self, tmp_path):
        """A video with global color shift in all frames should fail."""
        src = str(tmp_path / "src.mp4")
        edited = str(tmp_path / "edited.mp4")
        self._make_video(src, color=(128, 128, 128))
        self._make_video(edited, color=(160, 128, 128))

        result = score_video_pair(src, edited, threshold=2.0)

        assert result["max_mean_delta_e"] > 2.0
        assert result["pass"] is False

    def test_result_has_expected_keys(self, tmp_path):
        src = str(tmp_path / "src.mp4")
        edited = str(tmp_path / "edited.mp4")
        self._make_video(src, color=(128, 128, 128))
        self._make_video(edited, color=(128, 128, 128))

        result = score_video_pair(src, edited)

        assert "video_pair_id" in result
        assert "mean_delta_e_per_frame" in result
        assert "max_mean_delta_e" in result
        assert "pass" in result
        assert "mask_coverage_ratio" in result

    @staticmethod
    def _make_video(path: str, color: tuple, num_frames: int = 10, h: int = 64, w: int = 64):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
        for _ in range(num_frames):
            frame = np.full((h, w, 3), color, dtype=np.uint8)
            writer.write(frame)
        writer.release()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_scorer.py -v
```
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# src/vid_color_filter/scorer.py
import os
import numpy as np
from vid_color_filter.frame_sampler import sample_frame_pairs
from vid_color_filter.mask_generator import generate_edit_mask
from vid_color_filter.color_diff import compute_mean_ciede2000


def score_video_pair(
    src_path: str,
    edited_path: str,
    num_frames: int = 16,
    threshold: float = 2.0,
    diff_threshold: float = 5.0,
    dilate_kernel: int = 21,
) -> dict:
    """Score a video editing pair for color contamination in unedited regions.

    Returns dict with scoring results and metadata.
    """
    pair_id = os.path.splitext(os.path.basename(src_path))[0]

    frame_pairs = sample_frame_pairs(src_path, edited_path, num_frames=num_frames)

    mean_des = []
    coverages = []

    for src_frame, edited_frame in frame_pairs:
        mask, coverage = generate_edit_mask(
            src_frame, edited_frame,
            diff_threshold=diff_threshold,
            dilate_kernel=dilate_kernel,
        )
        coverages.append(coverage)

        mean_de = compute_mean_ciede2000(src_frame, edited_frame, mask)
        mean_des.append(mean_de)

    # Filter out NaN frames (fully masked)
    valid_des = [d for d in mean_des if not np.isnan(d)]
    max_mean_de = max(valid_des) if valid_des else float("nan")

    return {
        "video_pair_id": pair_id,
        "mean_delta_e_per_frame": mean_des,
        "max_mean_delta_e": max_mean_de,
        "pass": max_mean_de < threshold if not np.isnan(max_mean_de) else False,
        "mask_coverage_ratio": max(coverages) if coverages else 0.0,
    }
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_scorer.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/vid_color_filter/scorer.py tests/test_scorer.py
git commit -m "feat: add video-level scoring pipeline"
```

---

### Task 6: CLI with Parallel Processing

**Files:**
- Create: `src/vid_color_filter/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_cli.py
import json
import csv
import numpy as np
import cv2
import pytest
from vid_color_filter.cli import run_batch


class TestRunBatch:
    def test_processes_video_pairs_and_writes_json(self, tmp_path):
        # Create two video pairs
        pairs = []
        for i in range(2):
            src = str(tmp_path / f"src_{i}.mp4")
            edited = str(tmp_path / f"edited_{i}.mp4")
            self._make_video(src, color=(128, 128, 128))
            self._make_video(edited, color=(128, 128, 128))
            pairs.append((src, edited))

        output = str(tmp_path / "results.jsonl")
        run_batch(pairs, output, num_workers=2, num_frames=4)

        with open(output) as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        for r in results:
            assert "max_mean_delta_e" in r
            assert "pass" in r

    @staticmethod
    def _make_video(path: str, color: tuple, num_frames: int = 10, h: int = 64, w: int = 64):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
        for _ in range(num_frames):
            frame = np.full((h, w, 3), color, dtype=np.uint8)
            writer.write(frame)
        writer.release()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli.py -v
```
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# src/vid_color_filter/cli.py
import argparse
import json
import os
from multiprocessing import Pool
from functools import partial
from vid_color_filter.scorer import score_video_pair


def _process_one(pair: tuple[str, str], num_frames: int, threshold: float) -> dict:
    """Process a single video pair. Designed for multiprocessing."""
    src_path, edited_path = pair
    return score_video_pair(
        src_path, edited_path,
        num_frames=num_frames,
        threshold=threshold,
    )


def run_batch(
    pairs: list[tuple[str, str]],
    output_path: str,
    num_workers: int = 8,
    num_frames: int = 16,
    threshold: float = 2.0,
) -> None:
    """Process multiple video pairs in parallel and write results to JSONL."""
    worker_fn = partial(_process_one, num_frames=num_frames, threshold=threshold)

    with Pool(num_workers) as pool, open(output_path, "w") as f:
        for result in pool.imap_unordered(worker_fn, pairs):
            f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Filter video editing pairs by color difference")
    parser.add_argument("--input-dir", required=True, help="Directory with src/edited video pairs")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--src-pattern", default="src_*.mp4", help="Glob pattern for source videos")
    parser.add_argument("--edited-pattern", default="edited_*.mp4", help="Glob pattern for edited videos")
    parser.add_argument("--num-frames", type=int, default=16, help="Frames to sample per video")
    parser.add_argument("--threshold", type=float, default=2.0, help="CIEDE2000 threshold for pass/fail")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    # Discover video pairs by matching filenames
    import glob
    src_files = sorted(glob.glob(os.path.join(args.input_dir, args.src_pattern)))
    edited_files = sorted(glob.glob(os.path.join(args.input_dir, args.edited_pattern)))

    if len(src_files) != len(edited_files):
        print(f"Warning: {len(src_files)} source videos vs {len(edited_files)} edited videos")

    pairs = list(zip(src_files, edited_files))
    print(f"Processing {len(pairs)} video pairs with {args.workers} workers...")

    run_batch(pairs, args.output, num_workers=args.workers, num_frames=args.num_frames, threshold=args.threshold)

    # Print summary
    with open(args.output) as f:
        results = [json.loads(line) for line in f]
    passed = sum(1 for r in results if r["pass"])
    print(f"Done. {passed}/{len(results)} pairs passed (threshold={args.threshold})")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_cli.py -v
```
Expected: 1 passed

**Step 5: Commit**

```bash
git add src/vid_color_filter/cli.py tests/test_cli.py
git commit -m "feat: add CLI with parallel batch processing"
```

---

### Task 7: Run All Tests & Final Verification

**Step 1: Run full test suite**

```bash
pytest tests/ -v
```
Expected: All 11 tests passed

**Step 2: Run a quick end-to-end smoke test**

```bash
python -c "
import cv2, numpy as np, tempfile, os
from vid_color_filter.scorer import score_video_pair

d = tempfile.mkdtemp()
for name, color in [('src.mp4', (128,128,128)), ('edited.mp4', (128,128,128))]:
    w = cv2.VideoWriter(os.path.join(d, name), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64,64))
    for _ in range(10):
        w.write(np.full((64,64,3), color, dtype=np.uint8))
    w.release()

r = score_video_pair(os.path.join(d,'src.mp4'), os.path.join(d,'edited.mp4'))
print(f'Score: {r[\"max_mean_delta_e\"]:.3f}, Pass: {r[\"pass\"]}')
"
```
Expected: `Score: 0.000, Pass: True`

**Step 3: Commit any final adjustments and tag**

```bash
git tag v0.1.0
```
