# Calibration Experiment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a calibration pipeline that runs the S-CIELAB scorer on 1000+ video pairs, generates HTML visual reports for boundary cases, supports human annotation, and finds optimal `global_threshold` / `local_threshold` via grid search + F1 evaluation.

**Architecture:** Four new modules: (1) `visualizer.py` renders PNG heatmaps/overlays from pipeline intermediates, (2) `report.py` generates self-contained HTML reports with embedded annotation JS, (3) `calibration.py` handles distribution analysis, grid search, boundary case selection, report building, and F1 evaluation, (4) `batch_scorer.py` modification to optionally retain intermediate tensors. CLI additions in `run.py` for `--visualize`/`--viz-dir`.

**Tech Stack:** matplotlib (new dependency for PNG rendering), existing PyTorch GPU pipeline, pure-Python HTML templating (no external deps).

**Spec:** `docs/superpowers/specs/2026-03-30-calibration-experiment-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add `matplotlib>=3.7` to dependencies |
| `src/vid_color_filter/gpu/visualizer.py` | Create | Render PNG visualizations (heatmaps, mask overlays, temporal maps) from GPU tensors |
| `src/vid_color_filter/report.py` | Create | Generate self-contained HTML reports (single-pair pages + index) with annotation JS |
| `src/vid_color_filter/calibration.py` | Create | Distribution analysis, grid search, boundary case filtering, build-reports, F1 evaluation |
| `src/vid_color_filter/gpu/batch_scorer.py` | Modify | `--visualize` mode: retain & return intermediate tensors (representative frames, ΔE maps, masks, temporal maps) |
| `run.py` | Modify | Add `--visualize`, `--viz-dir` CLI flags; call visualizer+report in scoring loop |
| `tests/test_visualizer.py` | Create | Tests for PNG visualization generation |
| `tests/test_report.py` | Create | Tests for HTML report generation |
| `tests/test_calibration.py` | Create | Tests for calibration analysis, grid search, F1 evaluation |

---

## Task 1: Add matplotlib dependency

**Files:**
- Modify: `pyproject.toml:5-10`

- [ ] **Step 1: Add matplotlib to dependencies**

In `pyproject.toml`, add `matplotlib>=3.7` to the `dependencies` list:

```toml
dependencies = [
    "numpy>=1.24",
    "opencv-python>=4.8",
    "scikit-image>=0.21",
    "torch>=2.0",
    "matplotlib>=3.7",
]
```

- [ ] **Step 2: Install updated dependencies**

Run: `pip install -e .`
Expected: Successfully installs matplotlib

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add matplotlib dependency for calibration visualization"
```

---

## Task 2: Visualizer — PNG rendering from GPU tensors

**Files:**
- Create: `src/vid_color_filter/gpu/visualizer.py`
- Create: `tests/test_visualizer.py`

This module takes GPU tensors (ΔE maps, masks, RGB frames, temporal maps) and saves PNG visualizations using matplotlib. All inputs are numpy arrays (caller moves from GPU). No GPU ops in this module.

### Step group A: Heatmap rendering

- [ ] **Step 1: Write failing test for `render_heatmap`**

```python
# tests/test_visualizer.py
import numpy as np
import os
import tempfile
from vid_color_filter.gpu.visualizer import render_heatmap


def test_render_heatmap_creates_png():
    """render_heatmap saves a PNG file with ΔE overlay on source frame."""
    src_frame = np.random.randint(0, 255, (100, 160, 3), dtype=np.uint8)
    de_map = np.random.rand(100, 160).astype(np.float32) * 5.0
    mask = np.zeros((100, 160), dtype=bool)
    mask[20:40, 30:50] = True  # some masked region

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "heatmap.png")
        render_heatmap(src_frame, de_map, mask, out_path, vmax=10.0)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000  # non-trivial PNG


def test_render_heatmap_respects_vmax():
    """Values above vmax are clipped in the colormap."""
    src_frame = np.zeros((50, 80, 3), dtype=np.uint8)
    de_map = np.full((50, 80), 15.0, dtype=np.float32)  # all above vmax
    mask = np.zeros((50, 80), dtype=bool)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "heatmap.png")
        render_heatmap(src_frame, de_map, mask, out_path, vmax=10.0)
        assert os.path.exists(out_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_visualizer.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `render_heatmap`**

```python
# src/vid_color_filter/gpu/visualizer.py
"""PNG visualization rendering for calibration reports."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_heatmap(
    src_frame: np.ndarray,
    de_map: np.ndarray,
    mask: np.ndarray,
    out_path: str,
    vmax: float = 10.0,
    cmap: str = "viridis",
    alpha: float = 0.6,
) -> None:
    """Render ΔE heatmap overlaid on source frame, with mask in gray.

    Args:
        src_frame: (H, W, 3) uint8 RGB source frame.
        de_map: (H, W) float32 per-pixel ΔE values.
        mask: (H, W) bool, True = masked (edit region).
        out_path: Path to save PNG.
        vmax: Colorbar max value. Values above are clipped.
        cmap: Matplotlib colormap name.
        alpha: Overlay transparency.
    """
    h, w = de_map.shape
    dpi = 100
    fig, ax = plt.subplots(1, 1, figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(src_frame)

    # Heatmap overlay
    de_display = np.clip(de_map, 0, vmax)
    im = ax.imshow(de_display, cmap=cmap, vmin=0, vmax=vmax, alpha=alpha)

    # Gray out masked regions
    if mask.any():
        mask_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        mask_overlay[mask] = [0.5, 0.5, 0.5, 0.5]
        ax.imshow(mask_overlay)

    fig.colorbar(im, ax=ax, label="ΔE", fraction=0.046, pad=0.04)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_visualizer.py -v`
Expected: PASS

### Step group B: Mask overlay and temporal maps

- [ ] **Step 5: Write failing tests for `render_mask_overlay` and `render_temporal_map`**

```python
# append to tests/test_visualizer.py
from vid_color_filter.gpu.visualizer import render_mask_overlay, render_temporal_map


def test_render_mask_overlay_creates_png():
    """render_mask_overlay saves mask overlay on source frame."""
    src_frame = np.random.randint(0, 255, (100, 160, 3), dtype=np.uint8)
    mask = np.zeros((100, 160), dtype=bool)
    mask[20:60, 40:120] = True

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "mask.png")
        render_mask_overlay(src_frame, mask, out_path, coverage=0.25)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000


def test_render_temporal_map_creates_png():
    """render_temporal_map saves a standalone heatmap (no frame underlay)."""
    temporal_map = np.random.rand(100, 160).astype(np.float32) * 3.0

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "median.png")
        render_temporal_map(temporal_map, out_path, vmax=10.0,
                           cmap="viridis", label="Temporal Median ΔE")
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_visualizer.py -v`
Expected: FAIL with ImportError

- [ ] **Step 7: Implement `render_mask_overlay` and `render_temporal_map`**

```python
# append to src/vid_color_filter/gpu/visualizer.py

def render_mask_overlay(
    src_frame: np.ndarray,
    mask: np.ndarray,
    out_path: str,
    coverage: float = 0.0,
    alpha: float = 0.4,
) -> None:
    """Render edit mask overlay on source frame.

    Args:
        src_frame: (H, W, 3) uint8 RGB.
        mask: (H, W) bool.
        out_path: Path to save PNG.
        coverage: Mask coverage ratio to annotate on image.
        alpha: Mask transparency.
    """
    h, w = mask.shape
    dpi = 100
    fig, ax = plt.subplots(1, 1, figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(src_frame)

    mask_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    mask_overlay[mask] = [1.0, 0.0, 0.0, alpha]
    ax.imshow(mask_overlay)

    ax.set_title(f"Mask Coverage: {coverage:.1%}", fontsize=10)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_temporal_map(
    temporal_map: np.ndarray,
    out_path: str,
    vmax: float = 10.0,
    cmap: str = "viridis",
    label: str = "ΔE",
) -> None:
    """Render a standalone temporal aggregation map.

    Args:
        temporal_map: (H, W) float32.
        out_path: Path to save PNG.
        vmax: Colorbar max value.
        cmap: Matplotlib colormap name.
        label: Colorbar label.
    """
    h, w = temporal_map.shape
    dpi = 100
    fig, ax = plt.subplots(1, 1, figsize=(w / dpi, h / dpi), dpi=dpi)

    display = np.clip(np.nan_to_num(temporal_map, nan=0.0), 0, vmax)
    im = ax.imshow(display, cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)

    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_visualizer.py -v`
Expected: PASS

### Step group C: Save raw frames + orchestrator function

- [ ] **Step 9: Write failing test for `save_frame` and `generate_pair_visualizations`**

```python
# append to tests/test_visualizer.py
from vid_color_filter.gpu.visualizer import save_frame, generate_pair_visualizations


def test_save_frame_creates_png():
    """save_frame saves an RGB numpy array as PNG."""
    frame = np.random.randint(0, 255, (100, 160, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "frame.png")
        save_frame(frame, out_path)
        assert os.path.exists(out_path)


def test_generate_pair_visualizations_creates_all_files():
    """generate_pair_visualizations creates full set of PNGs for a video pair."""
    n_repr = 3
    h, w = 50, 80
    data = {
        "video_pair_id": "test_clip",
        "src_frames_repr": np.random.randint(0, 255, (n_repr, h, w, 3), dtype=np.uint8),
        "edit_frames_repr": np.random.randint(0, 255, (n_repr, h, w, 3), dtype=np.uint8),
        "de_maps_repr": np.random.rand(n_repr, h, w).astype(np.float32) * 3.0,
        "masks_repr": np.zeros((n_repr, h, w), dtype=bool),
        "coverages_repr": [0.1, 0.15, 0.12],
        "median_map": np.random.rand(h, w).astype(np.float32) * 2.0,
        "iqr_map": np.random.rand(h, w).astype(np.float32) * 1.0,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        generate_pair_visualizations(data, tmpdir, vmax=10.0)
        pair_dir = os.path.join(tmpdir, "test_clip")
        assert os.path.isdir(pair_dir)
        for i in range(n_repr):
            assert os.path.exists(os.path.join(pair_dir, f"src_frame_{i}.png"))
            assert os.path.exists(os.path.join(pair_dir, f"edit_frame_{i}.png"))
            assert os.path.exists(os.path.join(pair_dir, f"heatmap_{i}.png"))
            assert os.path.exists(os.path.join(pair_dir, f"mask_{i}.png"))
        assert os.path.exists(os.path.join(pair_dir, "median_map.png"))
        assert os.path.exists(os.path.join(pair_dir, "iqr_map.png"))
```

- [ ] **Step 10: Run tests to verify they fail**

Run: `pytest tests/test_visualizer.py::test_save_frame_creates_png tests/test_visualizer.py::test_generate_pair_visualizations_creates_all_files -v`
Expected: FAIL with ImportError

- [ ] **Step 11: Implement `save_frame` and `generate_pair_visualizations`**

```python
# append to src/vid_color_filter/gpu/visualizer.py
import os


def save_frame(frame: np.ndarray, out_path: str) -> None:
    """Save an RGB uint8 frame as PNG.

    Args:
        frame: (H, W, 3) uint8 RGB.
        out_path: Path to save PNG.
    """
    import cv2
    # OpenCV uses BGR, so convert from RGB
    cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def generate_pair_visualizations(
    data: dict,
    output_dir: str,
    vmax: float = 10.0,
) -> None:
    """Generate all PNG visualizations for a single video pair.

    Args:
        data: Dict with keys:
            - video_pair_id: str
            - src_frames_repr: (K, H, W, 3) uint8
            - edit_frames_repr: (K, H, W, 3) uint8
            - de_maps_repr: (K, H, W) float32
            - masks_repr: (K, H, W) bool
            - coverages_repr: list[float]
            - median_map: (H, W) float32
            - iqr_map: (H, W) float32
        output_dir: Base output directory. Creates {output_dir}/{video_pair_id}/.
        vmax: Colorbar max value (fixed at 10.0 for calibration).
    """
    pair_id = data["video_pair_id"]
    pair_dir = os.path.join(output_dir, pair_id)
    os.makedirs(pair_dir, exist_ok=True)

    n_repr = len(data["src_frames_repr"])

    for i in range(n_repr):
        save_frame(data["src_frames_repr"][i], os.path.join(pair_dir, f"src_frame_{i}.png"))
        save_frame(data["edit_frames_repr"][i], os.path.join(pair_dir, f"edit_frame_{i}.png"))
        render_heatmap(
            data["src_frames_repr"][i], data["de_maps_repr"][i],
            data["masks_repr"][i], os.path.join(pair_dir, f"heatmap_{i}.png"),
            vmax=vmax,
        )
        render_mask_overlay(
            data["src_frames_repr"][i], data["masks_repr"][i],
            os.path.join(pair_dir, f"mask_{i}.png"),
            coverage=data["coverages_repr"][i],
        )

    render_temporal_map(
        data["median_map"], os.path.join(pair_dir, "median_map.png"),
        vmax=vmax, cmap="viridis", label="Temporal Median ΔE",
    )
    render_temporal_map(
        data["iqr_map"], os.path.join(pair_dir, "iqr_map.png"),
        vmax=3.0, cmap="magma", label="Temporal IQR",
    )
```

- [ ] **Step 12: Run all visualizer tests**

Run: `pytest tests/test_visualizer.py -v`
Expected: All PASS

- [ ] **Step 13: Commit**

```bash
git add src/vid_color_filter/gpu/visualizer.py tests/test_visualizer.py
git commit -m "feat: add PNG visualization renderer for calibration reports"
```

---

## Task 3: HTML report generator

**Files:**
- Create: `src/vid_color_filter/report.py`
- Create: `tests/test_report.py`

This module generates self-contained HTML pages. Single-pair report pages reference PNGs via relative paths. The index page includes annotation JS using `localStorage` and an export button.

### Step group A: Single-pair report page

- [ ] **Step 1: Write failing test for `generate_pair_report`**

```python
# tests/test_report.py
import os
import tempfile
from vid_color_filter.report import generate_pair_report


def test_generate_pair_report_creates_html():
    """generate_pair_report creates a self-contained HTML report."""
    scores = {
        "video_pair_id": "clip_001",
        "global_shift_score": 1.5,
        "local_diff_score": 2.3,
        "temporal_instability": 0.4,
        "mask_coverage_ratio": 0.15,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        pair_dir = os.path.join(tmpdir, "clip_001")
        os.makedirs(pair_dir)
        # Create dummy PNGs
        for i in range(3):
            for name in ["src_frame", "edit_frame", "heatmap", "mask"]:
                open(os.path.join(pair_dir, f"{name}_{i}.png"), "w").close()
        open(os.path.join(pair_dir, "median_map.png"), "w").close()
        open(os.path.join(pair_dir, "iqr_map.png"), "w").close()

        generate_pair_report(scores, pair_dir, n_frames=3)

        report_path = os.path.join(pair_dir, "report.html")
        assert os.path.exists(report_path)
        html = open(report_path).read()
        assert "clip_001" in html
        assert "1.5" in html  # global_shift_score
        assert "src_frame_0.png" in html  # relative image path
        assert "localStorage" in html  # annotation JS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report.py::test_generate_pair_report_creates_html -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `generate_pair_report`**

```python
# src/vid_color_filter/report.py
"""HTML report generation for calibration annotation."""

from __future__ import annotations

import os


def generate_pair_report(
    scores: dict,
    pair_dir: str,
    n_frames: int = 5,
) -> None:
    """Generate an HTML report page for a single video pair.

    Args:
        scores: Dict with video_pair_id, global_shift_score, local_diff_score,
                temporal_instability, mask_coverage_ratio.
        pair_dir: Directory containing PNG visualizations. report.html is written here.
        n_frames: Number of representative frames.
    """
    pair_id = scores["video_pair_id"]
    global_s = scores.get("global_shift_score", float("nan"))
    local_s = scores.get("local_diff_score", float("nan"))
    instab = scores.get("temporal_instability", float("nan"))
    coverage = scores.get("mask_coverage_ratio", float("nan"))

    # Build frame switching HTML
    frame_tabs = "".join(
        f'<button class="tab-btn" onclick="switchFrame({i})"'
        f'{" style=\\"font-weight:bold\\"" if i == 0 else ""}>'
        f"Frame {i}</button> "
        for i in range(n_frames)
    )

    frame_sections = ""
    for i in range(n_frames):
        display = "block" if i == 0 else "none"
        frame_sections += f"""
        <div class="frame-set" id="frame-{i}" style="display:{display}">
          <div class="row">
            <div class="col"><h4>Source</h4><img src="src_frame_{i}.png"></div>
            <div class="col"><h4>Edited</h4><img src="edit_frame_{i}.png"></div>
          </div>
          <div class="row">
            <div class="col"><h4>ΔE Heatmap</h4><img src="heatmap_{i}.png"></div>
            <div class="col"><h4>Edit Mask</h4><img src="mask_{i}.png"></div>
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{pair_id} - Calibration Report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
table {{ border-collapse: collapse; margin: 10px 0; }}
th, td {{ border: 1px solid #444; padding: 6px 12px; text-align: left; }}
th {{ background: #16213e; }}
.row {{ display: flex; gap: 10px; margin: 10px 0; }}
.col {{ flex: 1; }}
.col img {{ width: 100%; }}
h4 {{ margin: 4px 0; color: #a0a0c0; }}
.tab-btn {{ padding: 6px 14px; cursor: pointer; background: #16213e; color: #e0e0e0;
            border: 1px solid #444; border-radius: 4px; }}
.tab-btn:hover {{ background: #1a3a5c; }}
.tab-btn.active {{ background: #0f3460; font-weight: bold; }}
.annotation {{ margin: 20px 0; padding: 16px; background: #16213e; border-radius: 8px; }}
.ann-btn {{ padding: 10px 30px; font-size: 16px; cursor: pointer; border: 2px solid #444;
            border-radius: 6px; margin-right: 10px; background: #1a1a2e; color: #e0e0e0; }}
.ann-btn:hover {{ opacity: 0.8; }}
.ann-btn.pass-selected {{ background: #1b5e20; border-color: #4caf50; color: #fff; }}
.ann-btn.fail-selected {{ background: #b71c1c; border-color: #f44336; color: #fff; }}
.temporal-row {{ display: flex; gap: 10px; margin: 10px 0; }}
.temporal-row .col img {{ width: 100%; }}
</style>
</head>
<body>
<h2>{pair_id}</h2>

<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Global Shift Score</td><td>{global_s:.3f}</td></tr>
<tr><td>Local Diff Score</td><td>{local_s:.3f}</td></tr>
<tr><td>Temporal Instability</td><td>{instab:.3f}</td></tr>
<tr><td>Mask Coverage</td><td>{coverage:.1%}</td></tr>
</table>

<h3>Frame Comparison</h3>
<div>{frame_tabs}</div>
{frame_sections}

<h3>Temporal Aggregation</h3>
<div class="temporal-row">
  <div class="col"><h4>Median ΔE Map</h4><img src="median_map.png"></div>
  <div class="col"><h4>IQR Map</h4><img src="iqr_map.png"></div>
</div>

<div class="annotation">
  <h3>Annotation</h3>
  <button class="ann-btn" id="btn-pass" onclick="annotate('pass')">✓ PASS</button>
  <button class="ann-btn" id="btn-fail" onclick="annotate('fail')">✗ FAIL</button>
  <span id="ann-status" style="margin-left:16px;"></span>
</div>

<p><a href="../index.html" style="color:#64b5f6;">← Back to Index</a></p>

<script>
const PAIR_ID = "{pair_id}";
const STORAGE_KEY = "calibration_annotation_" + PAIR_ID;

function switchFrame(idx) {{
  document.querySelectorAll('.frame-set').forEach(el => el.style.display = 'none');
  document.getElementById('frame-' + idx).style.display = 'block';
  document.querySelectorAll('.tab-btn').forEach((btn, i) => {{
    btn.classList.toggle('active', i === idx);
  }});
}}

function annotate(label) {{
  const data = {{ label: label, timestamp: new Date().toISOString() }};
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  updateUI(label);
}}

function updateUI(label) {{
  document.getElementById('btn-pass').className = 'ann-btn' + (label === 'pass' ? ' pass-selected' : '');
  document.getElementById('btn-fail').className = 'ann-btn' + (label === 'fail' ? ' fail-selected' : '');
  document.getElementById('ann-status').textContent = 'Labeled: ' + label.toUpperCase();
}}

// Restore on load
const saved = localStorage.getItem(STORAGE_KEY);
if (saved) {{
  try {{ updateUI(JSON.parse(saved).label); }} catch(e) {{}}
}}

// Initial tab state
document.querySelector('.tab-btn').classList.add('active');
</script>
</body>
</html>"""

    with open(os.path.join(pair_dir, "report.html"), "w") as f:
        f.write(html)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_report.py::test_generate_pair_report_creates_html -v`
Expected: PASS

### Step group B: Index page with export

- [ ] **Step 5: Write failing test for `generate_index_page`**

```python
# append to tests/test_report.py
from vid_color_filter.report import generate_index_page


def test_generate_index_page_creates_html():
    """generate_index_page creates an index with links and export button."""
    cases = [
        {"video_pair_id": "clip_001", "global_shift_score": 2.5, "local_diff_score": 1.0,
         "temporal_instability": 0.3},
        {"video_pair_id": "clip_002", "global_shift_score": 1.0, "local_diff_score": 4.0,
         "temporal_instability": 0.8},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        reports_dir = os.path.join(tmpdir, "reports")
        os.makedirs(reports_dir)
        generate_index_page(cases, reports_dir)

        index_path = os.path.join(reports_dir, "index.html")
        assert os.path.exists(index_path)
        html = open(index_path).read()
        assert "clip_001" in html
        assert "clip_002" in html
        assert "exportAnnotations" in html  # export function
        assert "localStorage" in html
        # Sorted by global_shift_score descending
        pos_001 = html.index("clip_001")
        pos_002 = html.index("clip_002")
        assert pos_001 < pos_002  # 2.5 > 1.0, so clip_001 first
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_report.py::test_generate_index_page_creates_html -v`
Expected: FAIL with ImportError

- [ ] **Step 7: Implement `generate_index_page`**

```python
# append to src/vid_color_filter/report.py
import json


def generate_index_page(
    cases: list[dict],
    reports_dir: str,
) -> None:
    """Generate the index.html page for browsing and annotating boundary cases.

    Args:
        cases: List of dicts with video_pair_id, global_shift_score,
               local_diff_score, temporal_instability.
        reports_dir: Directory to write index.html into. Pair report dirs
                     are subdirectories here.
    """
    # Sort by global_shift_score descending
    sorted_cases = sorted(cases, key=lambda c: c.get("global_shift_score", 0), reverse=True)

    pair_ids_json = json.dumps([c["video_pair_id"] for c in sorted_cases])

    rows = ""
    for c in sorted_cases:
        pid = c["video_pair_id"]
        gs = c.get("global_shift_score", float("nan"))
        ld = c.get("local_diff_score", float("nan"))
        ti = c.get("temporal_instability", float("nan"))
        rows += f"""<tr id="row-{pid}">
  <td><a href="{pid}/report.html" style="color:#64b5f6">{pid}</a></td>
  <td>{gs:.3f}</td>
  <td>{ld:.3f}</td>
  <td>{ti:.3f}</td>
  <td class="ann-cell" id="ann-{pid}">—</td>
</tr>\n"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Calibration — Boundary Cases</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #444; padding: 8px 12px; text-align: left; }}
th {{ background: #16213e; cursor: pointer; user-select: none; }}
th:hover {{ background: #1a3a5c; }}
tr:hover {{ background: #16213e44; }}
.pass {{ color: #4caf50; font-weight: bold; }}
.fail {{ color: #f44336; font-weight: bold; }}
.toolbar {{ margin: 16px 0; display: flex; align-items: center; gap: 16px; }}
.export-btn {{ padding: 10px 24px; font-size: 14px; cursor: pointer;
               background: #0f3460; color: #e0e0e0; border: 1px solid #444;
               border-radius: 6px; }}
.export-btn:hover {{ background: #1a3a5c; }}
</style>
</head>
<body>
<h1>Calibration — Boundary Cases</h1>
<div class="toolbar">
  <span id="progress">Annotated: 0 / {len(sorted_cases)}</span>
  <button class="export-btn" onclick="exportAnnotations()">Export annotations.json</button>
</div>

<table id="cases-table">
<thead>
<tr>
  <th onclick="sortTable(0)">Video Pair ID</th>
  <th onclick="sortTable(1)">Global Shift</th>
  <th onclick="sortTable(2)">Local Diff</th>
  <th onclick="sortTable(3)">Instability</th>
  <th>Annotation</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>

<script>
const PAIR_IDS = {pair_ids_json};

function loadAnnotations() {{
  let count = 0;
  PAIR_IDS.forEach(pid => {{
    const raw = localStorage.getItem("calibration_annotation_" + pid);
    if (raw) {{
      try {{
        const data = JSON.parse(raw);
        const cell = document.getElementById("ann-" + pid);
        if (cell) {{
          cell.textContent = data.label.toUpperCase();
          cell.className = "ann-cell " + data.label;
        }}
        count++;
      }} catch(e) {{}}
    }}
  }});
  document.getElementById("progress").textContent = "Annotated: " + count + " / " + PAIR_IDS.length;
}}

function exportAnnotations() {{
  const annotations = [];
  PAIR_IDS.forEach(pid => {{
    const raw = localStorage.getItem("calibration_annotation_" + pid);
    if (raw) {{
      try {{
        const data = JSON.parse(raw);
        annotations.push({{ video_pair_id: pid, label: data.label, timestamp: data.timestamp }});
      }} catch(e) {{}}
    }}
  }});
  const blob = new Blob([JSON.stringify(annotations, null, 2)], {{ type: "application/json" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "annotations.json";
  a.click();
  URL.revokeObjectURL(url);
}}

function sortTable(colIdx) {{
  const table = document.getElementById("cases-table");
  const tbody = table.querySelector("tbody");
  const rows = Array.from(tbody.querySelectorAll("tr"));
  const isNumeric = colIdx > 0 && colIdx < 4;

  rows.sort((a, b) => {{
    let va = a.cells[colIdx].textContent.trim();
    let vb = b.cells[colIdx].textContent.trim();
    if (isNumeric) return parseFloat(vb) - parseFloat(va);
    return va.localeCompare(vb);
  }});

  // Toggle direction
  if (table.dataset.sortCol === String(colIdx) && table.dataset.sortDir === "desc") {{
    rows.reverse();
    table.dataset.sortDir = "asc";
  }} else {{
    table.dataset.sortCol = String(colIdx);
    table.dataset.sortDir = "desc";
  }}

  rows.forEach(r => tbody.appendChild(r));
}}

loadAnnotations();
// Refresh annotations when tab becomes visible (returning from report page)
document.addEventListener("visibilitychange", () => {{
  if (!document.hidden) loadAnnotations();
}});
</script>
</body>
</html>"""

    with open(os.path.join(reports_dir, "index.html"), "w") as f:
        f.write(html)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_report.py -v`
Expected: All PASS

### Step group C: Error report page

- [ ] **Step 9: Write failing test for `generate_error_report`**

```python
# append to tests/test_report.py
from vid_color_filter.report import generate_error_report


def test_generate_error_report_creates_html():
    """generate_error_report creates a placeholder page for failed visualizations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pair_dir = os.path.join(tmpdir, "clip_bad")
        os.makedirs(pair_dir)
        generate_error_report("clip_bad", "Video file corrupted", pair_dir)

        report_path = os.path.join(pair_dir, "report.html")
        assert os.path.exists(report_path)
        html = open(report_path).read()
        assert "clip_bad" in html
        assert "corrupted" in html
```

- [ ] **Step 10: Implement `generate_error_report`**

```python
# append to src/vid_color_filter/report.py

def generate_error_report(
    video_pair_id: str,
    error_message: str,
    pair_dir: str,
) -> None:
    """Generate a placeholder report page for a pair that failed visualization.

    Args:
        video_pair_id: The video pair identifier.
        error_message: Error description.
        pair_dir: Directory to write report.html.
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{video_pair_id} - Error</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
.error {{ background: #3e1a1a; border: 1px solid #f44336; padding: 20px; border-radius: 8px; margin: 20px 0; }}
</style>
</head>
<body>
<h2>{video_pair_id}</h2>
<div class="error">
<h3>Visualization Failed</h3>
<p>{error_message}</p>
</div>
<p><a href="../index.html" style="color:#64b5f6;">← Back to Index</a></p>
</body>
</html>"""

    os.makedirs(pair_dir, exist_ok=True)
    with open(os.path.join(pair_dir, "report.html"), "w") as f:
        f.write(html)
```

- [ ] **Step 11: Run all report tests**

Run: `pytest tests/test_report.py -v`
Expected: All PASS

- [ ] **Step 12: Commit**

```bash
git add src/vid_color_filter/report.py tests/test_report.py
git commit -m "feat: add HTML report generator with annotation JS for calibration"
```

---

## Task 4: Calibration analysis module

**Files:**
- Create: `src/vid_color_filter/calibration.py`
- Create: `tests/test_calibration.py`

Handles: loading scores JSONL, distribution analysis (matplotlib → base64 HTML), grid search (pass rate + F1 heatmap), boundary case filtering, and the `build-reports` / `evaluate` subcommands.

### Step group A: JSONL loading + distribution analysis

- [ ] **Step 1: Write failing test for `load_scores` and `generate_distribution_html`**

```python
# tests/test_calibration.py
import json
import os
import tempfile
from vid_color_filter.calibration import load_scores, generate_distribution_html


def _write_scores_jsonl(path, n=50):
    """Helper: write dummy scores JSONL."""
    import random
    random.seed(42)
    with open(path, "w") as f:
        for i in range(n):
            record = {
                "video_pair_id": f"clip_{i:03d}",
                "global_shift_score": random.uniform(0.3, 4.5),
                "local_diff_score": random.uniform(0.5, 7.0),
                "temporal_instability": random.uniform(0.05, 1.5),
                "mask_coverage_ratio": random.uniform(0.0, 0.5),
                "pass": True,
            }
            f.write(json.dumps(record) + "\n")


def test_load_scores_reads_jsonl():
    """load_scores reads a JSONL file and returns a list of dicts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "scores.jsonl")
        _write_scores_jsonl(path, n=10)
        scores = load_scores(path)
        assert len(scores) == 10
        assert all("global_shift_score" in s for s in scores)


def test_generate_distribution_html():
    """generate_distribution_html creates an HTML file with embedded charts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scores_path = os.path.join(tmpdir, "scores.jsonl")
        _write_scores_jsonl(scores_path, n=30)
        scores = load_scores(scores_path)
        out_path = os.path.join(tmpdir, "distribution.html")
        generate_distribution_html(scores, out_path)
        assert os.path.exists(out_path)
        html = open(out_path).read()
        assert "base64" in html  # embedded PNG
        assert "Global Shift" in html or "global" in html.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_calibration.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `load_scores` and `generate_distribution_html`**

```python
# src/vid_color_filter/calibration.py
"""Calibration analysis: distribution, grid search, boundary selection, F1 evaluation."""

from __future__ import annotations

import base64
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_scores(path: str) -> list[dict]:
    """Load scores from a JSONL file.

    Args:
        path: Path to JSONL file.
    Returns:
        List of score dicts.
    """
    scores = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                scores.append(json.loads(line))
    return scores


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return encoded


def generate_distribution_html(scores: list[dict], out_path: str) -> None:
    """Generate distribution analysis HTML with embedded charts.

    Creates histograms for global_shift_score, local_diff_score,
    temporal_instability, and a 2D scatter plot.

    Args:
        scores: List of score dicts from load_scores.
        out_path: Path to write HTML file.
    """
    gs = [s["global_shift_score"] for s in scores if "global_shift_score" in s]
    ld = [s["local_diff_score"] for s in scores if "local_diff_score" in s]
    ti = [s["temporal_instability"] for s in scores if "temporal_instability" in s]

    images = []

    # Global shift histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gs, bins=50, color="#4fc3f7", edgecolor="#222")
    ax.set_xlabel("Global Shift Score (ΔE)")
    ax.set_ylabel("Count")
    ax.set_title("Global Shift Score Distribution")
    images.append(("Global Shift Score", _fig_to_base64(fig)))

    # Local diff histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ld, bins=50, color="#81c784", edgecolor="#222")
    ax.set_xlabel("Local Diff Score (ΔE)")
    ax.set_ylabel("Count")
    ax.set_title("Local Difference Score Distribution")
    images.append(("Local Difference Score", _fig_to_base64(fig)))

    # Temporal instability histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ti, bins=50, color="#ffb74d", edgecolor="#222")
    ax.set_xlabel("Temporal Instability")
    ax.set_ylabel("Count")
    ax.set_title("Temporal Instability Distribution")
    images.append(("Temporal Instability", _fig_to_base64(fig)))

    # 2D scatter: global vs local
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(gs, ld, alpha=0.5, s=10, c="#e0e0e0")
    ax.set_xlabel("Global Shift Score (ΔE)")
    ax.set_ylabel("Local Diff Score (ΔE)")
    ax.set_title("Global Shift vs Local Difference")
    ax.grid(True, alpha=0.3)
    images.append(("Global vs Local Scatter", _fig_to_base64(fig)))

    # Build HTML
    sections = "\n".join(
        f'<h3>{title}</h3>\n<img src="data:image/png;base64,{b64}" style="max-width:100%">'
        for title, b64 in images
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Score Distribution Analysis</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
img {{ margin: 10px 0; }}
</style>
</head>
<body>
<h1>Score Distribution Analysis</h1>
<p>Total video pairs: {len(scores)}</p>
{sections}
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_calibration.py -v`
Expected: PASS

### Step group B: Grid search preview

- [ ] **Step 5: Write failing test for `grid_search_preview`**

```python
# append to tests/test_calibration.py
from vid_color_filter.calibration import grid_search_preview


def test_grid_search_preview():
    """grid_search_preview computes pass rates for a grid of threshold pairs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scores_path = os.path.join(tmpdir, "scores.jsonl")
        _write_scores_jsonl(scores_path, n=50)
        scores = load_scores(scores_path)
        out_path = os.path.join(tmpdir, "grid_preview.html")
        grid_search_preview(scores, out_path)
        assert os.path.exists(out_path)
        html = open(out_path).read()
        assert "Pass Rate" in html or "pass" in html.lower()
        assert "base64" in html  # embedded heatmap
```

- [ ] **Step 6: Implement `grid_search_preview`**

```python
# append to src/vid_color_filter/calibration.py

# Grid search ranges (module-level constants)
GLOBAL_THRESHOLDS = [round(0.5 + i * 0.25, 2) for i in range(19)]  # 0.5 to 5.0
LOCAL_THRESHOLDS = [round(1.0 + i * 0.5, 1) for i in range(15)]    # 1.0 to 8.0


def grid_search_preview(scores: list[dict], out_path: str) -> None:
    """Generate grid search preview showing pass rate for each threshold combination.

    Args:
        scores: List of score dicts.
        out_path: Path to write HTML file.
    """
    gs_vals = np.array([s["global_shift_score"] for s in scores])
    ld_vals = np.array([s["local_diff_score"] for s in scores])
    n = len(scores)

    # Compute pass rate matrix
    pass_rates = np.zeros((len(LOCAL_THRESHOLDS), len(GLOBAL_THRESHOLDS)))
    for i, lt in enumerate(LOCAL_THRESHOLDS):
        for j, gt in enumerate(GLOBAL_THRESHOLDS):
            n_pass = np.sum((gs_vals < gt) & (ld_vals < lt))
            pass_rates[i, j] = n_pass / n if n > 0 else 0

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pass_rates, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(GLOBAL_THRESHOLDS)))
    ax.set_xticklabels([f"{v:.2f}" for v in GLOBAL_THRESHOLDS], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(LOCAL_THRESHOLDS)))
    ax.set_yticklabels([f"{v:.1f}" for v in LOCAL_THRESHOLDS], fontsize=7)
    ax.set_xlabel("Global Threshold (ΔE)")
    ax.set_ylabel("Local Threshold (ΔE)")
    ax.set_title(f"Pass Rate by Threshold Combination (n={n})")
    fig.colorbar(im, ax=ax, label="Pass Rate", fraction=0.046, pad=0.04)
    fig.tight_layout()

    b64 = _fig_to_base64(fig)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Grid Search Preview</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
img {{ max-width: 100%; }}
</style>
</head>
<body>
<h1>Grid Search Preview — Pass Rate</h1>
<p>Total video pairs: {n}</p>
<p>Global threshold range: {GLOBAL_THRESHOLDS[0]} – {GLOBAL_THRESHOLDS[-1]} (step 0.25)</p>
<p>Local threshold range: {LOCAL_THRESHOLDS[0]} – {LOCAL_THRESHOLDS[-1]} (step 0.5)</p>
<img src="data:image/png;base64,{b64}">
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_calibration.py::test_grid_search_preview -v`
Expected: PASS

### Step group C: Boundary case selection

- [ ] **Step 8: Write failing test for `select_boundary_cases`**

```python
# append to tests/test_calibration.py
from vid_color_filter.calibration import select_boundary_cases


def test_select_boundary_cases():
    """select_boundary_cases finds cases near candidate thresholds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scores_path = os.path.join(tmpdir, "scores.jsonl")
        _write_scores_jsonl(scores_path, n=100)
        scores = load_scores(scores_path)
        out_dir = tmpdir

        boundary, subset_csv = select_boundary_cases(scores, out_dir)
        assert len(boundary) > 0
        assert len(boundary) <= len(scores)
        assert all("video_pair_id" in b for b in boundary)

        # boundary_cases.json written
        bc_path = os.path.join(out_dir, "boundary_cases.json")
        assert os.path.exists(bc_path)

        # boundary_subset.csv written
        assert os.path.exists(subset_csv)
```

- [ ] **Step 9: Implement `select_boundary_cases`**

```python
# append to src/vid_color_filter/calibration.py
import csv


def select_boundary_cases(
    scores: list[dict],
    output_dir: str,
    global_range: tuple[float, float] = (1.5, 3.0),
    local_range: tuple[float, float] = (2.0, 5.0),
    margin: float = 0.5,
) -> tuple[list[dict], str]:
    """Select boundary cases whose scores fall near candidate threshold ranges.

    A video pair is a boundary case if its global_shift_score is within
    [global_range[0] - margin, global_range[1] + margin] OR its local_diff_score
    is within [local_range[0] - margin, local_range[1] + margin].

    Args:
        scores: List of score dicts.
        output_dir: Directory to write boundary_cases.json and boundary_subset.csv.
        global_range: (min, max) of the interesting global threshold search range.
        local_range: (min, max) of the interesting local threshold search range.
        margin: Extra margin around the ranges.
    Returns:
        (boundary_cases, subset_csv_path)
    """
    g_lo = global_range[0] - margin
    g_hi = global_range[1] + margin
    l_lo = local_range[0] - margin
    l_hi = local_range[1] + margin

    boundary = []
    for s in scores:
        gs = s.get("global_shift_score", float("nan"))
        ld = s.get("local_diff_score", float("nan"))
        if (g_lo <= gs <= g_hi) or (l_lo <= ld <= l_hi):
            boundary.append(s)

    # Write boundary_cases.json
    bc_path = os.path.join(output_dir, "boundary_cases.json")
    with open(bc_path, "w") as f:
        json.dump(boundary, f, indent=2)

    # Write boundary_subset.csv for re-running pipeline
    csv_path = os.path.join(output_dir, "boundary_subset.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video1_path", "video2_path"])
        writer.writeheader()
        for s in boundary:
            pid = s["video_pair_id"]
            # video_pair_id is typically the filename stem; original paths
            # must be recovered from the original CSV or scores JSONL.
            # If the scores contain src_path/edited_path, use those.
            writer.writerow({
                "video1_path": s.get("src_path", pid),
                "video2_path": s.get("edited_path", pid),
            })

    return boundary, csv_path
```

- [ ] **Step 10: Run test to verify it passes**

Run: `pytest tests/test_calibration.py::test_select_boundary_cases -v`
Expected: PASS

### Step group D: F1 evaluation (post-annotation)

- [ ] **Step 11: Write failing test for `evaluate_annotations`**

```python
# append to tests/test_calibration.py
from vid_color_filter.calibration import evaluate_annotations


def test_evaluate_annotations():
    """evaluate_annotations computes F1 for each threshold combo and finds best."""
    scores = [
        {"video_pair_id": "a", "global_shift_score": 1.0, "local_diff_score": 1.0},
        {"video_pair_id": "b", "global_shift_score": 3.0, "local_diff_score": 2.0},
        {"video_pair_id": "c", "global_shift_score": 2.0, "local_diff_score": 5.0},
        {"video_pair_id": "d", "global_shift_score": 0.5, "local_diff_score": 0.5},
    ]
    annotations = [
        {"video_pair_id": "a", "label": "pass"},
        {"video_pair_id": "b", "label": "fail"},
        {"video_pair_id": "c", "label": "fail"},
        {"video_pair_id": "d", "label": "pass"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "grid_results.html")
        best = evaluate_annotations(scores, annotations, out_path)
        assert "best_global_threshold" in best
        assert "best_local_threshold" in best
        assert "best_f1" in best
        assert 0 <= best["best_f1"] <= 1
        assert os.path.exists(out_path)
```

- [ ] **Step 12: Implement `evaluate_annotations`**

```python
# append to src/vid_color_filter/calibration.py

def evaluate_annotations(
    scores: list[dict],
    annotations: list[dict],
    out_path: str,
) -> dict:
    """Evaluate threshold grid against human annotations, find best F1.

    F1 is computed only over annotated pairs. "pass" = positive class.

    Args:
        scores: List of score dicts.
        annotations: List of {"video_pair_id": str, "label": "pass"|"fail"}.
        out_path: Path to write grid_search_results.html.
    Returns:
        Dict with best_global_threshold, best_local_threshold, best_f1.
    """
    # Build lookup: pair_id -> (global_shift, local_diff)
    score_map = {
        s["video_pair_id"]: (s["global_shift_score"], s["local_diff_score"])
        for s in scores
    }
    # Build ground truth: pair_id -> bool (True = pass)
    labels = {a["video_pair_id"]: a["label"] == "pass" for a in annotations}

    # Only evaluate pairs that are in both scores and annotations
    eval_ids = [pid for pid in labels if pid in score_map]
    if not eval_ids:
        return {"best_global_threshold": None, "best_local_threshold": None, "best_f1": 0.0}

    f1_matrix = np.zeros((len(LOCAL_THRESHOLDS), len(GLOBAL_THRESHOLDS)))
    prec_matrix = np.zeros_like(f1_matrix)
    rec_matrix = np.zeros_like(f1_matrix)

    for i, lt in enumerate(LOCAL_THRESHOLDS):
        for j, gt_thresh in enumerate(GLOBAL_THRESHOLDS):
            tp = fp = fn = tn = 0
            for pid in eval_ids:
                gs_val, ld_val = score_map[pid]
                pred_pass = (gs_val < gt_thresh) and (ld_val < lt)
                actual_pass = labels[pid]
                if pred_pass and actual_pass:
                    tp += 1
                elif pred_pass and not actual_pass:
                    fp += 1
                elif not pred_pass and actual_pass:
                    fn += 1
                else:
                    tn += 1
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            f1_matrix[i, j] = f1
            prec_matrix[i, j] = prec
            rec_matrix[i, j] = rec

    # Find best
    best_idx = np.unravel_index(np.argmax(f1_matrix), f1_matrix.shape)
    best_lt = LOCAL_THRESHOLDS[best_idx[0]]
    best_gt = GLOBAL_THRESHOLDS[best_idx[1]]
    best_f1 = float(f1_matrix[best_idx])

    # Generate heatmap HTML
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, data, title in zip(axes, [f1_matrix, prec_matrix, rec_matrix],
                                     ["F1 Score", "Precision", "Recall"]):
        im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(GLOBAL_THRESHOLDS)))
        ax.set_xticklabels([f"{v:.2f}" for v in GLOBAL_THRESHOLDS], rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(LOCAL_THRESHOLDS)))
        ax.set_yticklabels([f"{v:.1f}" for v in LOCAL_THRESHOLDS], fontsize=6)
        ax.set_xlabel("Global Threshold")
        ax.set_ylabel("Local Threshold")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Mark best on F1 plot
    axes[0].plot(best_idx[1], best_idx[0], "k*", markersize=15)
    fig.suptitle(
        f"Best: global={best_gt:.2f}, local={best_lt:.1f}, F1={best_f1:.3f} "
        f"(n={len(eval_ids)} annotated pairs)",
        fontsize=12,
    )
    fig.tight_layout()
    b64 = _fig_to_base64(fig)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Grid Search Results — F1 Evaluation</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
img {{ max-width: 100%; }}
.best {{ background: #1b5e20; padding: 16px; border-radius: 8px; margin: 16px 0; font-size: 18px; }}
</style>
</head>
<body>
<h1>Grid Search Results — F1 Evaluation</h1>
<div class="best">
  Best threshold: global = {best_gt:.2f}, local = {best_lt:.1f}<br>
  F1 = {best_f1:.3f} (on {len(eval_ids)} annotated pairs)
</div>
<img src="data:image/png;base64,{b64}">
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)

    return {
        "best_global_threshold": best_gt,
        "best_local_threshold": best_lt,
        "best_f1": best_f1,
    }
```

- [ ] **Step 13: Run all calibration tests**

Run: `pytest tests/test_calibration.py -v`
Expected: All PASS

- [ ] **Step 14: Commit**

```bash
git add src/vid_color_filter/calibration.py tests/test_calibration.py
git commit -m "feat: add calibration analysis module with grid search and F1 evaluation"
```

---

## Task 5: Calibration CLI (`__main__` entry point)

**Files:**
- Modify: `src/vid_color_filter/calibration.py`
- Create: `tests/test_calibration_cli.py`

Add `if __name__ == "__main__"` and argparse subcommands (`analyze`, `build-reports`, `evaluate`) so the module can be invoked as `python -m vid_color_filter.calibration <subcommand>`.

- [ ] **Step 1: Write failing test for CLI**

```python
# tests/test_calibration_cli.py
import json
import os
import subprocess
import sys
import tempfile


def _write_scores(path, n=30):
    import random
    random.seed(42)
    with open(path, "w") as f:
        for i in range(n):
            record = {
                "video_pair_id": f"clip_{i:03d}",
                "global_shift_score": random.uniform(0.3, 4.5),
                "local_diff_score": random.uniform(0.5, 7.0),
                "temporal_instability": random.uniform(0.05, 1.5),
                "mask_coverage_ratio": random.uniform(0.0, 0.5),
                "src_path": f"/data/src/clip_{i:03d}.mp4",
                "edited_path": f"/data/edit/clip_{i:03d}.mp4",
            }
            f.write(json.dumps(record) + "\n")


def test_analyze_subcommand():
    """python -m vid_color_filter.calibration analyze produces expected outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scores_path = os.path.join(tmpdir, "scores.jsonl")
        _write_scores(scores_path)

        result = subprocess.run(
            [sys.executable, "-m", "vid_color_filter.calibration", "analyze",
             "--scores", scores_path, "--output-dir", tmpdir],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        assert os.path.exists(os.path.join(tmpdir, "distribution.html"))
        assert os.path.exists(os.path.join(tmpdir, "grid_search_preview.html"))
        assert os.path.exists(os.path.join(tmpdir, "boundary_cases.json"))
        assert os.path.exists(os.path.join(tmpdir, "boundary_subset.csv"))


def test_evaluate_subcommand():
    """python -m vid_color_filter.calibration evaluate produces F1 results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scores_path = os.path.join(tmpdir, "scores.jsonl")
        _write_scores(scores_path, n=10)

        annotations = [
            {"video_pair_id": f"clip_{i:03d}", "label": "pass" if i < 5 else "fail"}
            for i in range(10)
        ]
        ann_path = os.path.join(tmpdir, "annotations.json")
        with open(ann_path, "w") as f:
            json.dump(annotations, f)

        result = subprocess.run(
            [sys.executable, "-m", "vid_color_filter.calibration", "evaluate",
             "--scores", scores_path, "--annotations", ann_path, "--output-dir", tmpdir],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert os.path.exists(os.path.join(tmpdir, "grid_search_results.html"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_calibration_cli.py -v`
Expected: FAIL (no __main__ block)

- [ ] **Step 3: Implement CLI with argparse subcommands**

```python
# append to src/vid_color_filter/calibration.py
import argparse

from vid_color_filter.report import generate_index_page


def _cmd_analyze(args):
    """Subcommand: analyze scores → distribution, grid preview, boundary cases."""
    scores = load_scores(args.scores)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loaded {len(scores)} scores from {args.scores}")

    generate_distribution_html(scores, os.path.join(args.output_dir, "distribution.html"))
    print("  → distribution.html")

    grid_search_preview(scores, os.path.join(args.output_dir, "grid_search_preview.html"))
    print("  → grid_search_preview.html")

    boundary, csv_path = select_boundary_cases(scores, args.output_dir)
    print(f"  → {len(boundary)} boundary cases → boundary_cases.json, {os.path.basename(csv_path)}")


def _cmd_build_reports(args):
    """Subcommand: build index.html after visualizations are generated."""
    bc_path = os.path.join(args.output_dir, "boundary_cases.json")
    if not os.path.exists(bc_path):
        bc_path = args.boundary_cases

    with open(bc_path) as f:
        boundary = json.load(f)

    reports_dir = args.viz_dir
    generate_index_page(boundary, reports_dir)
    print(f"  → index.html ({len(boundary)} cases)")


def _cmd_evaluate(args):
    """Subcommand: evaluate annotations → F1 grid search results."""
    scores = load_scores(args.scores)
    with open(args.annotations) as f:
        annotations = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "grid_search_results.html")
    best = evaluate_annotations(scores, annotations, out_path)
    print(f"Best: global={best['best_global_threshold']}, "
          f"local={best['best_local_threshold']}, F1={best['best_f1']:.3f}")
    print(f"  → grid_search_results.html")


def main():
    parser = argparse.ArgumentParser(description="Calibration analysis tools")
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze score distribution and select boundary cases")
    p_analyze.add_argument("--scores", required=True, help="Path to scores.jsonl")
    p_analyze.add_argument("--output-dir", required=True, help="Output directory")

    # build-reports
    p_build = sub.add_parser("build-reports", help="Build index.html after visualizations")
    p_build.add_argument("--boundary-cases", default="boundary_cases.json",
                         help="Path to boundary_cases.json")
    p_build.add_argument("--viz-dir", required=True, help="Directory with visualization reports")
    p_build.add_argument("--output-dir", default=".", help="Base output directory")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate annotations with F1 grid search")
    p_eval.add_argument("--scores", required=True, help="Path to scores.jsonl")
    p_eval.add_argument("--annotations", required=True, help="Path to annotations.json")
    p_eval.add_argument("--output-dir", required=True, help="Output directory")

    args = parser.parse_args()
    if args.command == "analyze":
        _cmd_analyze(args)
    elif args.command == "build-reports":
        _cmd_build_reports(args)
    elif args.command == "evaluate":
        _cmd_evaluate(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all calibration CLI tests**

Run: `pytest tests/test_calibration_cli.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/vid_color_filter/calibration.py tests/test_calibration_cli.py
git commit -m "feat: add calibration CLI subcommands (analyze, build-reports, evaluate)"
```

---

## Task 6: Modify batch_scorer to support `--visualize` mode

**Files:**
- Modify: `src/vid_color_filter/gpu/batch_scorer.py:13-26` (signature), `109-215` (`_score_scielab`)
- Modify: `tests/test_gpu_scorer.py`

When `visualize=True`, `_score_scielab` retains intermediate tensors (representative frames, ΔE maps, masks, temporal maps) and returns them in the result dict. When `visualize=False`, behavior is unchanged.

- [ ] **Step 1: Write failing test for visualize mode**

```python
# append to tests/test_gpu_scorer.py
import torch


def test_score_scielab_visualize_returns_intermediates():
    """With visualize=True, result dict contains intermediate tensors."""
    pytest = __import__("pytest")
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    n, h, w = 8, 64, 96
    src = torch.randint(0, 255, (n, h, w, 3), dtype=torch.uint8, device="cuda")
    edited = torch.randint(0, 255, (n, h, w, 3), dtype=torch.uint8, device="cuda")

    from vid_color_filter.gpu.batch_scorer import score_video_pair_gpu

    result = score_video_pair_gpu(
        src, edited, use_scielab=True, visualize=True, chunk_size=4,
    )

    assert "src_frames_repr" in result
    assert "edit_frames_repr" in result
    assert "de_maps_repr" in result
    assert "masks_repr" in result
    assert "coverages_repr" in result
    assert "median_map" in result
    assert "iqr_map" in result
    assert result["src_frames_repr"].shape[0] <= 5  # at most 5 representative frames


def test_score_scielab_no_visualize_no_intermediates():
    """With visualize=False (default), no intermediate tensors in result."""
    pytest = __import__("pytest")
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    n, h, w = 8, 64, 96
    src = torch.randint(0, 255, (n, h, w, 3), dtype=torch.uint8, device="cuda")
    edited = torch.randint(0, 255, (n, h, w, 3), dtype=torch.uint8, device="cuda")

    from vid_color_filter.gpu.batch_scorer import score_video_pair_gpu

    result = score_video_pair_gpu(
        src, edited, use_scielab=True, visualize=False, chunk_size=4,
    )

    assert "src_frames_repr" not in result
    assert "median_map" not in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gpu_scorer.py::test_score_scielab_visualize_returns_intermediates -v`
Expected: FAIL (unexpected keyword argument 'visualize')

- [ ] **Step 3: Add `visualize` parameter to `score_video_pair_gpu` and `_score_scielab`**

Modify `score_video_pair_gpu` signature (line 13) to add `visualize: bool = False` parameter. Pass it through to `_score_scielab`.

Modify `_score_scielab` to:
1. Add `visualize: bool = False` parameter
2. After computing `per_frame_mean_des` (which are already available), select representative frame indices: sort by mean ΔE, pick indices for max, min, median, plus 2 evenly spaced. Cap at `min(5, N)`.
3. If `visualize=True`, after temporal aggregation, add these keys to the result dict:
   - `src_frames_repr`: `src_frames[repr_indices].cpu().numpy()` — (K, H, W, 3) uint8
   - `edit_frames_repr`: `edited_frames[repr_indices].cpu().numpy()` — (K, H, W, 3) uint8
   - `de_maps_repr`: `de_maps[repr_indices].cpu().numpy()` — (K, H, W) float32
   - `masks_repr`: `masks[repr_indices].cpu().numpy()` — (K, H, W) bool
   - `coverages_repr`: `[coverages[i] for i in repr_indices]` — list of floats
   - `median_map`: `median_map.cpu().numpy()` — (H, W) float32
   - `iqr_map`: `iqr_map.cpu().numpy()` — (H, W) float32

Key implementation detail: `de_maps` and `masks` are already accumulated as `(N, H, W)` tensors before `temporal_aggregate` is called — use these directly. `median_map` and `iqr_map` are returned by `temporal_aggregate`.

Representative frame selection logic:

```python
def _select_representative_indices(mean_des: list[float], max_repr: int = 5) -> list[int]:
    """Select representative frame indices based on per-frame mean ΔE.

    Picks: max ΔE frame, min ΔE frame, median ΔE frame, plus evenly spaced.
    """
    n = len(mean_des)
    if n <= max_repr:
        return list(range(n))

    sorted_indices = sorted(range(n), key=lambda i: mean_des[i])
    selected = {
        sorted_indices[-1],   # max
        sorted_indices[0],    # min
        sorted_indices[n // 2],  # median
    }

    # Fill remaining with evenly spaced
    remaining = max_repr - len(selected)
    if remaining > 0:
        step = n / (remaining + 1)
        for k in range(1, remaining + 1):
            idx = int(k * step)
            selected.add(min(idx, n - 1))

    return sorted(selected)[:max_repr]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gpu_scorer.py -v`
Expected: All PASS (including existing tests)

- [ ] **Step 5: Commit**

```bash
git add src/vid_color_filter/gpu/batch_scorer.py tests/test_gpu_scorer.py
git commit -m "feat: add --visualize mode to batch_scorer for intermediate retention"
```

---

## Task 7: Wire `--visualize` and `--viz-dir` into `run.py`

**Files:**
- Modify: `run.py:29-105` (argument parsing), `run.py:171-183` (scoring loop)

- [ ] **Step 1: Add CLI arguments**

Add to `parse_args()` in `run.py`:

```python
parser.add_argument("--visualize", action="store_true",
                    help="Generate PNG visualizations for each scored pair")
parser.add_argument("--viz-dir", type=str, default=None,
                    help="Output directory for visualizations (required with --visualize)")
```

Add validation after parsing:

```python
if args.visualize and not args.viz_dir:
    parser.error("--viz-dir is required when using --visualize")
```

- [ ] **Step 2: Integrate visualization into scoring loop**

In the main scoring loop (around line 171-183), after `score_video_pair_gpu` returns a result, add:

```python
if args.visualize and "src_frames_repr" in result:
    from vid_color_filter.gpu.visualizer import generate_pair_visualizations
    from vid_color_filter.report import generate_pair_report

    try:
        generate_pair_visualizations(result, args.viz_dir, vmax=10.0)
        pair_dir = os.path.join(args.viz_dir, result["video_pair_id"])
        generate_pair_report(result, pair_dir, n_frames=len(result["src_frames_repr"]))
    except Exception as e:
        from vid_color_filter.report import generate_error_report
        pair_dir = os.path.join(args.viz_dir, result["video_pair_id"])
        generate_error_report(result["video_pair_id"], str(e), pair_dir)

    # Remove intermediate tensors before writing to JSONL
    for key in ["src_frames_repr", "edit_frames_repr", "de_maps_repr",
                "masks_repr", "coverages_repr", "median_map", "iqr_map"]:
        result.pop(key, None)
```

Pass `visualize=args.visualize` to `score_video_pair_gpu()`.

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `pytest tests/ -v --ignore=tests/test_gpu_scorer.py`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add run.py
git commit -m "feat: wire --visualize and --viz-dir flags into run.py scoring loop"
```

---

## Task 8: Store `src_path`/`edited_path` in JSONL for boundary subset CSV

**Files:**
- Modify: `run.py` (scoring loop, around line 171-183)

The `boundary_subset.csv` needs original video paths. Currently the JSONL only stores `video_pair_id`. We need to also store the source and edited video paths.

- [ ] **Step 1: Add src_path and edited_path to result dict before JSONL write**

In `run.py`, before writing the result to JSONL, add:

```python
result["src_path"] = str(src_path)
result["edited_path"] = str(edited_path)
```

This uses the path variables already available in the scoring loop.

- [ ] **Step 2: Run existing tests**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add run.py
git commit -m "feat: include src_path/edited_path in JSONL output for calibration subset"
```

---

## Task 9: End-to-end integration test

**Files:**
- Create: `tests/test_calibration_e2e.py`

A test that exercises the full workflow with synthetic data (no GPU, no real video files). Tests the analyze → build-reports → evaluate pipeline.

- [ ] **Step 1: Write end-to-end test**

```python
# tests/test_calibration_e2e.py
import json
import os
import subprocess
import sys
import tempfile


def test_full_calibration_workflow():
    """Full workflow: analyze → build-reports → evaluate."""
    import random
    random.seed(123)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Create synthetic scores.jsonl
        scores_path = os.path.join(tmpdir, "scores.jsonl")
        with open(scores_path, "w") as f:
            for i in range(100):
                record = {
                    "video_pair_id": f"clip_{i:03d}",
                    "global_shift_score": random.uniform(0.2, 5.0),
                    "local_diff_score": random.uniform(0.3, 8.0),
                    "temporal_instability": random.uniform(0.05, 2.0),
                    "mask_coverage_ratio": random.uniform(0.0, 0.5),
                    "src_path": f"/data/src/clip_{i:03d}.mp4",
                    "edited_path": f"/data/edit/clip_{i:03d}.mp4",
                }
                f.write(json.dumps(record) + "\n")

        # Step 2: Run analyze
        result = subprocess.run(
            [sys.executable, "-m", "vid_color_filter.calibration", "analyze",
             "--scores", scores_path, "--output-dir", tmpdir],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert os.path.exists(os.path.join(tmpdir, "boundary_cases.json"))

        # Step 3: Create dummy viz directories for boundary cases
        with open(os.path.join(tmpdir, "boundary_cases.json")) as f:
            boundary = json.load(f)
        reports_dir = os.path.join(tmpdir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        for case in boundary:
            pair_dir = os.path.join(reports_dir, case["video_pair_id"])
            os.makedirs(pair_dir, exist_ok=True)
            # Create minimal report.html placeholder
            with open(os.path.join(pair_dir, "report.html"), "w") as f:
                f.write("<html><body>placeholder</body></html>")

        # Step 4: Build reports (index page)
        result = subprocess.run(
            [sys.executable, "-m", "vid_color_filter.calibration", "build-reports",
             "--boundary-cases", os.path.join(tmpdir, "boundary_cases.json"),
             "--viz-dir", reports_dir,
             "--output-dir", tmpdir],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert os.path.exists(os.path.join(reports_dir, "index.html"))

        # Step 5: Create synthetic annotations
        annotations = []
        for case in boundary[:20]:  # annotate first 20
            gs = case["global_shift_score"]
            annotations.append({
                "video_pair_id": case["video_pair_id"],
                "label": "pass" if gs < 2.0 else "fail",
            })
        ann_path = os.path.join(tmpdir, "annotations.json")
        with open(ann_path, "w") as f:
            json.dump(annotations, f)

        # Step 6: Evaluate
        result = subprocess.run(
            [sys.executable, "-m", "vid_color_filter.calibration", "evaluate",
             "--scores", scores_path, "--annotations", ann_path,
             "--output-dir", tmpdir],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert os.path.exists(os.path.join(tmpdir, "grid_search_results.html"))
        assert "F1" in result.stdout or "Best" in result.stdout
```

- [ ] **Step 2: Run the end-to-end test**

Run: `pytest tests/test_calibration_e2e.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_calibration_e2e.py
git commit -m "test: add end-to-end calibration workflow integration test"
```

---

## Task 10: Final verification

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/test_gpu_scorer.py`
Expected: All PASS (GPU tests skipped if no GPU)

- [ ] **Step 2: Run GPU tests (if GPU available)**

Run: `pytest tests/test_gpu_scorer.py -v`
Expected: All PASS

- [ ] **Step 3: Verify CLI help**

Run: `python -m vid_color_filter.calibration --help`
Run: `python -m vid_color_filter.calibration analyze --help`
Expected: Help text shows all expected arguments

- [ ] **Step 4: Final commit if any cleanup needed**

```bash
git add -A && git commit -m "chore: final cleanup for calibration experiment"
```
