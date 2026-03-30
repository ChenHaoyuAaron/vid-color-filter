"""Tests for GPU visualizer PNG rendering."""

import os

import numpy as np
import pytest

from vid_color_filter.gpu.visualizer import (
    generate_pair_visualizations,
    render_heatmap,
    render_mask_overlay,
    render_temporal_map,
    save_frame,
)


@pytest.fixture
def sample_frame():
    """Create a 120x160 RGB uint8 frame with non-trivial content."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)


@pytest.fixture
def sample_de_map():
    """Create a 120x160 float32 delta-E map."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 15.0, (120, 160)).astype(np.float32)


@pytest.fixture
def sample_mask():
    """Create a 120x160 boolean mask (True = valid)."""
    mask = np.ones((120, 160), dtype=bool)
    mask[0:30, 0:40] = False  # top-left corner masked out
    return mask


@pytest.fixture
def sample_temporal_map():
    """Create a 120x160 float32 temporal map with some NaNs."""
    rng = np.random.default_rng(42)
    tmap = rng.uniform(0.0, 8.0, (120, 160)).astype(np.float32)
    tmap[50:60, 70:80] = np.nan
    return tmap


class TestRenderHeatmap:
    def test_render_heatmap_creates_png(
        self, tmp_path, sample_frame, sample_de_map, sample_mask
    ):
        out = tmp_path / "heatmap.png"
        render_heatmap(sample_frame, sample_de_map, sample_mask, str(out))
        assert out.exists()
        assert out.stat().st_size > 1000  # non-trivial PNG

    def test_render_heatmap_respects_vmax(
        self, tmp_path, sample_frame, sample_de_map, sample_mask
    ):
        """Values above vmax should be clipped; lower vmax => different render."""
        out_low = tmp_path / "heatmap_low.png"
        out_high = tmp_path / "heatmap_high.png"
        render_heatmap(
            sample_frame, sample_de_map, sample_mask, str(out_low), vmax=2.0
        )
        render_heatmap(
            sample_frame, sample_de_map, sample_mask, str(out_high), vmax=20.0
        )
        assert out_low.exists()
        assert out_high.exists()
        # Different vmax should produce different file sizes (different pixel values)
        assert out_low.stat().st_size != out_high.stat().st_size


class TestRenderMaskOverlay:
    def test_render_mask_overlay_creates_png(
        self, tmp_path, sample_frame, sample_mask
    ):
        out = tmp_path / "mask.png"
        render_mask_overlay(sample_frame, sample_mask, str(out), coverage=0.85)
        assert out.exists()
        assert out.stat().st_size > 1000


class TestRenderTemporalMap:
    def test_render_temporal_map_creates_png(self, tmp_path, sample_temporal_map):
        out = tmp_path / "temporal.png"
        render_temporal_map(sample_temporal_map, str(out))
        assert out.exists()
        assert out.stat().st_size > 1000


class TestSaveFrame:
    def test_save_frame_creates_png(self, tmp_path, sample_frame):
        out = tmp_path / "frame.png"
        save_frame(sample_frame, str(out))
        assert out.exists()
        assert out.stat().st_size > 500


class TestGeneratePairVisualizations:
    def test_generate_pair_visualizations_creates_all_files(
        self, tmp_path, sample_frame, sample_de_map, sample_mask, sample_temporal_map
    ):
        n_frames = 3
        data = {
            "video_pair_id": "test_pair_001",
            "src_frames_repr": [sample_frame.copy() for _ in range(n_frames)],
            "edit_frames_repr": [sample_frame.copy() for _ in range(n_frames)],
            "de_maps_repr": [sample_de_map.copy() for _ in range(n_frames)],
            "masks_repr": [sample_mask.copy() for _ in range(n_frames)],
            "coverages_repr": [0.8, 0.9, 0.75],
            "median_map": sample_temporal_map.copy(),
            "iqr_map": sample_temporal_map.copy(),
        }
        generate_pair_visualizations(data, str(tmp_path), vmax=10.0)

        pair_dir = tmp_path / "test_pair_001"
        assert pair_dir.is_dir()

        # Check per-frame files
        for i in range(n_frames):
            assert (pair_dir / f"src_frame_{i:02d}.png").exists()
            assert (pair_dir / f"edit_frame_{i:02d}.png").exists()
            assert (pair_dir / f"de_heatmap_{i:02d}.png").exists()
            assert (pair_dir / f"mask_overlay_{i:02d}.png").exists()

        # Check temporal maps
        assert (pair_dir / "median_map.png").exists()
        assert (pair_dir / "iqr_map.png").exists()
