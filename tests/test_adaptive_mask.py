import numpy as np
import pytest
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _make_lab_pair(h=64, w=64, edit_region=None, edit_shift=50.0):
    src = torch.full((1, h, w, 3), 50.0, device=DEVICE)
    edited = src.clone()
    if edit_region is not None:
        r = edit_region
        edited[0, r[0]:r[1], r[2]:r[3], :] += edit_shift
    return src, edited

class TestOtsuThreshold:
    def test_returns_float(self):
        from vid_color_filter.gpu.adaptive_mask import otsu_threshold
        values = torch.randn(1000, device=DEVICE).abs()
        t = otsu_threshold(values)
        assert isinstance(t, float) or t.dim() == 0

    def test_separates_bimodal(self):
        from vid_color_filter.gpu.adaptive_mask import otsu_threshold
        low = torch.full((500,), 1.0, device=DEVICE)
        high = torch.full((500,), 10.0, device=DEVICE)
        values = torch.cat([low, high])
        t = otsu_threshold(values)
        assert 2.0 < float(t) < 9.0

class TestAdaptiveMask:
    def test_identical_frames_empty_mask(self):
        from vid_color_filter.gpu.adaptive_mask import generate_adaptive_mask
        src, edited = _make_lab_pair()
        masks, coverages = generate_adaptive_mask(src, src)
        assert not masks.any()
        assert coverages[0].item() == pytest.approx(0.0, abs=0.01)

    def test_detects_large_edit(self):
        from vid_color_filter.gpu.adaptive_mask import generate_adaptive_mask
        src, edited = _make_lab_pair(edit_region=(20, 44, 20, 44), edit_shift=50.0)
        masks, coverages = generate_adaptive_mask(src, edited)
        assert masks[0, 30, 30].item() == True
        assert coverages[0].item() > 0.05

    def test_output_shapes(self):
        from vid_color_filter.gpu.adaptive_mask import generate_adaptive_mask
        src, edited = _make_lab_pair(h=100, w=80)
        masks, coverages = generate_adaptive_mask(src, edited)
        assert masks.shape == (1, 100, 80)
        assert masks.dtype == torch.bool
        assert coverages.shape == (1,)

    def test_fixed_threshold_fallback(self):
        from vid_color_filter.gpu.adaptive_mask import generate_adaptive_mask
        src, edited = _make_lab_pair(edit_region=(20, 44, 20, 44), edit_shift=50.0)
        masks, coverages = generate_adaptive_mask(src, edited, diff_threshold=5.0)
        assert masks[0, 30, 30].item() == True

    def test_batch_processing(self):
        from vid_color_filter.gpu.adaptive_mask import generate_adaptive_mask
        src = torch.full((4, 64, 64, 3), 50.0, device=DEVICE)
        edited = src.clone()
        edited[2, 20:40, 20:40, :] += 50.0
        masks, coverages = generate_adaptive_mask(src, edited)
        assert masks.shape == (4, 64, 64)
        assert coverages[0].item() < coverages[2].item()
