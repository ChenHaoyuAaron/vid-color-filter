import numpy as np
import pytest
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TestTemporalAggregate:
    def test_output_shapes(self):
        from vid_color_filter.gpu.temporal_aggregator import temporal_aggregate
        de_maps = torch.rand(16, 64, 64, device=DEVICE)
        masks = torch.zeros(16, 64, 64, dtype=torch.bool, device=DEVICE)
        median_map, iqr_map = temporal_aggregate(de_maps, masks)
        assert median_map.shape == (64, 64)
        assert iqr_map.shape == (64, 64)

    def test_stable_signal_low_iqr(self):
        from vid_color_filter.gpu.temporal_aggregator import temporal_aggregate
        de_maps = torch.full((32, 64, 64), 2.0, device=DEVICE)
        masks = torch.zeros(32, 64, 64, dtype=torch.bool, device=DEVICE)
        median_map, iqr_map = temporal_aggregate(de_maps, masks)
        assert median_map[32, 32].item() == pytest.approx(2.0, abs=0.1)
        assert iqr_map[32, 32].item() == pytest.approx(0.0, abs=0.1)

    def test_noisy_signal_high_iqr(self):
        from vid_color_filter.gpu.temporal_aggregator import temporal_aggregate
        torch.manual_seed(42)
        de_maps = torch.randn(32, 64, 64, device=DEVICE).abs() * 5
        masks = torch.zeros(32, 64, 64, dtype=torch.bool, device=DEVICE)
        _, iqr_map = temporal_aggregate(de_maps, masks)
        assert iqr_map.mean().item() > 1.0

    def test_masked_pixels_excluded(self):
        from vid_color_filter.gpu.temporal_aggregator import temporal_aggregate
        de_maps = torch.full((10, 64, 64), 3.0, device=DEVICE)
        masks = torch.zeros(10, 64, 64, dtype=torch.bool, device=DEVICE)
        masks[:8, 0, 0] = True
        median_map, _ = temporal_aggregate(de_maps, masks)
        assert torch.isnan(median_map[0, 0])

    def test_partially_masked_pixel_uses_unmasked(self):
        from vid_color_filter.gpu.temporal_aggregator import temporal_aggregate
        de_maps = torch.full((10, 64, 64), 3.0, device=DEVICE)
        de_maps[0, 10, 10] = 100.0
        masks = torch.zeros(10, 64, 64, dtype=torch.bool, device=DEVICE)
        masks[0, 10, 10] = True
        median_map, _ = temporal_aggregate(de_maps, masks)
        assert median_map[10, 10].item() == pytest.approx(3.0, abs=0.1)

class TestComputeScores:
    def test_zero_de_all_pass(self):
        from vid_color_filter.gpu.temporal_aggregator import compute_scores
        median_map = torch.zeros(64, 64, device=DEVICE)
        iqr_map = torch.zeros(64, 64, device=DEVICE)
        scores = compute_scores(median_map, iqr_map)
        assert scores["pass_global"] == True
        assert scores["pass_local"] == True
        assert scores["pass"] == True

    def test_high_global_shift_fails(self):
        from vid_color_filter.gpu.temporal_aggregator import compute_scores
        median_map = torch.full((64, 64), 5.0, device=DEVICE)
        iqr_map = torch.zeros(64, 64, device=DEVICE)
        scores = compute_scores(median_map, iqr_map, global_threshold=2.0)
        assert scores["pass_global"] == False
        assert scores["pass"] == False

    def test_high_local_diff_fails(self):
        from vid_color_filter.gpu.temporal_aggregator import compute_scores
        median_map = torch.full((64, 64), 1.0, device=DEVICE)
        median_map[0:5, 0:5] = 10.0
        iqr_map = torch.zeros(64, 64, device=DEVICE)
        scores = compute_scores(median_map, iqr_map, local_threshold=3.0)
        assert scores["pass_local"] == False
        assert scores["pass"] == False

    def test_scores_are_floats(self):
        from vid_color_filter.gpu.temporal_aggregator import compute_scores
        median_map = torch.rand(64, 64, device=DEVICE)
        iqr_map = torch.rand(64, 64, device=DEVICE)
        scores = compute_scores(median_map, iqr_map)
        assert isinstance(scores["global_shift_score"], float)
        assert isinstance(scores["local_diff_score"], float)
        assert isinstance(scores["temporal_instability"], float)
