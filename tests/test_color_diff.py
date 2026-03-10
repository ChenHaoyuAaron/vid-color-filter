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
        edited[:32, :] = [255, 0, 0]
        mask = np.zeros((64, 64), dtype=bool)
        mask[:32, :] = True

        result = compute_mean_ciede2000(src, edited, mask)

        assert result == pytest.approx(0.0, abs=0.01)

    def test_all_masked_returns_nan(self):
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=bool)

        result = compute_mean_ciede2000(frame, frame.copy(), mask)

        assert np.isnan(result)
