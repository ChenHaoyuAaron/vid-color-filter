import numpy as np
import pytest
from vid_color_filter.mask_generator import generate_edit_mask


class TestGenerateEditMask:
    def test_identical_frames_produce_empty_mask(self):
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask, coverage = generate_edit_mask(frame, frame.copy())

        assert mask.shape == (64, 64)
        assert mask.dtype == bool
        assert not mask.any()
        assert coverage == 0.0

    def test_detects_edited_region(self):
        src = np.full((64, 64, 3), 128, dtype=np.uint8)
        edited = src.copy()
        edited[20:40, 20:40] = [255, 0, 0]

        mask, coverage = generate_edit_mask(src, edited)

        assert mask[30, 30] == True
        assert coverage > 0.05

    def test_dilation_expands_mask(self):
        src = np.full((100, 100, 3), 128, dtype=np.uint8)
        edited = src.copy()
        # Use a region larger than min_component_size to survive filtering
        edited[45:55, 45:55] = [255, 0, 0]

        mask_no_dilate, _ = generate_edit_mask(src, edited, dilate_kernel=1)
        mask_dilated, _ = generate_edit_mask(src, edited, dilate_kernel=21)

        assert mask_dilated.sum() > mask_no_dilate.sum()

    def test_high_coverage_flagged(self):
        src = np.full((64, 64, 3), 128, dtype=np.uint8)
        edited = np.full((64, 64, 3), 200, dtype=np.uint8)

        mask, coverage = generate_edit_mask(src, edited)

        assert coverage > 0.8
