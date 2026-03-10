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
