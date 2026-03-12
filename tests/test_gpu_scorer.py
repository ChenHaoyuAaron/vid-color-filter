import numpy as np
import cv2
import pytest
import torch

from vid_color_filter.gpu.batch_scorer import score_video_pair_gpu
from vid_color_filter.frame_sampler import sample_frames_as_tensors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_video(path: str, color: tuple, num_frames: int = 10, h: int = 64, w: int = 64):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
    for _ in range(num_frames):
        frame = np.full((h, w, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_rgb_tensor(color, batch=8, h=64, w=64):
    return torch.full((batch, h, w, 3), 0, dtype=torch.uint8, device=DEVICE).fill_(0) + \
           torch.tensor(color, dtype=torch.uint8, device=DEVICE)


class TestScoreVideoPairGPU:
    def test_identical_frames_pass(self):
        t = _make_rgb_tensor((128, 128, 128))
        result = score_video_pair_gpu(t, t.clone(), src_path="test.mp4")
        assert result["max_mean_delta_e"] == pytest.approx(0.0, abs=0.1)
        assert result["pass"] is True

    def test_shifted_frames_fail(self):
        src = _make_rgb_tensor((128, 128, 128))
        edited = _make_rgb_tensor((180, 128, 128))
        result = score_video_pair_gpu(src, edited, src_path="test.mp4", threshold=2.0)
        assert result["max_mean_delta_e"] > 2.0
        assert result["pass"] is False

    def test_result_keys(self):
        t = _make_rgb_tensor((128, 128, 128))
        result = score_video_pair_gpu(t, t.clone(), src_path="test.mp4")
        assert "video_pair_id" in result
        assert "mean_delta_e_per_frame" in result
        assert "max_mean_delta_e" in result
        assert "pass" in result
        assert "mask_coverage_ratio" in result

    def test_metric_selection(self):
        src = _make_rgb_tensor((128, 128, 128))
        edited = _make_rgb_tensor((150, 128, 128))
        for metric in ("cie76", "cie94", "ciede2000"):
            result = score_video_pair_gpu(src, edited, metric=metric)
            assert result["max_mean_delta_e"] > 0.0

    def test_with_video_files(self, tmp_path):
        """End-to-end: read video files -> tensor -> GPU score."""
        src = str(tmp_path / "src.mp4")
        edited = str(tmp_path / "edited.mp4")
        _make_video(src, color=(128, 128, 128))
        _make_video(edited, color=(128, 128, 128))

        src_t, edited_t = sample_frames_as_tensors(src, edited, num_frames=4, device=DEVICE)
        assert src_t is not None
        result = score_video_pair_gpu(src_t, edited_t, src_path=src)
        assert result["pass"] is True
