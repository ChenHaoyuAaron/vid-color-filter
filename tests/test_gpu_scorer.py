import numpy as np
import cv2
import pytest
import torch

from vid_color_filter.gpu.batch_scorer import score_video_pair_gpu, _select_representative_indices
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


class TestNewScoringPipeline:
    def test_output_has_new_fields(self):
        src = torch.full((16, 64, 64, 3), 128, dtype=torch.uint8, device=DEVICE)
        edited = src.clone()
        result = score_video_pair_gpu(src, edited, src_path="test.mp4", use_scielab=True)
        assert "global_shift_score" in result
        assert "local_diff_score" in result
        assert "temporal_instability" in result
        assert "pass_global" in result
        assert "pass_local" in result
        assert "pass" in result
        assert "per_frame_mean_delta_e" in result
        assert "mean_delta_e_per_frame" in result
        assert "mask_coverage_ratio" in result

    def test_identical_videos_pass(self):
        src = torch.full((16, 64, 64, 3), 128, dtype=torch.uint8, device=DEVICE)
        result = score_video_pair_gpu(src, src.clone(), src_path="test.mp4", use_scielab=True)
        assert result["pass"] == True
        assert result["global_shift_score"] < 1.0

    def test_color_shifted_video_detected(self):
        src = torch.full((16, 64, 64, 3), 128, dtype=torch.uint8, device=DEVICE)
        edited = torch.full((16, 64, 64, 3), 148, dtype=torch.uint8, device=DEVICE)
        result = score_video_pair_gpu(src, edited, src_path="test.mp4", use_scielab=True, global_threshold=1.0)
        assert result["global_shift_score"] > 1.0
        assert result["pass_global"] == False

    def test_backward_compat_mode(self):
        src = torch.full((8, 64, 64, 3), 128, dtype=torch.uint8, device=DEVICE)
        result = score_video_pair_gpu(src, src.clone(), src_path="test.mp4")
        assert "max_mean_delta_e" in result
        assert "mean_delta_e_per_frame" in result

    def test_visualize_returns_intermediates(self):
        src = torch.full((8, 64, 64, 3), 128, dtype=torch.uint8, device=DEVICE)
        edited = torch.full((8, 64, 64, 3), 140, dtype=torch.uint8, device=DEVICE)
        result = score_video_pair_gpu(
            src, edited, src_path="test.mp4",
            use_scielab=True, visualize=True, chunk_size=4,
        )
        assert "src_frames_repr" in result
        assert "edit_frames_repr" in result
        assert "de_maps_repr" in result
        assert "masks_repr" in result
        assert "coverages_repr" in result
        assert "median_map" in result
        assert "iqr_map" in result
        assert result["src_frames_repr"].shape[0] <= 5
        assert result["median_map"].ndim == 2

    def test_no_visualize_no_intermediates(self):
        src = torch.full((8, 64, 64, 3), 128, dtype=torch.uint8, device=DEVICE)
        result = score_video_pair_gpu(
            src, src.clone(), src_path="test.mp4",
            use_scielab=True, visualize=False, chunk_size=4,
        )
        assert "src_frames_repr" not in result
        assert "median_map" not in result


class TestSelectRepresentativeIndices:
    def test_fewer_than_max(self):
        indices = _select_representative_indices([1.0, 2.0, 3.0], max_repr=5)
        assert indices == [0, 1, 2]

    def test_exactly_max(self):
        indices = _select_representative_indices([1.0, 2.0, 3.0, 4.0, 5.0], max_repr=5)
        assert indices == [0, 1, 2, 3, 4]

    def test_more_than_max(self):
        mean_des = [0.5, 3.0, 1.5, 4.0, 0.1, 2.0, 5.0, 1.0]
        indices = _select_representative_indices(mean_des, max_repr=5)
        assert len(indices) == 5
        assert 4 in indices  # min (0.1)
        assert 6 in indices  # max (5.0)
        assert all(0 <= i < 8 for i in indices)
        assert indices == sorted(indices)  # sorted order
