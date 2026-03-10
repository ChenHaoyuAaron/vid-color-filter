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
