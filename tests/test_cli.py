import json
import numpy as np
import cv2
import pytest
from vid_color_filter.cli import run_batch


class TestRunBatch:
    def test_processes_video_pairs_and_writes_jsonl(self, tmp_path):
        pairs = []
        for i in range(2):
            src = str(tmp_path / f"src_{i}.mp4")
            edited = str(tmp_path / f"edited_{i}.mp4")
            self._make_video(src, color=(128, 128, 128))
            self._make_video(edited, color=(128, 128, 128))
            pairs.append((src, edited))

        output = str(tmp_path / "results.jsonl")
        run_batch(pairs, output, num_workers=2, num_frames=4)

        with open(output) as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        for r in results:
            assert "max_mean_delta_e" in r
            assert "pass" in r

    @staticmethod
    def _make_video(path: str, color: tuple, num_frames: int = 10, h: int = 64, w: int = 64):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
        for _ in range(num_frames):
            frame = np.full((h, w, 3), color, dtype=np.uint8)
            writer.write(frame)
        writer.release()
