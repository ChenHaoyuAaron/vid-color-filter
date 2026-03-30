import json
import os
import subprocess
import sys
import tempfile


def _write_scores(path, n=30):
    import random
    random.seed(42)
    with open(path, "w") as f:
        for i in range(n):
            record = {
                "video_pair_id": f"clip_{i:03d}",
                "global_shift_score": random.uniform(0.3, 4.5),
                "local_diff_score": random.uniform(0.5, 7.0),
                "temporal_instability": random.uniform(0.05, 1.5),
                "mask_coverage_ratio": random.uniform(0.0, 0.5),
                "src_path": f"/data/src/clip_{i:03d}.mp4",
                "edited_path": f"/data/edit/clip_{i:03d}.mp4",
            }
            f.write(json.dumps(record) + "\n")


def test_analyze_subcommand():
    """python -m vid_color_filter.calibration analyze produces expected outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scores_path = os.path.join(tmpdir, "scores.jsonl")
        _write_scores(scores_path)

        result = subprocess.run(
            [sys.executable, "-m", "vid_color_filter.calibration", "analyze",
             "--scores", scores_path, "--output-dir", tmpdir],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        assert os.path.exists(os.path.join(tmpdir, "distribution.html"))
        assert os.path.exists(os.path.join(tmpdir, "grid_search_preview.html"))
        assert os.path.exists(os.path.join(tmpdir, "boundary_cases.json"))
        assert os.path.exists(os.path.join(tmpdir, "boundary_subset.csv"))


def test_build_reports_subcommand():
    """python -m vid_color_filter.calibration build-reports creates index.html."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create boundary_cases.json
        boundary = [
            {"video_pair_id": "clip_001", "global_shift_score": 2.0,
             "local_diff_score": 3.0, "temporal_instability": 0.5},
        ]
        bc_path = os.path.join(tmpdir, "boundary_cases.json")
        with open(bc_path, "w") as f:
            json.dump(boundary, f)

        reports_dir = os.path.join(tmpdir, "reports")
        os.makedirs(reports_dir)

        result = subprocess.run(
            [sys.executable, "-m", "vid_color_filter.calibration", "build-reports",
             "--boundary-cases", bc_path, "--viz-dir", reports_dir],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert os.path.exists(os.path.join(reports_dir, "index.html"))


def test_evaluate_subcommand():
    """python -m vid_color_filter.calibration evaluate produces F1 results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scores_path = os.path.join(tmpdir, "scores.jsonl")
        _write_scores(scores_path, n=10)

        annotations = [
            {"video_pair_id": f"clip_{i:03d}", "label": "pass" if i < 5 else "fail"}
            for i in range(10)
        ]
        ann_path = os.path.join(tmpdir, "annotations.json")
        with open(ann_path, "w") as f:
            json.dump(annotations, f)

        result = subprocess.run(
            [sys.executable, "-m", "vid_color_filter.calibration", "evaluate",
             "--scores", scores_path, "--annotations", ann_path, "--output-dir", tmpdir],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert os.path.exists(os.path.join(tmpdir, "grid_search_results.html"))
        assert "Best" in result.stdout or "F1" in result.stdout
