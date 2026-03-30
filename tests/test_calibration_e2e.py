"""End-to-end integration test for the calibration workflow pipeline.

Tests the full sequence: generate scores -> analyze -> build-reports -> evaluate.
"""

import json
import os
import random
import subprocess
import sys
import tempfile


def _generate_synthetic_scores(path, n=50):
    """Generate synthetic score records with varied distributions.

    Creates a mix of clearly-passing, clearly-failing, and boundary cases.
    """
    random.seed(42)
    records = []
    for i in range(n):
        if i < 15:
            # Clearly passing: low scores
            gs = random.uniform(0.3, 1.2)
            ld = random.uniform(0.5, 1.8)
            ti = random.uniform(0.05, 0.3)
        elif i < 30:
            # Clearly failing: high scores
            gs = random.uniform(3.5, 5.0)
            ld = random.uniform(5.5, 8.0)
            ti = random.uniform(0.8, 1.5)
        else:
            # Boundary region: ambiguous scores
            gs = random.uniform(1.5, 3.5)
            ld = random.uniform(2.0, 5.5)
            ti = random.uniform(0.2, 0.9)

        record = {
            "video_pair_id": f"clip_{i:03d}",
            "global_shift_score": round(gs, 4),
            "local_diff_score": round(ld, 4),
            "temporal_instability": round(ti, 4),
            "mask_coverage_ratio": round(random.uniform(0.0, 0.5), 4),
            "src_path": f"/data/src/clip_{i:03d}.mp4",
            "edited_path": f"/data/edit/clip_{i:03d}.mp4",
        }
        records.append(record)

    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return records


def test_calibration_e2e_pipeline():
    """Test the entire calibration workflow end-to-end via CLI subcommands."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scores_path = os.path.join(tmpdir, "scores.jsonl")
        analyze_dir = os.path.join(tmpdir, "analyze_out")
        reports_dir = os.path.join(tmpdir, "reports")
        eval_dir = os.path.join(tmpdir, "eval_out")

        # ------------------------------------------------------------------
        # Step 1: Generate synthetic scores JSONL
        # ------------------------------------------------------------------
        records = _generate_synthetic_scores(scores_path, n=50)
        assert os.path.exists(scores_path)
        with open(scores_path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 50

        # Verify score structure
        first = json.loads(lines[0])
        for key in ("video_pair_id", "global_shift_score", "local_diff_score",
                     "temporal_instability", "mask_coverage_ratio",
                     "src_path", "edited_path"):
            assert key in first, f"Missing key {key} in score record"

        # ------------------------------------------------------------------
        # Step 2: Run analyze subcommand
        # ------------------------------------------------------------------
        result = subprocess.run(
            [sys.executable, "-m", "vid_color_filter.calibration", "analyze",
             "--scores", scores_path, "--output-dir", analyze_dir],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"analyze failed: {result.stderr}"

        # Verify all expected outputs exist
        distribution_html = os.path.join(analyze_dir, "distribution.html")
        grid_preview_html = os.path.join(analyze_dir, "grid_search_preview.html")
        boundary_json = os.path.join(analyze_dir, "boundary_cases.json")
        boundary_csv = os.path.join(analyze_dir, "boundary_subset.csv")

        assert os.path.exists(distribution_html), "distribution.html not created"
        assert os.path.exists(grid_preview_html), "grid_search_preview.html not created"
        assert os.path.exists(boundary_json), "boundary_cases.json not created"
        assert os.path.exists(boundary_csv), "boundary_subset.csv not created"

        # Verify distribution.html contains embedded charts
        with open(distribution_html) as f:
            dist_content = f.read()
        assert "data:image/png;base64," in dist_content
        assert "Score Distributions" in dist_content

        # Verify grid_search_preview.html
        with open(grid_preview_html) as f:
            grid_content = f.read()
        assert "data:image/png;base64," in grid_content
        assert "Grid Search Preview" in grid_content

        # Verify boundary_cases.json has valid content
        with open(boundary_json) as f:
            boundary_cases = json.load(f)
        assert isinstance(boundary_cases, list)
        assert len(boundary_cases) > 0, "No boundary cases selected"
        for bc in boundary_cases:
            assert "video_pair_id" in bc

        # Verify boundary_subset.csv has header and rows
        with open(boundary_csv) as f:
            csv_lines = f.readlines()
        assert len(csv_lines) >= 2, "CSV should have header + at least 1 row"
        assert "video1_path" in csv_lines[0]
        assert "video2_path" in csv_lines[0]

        # ------------------------------------------------------------------
        # Step 3: Run build-reports subcommand
        # ------------------------------------------------------------------
        os.makedirs(reports_dir, exist_ok=True)

        result = subprocess.run(
            [sys.executable, "-m", "vid_color_filter.calibration", "build-reports",
             "--boundary-cases", boundary_json, "--viz-dir", reports_dir],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"build-reports failed: {result.stderr}"

        index_html = os.path.join(reports_dir, "index.html")
        assert os.path.exists(index_html), "index.html not created by build-reports"
        with open(index_html) as f:
            index_content = f.read()
        assert len(index_content) > 0, "index.html is empty"

        # ------------------------------------------------------------------
        # Step 4: Create mock annotations from boundary cases
        # ------------------------------------------------------------------
        # Use deterministic labeling: pass if global_shift_score < 2.5 AND
        # local_diff_score < 3.5, else fail.
        annotations = []
        for bc in boundary_cases:
            label = (
                "pass"
                if bc["global_shift_score"] < 2.5 and bc["local_diff_score"] < 3.5
                else "fail"
            )
            annotations.append({
                "video_pair_id": bc["video_pair_id"],
                "label": label,
            })

        ann_path = os.path.join(tmpdir, "annotations.json")
        with open(ann_path, "w") as f:
            json.dump(annotations, f)

        # Verify we have both pass and fail labels
        pass_count = sum(1 for a in annotations if a["label"] == "pass")
        fail_count = sum(1 for a in annotations if a["label"] == "fail")
        assert pass_count > 0, "Need at least one pass annotation"
        assert fail_count > 0, "Need at least one fail annotation"

        # ------------------------------------------------------------------
        # Step 5: Run evaluate subcommand
        # ------------------------------------------------------------------
        result = subprocess.run(
            [sys.executable, "-m", "vid_color_filter.calibration", "evaluate",
             "--scores", scores_path, "--annotations", ann_path,
             "--output-dir", eval_dir],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"evaluate failed: {result.stderr}"

        grid_results_html = os.path.join(eval_dir, "grid_search_results.html")
        assert os.path.exists(grid_results_html), "grid_search_results.html not created"

        # Verify evaluate output mentions best F1
        assert "Best" in result.stdout or "F1" in result.stdout, (
            f"Expected F1/Best in stdout, got: {result.stdout}"
        )

        # Verify grid_search_results.html has content
        with open(grid_results_html) as f:
            eval_content = f.read()
        assert "data:image/png;base64," in eval_content
        assert "Annotation Evaluation" in eval_content
