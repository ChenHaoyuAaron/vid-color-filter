import json
import csv
import os
import random

import pytest

from vid_color_filter.calibration import (
    load_scores,
    generate_distribution_html,
    grid_search_preview,
    select_boundary_cases,
    evaluate_annotations,
    GLOBAL_THRESHOLDS,
    LOCAL_THRESHOLDS,
)


def _write_scores_jsonl(path, n=50):
    """Write synthetic scores JSONL with random values (seed 42)."""
    rng = random.Random(42)
    records = []
    for i in range(n):
        rec = {
            "video_pair_id": f"pair_{i:03d}",
            "src_path": f"/videos/src_{i:03d}.mp4",
            "edited_path": f"/videos/edited_{i:03d}.mp4",
            "global_shift_score": round(rng.uniform(0.0, 6.0), 3),
            "local_diff_score": round(rng.uniform(0.0, 10.0), 3),
            "temporal_instability": round(rng.uniform(0.0, 3.0), 3),
        }
        records.append(rec)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return records


class TestLoadScores:
    def test_load_scores_reads_jsonl(self, tmp_path):
        jsonl_path = tmp_path / "scores.jsonl"
        written = _write_scores_jsonl(str(jsonl_path), n=50)
        scores = load_scores(str(jsonl_path))
        assert len(scores) == 50
        assert isinstance(scores, list)
        assert isinstance(scores[0], dict)
        assert scores[0]["video_pair_id"] == written[0]["video_pair_id"]
        assert "global_shift_score" in scores[0]
        assert "local_diff_score" in scores[0]
        assert "temporal_instability" in scores[0]


class TestGenerateDistributionHtml:
    def test_generate_distribution_html(self, tmp_path):
        jsonl_path = tmp_path / "scores.jsonl"
        _write_scores_jsonl(str(jsonl_path), n=50)
        scores = load_scores(str(jsonl_path))

        out_path = str(tmp_path / "distribution.html")
        generate_distribution_html(scores, out_path)

        assert os.path.exists(out_path)
        with open(out_path) as f:
            html = f.read()
        # Should contain base64-encoded images
        assert "data:image/png;base64," in html
        # Should be dark themed
        assert "background" in html.lower()


class TestGridSearchPreview:
    def test_grid_search_preview(self, tmp_path):
        jsonl_path = tmp_path / "scores.jsonl"
        _write_scores_jsonl(str(jsonl_path), n=50)
        scores = load_scores(str(jsonl_path))

        out_path = str(tmp_path / "grid_search.html")
        grid_search_preview(scores, out_path)

        assert os.path.exists(out_path)
        with open(out_path) as f:
            html = f.read()
        assert "data:image/png;base64," in html


class TestSelectBoundaryCases:
    def test_select_boundary_cases(self, tmp_path):
        jsonl_path = tmp_path / "scores.jsonl"
        _write_scores_jsonl(str(jsonl_path), n=50)
        scores = load_scores(str(jsonl_path))

        output_dir = str(tmp_path / "boundary_output")
        os.makedirs(output_dir, exist_ok=True)

        boundary_cases, csv_path = select_boundary_cases(scores, output_dir)

        # Should return non-empty results
        assert len(boundary_cases) > 0
        assert isinstance(boundary_cases[0], dict)

        # boundary_cases.json should exist
        json_path = os.path.join(output_dir, "boundary_cases.json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            loaded = json.load(f)
        assert len(loaded) == len(boundary_cases)

        # boundary_subset.csv should exist
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0
        assert "video1_path" in rows[0]
        assert "video2_path" in rows[0]


class TestEvaluateAnnotations:
    def test_evaluate_annotations(self, tmp_path):
        jsonl_path = tmp_path / "scores.jsonl"
        _write_scores_jsonl(str(jsonl_path), n=50)
        scores = load_scores(str(jsonl_path))

        # Create annotations for a subset of pairs
        annotations = []
        for s in scores[:20]:
            # Label as pass if both scores are low
            label = "pass" if s["global_shift_score"] < 3.0 and s["local_diff_score"] < 5.0 else "fail"
            annotations.append({
                "video_pair_id": s["video_pair_id"],
                "label": label,
            })

        out_path = str(tmp_path / "evaluation.html")
        result = evaluate_annotations(scores, annotations, out_path)

        assert "best_global_threshold" in result
        assert "best_local_threshold" in result
        assert "best_f1" in result
        assert isinstance(result["best_f1"], float)
        assert 0.0 <= result["best_f1"] <= 1.0
        assert os.path.exists(out_path)
