"""Tests for the HTML report generator."""

import json
import os
from pathlib import Path

import pytest

from vid_color_filter.report import (
    generate_error_report,
    generate_index_page,
    generate_pair_report,
)


@pytest.fixture
def pair_dir(tmp_path):
    """Create a temporary pair directory."""
    d = tmp_path / "vp_001"
    d.mkdir()
    return d


@pytest.fixture
def reports_dir(tmp_path):
    """Create a temporary reports directory."""
    return tmp_path


@pytest.fixture
def sample_scores():
    return {
        "video_pair_id": "vp_001",
        "global_shift_score": 0.85,
        "local_diff_score": 0.42,
        "temporal_instability": 0.13,
        "mask_coverage_ratio": 0.67,
    }


@pytest.fixture
def sample_cases():
    return [
        {
            "video_pair_id": "vp_001",
            "global_shift_score": 0.85,
            "local_diff_score": 0.42,
            "temporal_instability": 0.13,
        },
        {
            "video_pair_id": "vp_002",
            "global_shift_score": 0.95,
            "local_diff_score": 0.31,
            "temporal_instability": 0.08,
        },
        {
            "video_pair_id": "vp_003",
            "global_shift_score": 0.55,
            "local_diff_score": 0.72,
            "temporal_instability": 0.25,
        },
    ]


class TestGeneratePairReport:
    def test_generate_pair_report_creates_html(self, sample_scores, pair_dir):
        generate_pair_report(sample_scores, pair_dir, n_frames=5)

        report_path = pair_dir / "report.html"
        assert report_path.exists()

        html = report_path.read_text()

        # Check it's valid HTML structure
        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "</html>" in html

        # Dark theme
        assert "#1a1a2e" in html
        assert "#e0e0e0" in html

        # Metrics table should contain score values
        assert "0.85" in html  # global_shift_score
        assert "0.42" in html  # local_diff_score
        assert "0.13" in html  # temporal_instability
        assert "0.67" in html  # mask_coverage_ratio

        # Frame comparison with tab switching for N frames
        for i in range(5):
            assert f"src_frame_{i}.png" in html

        # Heatmap section
        assert "heatmap" in html.lower()

        # Mask section
        assert "mask" in html.lower()

        # Temporal maps (median + IQR side by side)
        assert "median" in html.lower()
        assert "iqr" in html.lower()

        # Annotation buttons (pass/fail)
        assert "pass" in html.lower()
        assert "fail" in html.lower()

        # localStorage key for annotations
        assert "calibration_annotation_vp_001" in html

        # Back link to ../index.html
        assert "../index.html" in html

    def test_generate_pair_report_default_n_frames(self, sample_scores, pair_dir):
        """Default n_frames should be 5."""
        generate_pair_report(sample_scores, pair_dir)

        html = (pair_dir / "report.html").read_text()
        assert "src_frame_4.png" in html
        assert "src_frame_5.png" not in html

    def test_generate_pair_report_custom_n_frames(self, sample_scores, pair_dir):
        """Support custom number of frames."""
        generate_pair_report(sample_scores, pair_dir, n_frames=3)

        html = (pair_dir / "report.html").read_text()
        assert "src_frame_2.png" in html
        assert "src_frame_3.png" not in html

    def test_generate_pair_report_relative_image_paths(self, sample_scores, pair_dir):
        """Image paths should be relative (no absolute paths)."""
        generate_pair_report(sample_scores, pair_dir, n_frames=2)

        html = (pair_dir / "report.html").read_text()
        # Should NOT contain absolute paths to images
        assert str(pair_dir) not in html


class TestGenerateIndexPage:
    def test_generate_index_page_creates_html(self, sample_cases, reports_dir):
        generate_index_page(sample_cases, reports_dir)

        index_path = reports_dir / "index.html"
        assert index_path.exists()

        html = index_path.read_text()

        # Check HTML structure
        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "</html>" in html

        # Dark theme matching pair reports
        assert "#1a1a2e" in html
        assert "#e0e0e0" in html

        # All video pair IDs present
        assert "vp_001" in html
        assert "vp_002" in html
        assert "vp_003" in html

        # Links to report pages
        assert "vp_001/report.html" in html
        assert "vp_002/report.html" in html
        assert "vp_003/report.html" in html

        # Sorted by global_shift_score descending:
        # vp_002 (0.95) should appear before vp_001 (0.85) before vp_003 (0.55)
        pos_002 = html.index("vp_002")
        pos_001 = html.index("vp_001")
        pos_003 = html.index("vp_003")
        assert pos_002 < pos_001 < pos_003

        # Sortable columns via JS
        assert "sort" in html.lower()

        # Annotation status from localStorage
        assert "localStorage" in html

        # Progress counter
        assert "annotated" in html.lower() or "progress" in html.lower()

        # Export button
        assert "export" in html.lower()
        assert "annotations.json" in html

        # visibilitychange listener
        assert "visibilitychange" in html

    def test_generate_index_page_score_values(self, sample_cases, reports_dir):
        """Score values should appear in the table."""
        generate_index_page(sample_cases, reports_dir)
        html = (reports_dir / "index.html").read_text()

        assert "0.95" in html
        assert "0.85" in html
        assert "0.55" in html


class TestGenerateErrorReport:
    def test_generate_error_report_creates_html(self, pair_dir):
        error_msg = "FFmpeg failed: codec not supported"
        generate_error_report("vp_001", error_msg, pair_dir)

        report_path = pair_dir / "report.html"
        assert report_path.exists()

        html = report_path.read_text()

        # HTML structure
        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "</html>" in html

        # Error message displayed
        assert "FFmpeg failed: codec not supported" in html

        # Red box / error styling
        assert "red" in html.lower() or "#ff" in html.lower() or "error" in html.lower()

        # Back link
        assert "../index.html" in html

        # Dark theme
        assert "#1a1a2e" in html

    def test_generate_error_report_escapes_html(self, pair_dir):
        """Error messages with HTML characters should be escaped."""
        error_msg = '<script>alert("xss")</script>'
        generate_error_report("vp_001", error_msg, pair_dir)

        html = (pair_dir / "report.html").read_text()
        # The raw script tag should NOT appear unescaped
        assert '<script>alert("xss")</script>' not in html
