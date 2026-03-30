"""Calibration analysis module for threshold tuning and F1 evaluation."""

import csv
import io
import json
import base64
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Grid-search threshold constants
# ---------------------------------------------------------------------------
GLOBAL_THRESHOLDS = [round(0.5 + i * 0.25, 2) for i in range(19)]  # 0.5 to 5.0
LOCAL_THRESHOLDS = [round(1.0 + i * 0.5, 1) for i in range(15)]    # 1.0 to 8.0

# ---------------------------------------------------------------------------
# Dark-themed HTML wrapper
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ background-color: #1e1e1e; color: #d4d4d4; font-family: sans-serif; padding: 20px; }}
h1, h2 {{ color: #e0e0e0; }}
img {{ max-width: 100%; margin: 10px 0; }}
.charts {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
.chart {{ text-align: center; }}
</style>
</head>
<body>
<h1>{title}</h1>
{body}
</body>
</html>
"""


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_scores(path) -> list[dict]:
    """Read JSONL file, return list of dicts."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_distribution_html(scores, out_path):
    """Generate HTML with distribution histograms and 2D scatter plot."""
    global_vals = [s["global_shift_score"] for s in scores]
    local_vals = [s["local_diff_score"] for s in scores]
    temporal_vals = [s["temporal_instability"] for s in scores]

    images = []

    # Histograms for each metric
    for vals, label in [
        (global_vals, "global_shift_score"),
        (local_vals, "local_diff_score"),
        (temporal_vals, "temporal_instability"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#1e1e1e")
        ax.set_facecolor("#2d2d2d")
        ax.hist(vals, bins=50, color="#4fc3f7", edgecolor="#1e1e1e")
        ax.set_title(label, color="#e0e0e0")
        ax.set_xlabel("Score", color="#d4d4d4")
        ax.set_ylabel("Count", color="#d4d4d4")
        ax.tick_params(colors="#d4d4d4")
        images.append((label, _fig_to_base64(fig)))

    # 2D scatter plot
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#1e1e1e")
    ax.set_facecolor("#2d2d2d")
    ax.scatter(global_vals, local_vals, alpha=0.6, c="#4fc3f7", s=20)
    ax.set_xlabel("global_shift_score", color="#d4d4d4")
    ax.set_ylabel("local_diff_score", color="#d4d4d4")
    ax.set_title("Global vs Local Scores", color="#e0e0e0")
    ax.tick_params(colors="#d4d4d4")
    images.append(("scatter", _fig_to_base64(fig)))

    body_parts = ['<div class="charts">']
    for label, b64 in images:
        body_parts.append(
            f'<div class="chart"><h2>{label}</h2>'
            f'<img src="data:image/png;base64,{b64}"></div>'
        )
    body_parts.append("</div>")

    html = _HTML_TEMPLATE.format(title="Score Distributions", body="\n".join(body_parts))
    with open(out_path, "w") as f:
        f.write(html)


def grid_search_preview(scores, out_path):
    """Generate heatmap of pass rates for (global, local) threshold pairs."""
    global_vals = np.array([s["global_shift_score"] for s in scores])
    local_vals = np.array([s["local_diff_score"] for s in scores])
    n = len(scores)

    pass_rates = np.zeros((len(LOCAL_THRESHOLDS), len(GLOBAL_THRESHOLDS)))
    for gi, gt in enumerate(GLOBAL_THRESHOLDS):
        for li, lt in enumerate(LOCAL_THRESHOLDS):
            passes = np.sum((global_vals < gt) & (local_vals < lt))
            pass_rates[li, gi] = passes / n

    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#1e1e1e")
    ax.set_facecolor("#2d2d2d")
    im = ax.imshow(pass_rates, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(GLOBAL_THRESHOLDS)))
    ax.set_xticklabels([str(t) for t in GLOBAL_THRESHOLDS], rotation=45, fontsize=7, color="#d4d4d4")
    ax.set_yticks(range(len(LOCAL_THRESHOLDS)))
    ax.set_yticklabels([str(t) for t in LOCAL_THRESHOLDS], fontsize=7, color="#d4d4d4")
    ax.set_xlabel("Global Threshold", color="#d4d4d4")
    ax.set_ylabel("Local Threshold", color="#d4d4d4")
    ax.set_title("Pass Rate Heatmap", color="#e0e0e0")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors="#d4d4d4")
    ax.tick_params(colors="#d4d4d4")

    b64 = _fig_to_base64(fig)

    body = (
        f'<div class="chart"><img src="data:image/png;base64,{b64}"></div>'
    )
    html = _HTML_TEMPLATE.format(title="Grid Search Preview", body=body)
    with open(out_path, "w") as f:
        f.write(html)


def select_boundary_cases(
    scores,
    output_dir,
    global_range=(1.5, 3.0),
    local_range=(2.0, 5.0),
    margin=0.5,
) -> tuple[list[dict], str]:
    """Select boundary cases near decision boundaries."""
    g_lo, g_hi = global_range[0] - margin, global_range[1] + margin
    l_lo, l_hi = local_range[0] - margin, local_range[1] + margin

    boundary_cases = [
        s for s in scores
        if (g_lo <= s["global_shift_score"] <= g_hi)
        or (l_lo <= s["local_diff_score"] <= l_hi)
    ]

    # Write boundary_cases.json
    json_path = os.path.join(output_dir, "boundary_cases.json")
    with open(json_path, "w") as f:
        json.dump(boundary_cases, f, indent=2)

    # Write boundary_subset.csv
    csv_path = os.path.join(output_dir, "boundary_subset.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video1_path", "video2_path"])
        writer.writeheader()
        for s in boundary_cases:
            writer.writerow({
                "video1_path": s["src_path"],
                "video2_path": s["edited_path"],
            })

    return boundary_cases, csv_path


def evaluate_annotations(scores, annotations, out_path) -> dict:
    """Evaluate threshold combos against human annotations, find best F1."""
    # Build lookup: video_pair_id -> score record
    score_map = {s["video_pair_id"]: s for s in scores}

    # Ground truth labels (using 'labels' to avoid shadowing)
    labels = {a["video_pair_id"]: a["label"] for a in annotations}

    # Annotated pair IDs that exist in scores
    annotated_ids = [pid for pid in labels if pid in score_map]

    # Compute F1, precision, recall for each threshold combo
    n_global = len(GLOBAL_THRESHOLDS)
    n_local = len(LOCAL_THRESHOLDS)
    f1_grid = np.zeros((n_local, n_global))
    precision_grid = np.zeros((n_local, n_global))
    recall_grid = np.zeros((n_local, n_global))

    best_f1 = -1.0
    best_gt_val = GLOBAL_THRESHOLDS[0]
    best_lt_val = LOCAL_THRESHOLDS[0]

    for gi, gt_thresh in enumerate(GLOBAL_THRESHOLDS):
        for li, lt_thresh in enumerate(LOCAL_THRESHOLDS):
            tp = fp = fn = tn = 0
            for pid in annotated_ids:
                s = score_map[pid]
                predicted_pass = (
                    s["global_shift_score"] < gt_thresh
                    and s["local_diff_score"] < lt_thresh
                )
                actual_pass = labels[pid] == "pass"

                if predicted_pass and actual_pass:
                    tp += 1
                elif predicted_pass and not actual_pass:
                    fp += 1
                elif not predicted_pass and actual_pass:
                    fn += 1
                else:
                    tn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            f1_grid[li, gi] = f1
            precision_grid[li, gi] = precision
            recall_grid[li, gi] = recall

            if f1 > best_f1:
                best_f1 = f1
                best_gt_val = gt_thresh
                best_lt_val = lt_thresh

    # Generate HTML with 3 side-by-side heatmaps
    images = []
    for grid, title in [
        (f1_grid, "F1 Score"),
        (precision_grid, "Precision"),
        (recall_grid, "Recall"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7), facecolor="#1e1e1e")
        ax.set_facecolor("#2d2d2d")
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n_global))
        ax.set_xticklabels([str(t) for t in GLOBAL_THRESHOLDS], rotation=45, fontsize=6, color="#d4d4d4")
        ax.set_yticks(range(n_local))
        ax.set_yticklabels([str(t) for t in LOCAL_THRESHOLDS], fontsize=6, color="#d4d4d4")
        ax.set_xlabel("Global Threshold", color="#d4d4d4")
        ax.set_ylabel("Local Threshold", color="#d4d4d4")
        ax.set_title(title, color="#e0e0e0")
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors="#d4d4d4")
        ax.tick_params(colors="#d4d4d4")

        # Mark best F1 with a star on the F1 plot
        if title == "F1 Score":
            best_gi = GLOBAL_THRESHOLDS.index(best_gt_val)
            best_li = LOCAL_THRESHOLDS.index(best_lt_val)
            ax.plot(best_gi, best_li, marker="*", color="white", markersize=20, markeredgecolor="black")

        images.append((title, _fig_to_base64(fig)))

    body_parts = ['<div class="charts">']
    for title, b64 in images:
        body_parts.append(
            f'<div class="chart"><h2>{title}</h2>'
            f'<img src="data:image/png;base64,{b64}"></div>'
        )
    body_parts.append("</div>")

    html = _HTML_TEMPLATE.format(title="Annotation Evaluation", body="\n".join(body_parts))
    with open(out_path, "w") as f:
        f.write(html)

    return {
        "best_global_threshold": best_gt_val,
        "best_local_threshold": best_lt_val,
        "best_f1": best_f1,
    }


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------

def _cmd_analyze(args):
    """Subcommand: analyze scores -> distribution, grid preview, boundary cases."""
    scores = load_scores(args.scores)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loaded {len(scores)} scores from {args.scores}")

    generate_distribution_html(scores, os.path.join(args.output_dir, "distribution.html"))
    print("  -> distribution.html")

    grid_search_preview(scores, os.path.join(args.output_dir, "grid_search_preview.html"))
    print("  -> grid_search_preview.html")

    boundary, csv_path = select_boundary_cases(scores, args.output_dir)
    print(f"  -> {len(boundary)} boundary cases -> boundary_cases.json, {os.path.basename(csv_path)}")


def _cmd_build_reports(args):
    """Subcommand: build index.html after visualizations are generated."""
    from vid_color_filter.report import generate_index_page

    with open(args.boundary_cases) as f:
        boundary = json.load(f)

    generate_index_page(boundary, args.viz_dir)
    print(f"  -> index.html ({len(boundary)} cases)")


def _cmd_evaluate(args):
    """Subcommand: evaluate annotations -> F1 grid search results."""
    scores = load_scores(args.scores)
    with open(args.annotations) as f:
        annotations = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "grid_search_results.html")
    best = evaluate_annotations(scores, annotations, out_path)
    print(f"Best: global={best['best_global_threshold']}, "
          f"local={best['best_local_threshold']}, F1={best['best_f1']:.3f}")
    print("  -> grid_search_results.html")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calibration analysis tools")
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze score distribution and select boundary cases")
    p_analyze.add_argument("--scores", required=True, help="Path to scores.jsonl")
    p_analyze.add_argument("--output-dir", required=True, help="Output directory")

    # build-reports
    p_build = sub.add_parser("build-reports", help="Build index.html after visualizations")
    p_build.add_argument("--boundary-cases", required=True, help="Path to boundary_cases.json")
    p_build.add_argument("--viz-dir", required=True, help="Directory with visualization reports")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate annotations with F1 grid search")
    p_eval.add_argument("--scores", required=True, help="Path to scores.jsonl")
    p_eval.add_argument("--annotations", required=True, help="Path to annotations.json")
    p_eval.add_argument("--output-dir", required=True, help="Output directory")

    args = parser.parse_args()
    if args.command == "analyze":
        _cmd_analyze(args)
    elif args.command == "build-reports":
        _cmd_build_reports(args)
    elif args.command == "evaluate":
        _cmd_evaluate(args)


if __name__ == "__main__":
    main()
