import os
import numpy as np
from vid_color_filter.frame_sampler import sample_frame_pairs
from vid_color_filter.mask_generator import generate_edit_mask
from vid_color_filter.color_diff import compute_mean_ciede2000


def score_video_pair(
    src_path: str,
    edited_path: str,
    num_frames: int = 16,
    threshold: float = 2.0,
    diff_threshold: float = 5.0,
    dilate_kernel: int = 21,
) -> dict:
    """Score a video editing pair for color contamination in unedited regions.

    Returns dict with scoring results and metadata.
    """
    pair_id = os.path.splitext(os.path.basename(src_path))[0]

    frame_pairs = sample_frame_pairs(src_path, edited_path, num_frames=num_frames)

    mean_des = []
    coverages = []

    for src_frame, edited_frame in frame_pairs:
        mask, coverage = generate_edit_mask(
            src_frame, edited_frame,
            diff_threshold=diff_threshold,
            dilate_kernel=dilate_kernel,
        )
        coverages.append(coverage)

        mean_de = compute_mean_ciede2000(src_frame, edited_frame, mask)

        # If the mask covers the entire frame (global shift), measure
        # delta E over all pixels since the whole frame is contaminated.
        if np.isnan(mean_de) and coverage >= 1.0:
            empty_mask = np.zeros(mask.shape, dtype=bool)
            mean_de = compute_mean_ciede2000(src_frame, edited_frame, empty_mask)

        mean_des.append(mean_de)

    # Filter out NaN frames (fully masked)
    valid_des = [d for d in mean_des if not np.isnan(d)]
    max_mean_de = max(valid_des) if valid_des else float("nan")

    return {
        "video_pair_id": pair_id,
        "mean_delta_e_per_frame": mean_des,
        "max_mean_delta_e": max_mean_de,
        "pass": max_mean_de < threshold if not np.isnan(max_mean_de) else False,
        "mask_coverage_ratio": max(coverages) if coverages else 0.0,
    }
