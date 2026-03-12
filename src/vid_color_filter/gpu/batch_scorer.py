import os
import torch
import numpy as np

from vid_color_filter.gpu.color_space import rgb_to_lab
from vid_color_filter.gpu.color_metrics import METRICS
from vid_color_filter.gpu.mask_generator import generate_edit_mask_gpu


def score_video_pair_gpu(
    src_frames: torch.Tensor,
    edited_frames: torch.Tensor,
    src_path: str = "",
    threshold: float = 2.0,
    diff_threshold: float = 5.0,
    dilate_kernel: int = 21,
    metric: str = "cie94",
) -> dict:
    """Score a video pair on GPU with batched frame processing.

    All sampled frames are processed in a single GPU pass:
    RGB->Lab conversion, mask generation, and color diff are batched.

    Args:
        src_frames: (N, H, W, 3) uint8 RGB tensor of source frames.
        edited_frames: (N, H, W, 3) uint8 RGB tensor of edited frames.
        src_path: Source video path (used for pair ID).
        threshold: Delta E threshold for pass/fail.
        diff_threshold: Lab distance threshold for mask binarization.
        dilate_kernel: Dilation kernel size for mask.
        metric: Color metric name ("cie76", "cie94", or "ciede2000").

    Returns:
        Dict with scoring results matching the original scorer output format.
    """
    pair_id = os.path.splitext(os.path.basename(src_path))[0] if src_path else ""

    metric_fn = METRICS[metric]

    src_lab = rgb_to_lab(src_frames)
    edited_lab = rgb_to_lab(edited_frames)

    masks, coverages = generate_edit_mask_gpu(
        src_lab, edited_lab,
        diff_threshold=diff_threshold,
        dilate_kernel=dilate_kernel,
    )

    mean_des = metric_fn(src_lab, edited_lab, masks)

    # Handle fully-masked frames: if coverage >= 1.0, measure over all pixels
    full_coverage = coverages >= 1.0
    nan_mask = torch.isnan(mean_des)
    needs_global = full_coverage & nan_mask
    if needs_global.any():
        global_des = metric_fn(src_lab, edited_lab, mask=None)
        mean_des = torch.where(needs_global, global_des, mean_des)

    mean_des_list = mean_des.cpu().tolist()
    coverages_list = coverages.cpu().tolist()

    valid_des = [d for d in mean_des_list if not np.isnan(d)]
    max_mean_de = max(valid_des) if valid_des else float("nan")

    return {
        "video_pair_id": pair_id,
        "mean_delta_e_per_frame": mean_des_list,
        "max_mean_delta_e": max_mean_de,
        "pass": max_mean_de < threshold if not np.isnan(max_mean_de) else False,
        "mask_coverage_ratio": max(coverages_list) if coverages_list else 0.0,
    }
