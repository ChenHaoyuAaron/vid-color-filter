import os
import torch
import numpy as np

from vid_color_filter.gpu.color_space import rgb_to_lab
from vid_color_filter.gpu.color_metrics import METRICS
from vid_color_filter.gpu.mask_generator import generate_edit_mask_gpu
from vid_color_filter.gpu.scielab import scielab_filter
from vid_color_filter.gpu.adaptive_mask import generate_adaptive_mask
from vid_color_filter.gpu.temporal_aggregator import temporal_aggregate, compute_scores


def score_video_pair_gpu(
    src_frames: torch.Tensor,
    edited_frames: torch.Tensor,
    src_path: str = "",
    threshold: float = 2.0,
    diff_threshold: float = 5.0,
    dilate_kernel: int = 21,
    metric: str = "cie94",
    use_scielab: bool = False,
    pixels_per_degree: float = 60.0,
    global_threshold: float | None = None,
    local_threshold: float = 3.0,
    chunk_size: int = 8,
) -> dict:
    """Score a video pair on GPU with batched frame processing.

    Two modes:
    - use_scielab=False (default): Legacy pipeline for backward compatibility.
    - use_scielab=True: New S-CIELAB temporal pipeline with adaptive masking.

    Args:
        src_frames: (N, H, W, 3) uint8 RGB tensor of source frames.
        edited_frames: (N, H, W, 3) uint8 RGB tensor of edited frames.
        src_path: Source video path (used for pair ID).
        threshold: Delta E threshold for pass/fail (legacy) or default global_threshold.
        diff_threshold: Lab distance threshold for mask binarization.
        dilate_kernel: Dilation kernel size for mask.
        metric: Color metric name ("cie76", "cie94", or "ciede2000").
        use_scielab: Enable S-CIELAB temporal pipeline.
        pixels_per_degree: Pixels per degree of visual angle for S-CIELAB.
        global_threshold: Threshold for global shift score (defaults to threshold).
        local_threshold: Threshold for local diff score.
        chunk_size: Number of frames to process per GPU chunk.

    Returns:
        Dict with scoring results.
    """
    if use_scielab:
        return _score_scielab(
            src_frames, edited_frames, src_path=src_path,
            threshold=threshold, diff_threshold=diff_threshold,
            dilate_kernel=dilate_kernel, metric=metric,
            pixels_per_degree=pixels_per_degree,
            global_threshold=global_threshold,
            local_threshold=local_threshold, chunk_size=chunk_size,
        )
    return _score_legacy(
        src_frames, edited_frames, src_path=src_path,
        threshold=threshold, diff_threshold=diff_threshold,
        dilate_kernel=dilate_kernel, metric=metric,
    )


def _score_legacy(
    src_frames, edited_frames, src_path="", threshold=2.0,
    diff_threshold=5.0, dilate_kernel=21, metric="cie94",
):
    """Legacy scoring pipeline (backward compatible)."""
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


def _score_scielab(
    src_frames, edited_frames, src_path="", threshold=2.0,
    diff_threshold=5.0, dilate_kernel=21, metric="cie94",
    pixels_per_degree=60.0, global_threshold=None,
    local_threshold=3.0, chunk_size=8,
):
    """S-CIELAB temporal scoring pipeline."""
    pair_id = os.path.splitext(os.path.basename(src_path))[0] if src_path else ""
    if global_threshold is None:
        global_threshold = threshold

    metric_fn = METRICS[metric]
    N = src_frames.shape[0]

    all_de_maps = []
    all_masks = []
    all_mean_des = []
    kernel_cache = {}

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        src_chunk = src_frames[start:end]
        edited_chunk = edited_frames[start:end]

        # Raw Lab for adaptive masking
        src_lab_raw = rgb_to_lab(src_chunk)
        edited_lab_raw = rgb_to_lab(edited_chunk)

        # Adaptive mask on raw Lab
        masks, _coverages = generate_adaptive_mask(
            src_lab_raw, edited_lab_raw,
            diff_threshold=diff_threshold,
            dilate_kernel=dilate_kernel,
        )

        # S-CIELAB filtered Lab for perceptual delta E
        src_lab_filtered = scielab_filter(src_chunk, pixels_per_degree, _cached_kernels=kernel_cache)
        edited_lab_filtered = scielab_filter(edited_chunk, pixels_per_degree, _cached_kernels=kernel_cache)

        # Per-pixel delta E maps (B, H, W)
        de_map = metric_fn(src_lab_filtered, edited_lab_filtered, mask=masks, reduce="none")
        # Per-frame mean delta E (B,)
        mean_de = metric_fn(src_lab_filtered, edited_lab_filtered, mask=masks, reduce="mean")

        # Handle fully-masked frames: fall back to global (unmasked) scoring
        full_coverage = _coverages >= 1.0
        nan_mask = torch.isnan(mean_de)
        needs_global = full_coverage & nan_mask
        if needs_global.any():
            global_de_map = metric_fn(src_lab_filtered, edited_lab_filtered, mask=None, reduce="none")
            global_mean_de = metric_fn(src_lab_filtered, edited_lab_filtered, mask=None, reduce="mean")
            # Replace NaN de_map frames with unmasked versions
            de_map = torch.where(
                needs_global.view(-1, 1, 1).expand_as(de_map),
                global_de_map, de_map,
            )
            mean_de = torch.where(needs_global, global_mean_de, mean_de)
            # Clear masks for fully-covered frames so temporal_aggregate uses all pixels
            masks = torch.where(
                needs_global.view(-1, 1, 1).expand_as(masks),
                torch.zeros_like(masks),
                masks,
            )

        all_de_maps.append(de_map)
        all_masks.append(masks)
        all_mean_des.append(mean_de)

    # Concatenate all chunks
    de_maps = torch.cat(all_de_maps, dim=0)   # (N, H, W)
    masks = torch.cat(all_masks, dim=0)         # (N, H, W)
    mean_des = torch.cat(all_mean_des, dim=0)   # (N,)

    # Temporal aggregation
    median_map, iqr_map = temporal_aggregate(de_maps, masks)
    scores = compute_scores(
        median_map, iqr_map,
        global_threshold=global_threshold,
        local_threshold=local_threshold,
    )

    # Build output with both new and backward-compat fields
    mean_des_list = mean_des.cpu().tolist()
    valid_des = [d for d in mean_des_list if not np.isnan(d)]
    max_mean_de = max(valid_des) if valid_des else float("nan")

    return {
        "video_pair_id": pair_id,
        # New fields
        "global_shift_score": scores["global_shift_score"],
        "local_diff_score": scores["local_diff_score"],
        "temporal_instability": scores["temporal_instability"],
        "pass_global": scores["pass_global"],
        "pass_local": scores["pass_local"],
        "pass": scores["pass"],
        # Backward compat fields
        "per_frame_mean_delta_e": mean_des_list,
        "mean_delta_e_per_frame": mean_des_list,
        "max_mean_delta_e": max_mean_de,
    }
