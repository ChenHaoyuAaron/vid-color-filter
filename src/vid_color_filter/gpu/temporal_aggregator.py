"""Temporal aggregation and multi-dimensional scoring for video color difference."""

import torch


def temporal_aggregate(de_maps, masks, min_unmasked_ratio=0.5):
    N, H, W = de_maps.shape
    de_nan = de_maps.clone()
    de_nan[masks] = float("nan")
    unmasked_count = (~masks).float().sum(dim=0)
    min_frames = N * min_unmasked_ratio
    excluded = unmasked_count < min_frames
    median_map = torch.nanmedian(de_nan, dim=0).values
    q75 = torch.nanquantile(de_nan, 0.75, dim=0)
    q25 = torch.nanquantile(de_nan, 0.25, dim=0)
    iqr_map = q75 - q25
    nan_val = torch.tensor(float("nan"), device=de_maps.device)
    median_map = torch.where(excluded, nan_val, median_map)
    iqr_map = torch.where(excluded, nan_val, iqr_map)
    return median_map, iqr_map


def compute_scores(median_map, iqr_map, global_threshold=2.0, local_threshold=3.0):
    valid = ~torch.isnan(median_map)
    valid_medians = median_map[valid]
    valid_iqrs = iqr_map[valid & ~torch.isnan(iqr_map)]
    if valid_medians.numel() == 0:
        return {
            "global_shift_score": float("nan"),
            "local_diff_score": float("nan"),
            "temporal_instability": float("nan"),
            "pass_global": False,
            "pass_local": False,
            "pass": False,
        }
    global_shift = float(torch.median(valid_medians).item())
    local_max = float(torch.max(valid_medians).item())
    local_diff = local_max - global_shift
    instability = float(valid_iqrs.mean().item()) if valid_iqrs.numel() > 0 else 0.0
    pass_global = global_shift < global_threshold
    pass_local = local_diff < local_threshold
    return {
        "global_shift_score": global_shift,
        "local_diff_score": local_diff,
        "temporal_instability": instability,
        "pass_global": pass_global,
        "pass_local": pass_local,
        "pass": pass_global and pass_local,
    }
