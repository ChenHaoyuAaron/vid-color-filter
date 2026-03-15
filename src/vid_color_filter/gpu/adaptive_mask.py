"""Adaptive edit region mask with Otsu thresholding and hysteresis expansion."""

import torch
import torch.nn.functional as F


def otsu_threshold(values, n_bins=256):
    v_min = values.min()
    v_max = values.max()
    if v_max - v_min < 1e-8:
        return v_max
    # Extend range slightly so max-value items don't all land in the last bin
    v_range = float(v_max - v_min)
    hist_min = float(v_min) - v_range * 0.01
    hist_max = float(v_max) + v_range * 0.01
    hist = torch.histc(values.float(), bins=n_bins, min=hist_min, max=hist_max)
    bin_edges = torch.linspace(hist_min, hist_max, n_bins + 1, device=values.device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = hist.sum()
    cum_sum = torch.cumsum(hist, dim=0)
    cum_mean = torch.cumsum(hist * bin_centers, dim=0)
    w0 = cum_sum / total
    w1 = 1.0 - w0
    mu0 = cum_mean / (cum_sum + 1e-10)
    mu1 = (cum_mean[-1] - cum_mean) / (total - cum_sum + 1e-10)
    variance = w0 * w1 * (mu0 - mu1) ** 2
    variance[0] = 0
    variance[-1] = 0
    max_var = variance.max()
    if max_var < 1e-10:
        return (v_min + v_max) / 2.0
    # When multiple bins share the max variance (e.g. bimodal with gap),
    # pick the midpoint among them for a balanced threshold.
    best_mask = (variance >= max_var - 1e-10)
    best_indices = best_mask.nonzero(as_tuple=True)[0]
    mid_idx = best_indices[len(best_indices) // 2]
    return bin_centers[mid_idx]


def _hysteresis_expand(seed_mask, low_mask, max_iterations=50):
    current = seed_mask.float().unsqueeze(1)
    low = low_mask.float().unsqueeze(1)
    for _ in range(max_iterations):
        expanded = F.max_pool2d(current, kernel_size=3, stride=1, padding=1)
        expanded = expanded * low
        if torch.equal(expanded, current):
            break
        current = expanded
    return current.squeeze(1).bool()


def generate_adaptive_mask(src_lab, edited_lab, diff_threshold=None, dilate_kernel=21):
    B, H, W, _ = src_lab.shape
    diff = src_lab - edited_lab
    dist = torch.sqrt((diff * diff).sum(dim=-1))
    masks_list = []
    for i in range(B):
        frame_dist = dist[i]
        if diff_threshold is not None:
            mask_i = frame_dist > diff_threshold
        else:
            flat = frame_dist.flatten()
            if flat.max() - flat.min() < 1e-6:
                masks_list.append(torch.zeros(H, W, dtype=torch.bool, device=src_lab.device))
                continue
            threshold_high = otsu_threshold(flat)
            threshold_low = threshold_high * 0.5
            seed = frame_dist > threshold_high
            low = frame_dist > threshold_low
            mask_i = _hysteresis_expand(seed.unsqueeze(0), low.unsqueeze(0)).squeeze(0)
        masks_list.append(mask_i)
    masks = torch.stack(masks_list)
    if dilate_kernel > 1:
        masks_float = masks.float().unsqueeze(1)
        pad = dilate_kernel // 2
        masks_float = F.max_pool2d(masks_float, kernel_size=dilate_kernel, stride=1, padding=pad)
        masks = masks_float.squeeze(1).bool()
    coverages = masks.float().mean(dim=(-2, -1))
    return masks, coverages
