"""S-CIELAB spatial filtering for perceptual color difference.

Implements Zhang & Wandell (1997) S-CIELAB algorithm:
RGB -> XYZ -> Poirson-Wandell opponent -> CSF filtering -> XYZ -> Lab
"""

import torch
import torch.nn.functional as F
from vid_color_filter.gpu.color_space import rgb_to_xyz, xyz_to_lab

_PW_MATRIX = torch.tensor([
    [0.9795, 1.5318, 0.1225],
    [-0.1071, 0.3122, 0.0215],
    [0.0383, 0.0023, 0.5765],
], dtype=torch.float32)

_PW_MATRIX_INV = torch.linalg.inv(_PW_MATRIX)

_CSF_PARAMS = [
    [(0.921, 0.0283), (0.105, 0.133), (-0.026, 4.336)],
    [(0.531, 0.0392), (0.330, 0.494)],
    [(0.488, 0.0536), (0.371, 0.386)],
]

def _make_gaussian_kernel_1d(sigma_pixels, size):
    radius = size // 2
    coords = torch.arange(size, dtype=torch.float32) - radius
    g = torch.exp(-0.5 * (coords / max(sigma_pixels, 1e-6)) ** 2)
    return g / g.sum()

_MAX_KERNEL_SIZE = 513

def build_csf_kernels(pixels_per_degree=60.0, device="cpu"):
    """Build per-component 1D Gaussian kernels for each CSF channel.

    Returns a list of 3 channel entries. Each channel entry is a list of
    (weight, kernel_1d) tuples — one per Gaussian component. Components must
    be applied as separate separable convolutions and then summed to avoid
    cross-term artifacts from combining multi-Gaussian kernels into one 1D
    kernel.
    """
    channel_kernels = []
    for channel_params in _CSF_PARAMS:
        components = []
        for weight, sigma_deg in channel_params:
            sigma_px = sigma_deg * pixels_per_degree
            radius = min(int(3 * sigma_px + 0.5), _MAX_KERNEL_SIZE // 2)
            if radius < 1:
                radius = 1
            size = 2 * radius + 1
            g1d = _make_gaussian_kernel_1d(sigma_px, size)
            components.append((weight, g1d.to(device)))
        channel_kernels.append(components)
    return channel_kernels

def xyz_to_opponent(xyz):
    mat = _PW_MATRIX.to(device=xyz.device, dtype=xyz.dtype)
    return xyz @ mat.T

def opponent_to_xyz(opp):
    inv_mat = _PW_MATRIX_INV.to(device=opp.device, dtype=opp.dtype)
    return opp @ inv_mat.T

def _apply_separable_1d(channel_4d, kernel_1d):
    """Apply a single 1D kernel separably (horizontal then vertical)."""
    K = kernel_1d.shape[0]
    pad = K // 2
    x = F.pad(channel_4d, (pad, pad, 0, 0), mode="replicate")
    kh = kernel_1d.reshape(1, 1, 1, K)
    x = F.conv2d(x, kh)
    x = F.pad(x, (0, 0, pad, pad), mode="replicate")
    kv = kernel_1d.reshape(1, 1, K, 1)
    x = F.conv2d(x, kv)
    return x


def _apply_csf_to_channel(channel, components):
    """Apply CSF as sum of separable Gaussian components.

    Each component is applied as its own separable pair and then the weighted
    results are summed. This avoids cross-term artifacts that would arise from
    combining multiple Gaussians into a single 1D kernel before separable
    convolution.
    """
    x = channel.unsqueeze(1)  # (B, 1, H, W)
    result = torch.zeros_like(x)
    for weight, kernel_1d in components:
        result = result + weight * _apply_separable_1d(x, kernel_1d)
    # Normalize so the total response sums to 1
    total_weight = sum(w for w, _ in components)
    if abs(total_weight) > 1e-6:
        result = result / total_weight
    return result.squeeze(1)

def scielab_filter(rgb, pixels_per_degree=60.0, _cached_kernels=None):
    xyz = rgb_to_xyz(rgb)
    opp = xyz_to_opponent(xyz)
    if _cached_kernels is not None and pixels_per_degree in _cached_kernels:
        kernels = _cached_kernels[pixels_per_degree]
    else:
        kernels = build_csf_kernels(pixels_per_degree, device=rgb.device)
        if _cached_kernels is not None:
            _cached_kernels[pixels_per_degree] = kernels
    filtered_channels = []
    for i in range(3):
        filtered = _apply_csf_to_channel(opp[..., i], kernels[i])
        filtered_channels.append(filtered)
    filtered_opp = torch.stack(filtered_channels, dim=-1)
    filtered_xyz = opponent_to_xyz(filtered_opp)
    return xyz_to_lab(filtered_xyz)
