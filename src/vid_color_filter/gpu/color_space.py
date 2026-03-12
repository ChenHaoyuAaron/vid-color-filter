import torch

# D65 illuminant reference white
_D65_X = 0.95047
_D65_Y = 1.00000
_D65_Z = 1.08883

# sRGB to XYZ matrix (D65)
_RGB_TO_XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float32)

_LAB_DELTA = 6.0 / 29.0
_LAB_DELTA_SQ = _LAB_DELTA ** 2
_LAB_DELTA_CU = _LAB_DELTA ** 3


def _srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    """Inverse sRGB companding: [0,1] sRGB -> linear RGB."""
    low = srgb / 12.92
    high = ((srgb + 0.055) / 1.055) ** 2.4
    return torch.where(srgb <= 0.04045, low, high)


def _lab_f(t: torch.Tensor) -> torch.Tensor:
    """CIE Lab transfer function."""
    return torch.where(
        t > _LAB_DELTA_CU,
        t.pow(1.0 / 3.0),
        t / (3.0 * _LAB_DELTA_SQ) + 4.0 / 29.0,
    )


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert batched RGB images to CIE Lab color space on GPU.

    Args:
        rgb: (B, H, W, 3) uint8 or float32 tensor in [0, 255] or [0, 1].

    Returns:
        (B, H, W, 3) float32 Lab tensor. L in [0,100], a/b in approx [-128,127].
    """
    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0
    elif rgb.max() > 1.0:
        rgb = rgb / 255.0

    linear = _srgb_to_linear(rgb)

    mat = _RGB_TO_XYZ.to(device=linear.device, dtype=linear.dtype)
    # (B, H, W, 3) @ (3, 3)^T -> (B, H, W, 3) XYZ
    xyz = linear @ mat.T

    x = xyz[..., 0] / _D65_X
    y = xyz[..., 1] / _D65_Y
    z = xyz[..., 2] / _D65_Z

    fx = _lab_f(x)
    fy = _lab_f(y)
    fz = _lab_f(z)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return torch.stack([L, a, b], dim=-1)
