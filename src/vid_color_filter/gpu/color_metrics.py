import torch
import math


def delta_e_cie76(
    lab1: torch.Tensor,
    lab2: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """CIE76 color difference (Euclidean distance in Lab).

    Args:
        lab1, lab2: (B, H, W, 3) Lab tensors.
        mask: Optional (B, H, W) bool tensor. True = excluded (edited) pixels.

    Returns:
        (B,) mean delta E per image in batch over unmasked pixels.
    """
    diff = lab1 - lab2
    de = torch.sqrt((diff * diff).sum(dim=-1))

    return _masked_mean_per_image(de, mask)


def delta_e_cie94(
    lab1: torch.Tensor,
    lab2: torch.Tensor,
    mask: torch.Tensor | None = None,
    k_L: float = 1.0,
    K1: float = 0.045,
    K2: float = 0.015,
) -> torch.Tensor:
    """CIE94 color difference (graphic arts weights).

    Args:
        lab1, lab2: (B, H, W, 3) Lab tensors.
        mask: Optional (B, H, W) bool tensor. True = excluded pixels.

    Returns:
        (B,) mean delta E per image in batch over unmasked pixels.
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    dL = L1 - L2
    C1 = torch.sqrt(a1 * a1 + b1 * b1)
    C2 = torch.sqrt(a2 * a2 + b2 * b2)
    dC = C1 - C2

    da = a1 - a2
    db = b1 - b2
    dH_sq = (da * da + db * db - dC * dC).clamp(min=0.0)

    S_L = 1.0
    S_C = 1.0 + K1 * C1
    S_H = 1.0 + K2 * C1

    term_L = dL / (k_L * S_L)
    term_C = dC / S_C
    term_H = dH_sq / (S_H * S_H)

    de = torch.sqrt(term_L * term_L + term_C * term_C + term_H)

    return _masked_mean_per_image(de, mask)


def delta_e_ciede2000(
    lab1: torch.Tensor,
    lab2: torch.Tensor,
    mask: torch.Tensor | None = None,
    k_L: float = 1.0,
    k_C: float = 1.0,
    k_H: float = 1.0,
) -> torch.Tensor:
    """CIEDE2000 color difference (full formula, GPU-vectorized).

    Args:
        lab1, lab2: (B, H, W, 3) Lab tensors.
        mask: Optional (B, H, W) bool tensor. True = excluded pixels.

    Returns:
        (B,) mean delta E per image in batch over unmasked pixels.
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1_ab = torch.sqrt(a1 * a1 + b1 * b1)
    C2_ab = torch.sqrt(a2 * a2 + b2 * b2)
    C_ab_mean = (C1_ab + C2_ab) / 2.0

    C_ab_mean_pow7 = C_ab_mean ** 7
    G = 0.5 * (1.0 - torch.sqrt(C_ab_mean_pow7 / (C_ab_mean_pow7 + 25.0 ** 7)))

    a1_prime = a1 * (1.0 + G)
    a2_prime = a2 * (1.0 + G)

    C1_prime = torch.sqrt(a1_prime * a1_prime + b1 * b1)
    C2_prime = torch.sqrt(a2_prime * a2_prime + b2 * b2)

    h1_prime = torch.atan2(b1, a1_prime) % (2 * math.pi)
    h2_prime = torch.atan2(b2, a2_prime) % (2 * math.pi)

    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    h_diff = h2_prime - h1_prime
    C_product = C1_prime * C2_prime

    dh_prime = torch.where(
        C_product == 0,
        torch.zeros_like(h_diff),
        torch.where(
            torch.abs(h_diff) <= math.pi,
            h_diff,
            torch.where(h_diff > math.pi, h_diff - 2 * math.pi, h_diff + 2 * math.pi),
        ),
    )
    dH_prime = 2.0 * torch.sqrt(C_product) * torch.sin(dh_prime / 2.0)

    L_prime_mean = (L1 + L2) / 2.0
    C_prime_mean = (C1_prime + C2_prime) / 2.0

    h_sum = h1_prime + h2_prime
    h_prime_mean = torch.where(
        C_product == 0,
        h_sum,
        torch.where(
            torch.abs(h_diff) <= math.pi,
            h_sum / 2.0,
            torch.where(h_sum < 2 * math.pi, (h_sum + 2 * math.pi) / 2.0, (h_sum - 2 * math.pi) / 2.0),
        ),
    )

    T = (
        1.0
        - 0.17 * torch.cos(h_prime_mean - math.radians(30))
        + 0.24 * torch.cos(2.0 * h_prime_mean)
        + 0.32 * torch.cos(3.0 * h_prime_mean + math.radians(6))
        - 0.20 * torch.cos(4.0 * h_prime_mean - math.radians(63))
    )

    L_mean_50_sq = (L_prime_mean - 50.0) ** 2
    S_L = 1.0 + 0.015 * L_mean_50_sq / torch.sqrt(20.0 + L_mean_50_sq)
    S_C = 1.0 + 0.045 * C_prime_mean
    S_H = 1.0 + 0.015 * C_prime_mean * T

    C_prime_mean_pow7 = C_prime_mean ** 7
    R_C = 2.0 * torch.sqrt(C_prime_mean_pow7 / (C_prime_mean_pow7 + 25.0 ** 7))

    theta = math.radians(30) * torch.exp(-((h_prime_mean / math.radians(1) - 275.0) / 25.0) ** 2)
    R_T = -torch.sin(2.0 * theta) * R_C

    term_L = dL_prime / (k_L * S_L)
    term_C = dC_prime / (k_C * S_C)
    term_H = dH_prime / (k_H * S_H)

    de = torch.sqrt(term_L ** 2 + term_C ** 2 + term_H ** 2 + R_T * term_C * term_H)

    return _masked_mean_per_image(de, mask)


METRICS = {
    "cie76": delta_e_cie76,
    "cie94": delta_e_cie94,
    "ciede2000": delta_e_ciede2000,
}


def _masked_mean_per_image(
    de: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Compute mean delta E per image, excluding masked pixels.

    Args:
        de: (B, H, W) delta E values.
        mask: Optional (B, H, W) bool. True = excluded.

    Returns:
        (B,) tensor of per-image means. NaN for fully masked images.
    """
    if mask is not None:
        valid = ~mask
        de = de * valid.float()
        counts = valid.float().sum(dim=(-2, -1))
        sums = de.sum(dim=(-2, -1))
        return torch.where(counts > 0, sums / counts, torch.tensor(float("nan"), device=de.device))
    else:
        return de.mean(dim=(-2, -1))
