import torch
import torch.nn.functional as F


def generate_edit_mask_gpu(
    src_lab: torch.Tensor,
    edited_lab: torch.Tensor,
    diff_threshold: float = 5.0,
    dilate_kernel: int = 21,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate edit masks for a batch of frame pairs on GPU.

    Uses Lab Euclidean distance for thresholding, then max_pool2d for
    morphological dilation (GPU-native, avoids CPU transfer).

    Connected component filtering is omitted: dilation merges nearby noise,
    and stray mask pixels have negligible impact on mean color diff of
    unmasked regions.

    Args:
        src_lab: (B, H, W, 3) Lab tensor for source frames.
        edited_lab: (B, H, W, 3) Lab tensor for edited frames.
        diff_threshold: Lab Euclidean distance threshold for binarization.
        dilate_kernel: Dilation kernel size (must be odd).

    Returns:
        masks: (B, H, W) bool tensor. True = edited region (excluded).
        coverages: (B,) float tensor. Fraction of masked pixels per frame.
    """
    diff = src_lab - edited_lab
    dist = torch.sqrt((diff * diff).sum(dim=-1))

    binary = (dist > diff_threshold).float()

    if dilate_kernel > 1:
        pad = dilate_kernel // 2
        # max_pool2d expects (N, C, H, W)
        x = binary.unsqueeze(1)
        dilated = F.max_pool2d(x, kernel_size=dilate_kernel, stride=1, padding=pad)
        binary = dilated.squeeze(1)

    masks = binary.bool()
    coverages = masks.float().mean(dim=(-2, -1))

    return masks, coverages
