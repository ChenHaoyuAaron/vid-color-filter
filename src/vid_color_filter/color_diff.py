import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000


def compute_mean_ciede2000(
    src_frame: np.ndarray,
    edited_frame: np.ndarray,
    edit_mask: np.ndarray,
) -> float:
    """Compute mean CIEDE2000 over unmasked (unedited) pixels.

    Args:
        src_frame: BGR source frame.
        edited_frame: BGR edited frame.
        edit_mask: Boolean mask where True = edited region (excluded).

    Returns:
        Mean CIEDE2000 of unmasked pixels. NaN if all pixels are masked.
    """
    unmasked = ~edit_mask
    if not unmasked.any():
        return float("nan")

    src_lab = rgb2lab(cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB))
    edited_lab = rgb2lab(cv2.cvtColor(edited_frame, cv2.COLOR_BGR2RGB))

    de = deltaE_ciede2000(src_lab[unmasked], edited_lab[unmasked])

    return float(np.mean(de))
