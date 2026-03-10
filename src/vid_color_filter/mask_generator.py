import cv2
import numpy as np
from skimage.color import rgb2lab


def generate_edit_mask(
    src_frame: np.ndarray,
    edited_frame: np.ndarray,
    diff_threshold: float = 5.0,
    dilate_kernel: int = 21,
    min_component_size: int = 100,
) -> tuple[np.ndarray, float]:
    """Generate a boolean mask of edited regions by frame differencing.

    Args:
        src_frame: BGR source frame.
        edited_frame: BGR edited frame.
        diff_threshold: Lab Euclidean distance threshold for binarization.
        dilate_kernel: Dilation kernel size in pixels.
        min_component_size: Minimum connected component area to keep.

    Returns:
        (mask, coverage_ratio) where mask is bool array (True = edited region).
    """
    src_lab = rgb2lab(cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB))
    edited_lab = rgb2lab(cv2.cvtColor(edited_frame, cv2.COLOR_BGR2RGB))

    diff = np.sqrt(np.sum((src_lab - edited_lab) ** 2, axis=2))

    binary = (diff > diff_threshold).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_component_size:
            binary[labels == i] = 0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel)
    )
    dilated = cv2.dilate(binary, kernel)

    mask = dilated.astype(bool)
    coverage = mask.sum() / mask.size

    return mask, coverage
