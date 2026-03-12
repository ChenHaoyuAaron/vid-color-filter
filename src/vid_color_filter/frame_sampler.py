import cv2
import numpy as np


def sample_frame_pairs(
    src_path: str,
    edited_path: str,
    num_frames: int = 16,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sample aligned frame pairs from source and edited videos.

    Returns list of (src_frame, edited_frame) tuples in BGR format.
    Frames are aligned by proportional index when frame counts differ.
    """
    src_cap = cv2.VideoCapture(src_path)
    edited_cap = cv2.VideoCapture(edited_path)

    src_total = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    edited_total = int(edited_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    actual_samples = min(num_frames, src_total, edited_total)
    src_indices = np.linspace(0, src_total - 1, actual_samples, dtype=int)
    edited_indices = np.linspace(0, edited_total - 1, actual_samples, dtype=int)

    pairs = []
    for src_idx, edited_idx in zip(src_indices, edited_indices):
        src_cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_idx))
        ret_s, src_frame = src_cap.read()

        edited_cap.set(cv2.CAP_PROP_POS_FRAMES, int(edited_idx))
        ret_e, edited_frame = edited_cap.read()

        if ret_s and ret_e:
            pairs.append((src_frame, edited_frame))

    src_cap.release()
    edited_cap.release()
    return pairs


def sample_frames_as_tensors(
    src_path: str,
    edited_path: str,
    num_frames: int = 16,
    device: str = "cuda",
):
    """Sample aligned frame pairs and return as stacked GPU tensors.

    Reads frames via OpenCV, converts BGR->RGB, and stacks into
    (N, H, W, 3) uint8 tensors ready for GPU color processing.

    Returns:
        (src_tensor, edited_tensor) each of shape (N, H, W, 3) on device.
        Returns (None, None) if no frames could be read.
    """
    import torch

    pairs = sample_frame_pairs(src_path, edited_path, num_frames=num_frames)
    if not pairs:
        return None, None

    src_frames = []
    edited_frames = []
    for src_bgr, edited_bgr in pairs:
        src_frames.append(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB))
        edited_frames.append(cv2.cvtColor(edited_bgr, cv2.COLOR_BGR2RGB))

    src_tensor = torch.from_numpy(np.stack(src_frames)).to(device)
    edited_tensor = torch.from_numpy(np.stack(edited_frames)).to(device)

    return src_tensor, edited_tensor
