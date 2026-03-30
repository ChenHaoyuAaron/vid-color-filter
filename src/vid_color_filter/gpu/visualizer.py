"""PNG visualization renderer for calibration reports.

Renders delta-E heatmaps, mask overlays, temporal maps, and raw frames
as PNG images using matplotlib and OpenCV. No GPU operations.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def render_heatmap(
    src_frame: np.ndarray,
    de_map: np.ndarray,
    mask: np.ndarray,
    out_path: str,
    vmax: float = 10.0,
    cmap: str = "viridis",
    alpha: float = 0.6,
) -> None:
    """Overlay a delta-E colormap on the source frame, graying out masked regions.

    Parameters
    ----------
    src_frame : (H, W, 3) uint8 RGB frame.
    de_map : (H, W) float32 delta-E values.
    mask : (H, W) bool — True means valid (not masked).
    out_path : Destination PNG path.
    vmax : Upper clamp for the colormap.
    cmap : Matplotlib colormap name.
    alpha : Overlay opacity.
    """
    h, w = src_frame.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)

    # Show source frame as background
    display = src_frame.copy()
    # Gray out masked (invalid) regions
    gray = np.mean(src_frame, axis=2, keepdims=True).astype(np.uint8)
    display[~mask] = np.broadcast_to(gray, src_frame.shape)[~mask]

    ax.imshow(display)

    # Overlay delta-E heatmap only on valid regions
    de_overlay = np.ma.array(de_map, mask=~mask)
    im = ax.imshow(de_overlay, cmap=cmap, vmin=0, vmax=vmax, alpha=alpha)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_mask_overlay(
    src_frame: np.ndarray,
    mask: np.ndarray,
    out_path: str,
    coverage: float = 0.0,
    alpha: float = 0.4,
) -> None:
    """Red semi-transparent mask overlay on source frame.

    Parameters
    ----------
    src_frame : (H, W, 3) uint8 RGB frame.
    mask : (H, W) bool — True means valid.
    out_path : Destination PNG path.
    coverage : Mask coverage fraction (0-1) shown in title.
    alpha : Overlay opacity for the mask region.
    """
    h, w = src_frame.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)

    ax.imshow(src_frame)

    # Create red overlay for masked-out (invalid) regions
    red_overlay = np.zeros((h, w, 4), dtype=np.float32)
    red_overlay[~mask, 0] = 1.0  # red channel
    red_overlay[~mask, 3] = alpha  # alpha channel

    ax.imshow(red_overlay)
    ax.set_title(f"Mask coverage: {coverage * 100:.1f}%", fontsize=10)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_temporal_map(
    temporal_map: np.ndarray,
    out_path: str,
    vmax: float = 10.0,
    cmap: str = "viridis",
    label: str = "\u0394E",
) -> None:
    """Render a standalone heatmap (no frame underlay).

    Parameters
    ----------
    temporal_map : (H, W) float32 map (may contain NaN).
    out_path : Destination PNG path.
    vmax : Upper clamp for the colormap.
    cmap : Matplotlib colormap name.
    label : Colorbar label.
    """
    clean_map = np.nan_to_num(temporal_map, nan=0.0)
    h, w = clean_map.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)

    im = ax.imshow(clean_map, cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_frame(frame: np.ndarray, out_path: str) -> None:
    """Save an RGB uint8 frame as PNG using OpenCV.

    Parameters
    ----------
    frame : (H, W, 3) uint8 RGB array.
    out_path : Destination PNG path.
    """
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)


def generate_pair_visualizations(
    data: dict,
    output_dir: str,
    vmax: float = 10.0,
) -> None:
    """Generate all PNGs for a video pair.

    Creates ``{output_dir}/{video_pair_id}/`` and writes:
    - src_frame_XX.png, edit_frame_XX.png  (per representative frame)
    - de_heatmap_XX.png, mask_overlay_XX.png  (per representative frame)
    - median_map.png, iqr_map.png  (temporal aggregates)

    Parameters
    ----------
    data : Dict with keys video_pair_id, src_frames_repr, edit_frames_repr,
           de_maps_repr, masks_repr, coverages_repr, median_map, iqr_map.
    output_dir : Root output directory.
    vmax : Upper clamp for delta-E heatmaps.
    """
    pair_id = data["video_pair_id"]
    pair_dir = Path(output_dir) / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)

    n_frames = len(data["src_frames_repr"])

    for i in range(n_frames):
        save_frame(
            data["src_frames_repr"][i],
            str(pair_dir / f"src_frame_{i:02d}.png"),
        )
        save_frame(
            data["edit_frames_repr"][i],
            str(pair_dir / f"edit_frame_{i:02d}.png"),
        )
        render_heatmap(
            data["src_frames_repr"][i],
            data["de_maps_repr"][i],
            data["masks_repr"][i],
            str(pair_dir / f"de_heatmap_{i:02d}.png"),
            vmax=vmax,
        )
        render_mask_overlay(
            data["src_frames_repr"][i],
            data["masks_repr"][i],
            str(pair_dir / f"mask_overlay_{i:02d}.png"),
            coverage=data["coverages_repr"][i],
        )

    # Temporal aggregate maps
    render_temporal_map(
        data["median_map"],
        str(pair_dir / "median_map.png"),
        vmax=vmax,
    )
    render_temporal_map(
        data["iqr_map"],
        str(pair_dir / "iqr_map.png"),
        vmax=3.0,
        cmap="magma",
        label="IQR",
    )
