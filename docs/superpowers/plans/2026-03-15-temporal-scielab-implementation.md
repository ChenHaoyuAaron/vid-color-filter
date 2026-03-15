# Temporal-Aware S-CIELAB Video Color Difference Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the frame-independent mean/max ΔE scoring with a temporal-aware S-CIELAB pipeline that separates true color differences from codec noise.

**Architecture:** Four new GPU modules (S-CIELAB filtering, adaptive masking, temporal aggregation, multi-dimensional scoring) integrated into the existing `batch_scorer.py` pipeline. Each module is independently testable with well-defined tensor interfaces. Frame-chunked processing supports large resolutions.

**Tech Stack:** PyTorch (GPU tensors, `F.conv2d`, `torch.nanmedian`, `torch.nanquantile`), existing `color_space.py` / `color_metrics.py` infrastructure.

**Spec:** `docs/superpowers/specs/2026-03-15-temporal-scielab-video-color-diff-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/vid_color_filter/gpu/color_space.py` | Modify | Add `rgb_to_xyz` (intermediate needed for S-CIELAB) |
| `src/vid_color_filter/gpu/color_metrics.py` | Modify | Add `reduce="none"` mode to return `(B,H,W)` per-pixel ΔE maps |
| `src/vid_color_filter/gpu/scielab.py` | Create | S-CIELAB: CSF kernel generation, Poirson-Wandell opponent transform, spatial filtering |
| `src/vid_color_filter/gpu/adaptive_mask.py` | Create | Otsu thresholding + hysteresis expansion for edit mask |
| `src/vid_color_filter/gpu/temporal_aggregator.py` | Create | Temporal median/IQR, global/local/instability scoring, pass/fail logic |
| `src/vid_color_filter/gpu/batch_scorer.py` | Modify | Wire new modules into pipeline, new output format |
| `run.py` | Modify | Add new CLI args, wire to scorer |
| `tests/test_scielab.py` | Create | S-CIELAB unit tests |
| `tests/test_adaptive_mask.py` | Create | Adaptive mask unit tests |
| `tests/test_temporal_aggregator.py` | Create | Temporal aggregation unit tests |
| `tests/test_gpu_color_metrics.py` | Modify | Add per-pixel mode tests |
| `tests/test_gpu_scorer.py` | Modify | Update end-to-end tests |

---

## Chunk 1: Foundation — color_space.rgb_to_xyz + color_metrics per-pixel mode

### Task 1: Add `rgb_to_xyz` to color_space.py

**Files:**
- Modify: `src/vid_color_filter/gpu/color_space.py`
- Test: `tests/test_gpu_color_metrics.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_gpu_color_metrics.py`, add at the top of the file after existing imports:

```python
from vid_color_filter.gpu.color_space import rgb_to_xyz
```

Add a new test class:

```python
class TestRgbToXyz:
    def test_shape_preserved(self):
        rgb = _make_rgb_tensor((128, 128, 128), (200, 100, 50))
        xyz = rgb_to_xyz(rgb)
        assert xyz.shape == (2, 64, 64, 3)

    def test_white_d65(self):
        """Pure white should map to D65 reference white."""
        white = _make_rgb_tensor((255, 255, 255))
        xyz = rgb_to_xyz(white)[0, 0, 0].cpu()
        assert xyz[0].item() == pytest.approx(0.95047, abs=0.01)
        assert xyz[1].item() == pytest.approx(1.0, abs=0.01)
        assert xyz[2].item() == pytest.approx(1.08883, abs=0.01)

    def test_black_zero(self):
        black = _make_rgb_tensor((0, 0, 0))
        xyz = rgb_to_xyz(black)[0, 0, 0].cpu()
        assert xyz[0].item() == pytest.approx(0.0, abs=0.001)
        assert xyz[1].item() == pytest.approx(0.0, abs=0.001)
        assert xyz[2].item() == pytest.approx(0.0, abs=0.001)

    def test_matches_skimage(self):
        from skimage.color import rgb2xyz
        rgb_np = np.array([[[128, 64, 200]]], dtype=np.uint8)
        expected = rgb2xyz(rgb_np / 255.0)
        rgb_t = torch.from_numpy(rgb_np).unsqueeze(0).to(DEVICE)
        result = rgb_to_xyz(rgb_t).cpu().numpy()[0]
        np.testing.assert_allclose(result, expected, atol=0.005)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gpu_color_metrics.py::TestRgbToXyz -v`
Expected: ImportError — `rgb_to_xyz` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

In `src/vid_color_filter/gpu/color_space.py`, add `rgb_to_xyz` after `_srgb_to_linear` and `xyz_to_lab` after `_lab_f`. Then refactor `rgb_to_lab` to use them:

```python
def rgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    """Convert batched RGB images to CIE XYZ color space on GPU.

    Args:
        rgb: (B, H, W, 3) uint8 or float32 tensor.

    Returns:
        (B, H, W, 3) float32 XYZ tensor.
    """
    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0
    elif rgb.max() > 1.0:
        rgb = rgb / 255.0

    linear = _srgb_to_linear(rgb)
    mat = _RGB_TO_XYZ.to(device=linear.device, dtype=linear.dtype)
    return linear @ mat.T


def xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor:
    """Convert batched XYZ images to CIE Lab color space.

    Args:
        xyz: (B, H, W, 3) float32 XYZ tensor.

    Returns:
        (B, H, W, 3) float32 Lab tensor.
    """
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
```

Then simplify `rgb_to_lab` to:

```python
def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert batched RGB images to CIE Lab color space on GPU."""
    return xyz_to_lab(rgb_to_xyz(rgb))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gpu_color_metrics.py::TestRgbToXyz -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vid_color_filter/gpu/color_space.py tests/test_gpu_color_metrics.py
git commit -m "feat: add rgb_to_xyz to GPU color_space module"
```

---

### Task 2: Add per-pixel ΔE mode to color_metrics.py

**Files:**
- Modify: `src/vid_color_filter/gpu/color_metrics.py`
- Test: `tests/test_gpu_color_metrics.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_gpu_color_metrics.py`:

```python
class TestPerPixelMode:
    def test_cie76_returns_spatial_map(self):
        lab1 = rgb_to_lab(_make_rgb_tensor((128, 128, 128)))
        lab2 = rgb_to_lab(_make_rgb_tensor((180, 128, 128)))
        result = delta_e_cie76(lab1, lab2, reduce="none")
        assert result.shape == (1, 64, 64)

    def test_ciede2000_returns_spatial_map(self):
        lab1 = rgb_to_lab(_make_rgb_tensor((128, 128, 128)))
        lab2 = rgb_to_lab(_make_rgb_tensor((180, 128, 128)))
        result = delta_e_ciede2000(lab1, lab2, reduce="none")
        assert result.shape == (1, 64, 64)

    def test_per_pixel_mean_matches_default(self):
        """Per-pixel mode mean should match default mean mode."""
        lab1 = rgb_to_lab(_make_rgb_tensor((100, 150, 200)))
        lab2 = rgb_to_lab(_make_rgb_tensor((110, 140, 210)))
        mean_result = delta_e_ciede2000(lab1, lab2)
        pixel_result = delta_e_ciede2000(lab1, lab2, reduce="none")
        assert pixel_result.mean(dim=(-2, -1))[0].item() == pytest.approx(
            mean_result[0].item(), rel=0.001
        )

    def test_per_pixel_with_mask_sets_nan(self):
        lab1 = rgb_to_lab(_make_rgb_tensor((128, 128, 128)))
        lab2 = rgb_to_lab(_make_rgb_tensor((180, 128, 128)))
        mask = torch.zeros(1, 64, 64, dtype=torch.bool, device=DEVICE)
        mask[0, :32, :] = True
        result = delta_e_ciede2000(lab1, lab2, mask=mask, reduce="none")
        assert torch.isnan(result[0, 0, 0])  # masked pixel -> NaN
        assert not torch.isnan(result[0, 32, 0])  # unmasked pixel -> value
```

Update the import at the top to also import `delta_e_cie76`:

```python
from vid_color_filter.gpu.color_metrics import (
    delta_e_cie76,
    delta_e_cie94,
    delta_e_ciede2000,
)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gpu_color_metrics.py::TestPerPixelMode -v`
Expected: TypeError — unexpected keyword argument `reduce`.

- [ ] **Step 3: Write minimal implementation**

In `src/vid_color_filter/gpu/color_metrics.py`, modify all three delta_e functions to accept a `reduce` parameter. The pattern is the same for all three — change the final return line.

For `delta_e_cie76`, change signature and return:

```python
def delta_e_cie76(
    lab1: torch.Tensor,
    lab2: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduce: str = "mean",
) -> torch.Tensor:
```

Replace the return line:

```python
    # old: return _masked_mean_per_image(de, mask)
    if reduce == "none":
        if mask is not None:
            de = torch.where(~mask, de, torch.tensor(float("nan"), device=de.device))
        return de
    return _masked_mean_per_image(de, mask)
```

For `delta_e_cie94`, same pattern — add `reduce: str = "mean"` after `K2`, replace return:

```python
def delta_e_cie94(
    lab1: torch.Tensor,
    lab2: torch.Tensor,
    mask: torch.Tensor | None = None,
    k_L: float = 1.0,
    K1: float = 0.045,
    K2: float = 0.015,
    reduce: str = "mean",
) -> torch.Tensor:
```

Replace the return line:

```python
    if reduce == "none":
        if mask is not None:
            de = torch.where(~mask, de, torch.tensor(float("nan"), device=de.device))
        return de
    return _masked_mean_per_image(de, mask)
```

For `delta_e_ciede2000`, add `reduce: str = "mean"` after `k_H`, replace return:

```python
def delta_e_ciede2000(
    lab1: torch.Tensor,
    lab2: torch.Tensor,
    mask: torch.Tensor | None = None,
    k_L: float = 1.0,
    k_C: float = 1.0,
    k_H: float = 1.0,
    reduce: str = "mean",
) -> torch.Tensor:
```

Replace the return line:

```python
    if reduce == "none":
        if mask is not None:
            de = torch.where(~mask, de, torch.tensor(float("nan"), device=de.device))
        return de
    return _masked_mean_per_image(de, mask)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gpu_color_metrics.py::TestPerPixelMode -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Run all existing tests to verify no regression**

Run: `pytest tests/test_gpu_color_metrics.py -v`
Expected: All tests PASS (existing tests use default `reduce="mean"` behavior).

- [ ] **Step 6: Commit**

```bash
git add src/vid_color_filter/gpu/color_metrics.py tests/test_gpu_color_metrics.py
git commit -m "feat: add per-pixel reduce='none' mode to delta_e functions"
```

---

## Chunk 2: S-CIELAB Spatial Filtering

### Task 3: Create S-CIELAB module — CSF kernels and opponent color transform

**Files:**
- Create: `src/vid_color_filter/gpu/scielab.py`
- Create: `tests/test_scielab.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scielab.py`:

```python
import numpy as np
import pytest
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_rgb_tensor(*colors, h=64, w=64):
    frames = []
    for c in colors:
        frames.append(np.full((h, w, 3), c, dtype=np.uint8))
    return torch.from_numpy(np.stack(frames)).to(DEVICE)


class TestCSFKernels:
    def test_build_csf_kernels_returns_three(self):
        from vid_color_filter.gpu.scielab import build_csf_kernels
        kernels = build_csf_kernels(pixels_per_degree=60, device=DEVICE)
        assert len(kernels) == 3  # O1, O2, O3

    def test_kernels_sum_to_one(self):
        from vid_color_filter.gpu.scielab import build_csf_kernels
        kernels = build_csf_kernels(pixels_per_degree=60, device=DEVICE)
        for k in kernels:
            assert k.sum().item() == pytest.approx(1.0, abs=0.01)

    def test_kernel_shapes_are_1d_and_odd(self):
        from vid_color_filter.gpu.scielab import build_csf_kernels
        kernels = build_csf_kernels(pixels_per_degree=60, device=DEVICE)
        for k in kernels:
            assert k.dim() == 1  # 1D kernels for separable convolution
            assert k.shape[0] % 2 == 1

    def test_achromatic_kernel_largest(self):
        """O1 achromatic kernel should be largest (highest freq sensitivity)."""
        from vid_color_filter.gpu.scielab import build_csf_kernels
        kernels = build_csf_kernels(pixels_per_degree=60, device=DEVICE)
        assert kernels[0].shape[0] >= kernels[1].shape[0]
        assert kernels[0].shape[0] >= kernels[2].shape[0]

    def test_kernel_size_capped(self):
        """Large ppd should not create prohibitively large kernels."""
        from vid_color_filter.gpu.scielab import build_csf_kernels, _MAX_KERNEL_SIZE
        kernels = build_csf_kernels(pixels_per_degree=120, device=DEVICE)
        for k in kernels:
            assert k.shape[0] <= _MAX_KERNEL_SIZE


class TestOpponentTransform:
    def test_xyz_to_opponent_shape(self):
        from vid_color_filter.gpu.scielab import xyz_to_opponent
        xyz = torch.rand(2, 64, 64, 3, device=DEVICE)
        opp = xyz_to_opponent(xyz)
        assert opp.shape == (2, 64, 64, 3)

    def test_roundtrip_xyz_opponent_xyz(self):
        from vid_color_filter.gpu.scielab import xyz_to_opponent, opponent_to_xyz
        xyz = torch.rand(2, 32, 32, 3, device=DEVICE) * 0.5
        recovered = opponent_to_xyz(xyz_to_opponent(xyz))
        torch.testing.assert_close(recovered, xyz, atol=1e-4, rtol=1e-4)


class TestSCIELABFilter:
    def test_output_shape_matches_input(self):
        from vid_color_filter.gpu.scielab import scielab_filter
        rgb = _make_rgb_tensor((128, 128, 128), (200, 100, 50))
        lab = scielab_filter(rgb, pixels_per_degree=60)
        assert lab.shape == (2, 64, 64, 3)

    def test_uniform_image_unchanged(self):
        """A solid-color image should be nearly unchanged by S-CIELAB filtering."""
        from vid_color_filter.gpu.scielab import scielab_filter
        from vid_color_filter.gpu.color_space import rgb_to_lab
        rgb = _make_rgb_tensor((128, 128, 128))
        lab_direct = rgb_to_lab(rgb)
        lab_scielab = scielab_filter(rgb, pixels_per_degree=60)
        # Center pixels (avoiding edge effects) should be very close
        center = slice(16, 48)
        torch.testing.assert_close(
            lab_scielab[0, center, center],
            lab_direct[0, center, center],
            atol=1.0, rtol=0.05,
        )

    def test_noise_is_suppressed(self):
        """S-CIELAB should reduce ΔE for per-pixel random noise."""
        from vid_color_filter.gpu.scielab import scielab_filter
        from vid_color_filter.gpu.color_space import rgb_to_lab
        from vid_color_filter.gpu.color_metrics import delta_e_ciede2000

        base = torch.full((1, 64, 64, 3), 128, dtype=torch.uint8, device=DEVICE)
        noise = base.clone()
        # Add random per-pixel noise (simulates codec noise)
        torch.manual_seed(42)
        noise = (noise.float() + torch.randn(1, 64, 64, 3, device=DEVICE) * 5).clamp(0, 255).to(torch.uint8)

        # Direct Lab ΔE
        de_direct = delta_e_ciede2000(rgb_to_lab(base), rgb_to_lab(noise))
        # S-CIELAB filtered ΔE
        de_filtered = delta_e_ciede2000(
            scielab_filter(base, pixels_per_degree=60),
            scielab_filter(noise, pixels_per_degree=60),
        )
        assert de_filtered[0].item() < de_direct[0].item()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scielab.py -v`
Expected: ImportError — `scielab` module does not exist.

- [ ] **Step 3: Implement CSF kernel builder**

Create `src/vid_color_filter/gpu/scielab.py`:

```python
"""S-CIELAB spatial filtering for perceptual color difference.

Implements the Zhang & Wandell (1997) S-CIELAB algorithm:
RGB -> XYZ -> Poirson-Wandell opponent -> CSF filtering -> XYZ -> Lab

The CSF filtering removes spatially imperceptible color differences,
making ΔE measurements more aligned with human perception.
"""

import torch
import torch.nn.functional as F

from vid_color_filter.gpu.color_space import rgb_to_xyz, xyz_to_lab

# Poirson-Wandell opponent color transform matrix (XYZ -> O1, O2, O3)
_PW_MATRIX = torch.tensor([
    [0.9795, 1.5318, 0.1225],
    [-0.1071, 0.3122, 0.0215],
    [0.0383, 0.0023, 0.5765],
], dtype=torch.float32)

# CSF parameters: list of (weight, sigma_degrees) tuples per channel
# From Zhang & Wandell (1997)
_CSF_PARAMS = [
    # O1 (achromatic): 3 Gaussians
    [(0.921, 0.0283), (0.105, 0.133), (-0.026, 4.336)],
    # O2 (red-green): 2 Gaussians
    [(0.531, 0.0392), (0.330, 0.494)],
    # O3 (blue-yellow): 2 Gaussians
    [(0.488, 0.0536), (0.371, 0.386)],
]

def _make_gaussian_kernel_1d(sigma_pixels: float, size: int) -> torch.Tensor:
    """Create a 1D Gaussian kernel at a given size."""
    radius = size // 2
    coords = torch.arange(size, dtype=torch.float32) - radius
    g = torch.exp(-0.5 * (coords / max(sigma_pixels, 1e-6)) ** 2)
    return g / g.sum()

# Maximum kernel diameter to avoid prohibitive convolution cost.
# The O1 achromatic CSF has sigma=4.336° which at 60ppd gives 260px,
# 3-sigma radius=780, diameter=1561. We cap and use separable 1D convs.
_MAX_KERNEL_SIZE = 513


def build_csf_kernels(
    pixels_per_degree: float = 60.0,
    device: str | torch.device = "cpu",
) -> list[torch.Tensor]:
    """Build CSF convolution kernels for the three opponent channels.

    Args:
        pixels_per_degree: Viewing condition parameter.
        device: Target device.

    Returns:
        List of 3 tensors [O1_kernel, O2_kernel, O3_kernel], each 1D (K,).
        Use separable convolution (H pass then V pass) for efficiency.
    """
    kernels = []
    for channel_params in _CSF_PARAMS:
        # Find the max kernel size, capped at _MAX_KERNEL_SIZE
        max_radius = 0
        for weight, sigma_deg in channel_params:
            sigma_px = sigma_deg * pixels_per_degree
            radius = min(int(3 * sigma_px + 0.5), _MAX_KERNEL_SIZE // 2)
            if radius < 1:
                radius = 1
            max_radius = max(max_radius, radius)

        size = 2 * max_radius + 1

        # Build each Gaussian component as 1D kernel at the common size,
        # then combine. We store 1D kernels for separable convolution.
        components = []
        for weight, sigma_deg in channel_params:
            sigma_px = sigma_deg * pixels_per_degree
            g1d = _make_gaussian_kernel_1d(sigma_px, size)
            components.append((weight, g1d))

        # Combine into a single 1D kernel (separable: apply H then V)
        combined_1d = torch.zeros(size, dtype=torch.float32)
        for weight, g1d in components:
            combined_1d += weight * g1d
        combined_1d = combined_1d / combined_1d.sum()

        kernels.append(combined_1d.to(device))

    return kernels


def xyz_to_opponent(xyz: torch.Tensor) -> torch.Tensor:
    """Convert XYZ to Poirson-Wandell opponent color space.

    Args:
        xyz: (B, H, W, 3) XYZ tensor.

    Returns:
        (B, H, W, 3) opponent tensor (O1, O2, O3).
    """
    mat = _PW_MATRIX.to(device=xyz.device, dtype=xyz.dtype)
    return xyz @ mat.T


def opponent_to_xyz(opp: torch.Tensor) -> torch.Tensor:
    """Convert Poirson-Wandell opponent colors back to XYZ.

    Args:
        opp: (B, H, W, 3) opponent tensor.

    Returns:
        (B, H, W, 3) XYZ tensor.
    """
    mat = _PW_MATRIX.to(device=opp.device, dtype=opp.dtype)
    inv_mat = torch.linalg.inv(mat)
    return opp @ inv_mat.T


def _apply_csf_to_channel(
    channel: torch.Tensor,
    kernel_1d: torch.Tensor,
) -> torch.Tensor:
    """Apply CSF kernel to a single channel via separable 1D convolutions.

    Uses two 1D passes (horizontal then vertical) instead of one 2D conv,
    reducing O(K^2) to O(2K) — critical for large CSF kernels.

    Args:
        channel: (B, H, W) tensor.
        kernel_1d: (K,) 1D kernel tensor.

    Returns:
        (B, H, W) filtered tensor.
    """
    K = kernel_1d.shape[0]
    pad = K // 2
    x = channel.unsqueeze(1)  # (B, 1, H, W)

    # Horizontal pass: kernel shape (1, 1, 1, K)
    kh = kernel_1d.reshape(1, 1, 1, K)
    x = F.conv2d(x, kh, padding=(0, pad))

    # Vertical pass: kernel shape (1, 1, K, 1)
    kv = kernel_1d.reshape(1, 1, K, 1)
    x = F.conv2d(x, kv, padding=(pad, 0))

    return x.squeeze(1)  # (B, H, W)


def scielab_filter(
    rgb: torch.Tensor,
    pixels_per_degree: float = 60.0,
    _cached_kernels: dict | None = None,
) -> torch.Tensor:
    """Apply S-CIELAB spatial filtering to RGB images, returning filtered Lab.

    Pipeline: RGB -> XYZ -> opponent -> CSF filter -> XYZ -> Lab

    Args:
        rgb: (B, H, W, 3) uint8 or float32 RGB tensor.
        pixels_per_degree: Viewing condition (default 60 for desktop monitor).
        _cached_kernels: Optional pre-built kernels dict for reuse.

    Returns:
        (B, H, W, 3) float32 Lab tensor after S-CIELAB filtering.
    """
    xyz = rgb_to_xyz(rgb)
    opp = xyz_to_opponent(xyz)

    if _cached_kernels is not None and pixels_per_degree in _cached_kernels:
        kernels = _cached_kernels[pixels_per_degree]
    else:
        kernels = build_csf_kernels(pixels_per_degree, device=rgb.device)
        if _cached_kernels is not None:
            _cached_kernels[pixels_per_degree] = kernels

    # Filter each opponent channel
    filtered_channels = []
    for i in range(3):
        filtered = _apply_csf_to_channel(opp[..., i], kernels[i])
        filtered_channels.append(filtered)

    filtered_opp = torch.stack(filtered_channels, dim=-1)  # (B, H, W, 3)
    filtered_xyz = opponent_to_xyz(filtered_opp)
    return xyz_to_lab(filtered_xyz)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scielab.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vid_color_filter/gpu/scielab.py tests/test_scielab.py
git commit -m "feat: add S-CIELAB spatial filtering module"
```

---

## Chunk 3: Adaptive Edit Region Mask

### Task 4: Create adaptive mask module with Otsu + hysteresis

**Files:**
- Create: `src/vid_color_filter/gpu/adaptive_mask.py`
- Create: `tests/test_adaptive_mask.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_adaptive_mask.py`:

```python
import numpy as np
import pytest
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_lab_pair(h=64, w=64, edit_region=None, edit_shift=50.0):
    """Create a source/edited Lab tensor pair with optional edit region.

    Returns (src_lab, edited_lab) both (1, H, W, 3).
    """
    src = torch.full((1, h, w, 3), 50.0, device=DEVICE)  # L=50, a=0, b=0
    edited = src.clone()
    if edit_region is not None:
        r = edit_region  # (row_start, row_end, col_start, col_end)
        edited[0, r[0]:r[1], r[2]:r[3], :] += edit_shift
    return src, edited


class TestOtsuThreshold:
    def test_returns_float(self):
        from vid_color_filter.gpu.adaptive_mask import otsu_threshold
        values = torch.randn(1000, device=DEVICE).abs()
        t = otsu_threshold(values)
        assert isinstance(t, float) or t.dim() == 0

    def test_separates_bimodal(self):
        """For a clearly bimodal distribution, Otsu should find the valley."""
        from vid_color_filter.gpu.adaptive_mask import otsu_threshold
        low = torch.full((500,), 1.0, device=DEVICE)
        high = torch.full((500,), 10.0, device=DEVICE)
        values = torch.cat([low, high])
        t = otsu_threshold(values)
        assert 2.0 < float(t) < 9.0


class TestAdaptiveMask:
    def test_identical_frames_empty_mask(self):
        from vid_color_filter.gpu.adaptive_mask import generate_adaptive_mask
        src, edited = _make_lab_pair()
        masks, coverages = generate_adaptive_mask(src, src)
        assert not masks.any()
        assert coverages[0].item() == pytest.approx(0.0, abs=0.01)

    def test_detects_large_edit(self):
        from vid_color_filter.gpu.adaptive_mask import generate_adaptive_mask
        src, edited = _make_lab_pair(edit_region=(20, 44, 20, 44), edit_shift=50.0)
        masks, coverages = generate_adaptive_mask(src, edited)
        assert masks[0, 30, 30].item() == True
        assert coverages[0].item() > 0.05

    def test_output_shapes(self):
        from vid_color_filter.gpu.adaptive_mask import generate_adaptive_mask
        src, edited = _make_lab_pair(h=100, w=80)
        masks, coverages = generate_adaptive_mask(src, edited)
        assert masks.shape == (1, 100, 80)
        assert masks.dtype == torch.bool
        assert coverages.shape == (1,)

    def test_fixed_threshold_fallback(self):
        """When diff_threshold is a float, use fixed threshold (legacy)."""
        from vid_color_filter.gpu.adaptive_mask import generate_adaptive_mask
        src, edited = _make_lab_pair(edit_region=(20, 44, 20, 44), edit_shift=50.0)
        masks, coverages = generate_adaptive_mask(src, edited, diff_threshold=5.0)
        assert masks[0, 30, 30].item() == True

    def test_batch_processing(self):
        from vid_color_filter.gpu.adaptive_mask import generate_adaptive_mask
        src = torch.full((4, 64, 64, 3), 50.0, device=DEVICE)
        edited = src.clone()
        edited[2, 20:40, 20:40, :] += 50.0  # Only frame 2 has edit
        masks, coverages = generate_adaptive_mask(src, edited)
        assert masks.shape == (4, 64, 64)
        assert coverages[0].item() < coverages[2].item()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adaptive_mask.py -v`
Expected: ImportError — module does not exist.

- [ ] **Step 3: Implement adaptive mask module**

Create `src/vid_color_filter/gpu/adaptive_mask.py`:

```python
"""Adaptive edit region mask with Otsu thresholding and hysteresis expansion.

Replaces the fixed-threshold mask generator with:
1. Otsu adaptive threshold from the Lab distance histogram
2. Hysteresis: high-threshold seeds expanded via low-threshold connectivity
3. Morphological dilation as safety boundary
"""

import torch
import torch.nn.functional as F


def otsu_threshold(
    values: torch.Tensor,
    n_bins: int = 256,
) -> torch.Tensor:
    """Compute Otsu's threshold on a 1D tensor of positive values.

    Args:
        values: 1D tensor of values to threshold.
        n_bins: Histogram bins.

    Returns:
        Scalar threshold tensor.
    """
    v_min = values.min()
    v_max = values.max()
    if v_max - v_min < 1e-8:
        return v_max  # uniform -> threshold above all

    hist = torch.histc(values.float(), bins=n_bins, min=float(v_min), max=float(v_max))
    bin_edges = torch.linspace(float(v_min), float(v_max), n_bins + 1, device=values.device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total = hist.sum()
    cum_sum = torch.cumsum(hist, dim=0)
    cum_mean = torch.cumsum(hist * bin_centers, dim=0)

    global_mean = cum_mean[-1] / total

    w0 = cum_sum / total
    w1 = 1.0 - w0
    mu0 = cum_mean / (cum_sum + 1e-10)
    mu1 = (cum_mean[-1] - cum_mean) / (total - cum_sum + 1e-10)

    variance = w0 * w1 * (mu0 - mu1) ** 2

    # Avoid edge bins
    variance[0] = 0
    variance[-1] = 0

    best_idx = variance.argmax()
    return bin_centers[best_idx]


def _hysteresis_expand(
    seed_mask: torch.Tensor,
    low_mask: torch.Tensor,
    max_iterations: int = 50,
) -> torch.Tensor:
    """Expand seed_mask into low_mask regions via connectivity.

    Args:
        seed_mask: (B, H, W) bool — high-confidence regions.
        low_mask: (B, H, W) bool — regions that could be included if connected to seeds.
        max_iterations: Maximum expansion iterations.

    Returns:
        (B, H, W) bool — expanded mask.
    """
    current = seed_mask.float().unsqueeze(1)  # (B, 1, H, W)
    low = low_mask.float().unsqueeze(1)

    for _ in range(max_iterations):
        # Dilate by 1 pixel using max_pool
        expanded = F.max_pool2d(current, kernel_size=3, stride=1, padding=1)
        # Only keep pixels that are also in low_mask
        expanded = expanded * low
        if torch.equal(expanded, current):
            break
        current = expanded

    return current.squeeze(1).bool()


def generate_adaptive_mask(
    src_lab: torch.Tensor,
    edited_lab: torch.Tensor,
    diff_threshold: float | None = None,
    dilate_kernel: int = 21,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate edit region masks with adaptive or fixed thresholding.

    Args:
        src_lab: (B, H, W, 3) Lab tensors of source frames.
        edited_lab: (B, H, W, 3) Lab tensors of edited frames.
        diff_threshold: If None, use Otsu adaptive. If float, use fixed threshold.
        dilate_kernel: Dilation kernel size (must be odd).

    Returns:
        masks: (B, H, W) bool tensor (True = edited/excluded).
        coverages: (B,) float tensor (fraction of masked pixels per frame).
    """
    B, H, W, _ = src_lab.shape

    # Per-pixel Lab distance
    diff = src_lab - edited_lab
    dist = torch.sqrt((diff * diff).sum(dim=-1))  # (B, H, W)

    masks_list = []
    for i in range(B):
        frame_dist = dist[i]  # (H, W)

        if diff_threshold is not None:
            # Fixed threshold (legacy mode)
            mask_i = frame_dist > diff_threshold
        else:
            # Otsu adaptive threshold
            flat = frame_dist.flatten()
            if flat.max() - flat.min() < 1e-6:
                # No variation -> no edits
                masks_list.append(torch.zeros(H, W, dtype=torch.bool, device=src_lab.device))
                continue

            threshold_high = otsu_threshold(flat)
            threshold_low = threshold_high * 0.5

            seed = frame_dist > threshold_high
            low = frame_dist > threshold_low

            mask_i = _hysteresis_expand(
                seed.unsqueeze(0), low.unsqueeze(0)
            ).squeeze(0)

        masks_list.append(mask_i)

    masks = torch.stack(masks_list)  # (B, H, W)

    # Morphological dilation
    if dilate_kernel > 1:
        masks_float = masks.float().unsqueeze(1)  # (B, 1, H, W)
        pad = dilate_kernel // 2
        masks_float = F.max_pool2d(masks_float, kernel_size=dilate_kernel, stride=1, padding=pad)
        masks = masks_float.squeeze(1).bool()

    coverages = masks.float().mean(dim=(-2, -1))  # (B,)

    return masks, coverages
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adaptive_mask.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vid_color_filter/gpu/adaptive_mask.py tests/test_adaptive_mask.py
git commit -m "feat: add adaptive Otsu + hysteresis edit mask module"
```

---

## Chunk 4: Temporal Aggregation and Multi-dimensional Scoring

### Task 5: Create temporal aggregator module

**Files:**
- Create: `src/vid_color_filter/gpu/temporal_aggregator.py`
- Create: `tests/test_temporal_aggregator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_temporal_aggregator.py`:

```python
import numpy as np
import pytest
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestTemporalAggregate:
    def test_output_shapes(self):
        from vid_color_filter.gpu.temporal_aggregator import temporal_aggregate
        de_maps = torch.rand(16, 64, 64, device=DEVICE)
        masks = torch.zeros(16, 64, 64, dtype=torch.bool, device=DEVICE)
        median_map, iqr_map = temporal_aggregate(de_maps, masks)
        assert median_map.shape == (64, 64)
        assert iqr_map.shape == (64, 64)

    def test_stable_signal_low_iqr(self):
        """Constant ΔE across frames -> IQR ≈ 0."""
        from vid_color_filter.gpu.temporal_aggregator import temporal_aggregate
        de_maps = torch.full((32, 64, 64), 2.0, device=DEVICE)
        masks = torch.zeros(32, 64, 64, dtype=torch.bool, device=DEVICE)
        median_map, iqr_map = temporal_aggregate(de_maps, masks)
        assert median_map[32, 32].item() == pytest.approx(2.0, abs=0.1)
        assert iqr_map[32, 32].item() == pytest.approx(0.0, abs=0.1)

    def test_noisy_signal_high_iqr(self):
        """High variance across frames -> high IQR."""
        from vid_color_filter.gpu.temporal_aggregator import temporal_aggregate
        torch.manual_seed(42)
        de_maps = torch.randn(32, 64, 64, device=DEVICE).abs() * 5
        masks = torch.zeros(32, 64, 64, dtype=torch.bool, device=DEVICE)
        _, iqr_map = temporal_aggregate(de_maps, masks)
        assert iqr_map.mean().item() > 1.0

    def test_masked_pixels_excluded(self):
        """Pixels masked in >50% of frames -> NaN in output."""
        from vid_color_filter.gpu.temporal_aggregator import temporal_aggregate
        de_maps = torch.full((10, 64, 64), 3.0, device=DEVICE)
        masks = torch.zeros(10, 64, 64, dtype=torch.bool, device=DEVICE)
        # Mask pixel (0,0) in 8 out of 10 frames (>50%)
        masks[:8, 0, 0] = True
        median_map, _ = temporal_aggregate(de_maps, masks)
        assert torch.isnan(median_map[0, 0])

    def test_partially_masked_pixel_uses_unmasked(self):
        """Pixel masked in <50% frames should use unmasked frames only."""
        from vid_color_filter.gpu.temporal_aggregator import temporal_aggregate
        de_maps = torch.full((10, 64, 64), 3.0, device=DEVICE)
        de_maps[0, 10, 10] = 100.0  # outlier in frame 0
        masks = torch.zeros(10, 64, 64, dtype=torch.bool, device=DEVICE)
        masks[0, 10, 10] = True  # mask the outlier frame
        median_map, _ = temporal_aggregate(de_maps, masks)
        assert median_map[10, 10].item() == pytest.approx(3.0, abs=0.1)


class TestComputeScores:
    def test_zero_de_all_pass(self):
        from vid_color_filter.gpu.temporal_aggregator import compute_scores
        median_map = torch.zeros(64, 64, device=DEVICE)
        iqr_map = torch.zeros(64, 64, device=DEVICE)
        scores = compute_scores(median_map, iqr_map)
        assert scores["pass_global"] == True
        assert scores["pass_local"] == True
        assert scores["pass"] == True

    def test_high_global_shift_fails(self):
        from vid_color_filter.gpu.temporal_aggregator import compute_scores
        median_map = torch.full((64, 64), 5.0, device=DEVICE)
        iqr_map = torch.zeros(64, 64, device=DEVICE)
        scores = compute_scores(median_map, iqr_map, global_threshold=2.0)
        assert scores["pass_global"] == False
        assert scores["pass"] == False

    def test_high_local_diff_fails(self):
        from vid_color_filter.gpu.temporal_aggregator import compute_scores
        median_map = torch.full((64, 64), 1.0, device=DEVICE)
        # Add localized high region
        median_map[0:5, 0:5] = 10.0
        iqr_map = torch.zeros(64, 64, device=DEVICE)
        scores = compute_scores(median_map, iqr_map, local_threshold=3.0)
        assert scores["pass_local"] == False
        assert scores["pass"] == False

    def test_scores_are_floats(self):
        from vid_color_filter.gpu.temporal_aggregator import compute_scores
        median_map = torch.rand(64, 64, device=DEVICE)
        iqr_map = torch.rand(64, 64, device=DEVICE)
        scores = compute_scores(median_map, iqr_map)
        assert isinstance(scores["global_shift_score"], float)
        assert isinstance(scores["local_diff_score"], float)
        assert isinstance(scores["temporal_instability"], float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_temporal_aggregator.py -v`
Expected: ImportError — module does not exist.

- [ ] **Step 3: Implement temporal aggregator**

Create `src/vid_color_filter/gpu/temporal_aggregator.py`:

```python
"""Temporal aggregation and multi-dimensional scoring for video color difference.

Takes per-frame, per-pixel ΔE maps and produces:
1. Temporal median map (stable color difference signal)
2. Temporal IQR map (frame-to-frame fluctuation measure)
3. Three scores: global shift, local difference, temporal instability
"""

import torch


def temporal_aggregate(
    de_maps: torch.Tensor,
    masks: torch.Tensor,
    min_unmasked_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute temporal median and IQR of per-pixel ΔE across frames.

    Args:
        de_maps: (N, H, W) per-pixel ΔE values across N frames.
        masks: (N, H, W) bool tensor. True = masked (excluded) pixel in that frame.
        min_unmasked_ratio: Minimum fraction of frames a pixel must be unmasked
            to be included in scoring. Default 0.5.

    Returns:
        median_map: (H, W) temporal median ΔE. NaN for excluded pixels.
        iqr_map: (H, W) temporal IQR. NaN for excluded pixels.
    """
    N, H, W = de_maps.shape

    # Set masked pixels to NaN for nanmedian/nanquantile
    de_nan = de_maps.clone()
    de_nan[masks] = float("nan")

    # Count unmasked frames per pixel
    unmasked_count = (~masks).float().sum(dim=0)  # (H, W)
    min_frames = N * min_unmasked_ratio
    excluded = unmasked_count < min_frames  # (H, W)

    # Temporal median
    median_map = torch.nanmedian(de_nan, dim=0).values  # (H, W)

    # Temporal IQR (Q75 - Q25)
    q75 = torch.nanquantile(de_nan, 0.75, dim=0)  # (H, W)
    q25 = torch.nanquantile(de_nan, 0.25, dim=0)  # (H, W)
    iqr_map = q75 - q25

    # Mark excluded pixels as NaN
    nan_val = torch.tensor(float("nan"), device=de_maps.device)
    median_map = torch.where(excluded, nan_val, median_map)
    iqr_map = torch.where(excluded, nan_val, iqr_map)

    return median_map, iqr_map


def compute_scores(
    median_map: torch.Tensor,
    iqr_map: torch.Tensor,
    global_threshold: float = 2.0,
    local_threshold: float = 3.0,
) -> dict:
    """Compute multi-dimensional color difference scores.

    Args:
        median_map: (H, W) temporal median ΔE map (may contain NaN).
        iqr_map: (H, W) temporal IQR map (may contain NaN).
        global_threshold: Pass/fail threshold for global shift score.
        local_threshold: Pass/fail threshold for local difference score.

    Returns:
        Dict with scores and pass/fail results.
    """
    # Filter out NaN pixels
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
    p95 = float(torch.quantile(valid_medians, 0.95).item())
    local_diff = p95 - global_shift
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_temporal_aggregator.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vid_color_filter/gpu/temporal_aggregator.py tests/test_temporal_aggregator.py
git commit -m "feat: add temporal aggregation and multi-dimensional scoring"
```

---

## Chunk 5: Pipeline Integration — batch_scorer + run.py + end-to-end tests

### Task 6: Update batch_scorer.py with new pipeline

**Prerequisite:** Tasks 1-5 must be complete and all tests passing.

**Files:**
- Modify: `src/vid_color_filter/gpu/batch_scorer.py`
- Modify: `tests/test_gpu_scorer.py`

- [ ] **Step 1: Write the failing end-to-end test**

Add to `tests/test_gpu_scorer.py` (keep existing tests, add new class):

```python
class TestNewScoringPipeline:
    def test_output_has_new_fields(self):
        """New pipeline should return global_shift_score, local_diff_score, etc."""
        src = torch.full((16, 64, 64, 3), 128, dtype=torch.uint8, device=DEVICE)
        edited = src.clone()
        result = score_video_pair_gpu(
            src, edited, src_path="test.mp4",
            use_scielab=True,
        )
        assert "global_shift_score" in result
        assert "local_diff_score" in result
        assert "temporal_instability" in result
        assert "pass_global" in result
        assert "pass_local" in result
        assert "pass" in result
        # Backward compat
        assert "per_frame_mean_delta_e" in result
        assert "mean_delta_e_per_frame" in result

    def test_identical_videos_pass(self):
        src = torch.full((16, 64, 64, 3), 128, dtype=torch.uint8, device=DEVICE)
        result = score_video_pair_gpu(
            src, src.clone(), src_path="test.mp4",
            use_scielab=True,
        )
        assert result["pass"] == True
        assert result["global_shift_score"] < 1.0

    def test_color_shifted_video_detected(self):
        """A uniform color shift should show up in global_shift_score."""
        src = torch.full((16, 64, 64, 3), 128, dtype=torch.uint8, device=DEVICE)
        edited = torch.full((16, 64, 64, 3), 148, dtype=torch.uint8, device=DEVICE)
        result = score_video_pair_gpu(
            src, edited, src_path="test.mp4",
            use_scielab=True, global_threshold=1.0,
        )
        assert result["global_shift_score"] > 1.0
        assert result["pass_global"] == False

    def test_backward_compat_mode(self):
        """Without use_scielab, output should match old format."""
        src = torch.full((8, 64, 64, 3), 128, dtype=torch.uint8, device=DEVICE)
        result = score_video_pair_gpu(src, src.clone(), src_path="test.mp4")
        assert "max_mean_delta_e" in result
        assert "mean_delta_e_per_frame" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gpu_scorer.py::TestNewScoringPipeline -v`
Expected: TypeError — unexpected keyword argument `use_scielab`.

- [ ] **Step 3: Update batch_scorer.py**

Replace the entire content of `src/vid_color_filter/gpu/batch_scorer.py`:

```python
import os
import torch
import numpy as np

from vid_color_filter.gpu.color_space import rgb_to_lab
from vid_color_filter.gpu.color_metrics import METRICS
from vid_color_filter.gpu.mask_generator import generate_edit_mask_gpu


def score_video_pair_gpu(
    src_frames: torch.Tensor,
    edited_frames: torch.Tensor,
    src_path: str = "",
    threshold: float = 2.0,
    diff_threshold: float | None = 5.0,
    dilate_kernel: int = 21,
    metric: str = "cie94",
    use_scielab: bool = False,
    pixels_per_degree: float = 60.0,
    global_threshold: float | None = None,
    local_threshold: float = 3.0,
    chunk_size: int = 8,
) -> dict:
    """Score a video pair on GPU with batched frame processing.

    Args:
        src_frames: (N, H, W, 3) uint8 RGB tensor of source frames.
        edited_frames: (N, H, W, 3) uint8 RGB tensor of edited frames.
        src_path: Source video path (used for pair ID).
        threshold: Delta E threshold for pass/fail (legacy / global alias).
        diff_threshold: Lab distance for mask. None = Otsu adaptive.
        dilate_kernel: Dilation kernel size.
        metric: Color metric name.
        use_scielab: If True, use new S-CIELAB temporal pipeline.
        pixels_per_degree: S-CIELAB viewing condition.
        global_threshold: Explicit global threshold (overrides threshold).
        local_threshold: Local difference threshold.
        chunk_size: Frames per processing chunk.

    Returns:
        Dict with scoring results.
    """
    pair_id = os.path.splitext(os.path.basename(src_path))[0] if src_path else ""

    if global_threshold is None:
        global_threshold = threshold

    if not use_scielab:
        return _score_legacy(
            src_frames, edited_frames, pair_id,
            threshold=threshold,
            diff_threshold=diff_threshold if diff_threshold is not None else 5.0,
            dilate_kernel=dilate_kernel,
            metric=metric,
        )

    return _score_scielab(
        src_frames, edited_frames, pair_id,
        diff_threshold=diff_threshold,
        dilate_kernel=dilate_kernel,
        metric=metric,
        pixels_per_degree=pixels_per_degree,
        global_threshold=global_threshold,
        local_threshold=local_threshold,
        chunk_size=chunk_size,
    )


def _score_legacy(
    src_frames, edited_frames, pair_id,
    threshold, diff_threshold, dilate_kernel, metric,
):
    """Original scoring pipeline (backward compatible)."""
    metric_fn = METRICS[metric]
    src_lab = rgb_to_lab(src_frames)
    edited_lab = rgb_to_lab(edited_frames)

    masks, coverages = generate_edit_mask_gpu(
        src_lab, edited_lab,
        diff_threshold=diff_threshold,
        dilate_kernel=dilate_kernel,
    )

    mean_des = metric_fn(src_lab, edited_lab, masks)

    full_coverage = coverages >= 1.0
    nan_mask = torch.isnan(mean_des)
    needs_global = full_coverage & nan_mask
    if needs_global.any():
        global_des = metric_fn(src_lab, edited_lab, mask=None)
        mean_des = torch.where(needs_global, global_des, mean_des)

    mean_des_list = mean_des.cpu().tolist()
    coverages_list = coverages.cpu().tolist()

    valid_des = [d for d in mean_des_list if not np.isnan(d)]
    max_mean_de = max(valid_des) if valid_des else float("nan")

    return {
        "video_pair_id": pair_id,
        "mean_delta_e_per_frame": mean_des_list,
        "max_mean_delta_e": max_mean_de,
        "pass": max_mean_de < threshold if not np.isnan(max_mean_de) else False,
        "mask_coverage_ratio": max(coverages_list) if coverages_list else 0.0,
    }


def _score_scielab(
    src_frames, edited_frames, pair_id,
    diff_threshold, dilate_kernel, metric,
    pixels_per_degree, global_threshold, local_threshold, chunk_size,
):
    """New S-CIELAB temporal scoring pipeline."""
    from vid_color_filter.gpu.scielab import scielab_filter
    from vid_color_filter.gpu.adaptive_mask import generate_adaptive_mask
    from vid_color_filter.gpu.temporal_aggregator import temporal_aggregate, compute_scores

    N = src_frames.shape[0]
    metric_fn = METRICS[metric]
    kernel_cache = {}

    all_de_maps = []
    all_masks = []
    per_frame_means = []
    all_coverages = []

    # Process in chunks to manage memory
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        src_chunk = src_frames[start:end]
        edited_chunk = edited_frames[start:end]

        # Raw Lab for masking (spec: mask uses raw Lab distance, not filtered)
        from vid_color_filter.gpu.color_space import rgb_to_lab
        src_lab_raw = rgb_to_lab(src_chunk)
        edited_lab_raw = rgb_to_lab(edited_chunk)

        # Adaptive mask on raw Lab
        masks, coverages = generate_adaptive_mask(
            src_lab_raw, edited_lab_raw,
            diff_threshold=diff_threshold,
            dilate_kernel=dilate_kernel,
        )

        # S-CIELAB filtered Lab for ΔE computation
        src_lab = scielab_filter(src_chunk, pixels_per_degree, _cached_kernels=kernel_cache)
        edited_lab = scielab_filter(edited_chunk, pixels_per_degree, _cached_kernels=kernel_cache)

        # Per-pixel ΔE on S-CIELAB filtered Lab, with raw-Lab mask
        de_map = metric_fn(src_lab, edited_lab, mask=masks, reduce="none")  # (chunk, H, W)
        per_frame_mean = metric_fn(src_lab, edited_lab, mask=masks)  # (chunk,)

        all_de_maps.append(de_map)
        all_masks.append(masks)
        per_frame_means.append(per_frame_mean)
        all_coverages.append(coverages)

    # Concatenate all chunks
    de_maps = torch.cat(all_de_maps, dim=0)  # (N, H, W)
    masks = torch.cat(all_masks, dim=0)  # (N, H, W)
    per_frame_mean_de = torch.cat(per_frame_means, dim=0)  # (N,)
    coverages = torch.cat(all_coverages, dim=0)  # (N,)

    # Temporal aggregation
    median_map, iqr_map = temporal_aggregate(de_maps, masks)

    # Multi-dimensional scoring
    scores = compute_scores(
        median_map, iqr_map,
        global_threshold=global_threshold,
        local_threshold=local_threshold,
    )

    per_frame_list = per_frame_mean_de.cpu().tolist()
    max_coverage = float(coverages.max().item())

    return {
        "video_pair_id": pair_id,
        "global_shift_score": scores["global_shift_score"],
        "local_diff_score": scores["local_diff_score"],
        "temporal_instability": scores["temporal_instability"],
        "pass_global": scores["pass_global"],
        "pass_local": scores["pass_local"],
        "pass": scores["pass"],
        "mask_coverage_ratio": max_coverage,
        # Backward compatibility fields
        "per_frame_mean_delta_e": per_frame_list,
        "mean_delta_e_per_frame": per_frame_list,
        "max_mean_delta_e": max(d for d in per_frame_list if not np.isnan(d)) if any(not np.isnan(d) for d in per_frame_list) else float("nan"),  # legacy compat
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gpu_scorer.py -v`
Expected: All tests PASS (both old and new).

- [ ] **Step 5: Commit**

```bash
git add src/vid_color_filter/gpu/batch_scorer.py tests/test_gpu_scorer.py
git commit -m "feat: integrate S-CIELAB temporal pipeline into batch_scorer"
```

---

### Task 7: Update run.py with new CLI parameters

**Files:**
- Modify: `run.py`

- [ ] **Step 1: Read current run.py argparse section**

Reference: `run.py` lines 40-70 contain the argparse setup.

- [ ] **Step 2: Add new CLI arguments**

In `run.py`, after the existing `--dilate-kernel` argument and before `args = parser.parse_args()`, add:

```python
    parser.add_argument(
        "--use-scielab", action="store_true", default=False,
        help="Enable S-CIELAB temporal pipeline (new scoring method).",
    )
    parser.add_argument(
        "--pixels-per-degree", type=float, default=60.0,
        help="S-CIELAB viewing condition. Desktop monitor ~60cm = 60 (default).",
    )
    parser.add_argument(
        "--global-threshold", type=float, default=None,
        help="Global color shift threshold. Defaults to --threshold value.",
    )
    parser.add_argument(
        "--local-threshold", type=float, default=3.0,
        help="Local color difference threshold (default: 3.0).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=8,
        help="Frames per GPU processing chunk (default: 8). Lower = less memory.",
    )
```

Change the `--diff-threshold` default from `5.0` to `None` and type handling:

```python
    parser.add_argument(
        "--diff-threshold", type=float, default=None,
        help="Edit mask threshold. None = Otsu adaptive (default). Float = fixed.",
    )
```

- [ ] **Step 3: Update the scoring call to pass new args**

In the scoring loop in `run.py`, update the `score_video_pair_gpu` call to pass new parameters:

```python
        result = score_video_pair_gpu(
            src_t, ed_t,
            src_path=src_path,
            threshold=args.threshold,
            diff_threshold=args.diff_threshold,
            dilate_kernel=args.dilate_kernel,
            metric=args.metric,
            use_scielab=args.use_scielab,
            pixels_per_degree=args.pixels_per_degree,
            global_threshold=args.global_threshold,
            local_threshold=args.local_threshold,
            chunk_size=args.chunk_size,
        )
```

- [ ] **Step 4: Update default --metric and --num-frames when --use-scielab is active**

Change `--metric` and `--num-frames` defaults to `None` so we can detect user-explicit values:

```python
    parser.add_argument("--num-frames", type=int, default=None, ...)
    parser.add_argument("--metric", choices=["cie76", "cie94", "ciede2000"], default=None, ...)
```

After `args = parser.parse_args()`, add:

```python
    # Apply context-appropriate defaults (None = user didn't specify)
    if args.num_frames is None:
        args.num_frames = 32 if args.use_scielab else 16
    if args.metric is None:
        args.metric = "ciede2000" if args.use_scielab else "cie94"
```

- [ ] **Step 5: Run a quick smoke test**

Run: `python run.py --help`
Expected: New arguments `--use-scielab`, `--pixels-per-degree`, `--global-threshold`, `--local-threshold`, `--chunk-size` visible in help output.

- [ ] **Step 6: Commit**

```bash
git add run.py
git commit -m "feat: add S-CIELAB CLI args to run.py"
```

---

### Task 8: Final regression test

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 2: Commit any fixes if needed**

If any tests fail, fix and commit.

- [ ] **Step 3: Final commit message**

If all green on first run, no commit needed here.
