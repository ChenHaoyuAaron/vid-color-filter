import numpy as np
import pytest
import torch

from vid_color_filter.gpu.color_space import rgb_to_lab, rgb_to_xyz
from vid_color_filter.gpu.color_metrics import (
    delta_e_cie76,
    delta_e_cie94,
    delta_e_ciede2000,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_rgb_tensor(*colors, h=64, w=64):
    """Create (B, H, W, 3) uint8 tensor from solid RGB color tuples."""
    frames = []
    for c in colors:
        frames.append(np.full((h, w, 3), c, dtype=np.uint8))
    return torch.from_numpy(np.stack(frames)).to(DEVICE)


class TestRgbToXyz:
    def test_shape_preserved(self):
        rgb = _make_rgb_tensor((128, 128, 128), (200, 100, 50))
        xyz = rgb_to_xyz(rgb)
        assert xyz.shape == (2, 64, 64, 3)

    def test_white_d65(self):
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


class TestRgbToLab:
    def test_shape_preserved(self):
        rgb = _make_rgb_tensor((128, 128, 128), (200, 100, 50))
        lab = rgb_to_lab(rgb)
        assert lab.shape == (2, 64, 64, 3)

    def test_matches_skimage(self):
        """Cross-validate GPU rgb_to_lab against skimage.color.rgb2lab."""
        from skimage.color import rgb2lab

        rgb_np = np.array([[[128, 64, 200]]], dtype=np.uint8)
        expected = rgb2lab(rgb_np)

        rgb_t = torch.from_numpy(rgb_np).unsqueeze(0).to(DEVICE)
        result = rgb_to_lab(rgb_t).cpu().numpy()[0]

        np.testing.assert_allclose(result, expected, atol=0.5)

    def test_black_and_white(self):
        black = _make_rgb_tensor((0, 0, 0))
        white = _make_rgb_tensor((255, 255, 255))
        lab_black = rgb_to_lab(black)[0, 0, 0].cpu()
        lab_white = rgb_to_lab(white)[0, 0, 0].cpu()
        assert lab_black[0].item() == pytest.approx(0.0, abs=0.5)
        assert lab_white[0].item() == pytest.approx(100.0, abs=0.5)


class TestCIE76:
    def test_identical_returns_zero(self):
        rgb = _make_rgb_tensor((128, 128, 128))
        lab = rgb_to_lab(rgb)
        result = delta_e_cie76(lab, lab)
        assert result[0].item() == pytest.approx(0.0, abs=0.01)

    def test_different_colors_positive(self):
        lab1 = rgb_to_lab(_make_rgb_tensor((128, 128, 128)))
        lab2 = rgb_to_lab(_make_rgb_tensor((180, 128, 128)))
        result = delta_e_cie76(lab1, lab2)
        assert result[0].item() > 0.0

    def test_mask_excludes_pixels(self):
        src = _make_rgb_tensor((128, 128, 128))
        edited = src.clone()
        edited[0, :32, :] = torch.tensor([255, 0, 0], dtype=torch.uint8, device=DEVICE)
        lab1 = rgb_to_lab(src)
        lab2 = rgb_to_lab(edited)
        mask = torch.zeros(1, 64, 64, dtype=torch.bool, device=DEVICE)
        mask[0, :32, :] = True
        result = delta_e_cie76(lab1, lab2, mask)
        assert result[0].item() == pytest.approx(0.0, abs=0.01)

    def test_matches_skimage_cie76(self):
        """Cross-validate against skimage deltaE_cie76."""
        from skimage.color import rgb2lab, deltaE_cie76

        c1 = np.full((64, 64, 3), 100, dtype=np.uint8)
        c2 = np.full((64, 64, 3), 150, dtype=np.uint8)
        lab1_np = rgb2lab(c1)
        lab2_np = rgb2lab(c2)
        expected = deltaE_cie76(lab1_np, lab2_np).mean()

        t1 = torch.from_numpy(c1).unsqueeze(0).to(DEVICE)
        t2 = torch.from_numpy(c2).unsqueeze(0).to(DEVICE)
        result = delta_e_cie76(rgb_to_lab(t1), rgb_to_lab(t2))
        assert result[0].item() == pytest.approx(float(expected), rel=0.05)


class TestCIE94:
    def test_identical_returns_zero(self):
        rgb = _make_rgb_tensor((128, 128, 128))
        lab = rgb_to_lab(rgb)
        result = delta_e_cie94(lab, lab)
        assert result[0].item() == pytest.approx(0.0, abs=0.01)

    def test_different_colors_positive(self):
        lab1 = rgb_to_lab(_make_rgb_tensor((128, 128, 128)))
        lab2 = rgb_to_lab(_make_rgb_tensor((180, 128, 128)))
        result = delta_e_cie94(lab1, lab2)
        assert result[0].item() > 0.0


class TestCIEDE2000:
    def test_identical_returns_zero(self):
        rgb = _make_rgb_tensor((128, 128, 128))
        lab = rgb_to_lab(rgb)
        result = delta_e_ciede2000(lab, lab)
        assert result[0].item() == pytest.approx(0.0, abs=0.01)

    def test_different_colors_positive(self):
        lab1 = rgb_to_lab(_make_rgb_tensor((128, 128, 128)))
        lab2 = rgb_to_lab(_make_rgb_tensor((180, 128, 128)))
        result = delta_e_ciede2000(lab1, lab2)
        assert result[0].item() > 0.0

    def test_matches_skimage_ciede2000(self):
        """Cross-validate against skimage deltaE_ciede2000."""
        from skimage.color import rgb2lab, deltaE_ciede2000

        c1 = np.full((64, 64, 3), 100, dtype=np.uint8)
        c2 = np.full((64, 64, 3), 150, dtype=np.uint8)
        lab1_np = rgb2lab(c1)
        lab2_np = rgb2lab(c2)
        expected = deltaE_ciede2000(lab1_np, lab2_np).mean()

        t1 = torch.from_numpy(c1).unsqueeze(0).to(DEVICE)
        t2 = torch.from_numpy(c2).unsqueeze(0).to(DEVICE)
        result = delta_e_ciede2000(rgb_to_lab(t1), rgb_to_lab(t2))
        assert result[0].item() == pytest.approx(float(expected), rel=0.1)

    def test_all_masked_returns_nan(self):
        rgb = _make_rgb_tensor((128, 128, 128))
        lab = rgb_to_lab(rgb)
        mask = torch.ones(1, 64, 64, dtype=torch.bool, device=DEVICE)
        result = delta_e_ciede2000(lab, lab, mask)
        assert torch.isnan(result[0])


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
        assert torch.isnan(result[0, 0, 0])
        assert not torch.isnan(result[0, 32, 0])
