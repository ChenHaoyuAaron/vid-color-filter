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
        assert len(kernels) == 3

    def test_kernels_sum_to_one(self):
        from vid_color_filter.gpu.scielab import build_csf_kernels
        kernels = build_csf_kernels(pixels_per_degree=60, device=DEVICE)
        for k in kernels:
            assert k.sum().item() == pytest.approx(1.0, abs=0.01)

    def test_kernel_shapes_are_1d_and_odd(self):
        from vid_color_filter.gpu.scielab import build_csf_kernels
        kernels = build_csf_kernels(pixels_per_degree=60, device=DEVICE)
        for k in kernels:
            assert k.dim() == 1
            assert k.shape[0] % 2 == 1

    def test_achromatic_kernel_largest(self):
        from vid_color_filter.gpu.scielab import build_csf_kernels
        kernels = build_csf_kernels(pixels_per_degree=60, device=DEVICE)
        assert kernels[0].shape[0] >= kernels[1].shape[0]
        assert kernels[0].shape[0] >= kernels[2].shape[0]

    def test_kernel_size_capped(self):
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
        from vid_color_filter.gpu.scielab import scielab_filter
        from vid_color_filter.gpu.color_space import rgb_to_lab
        rgb = _make_rgb_tensor((128, 128, 128))
        lab_direct = rgb_to_lab(rgb)
        lab_scielab = scielab_filter(rgb, pixels_per_degree=60)
        center = slice(16, 48)
        torch.testing.assert_close(
            lab_scielab[0, center, center],
            lab_direct[0, center, center],
            atol=1.0, rtol=0.05,
        )

    def test_noise_is_suppressed(self):
        from vid_color_filter.gpu.scielab import scielab_filter
        from vid_color_filter.gpu.color_space import rgb_to_lab
        from vid_color_filter.gpu.color_metrics import delta_e_ciede2000
        base = torch.full((1, 64, 64, 3), 128, dtype=torch.uint8, device=DEVICE)
        noise = base.clone()
        torch.manual_seed(42)
        noise = (noise.float() + torch.randn(1, 64, 64, 3, device=DEVICE) * 5).clamp(0, 255).to(torch.uint8)
        de_direct = delta_e_ciede2000(rgb_to_lab(base), rgb_to_lab(noise))
        de_filtered = delta_e_ciede2000(
            scielab_filter(base, pixels_per_degree=60),
            scielab_filter(noise, pixels_per_degree=60),
        )
        assert de_filtered[0].item() < de_direct[0].item()
