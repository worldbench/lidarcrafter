import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# components_from_spherical_harmonics() is from nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/utils/math.py


def components_from_spherical_harmonics(levels: int, directions):
    """
    Returns value for each component of spherical harmonics.

    Args:
        levels: Number of spherical harmonic levels to compute.
        directions: Spherical harmonic coefficients
    """
    num_components = levels**2
    components = torch.zeros(
        (*directions.shape[:-1], num_components), device=directions.device
    )

    # assert 1 <= levels <= 5, f"SH levels must be in [1,4], got {levels}"
    assert (
        directions.shape[-1] == 3
    ), f"Direction input should have three dimensions. Got {directions.shape[-1]}"

    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    xx = x**2
    yy = y**2
    zz = z**2

    # l0
    components[..., 0] = 0.28209479177387814

    # l1
    if levels > 1:
        components[..., 1] = 0.4886025119029199 * y
        components[..., 2] = 0.4886025119029199 * z
        components[..., 3] = 0.4886025119029199 * x

    # l2
    if levels > 2:
        components[..., 4] = 1.0925484305920792 * x * y
        components[..., 5] = 1.0925484305920792 * y * z
        components[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
        components[..., 7] = 1.0925484305920792 * x * z
        components[..., 8] = 0.5462742152960396 * (xx - yy)

    # l3
    if levels > 3:
        components[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
        components[..., 10] = 2.890611442640554 * x * y * z
        components[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
        components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
        components[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
        components[..., 14] = 1.445305721320277 * z * (xx - yy)
        components[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

    # l4
    if levels > 4:
        components[..., 16] = 2.5033429417967046 * x * y * (xx - yy)
        components[..., 17] = 1.7701307697799304 * y * z * (3 * xx - yy)
        components[..., 18] = 0.9461746957575601 * x * y * (7 * zz - 1)
        components[..., 19] = 0.6690465435572892 * y * z * (7 * zz - 3)
        components[..., 20] = 0.10578554691520431 * (35 * zz * zz - 30 * zz + 3)
        components[..., 21] = 0.6690465435572892 * x * z * (7 * zz - 3)
        components[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
        components[..., 23] = 1.7701307697799304 * x * z * (xx - 3 * yy)
        components[..., 24] = 0.6258357354491761 * (
            xx * (xx - 3 * yy) - yy * (3 * xx - yy)
        )

    return components


def generate_polar_coords(H: int, W: int, device: torch.device = "cpu") -> torch.Tensor:
    """
    theta: azimuthal angle in [-pi, pi]
    phi: polar angle in [0, pi]
    """
    phi = (0.5 - torch.arange(H, device=device) / H) * torch.pi
    theta = (1 - torch.arange(W, device=device) / W) * 2 * torch.pi - torch.pi
    [phi, theta] = torch.meshgrid([phi, theta], indexing="ij")
    angles = torch.stack([phi, theta])
    return angles[None]


class SphericalHarmonics(nn.Module):
    def __init__(self, levels: int = 4) -> None:
        super().__init__()
        self.levels = levels
        self.extra_ch = levels**2

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        B, _, H, W = coords.shape
        coords = coords.repeat_interleave(B, dim=0)
        phi, theta = coords[:, 0], coords[:, 1]
        directions = torch.stack(
            [
                torch.cos(theta) * torch.cos(phi),
                -torch.sin(theta) * torch.cos(phi),
                torch.sin(phi),
            ],
            dim=-1,
        )
        basis = components_from_spherical_harmonics(
            levels=self.levels, directions=directions
        )
        basis = basis.permute(0, 3, 1, 2).repeat_interleave(B, dim=0)
        return basis

    def extra_repr(self):
        return f"levels={self.levels}"


class FourierFeatures(nn.Module):
    def __init__(
        self,
        resolution,
    ):
        super().__init__()
        self.resolution = resolution

        self.L_h = int(np.ceil(np.log2(self.resolution[0])))
        self.L_w = int(np.ceil(np.log2(self.resolution[1])))

        freqs_h = torch.arange(self.L_h).exp2()
        freqs_h = torch.cat([freqs_h, torch.zeros(self.L_w)])
        freqs_w = torch.arange(self.L_w).exp2()
        freqs_w = torch.cat([torch.zeros(self.L_h), freqs_w])
        freqs = torch.stack([freqs_h, freqs_w], dim=-1)
        phase = torch.zeros(len(freqs_h))
        self.register_buffer("freqs", freqs[..., None, None])
        self.register_buffer("phase", phase)
        self.extra_ch = int(len(freqs_h) * 2)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        B, _, H, W = coords.shape
        coords = coords.repeat_interleave(B, dim=0)
        coords = F.conv2d(coords, weight=self.freqs, bias=self.phase)
        encoded = torch.cat([coords.sin(), coords.cos()], dim=1)
        return encoded

    def extra_repr(self):
        return f"shape={self.resolution}, num_freqs={self.extra_ch}, L=({self.L_h}, {self.L_w})"
