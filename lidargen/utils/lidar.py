from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def get_hdl64e_linear_ray_angles(
    H: int = 64, W: int = 2048, device: torch.device = "cpu"
):
    h_up, h_down = 3, -25
    w_left, w_right = 180, -180
    elevation = 1 - torch.arange(H, device=device) / H  # [0, 1]
    elevation = elevation * (h_up - h_down) + h_down  # [-25, 3]
    azimuth = 1 - torch.arange(W, device=device) / W  # [0, 1]
    azimuth = azimuth * (w_left - w_right) + w_right  # [-180, 180]
    [elevation, azimuth] = torch.meshgrid([elevation, azimuth], indexing="ij")
    angles = torch.stack([elevation, azimuth])[None].deg2rad()
    return angles

def get_linear_ray_angles(
    H: int = 64, W: int = 2048, fov_up=10, fov_down=-30, device: torch.device = "cpu"
):
    w_left, w_right = 180, -180
    elevation = 1 - torch.arange(H, device=device) / H  # [0, 1]
    elevation = elevation * (fov_up - fov_down) + fov_down  # [-25, 3]
    azimuth = 1 - torch.arange(W, device=device) / W  # [0, 1]
    azimuth = azimuth * (w_left - w_right) + w_right  # [-180, 180]
    [elevation, azimuth] = torch.meshgrid([elevation, azimuth], indexing="ij")
    angles = torch.stack([elevation, azimuth])[None].deg2rad()
    return angles

class LiDARUtility(nn.Module):
    def __init__(
        self,
        resolution: tuple[int],
        depth_format: Literal["log_depth", "inverse_depth", "depth"],
        min_depth: float,
        max_depth: float,
        ray_angles: torch.Tensor = None,
    ):
        super().__init__()
        assert depth_format in ("log_depth", "inverse_depth", "depth")
        self.resolution = resolution
        self.depth_format = depth_format
        self.min_depth = min_depth
        self.max_depth = max_depth
        if ray_angles is None:
            # ray_angles = get_hdl64e_linear_ray_angles(*resolution)
            raise NotImplementedError
        else:
            assert ray_angles.ndim == 4 and ray_angles.shape[1] == 2
        ray_angles = F.interpolate(
            ray_angles,
            size=self.resolution,
            mode="nearest-exact",
        )
        self.register_buffer("ray_angles", ray_angles.float())

    @staticmethod
    def denormalize(x: torch.Tensor) -> torch.Tensor:
        """Scale from [-1, +1] to [0, 1]"""
        return (x + 1) / 2

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        """Scale from [0, 1] to [-1, +1]"""
        return x * 2 - 1

    @torch.no_grad()
    def to_xyz(self, metric: torch.Tensor) -> torch.Tensor:
        assert metric.dim() == 4
        mask = (metric > self.min_depth) & (metric < self.max_depth)
        phi = self.ray_angles[:, [0]]
        theta = self.ray_angles[:, [1]]
        grid_x = metric * phi.cos() * theta.cos()
        grid_y = metric * phi.cos() * theta.sin()
        grid_z = metric * phi.sin()
        xyz = torch.cat((grid_x, grid_y, grid_z), dim=1)
        xyz = xyz * mask.float()
        return xyz

    @torch.no_grad()
    def convert_depth(
        self,
        metric: torch.Tensor,
        mask: torch.Tensor | None = None,
        depth_format: str = None,
    ) -> torch.Tensor:
        """
        Convert metric depth in [0, `max_depth`] to normalized depth in [0, 1].
        """
        if depth_format is None:
            depth_format = self.depth_format
        if mask is None:
            mask = self.get_mask(metric)
        if depth_format == "log_depth":
            normalized = torch.log2(metric + 1) / np.log2(self.max_depth + 1)
        elif depth_format == "inverse_depth":
            normalized = self.min_depth / metric.add(1e-8)
        elif depth_format == "depth":
            normalized = metric.div(self.max_depth)
        else:
            raise ValueError
        normalized = normalized.clamp(0, 1) * mask
        return normalized

    @torch.no_grad()
    def revert_depth(
        self,
        normalized: torch.Tensor,
        image_format: str = None,
    ) -> torch.Tensor:
        """
        Revert normalized depth in [0, 1] back to metric depth in [0, `max_depth`].
        """
        if image_format is None:
            image_format = self.depth_format
        if image_format == "log_depth":
            metric = torch.exp2(normalized * np.log2(self.max_depth + 1)) - 1
        elif image_format == "inverse_depth":
            metric = self.min_depth / normalized.add(1e-8)
        elif image_format == "depth":
            metric = normalized.mul(self.max_depth)
        else:
            raise ValueError
        return metric * self.get_mask(metric)

    def get_mask(self, metric):
        mask = (metric > self.min_depth) & (metric < self.max_depth)
        return mask.float()
    

@torch.no_grad()
def save_points(
    tensor, fp):
    
    ndarray = tensor.detach().cpu().numpy()
    np.savetxt(fp, ndarray)
