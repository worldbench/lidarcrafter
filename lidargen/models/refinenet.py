from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def _n_tuple(x: Iterable | int, N: int) -> tuple[int]:
    if isinstance(x, Iterable):
        assert len(x) == N
        return x
    else:
        return (x,) * N


class CircularConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, padding_mode="circular")


class InstanceNorm2dPlus(nn.InstanceNorm2d):
    def __init__(self, num_features: int, bias=True) -> None:
        super().__init__(num_features, affine=False, track_running_stats=False)
        self.post_affine = nn.Conv2d(
            num_features, num_features, 1, 1, 0, groups=num_features, bias=bias
        )
        self.alpha = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        self.alpha.data.normal_(1, 0.02)
        self.post_affine.weight.data.normal_(1, 0.02)
        if self.post_affine.bias is not None:
            self.post_affine.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        v, m = torch.var_mean(mean, dim=1, keepdim=True)
        mean = (mean - m) / v.add(1e-5).sqrt()
        h = super().forward(x) * self.alpha * mean
        h = self.post_affine(h)
        return h


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample=None,
        dilation=1,
    ):
        super().__init__()

        kwargs = dict(
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        )

        mid_channels = in_channels if resample == "down" else out_channels
        self.norm1 = InstanceNorm2dPlus(in_channels)
        self.elu1 = nn.ELU()
        self.conv1 = CircularConv2d(in_channels, mid_channels, **kwargs)
        self.norm2 = InstanceNorm2dPlus(mid_channels)
        self.elu2 = nn.ELU()
        self.conv2 = CircularConv2d(mid_channels, out_channels, **kwargs)

        # skip connection
        if in_channels != out_channels or resample is not None:
            self.skip = CircularConv2d(
                in_channels,
                out_channels,
                1 if dilation == 1 else 3,
                1,
                0 if dilation == 1 else dilation,
                1 if dilation == 1 else dilation,
            )
            if dilation == 1 and resample is not None:
                self.conv2 = nn.Sequential(self.conv2, nn.AvgPool2d(2, 2, 0))
                self.skip = nn.Sequential(self.skip, nn.AvgPool2d(2, 2, 0))
        else:
            self.skip = nn.Identity()

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.elu1(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.elu2(h)
        h = self.conv2(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.residual(x)


class ResidualConvUnit(nn.Module):
    def __init__(self, channels: int, num_blocks: int = 2, num_stages: int = 2):
        super().__init__()
        self.units = nn.Sequential()
        for _ in range(num_blocks):
            residual = nn.Sequential()
            for _ in range(num_stages):
                residual.append(nn.ELU())
                residual.append(CircularConv2d(channels, channels, 3, 1, 1, bias=False))
            self.units.append(residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for residual in self.units:
            x = x + residual(x)
        return x


class ChainedResidualPooling(nn.Module):
    def __init__(self, channels: int, num_stages: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_stages):
            self.convs.append(
                nn.Sequential(
                    nn.MaxPool2d(5, 1, 2),
                    CircularConv2d(channels, channels, 3, 1, 1, bias=False),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(x)
        for conv in self.convs:
            h = h + conv(h)
        return h


class RefineBlock(nn.Module):
    def __init__(self, in_channels, out_channels: int, num_end_blocks: int = 1):
        super().__init__()
        assert isinstance(in_channels, Iterable)
        self.adaptive_convs = nn.ModuleList()
        for c in in_channels:
            layers = [ResidualConvUnit(c)]
            if len(in_channels) > 1:
                layers += [CircularConv2d(c, out_channels, 3, 1, 1)]
            self.adaptive_convs.append(nn.Sequential(*layers))
        self.crp = ChainedResidualPooling(out_channels)
        self.output_conv = ResidualConvUnit(out_channels, num_blocks=num_end_blocks)

    def forward(
        self, xs: tuple[torch.Tensor, ...], shape: tuple[int, int]
    ) -> torch.Tensor:
        h = 0
        for rcu, x in zip(self.adaptive_convs, xs):
            h += F.interpolate(rcu(x), size=shape, mode="bilinear", align_corners=True)
        h = self.crp(h)
        h = self.output_conv(h)
        return h


class LiDARGenRefineNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        resolution: tuple[int, int] | int,
        out_channels: int = None,
        base_channels: int = 128,
        channel_multiplier: tuple[int] | int = (1, 2, 2, 2),
        coords_embedding="polar_coordinates",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.resolution = _n_tuple(resolution, 2)
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        assert coords_embedding == "polar_coordinates"
        H, W = self.resolution
        phi = torch.linspace(0, 1, H)
        theta = torch.linspace(0, 1, W)
        [phi, theta] = torch.meshgrid([phi, theta], indexing="ij")
        coords = torch.stack([phi, theta])[None]
        self.register_buffer("coords", coords)
        in_channels += self.coords.shape[1]

        updown_levels = 4
        channel_multiplier = _n_tuple(channel_multiplier, updown_levels)
        C = [base_channels] + [base_channels * m for m in channel_multiplier]

        self.in_conv = nn.Conv2d(in_channels, C[0], 3, 1, 1)
        self.d_block1 = nn.Sequential(
            ResidualBlock(C[0], C[1], resample=None),
            ResidualBlock(C[1], C[1], resample=None),
        )
        self.d_block2 = nn.Sequential(
            ResidualBlock(C[1], C[2], resample="down"),
            ResidualBlock(C[2], C[2], resample=None),
        )
        self.d_block3 = nn.Sequential(
            ResidualBlock(C[2], C[3], resample="down", dilation=2),
            ResidualBlock(C[3], C[3], resample=None, dilation=2),
        )
        self.d_block4 = nn.Sequential(
            ResidualBlock(C[3], C[4], resample="down", dilation=4),
            ResidualBlock(C[4], C[4], resample=None, dilation=4),
        )
        self.u_block4 = RefineBlock([C[4]], C[3])
        self.u_block3 = RefineBlock([C[3], C[3]], C[2])
        self.u_block2 = RefineBlock([C[2], C[2]], C[1])
        self.u_block1 = RefineBlock([C[1], C[1]], C[0], 3)
        self.out_conv = nn.Sequential(
            InstanceNorm2dPlus(C[0]),
            nn.ELU(),
            nn.Conv2d(C[0], self.out_channels, 3, 1, 1),
        )

    def forward(self, images: torch.Tensor, _timesteps: torch.Tensor) -> torch.Tensor:
        h = images

        coords = self.coords.repeat_interleave(h.shape[0], dim=0)
        h = torch.cat([h, coords], dim=1)

        h = self.in_conv(h)
        h1 = self.d_block1(h)
        h2 = self.d_block2(h1)
        h3 = self.d_block3(h2)
        h4 = self.d_block4(h3)
        h = self.u_block4([h4], h4.shape[2:])
        h = self.u_block3([h3, h], h3.shape[2:])
        h = self.u_block2([h2, h], h2.shape[2:])
        h = self.u_block1([h1, h], h1.shape[2:])
        h = self.out_conv(h)

        # score rescaling is omitted

        return h
