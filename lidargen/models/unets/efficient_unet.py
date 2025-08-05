# =============================================================================
# This code is from:
# https://github.com/kazuto1011/r2dm/blob/main/models/efficient_unet.py
# =============================================================================

from typing import Iterable, Literal

import einops
import numpy as np
import torch
from torch import nn

from . import encoding, ops


def _join(*tensors) -> torch.Tensor:
    return torch.cat(tensors, dim=1)


def _n_tuple(x: Iterable | int, N: int) -> tuple[int]:
    if isinstance(x, Iterable):
        assert len(x) == N
        return x
    else:
        return (x,) * N


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        gn_eps: float = 1e-6,
        gn_num_groups: int = 8,
        scale: float = 1 / np.sqrt(2),
    ):
        super().__init__()
        self.norm = nn.GroupNorm(gn_num_groups, in_channels, gn_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn.out_proj.apply(ops.zero_out)
        self.register_buffer("scale", torch.tensor(scale).float())

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        B, C, H, W = h.shape
        h = einops.rearrange(h, "B C H W -> B (H W) C")
        h, _ = self.attn(query=h, key=h, value=h, need_weights=False)
        h = einops.rearrange(h, "B (H W) C -> B C H W", H=H, W=W)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.residual(x)
        h = h * self.scale
        return h


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int | None,
        gn_num_groups: int = 8,
        gn_eps: float = 1e-6,
        scale: float = 1 / np.sqrt(2),
        dropout: float = 0.0,
        ring: bool = False,
    ):
        super().__init__()
        self.has_emb = emb_channels is not None

        # layer 1
        self.norm1 = nn.GroupNorm(gn_num_groups, in_channels, gn_eps)
        self.silu1 = nn.SiLU()
        self.conv1 = ops.Conv2d(in_channels, out_channels, 3, 1, 1, ring=ring)

        # layer 2
        if self.has_emb:
            self.norm2 = ops.AdaGN(emb_channels, out_channels, gn_num_groups, gn_eps)
        else:
            self.norm2 = nn.GroupNorm(gn_num_groups, out_channels, gn_eps)
        self.silu2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)
        self.conv2 = ops.Conv2d(out_channels, out_channels, 3, 1, 1, ring=ring)
        self.conv2.apply(ops.zero_out)

        # skip connection
        self.skip = (
            ops.Conv2d(in_channels, out_channels, 1, 1, 0)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.register_buffer("scale", torch.tensor(scale).float())

    def residual(
        self, x: torch.Tensor, emb: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.silu1(h)
        h = self.conv1(h)
        h = self.norm2(h, emb) if self.has_emb else self.norm2(h)
        h = self.silu2(h)
        h = self.drop2(h)
        h = self.conv2(h)
        return h

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.skip(x) + self.residual(x, emb)
        h = h * self.scale
        return h


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_residual_blocks: int,
        emb_channels: int,
        gn_num_groups: int = 8,
        gn_eps: float = 1e-6,
        attn: bool = False,
        attn_num_heads: int = 8,
        up: int = 1,
        down: int = 1,
        dropout: float = 0.0,
        ring: bool = False,
    ):
        super().__init__()

        # downsampling
        self.downsample = (
            nn.Sequential(
                ops.Conv2d(in_channels, out_channels, 3, 1, 1, ring=ring),
                ops.Resample(down=down, ring=ring),
            )
            if down > 1
            else nn.Identity()
        )

        # resnet blocks x N
        self.residual_blocks = ops.ConditionalSequential()
        for i in range(num_residual_blocks):
            self.residual_blocks.append(
                ResidualBlock(
                    in_channels=out_channels if i != 0 or down > 1 else in_channels,
                    out_channels=out_channels,
                    emb_channels=emb_channels,
                    gn_num_groups=gn_num_groups,
                    gn_eps=gn_eps,
                    dropout=dropout,
                    ring=ring,
                )
            )

        # self-attention
        self.self_attn_block = (
            SelfAttentionBlock(
                in_channels=out_channels,
                num_heads=attn_num_heads,
                gn_eps=gn_eps,
                gn_num_groups=gn_num_groups,
            )
            if attn
            else nn.Identity()
        )

        # upsampling
        self.upsample = (
            nn.Sequential(
                ops.Resample(up=up, ring=ring),
                ops.Conv2d(out_channels, out_channels, 3, 1, 1, ring=ring),
            )
            if up > 1
            else nn.Identity()
        )

    def forward(
        self, h: torch.Tensor, temb: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.downsample(h)
        h = self.residual_blocks(h, temb)
        h = self.self_attn_block(h)
        h = self.upsample(h)
        return h


class EfficientUNet(nn.Module):
    """
    Re-implementation of Efficient U-Net (https://arxiv.org/abs/2205.11487)
    + Our modification for LiDAR domain
    """

    def __init__(
        self,
        in_channels: int,
        resolution: tuple[int, int] | int,
        out_channels: int | None = None,  # == in_channels if None
        base_channels: int = 128,
        temb_channels: int = None,
        channel_multiplier: tuple[int] | int = (1, 2, 4, 8),
        num_residual_blocks: tuple[int] | int = (3, 3, 3, 3),
        gn_num_groups: int = 32 // 4,
        gn_eps: float = 1e-6,
        attn_num_heads: int = 8,
        coords_encoding: Literal[
            "spherical_harmonics", "polar_coordinates", "fourier_features", None
        ] = "spherical_harmonics",
        ring: bool = True,
    ):
        super().__init__()
        self.resolution = _n_tuple(resolution, 2)
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        temb_channels = base_channels * 4 if temb_channels is None else temb_channels

        # spatial coords embedding
        coords = encoding.generate_polar_coords(*self.resolution)
        self.register_buffer("coords", coords)
        self.coords_encoding = None
        if coords_encoding == "spherical_harmonics":
            self.coords_encoding = encoding.SphericalHarmonics(levels=5)
            in_channels += self.coords_encoding.extra_ch
        elif coords_encoding == "polar_coordinates":
            self.coords_encoding = nn.Identity()
            in_channels += coords.shape[1]
        elif coords_encoding == "fourier_features":
            self.coords_encoding = encoding.FourierFeatures(self.resolution)
            in_channels += self.coords_encoding.extra_ch

        # timestep embedding
        self.time_embedding = nn.Sequential(
            ops.SinusoidalPositionalEmbedding(base_channels),
            nn.Linear(base_channels, temb_channels),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels),
        )

        # parameters for up/down-sampling blocks
        updown_levels = 4
        channel_multiplier = _n_tuple(channel_multiplier, updown_levels)
        C = [base_channels] + [base_channels * m for m in channel_multiplier]
        N = _n_tuple(num_residual_blocks, updown_levels)

        cfgs = dict(
            emb_channels=temb_channels,
            gn_num_groups=gn_num_groups,
            gn_eps=gn_eps,
            attn_num_heads=attn_num_heads,
            dropout=0.0,
            ring=ring,
        )

        # downsampling blocks
        self.in_conv = ops.Conv2d(in_channels, C[0], 3, 1, 1, ring=ring)
        self.d_block1 = Block(C[0], C[1], N[0], **cfgs)
        self.d_block2 = Block(C[1], C[2], N[1], down=2, **cfgs)
        self.d_block3 = Block(C[2], C[3], N[2], down=2, **cfgs)
        self.d_block4 = Block(C[3], C[4], N[3], down=2, attn=True, **cfgs)

        # upsampling blocks
        self.u_block4 = Block(C[4], C[3], N[3], up=2, attn=True, **cfgs)
        self.u_block3 = Block(C[3] + C[3], C[2], N[2], up=2, **cfgs)
        self.u_block2 = Block(C[2] + C[2], C[1], N[1], up=2, **cfgs)
        self.u_block1 = Block(C[1] + C[1], C[0], N[0], **cfgs)
        self.out_conv = ops.Conv2d(C[0], self.out_channels, 3, 1, 1, ring=ring)
        self.out_conv.apply(ops.zero_out)

    def forward(self, images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        h = images

        # timestep embedding
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].repeat_interleave(h.shape[0], dim=0)
        temb = self.time_embedding(timesteps.to(h))

        # spatial embedding
        if self.coords_encoding is not None:
            cenc = self.coords_encoding(self.coords)
            cenc = cenc.repeat_interleave(h.shape[0], dim=0)
            h = torch.cat([h, cenc], dim=1)

        # u-net part
        h = self.in_conv(h)
        h1 = self.d_block1(h, temb)
        h2 = self.d_block2(h1, temb)
        h3 = self.d_block3(h2, temb)
        h4 = self.d_block4(h3, temb)
        h = self.u_block4(h4, temb)
        h = self.u_block3(_join(h, h3), temb)
        h = self.u_block2(_join(h, h2), temb)
        h = self.u_block1(_join(h, h1), temb)
        h = self.out_conv(h)

        return h
