# =============================================================================
# This code is based on:
# https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/models/image_transformer_v2.py
#
# Call `natten.use_fused_na(True)` for acceleration if running on GPUs.
# =============================================================================

import math
from typing import List, Literal

import einops
import natten
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn.modules.utils import _pair

from ..unets import encoding, ops

torch.backends.cuda.enable_flash_sdp(True)


# =============================================================================
# Normalization
# =============================================================================


class RMSNorm(torch.nn.Module):
    # from torchtune
    def __init__(self, in_dim: int, scale: bool = True, eps: float = 1e-6):
        super().__init__()
        self.in_dim = in_dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(in_dim)) if scale else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        x_normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_normed * self.scale).to(x)

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}"


class AdaRMSNorm(RMSNorm):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
    ):
        super().__init__(in_dim, scale=False)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, in_dim, bias=False).apply(ops.zero_out),
            Rearrange("B C -> B 1 1 C"),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return super().forward(x) * (1 + self.proj(emb))


# =============================================================================
# Global/local self-attention blocks
# =============================================================================


class AxialRoPE(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_harmonics: List[int]):
        super().__init__()
        freqs_h = self.setup_freqs(num_heads * dim // 4, max_harmonics[0])
        freqs_w = self.setup_freqs(num_heads * dim // 4, max_harmonics[1])
        self.register_buffer("freqs_h", freqs_h.view(dim // 4, num_heads).T)
        self.register_buffer("freqs_w", freqs_w.view(dim // 4, num_heads).T)

    def setup_freqs(self, dim: int, max_harmonics: int):
        # limit the spatial frequency to harmonics
        return torch.linspace(math.log(1), math.log(max_harmonics), dim).exp().round()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = einops.rearrange(coords, "b c h w -> b h w c")
        radian_h = coords[..., None, [0]] * self.freqs_h
        radian_w = coords[..., None, [1]] * self.freqs_w
        return torch.cat((radian_h, radian_w), dim=-1)

    @staticmethod
    def rotate(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 * theta.cos() - x2 * theta.sin()
        y2 = x1 * theta.sin() + x2 * theta.cos()
        return torch.cat((y1, y2), dim=-1)

    def extra_repr(self) -> str:
        return (
            f"freqs_h={tuple(self.freqs_h.shape)}, freqs_w={tuple(self.freqs_w.shape)}"
        )


class GlobalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_max_harmonics: List[int] = (1, 1),
        bias=False,
        eps=1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.eps = eps

        self.norm = AdaRMSNorm(dim, embed_dim)
        self.scale = nn.Parameter(torch.full([self.num_heads, 1], math.log(10.0)))
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.rope = AxialRoPE(self.head_dim, num_heads, rope_max_harmonics)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim, bias=bias).apply(ops.zero_out)

    def scale_qk(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        scale = self.scale.clamp(max=math.log(100)).exp().sqrt()
        q = (F.normalize(q, p=2, dim=-1, eps=self.eps) * scale).to(q.dtype)
        k = (F.normalize(k, p=2, dim=-1, eps=self.eps) * scale).to(k.dtype)
        return q, k

    def apply_rope_qk(self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor):
        theta = self.rope(coords)
        d = theta.shape[-1] * 2
        assert (q.shape[-1] >= d) and (k.shape[-1] >= d)
        q[..., :d] = self.rope.rotate(q[..., :d], theta).to(q.dtype)
        k[..., :d] = self.rope.rotate(k[..., :d], theta).to(k.dtype)
        return q, k

    def residual(
        self, x: torch.Tensor, coords: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        B, H, W, C = x.shape
        h = self.norm(x, emb)
        qkv = self.qkv_proj(h)
        q, k, v = einops.rearrange(
            qkv, "B H W (T N D) -> T B H W N D", T=3, D=self.head_dim
        )
        q, k = self.scale_qk(q, k)
        q, k = self.apply_rope_qk(q, k, coords)
        q = einops.rearrange(q, "B H W N D -> B N (H W) D")
        k = einops.rearrange(k, "B H W N D -> B N (H W) D")
        v = einops.rearrange(v, "B H W N D -> B N (H W) D")
        h = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        h = einops.rearrange(h, "B N (H W) D -> B H W (N D)", H=H, W=W)
        h = self.dropout(h)
        h = self.out_proj(h)
        return h

    def forward(
        self, x: torch.Tensor, coords: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        return x + self.residual(x, coords, emb)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, num_heads={self.num_heads}"


class CircularNeighborhoodSelfAttentionBlock(GlobalSelfAttentionBlock):
    def __init__(
        self,
        dim: int,
        embed_dim: int,
        num_heads: int,
        kernel_size: List[int],
        dilation: List[int] = 1,
        dropout: float = 0.0,
        rope_max_harmonics: List[int] = (1, 1),
    ):
        super().__init__(
            dim=dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            rope_max_harmonics=rope_max_harmonics,
        )
        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)

    def before_attn(self, q, k, v):
        # horizontal circular padding
        padding = self.kernel_size[1] // 2
        q = F.pad(q, (0, 0, 0, 0, padding, padding), mode="circular")
        k = F.pad(k, (0, 0, 0, 0, padding, padding), mode="circular")
        v = F.pad(v, (0, 0, 0, 0, padding, padding), mode="circular")
        return q, k, v

    def after_attn(self, x):
        # crop the padded area
        padding = self.kernel_size[1] // 2
        x = x[:, :, padding:-padding]
        return x

    def residual(
        self, x: torch.Tensor, coords: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        h = self.norm(x, emb)
        qkv = self.qkv_proj(h)
        if natten.context.is_fna_enabled():
            q, k, v = einops.rearrange(
                qkv, "B H W (T N D) -> T B H W N D", T=3, D=self.head_dim
            )
            q, k = self.scale_qk(q, k)
            q, k = self.apply_rope_qk(q, k, coords)
            q, k, v = self.before_attn(q, k, v)
            h = natten.functional.na2d(
                query=q,
                key=k,
                value=v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                scale=1.0,
            )
            h = einops.rearrange(h, "B H W N D -> B H W (N D)")
            h = self.after_attn(h)
        else:
            q, k, v = einops.rearrange(
                qkv, "B H W (T N D) -> T B H W N D", T=3, D=self.head_dim
            )
            q, k = self.scale_qk(q, k)
            q, k = self.apply_rope_qk(q, k, coords)
            q, k, v = self.before_attn(q, k, v)
            q = einops.rearrange(q, "B H W N D -> B N H W D")
            k = einops.rearrange(k, "B H W N D -> B N H W D")
            v = einops.rearrange(v, "B H W N D -> B N H W D")
            qk = natten.functional.na2d_qk(q, k, self.kernel_size)
            a = qk.softmax(dim=-1).to(v.dtype)
            h = natten.functional.na2d_av(a, v, self.kernel_size)
            h = einops.rearrange(h, "B N H W D -> B H W (N D)")
            h = self.after_attn(h)
        h = self.dropout(h)
        h = self.out_proj(h)
        return h

    def forward(
        self, x: torch.Tensor, coords: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        return x + self.residual(x, coords, emb)

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, "
            + f"dilation={self.dilation}, "
        )


# =============================================================================
# Up/down resampling
# =============================================================================


class PatchMerging(nn.Sequential):
    def __init__(self, dim: int):
        super().__init__(
            Rearrange("B (H P1) (W P2) C -> B H W (P1 P2 C)", P1=2, P2=2),
            nn.Linear(4 * dim, 2 * dim, bias=False),
        )


class PatchExpanding(nn.Module):
    # The interpolation coefficient `alpha` is constrained in [0,1] via sigmoid
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2, bias=False)
        self.alpha = nn.Parameter(torch.zeros(dim // 2))

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = self.linear(x)
        x = einops.rearrange(x, "B H W (P1 P2 C) -> B (H P1) (W P2) C", P1=2, P2=2)
        x = torch.lerp(skip, x, self.alpha.sigmoid().to(x))
        return x


class Tokenizer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: List[int],
    ):
        patch_size = _pair(patch_size)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
                bias=False,
            ),
            Rearrange("B C H W -> B H W C"),
        )


class Detokenizer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: List[int],
    ):
        patch_size = _pair(patch_size)
        super().__init__(
            RMSNorm(in_channels),
            nn.Linear(
                in_channels,
                out_channels * patch_size[0] * patch_size[1],
                bias=False,
            ).apply(ops.zero_out),
            Rearrange(
                "B H W (P1 P2 C) -> B C (H P1) (W P2)",
                P1=patch_size[0],
                P2=patch_size[1],
            ),
        )


# =============================================================================
# Main building blocks
# =============================================================================


class GEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features * 2, bias=bias)

    def forward(self, x):
        h = super().forward(x)
        h, gate = h.chunk(2, dim=-1)
        return h * F.gelu(gate)


class FeedForwardNetwork(nn.Module):
    def __init__(self, dim, mid_dim, embed_dim, dropout=0.0):
        super().__init__()
        self.adarms = AdaRMSNorm(dim, embed_dim)
        self.gegelu = GEGLU(dim, mid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(mid_dim, dim, bias=False).apply(ops.zero_out)

    def residual(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.adarms(x, emb)
        x = self.gegelu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x, emb)


class Block(nn.Module):
    def __init__(
        self,
        in_dim: int,
        time_embed_dim: int,
        num_heads: int,
        attn_type: Literal["local", "global"] = "global",
        kernel_size: List[int] = None,
        dilation: List[int] = 1,
        rope_max_harmonics: List[int] = (1, 1),
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if attn_type == "global":
            self.residual_attn = GlobalSelfAttentionBlock(
                dim=in_dim,
                embed_dim=time_embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                rope_max_harmonics=rope_max_harmonics,
            )
        elif attn_type == "local":
            self.residual_attn = CircularNeighborhoodSelfAttentionBlock(
                dim=in_dim,
                embed_dim=time_embed_dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
                rope_max_harmonics=rope_max_harmonics,
            )

        self.residual_ffn = FeedForwardNetwork(
            dim=in_dim,
            mid_dim=int(in_dim * mlp_ratio),
            embed_dim=time_embed_dim,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, coords: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        x = self.residual_attn(x, coords, emb)
        x = self.residual_ffn(x, emb)
        return x


class RandomFourierFeatures(nn.Module):
    def __init__(self, dim, std=1.0):
        super().__init__()
        self.register_buffer("freqs", torch.randn(dim // 2) * std)
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, timestep):
        h = timestep.ger(2 * torch.pi * self.freqs)
        h = torch.cat([h.cos(), h.sin()], dim=1)
        h = self.linear(h)
        return h


class MappingFeedForwardNetwork(nn.Module):
    def __init__(self, dim, mid_dim, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.gegelu = GEGLU(dim, mid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(mid_dim, dim, bias=False).apply(ops.zero_out)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.gegelu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)


class MappingNetwork(nn.Sequential):
    def __init__(self, dim, mid_dim, depth=1, dropout=0.0):
        super().__init__(
            RMSNorm(dim),
            *[MappingFeedForwardNetwork(dim, mid_dim, dropout) for _ in range(depth)],
            RMSNorm(dim),
        )


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, method: str, out_dim: int, resolution: List[int]):
        super().__init__()
        self.resolution = resolution
        if method == "spherical_harmonics":
            self.embedding = encoding.SphericalHarmonics(levels=5)
            emb_dim = self.embedding.extra_ch
        elif method == "fourier_features":
            self.embedding = encoding.FourierFeatures(self.resolution)
            emb_dim = self.embedding.extra_ch
        elif method == "polar_coordinates":
            self.embedding = nn.Identity()
            emb_dim = 2
        else:
            raise ValueError(f"Unknown positional embedding method: {method}")
        self.linear = nn.Linear(emb_dim, out_dim, bias=False)

    def forward(self, coords: torch.Tensor):
        h = self.embedding(coords)
        h = einops.rearrange(h, "B C H W -> B H W C")
        h = self.linear(h)
        return h


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, out_dim: int, resolution: List[int]):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, *resolution, out_dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, coords: torch.Tensor):
        h = self.embedding
        return h

    def extra_repr(self):
        return f"resolution={tuple(self.embedding.shape[1:3])}, out_dim={self.embedding.shape[3]}"


class HDiT(nn.Module):
    def __init__(
        self,
        resolution: List[int],
        in_channels: int,
        out_channels: int | None = None,
        base_channels: int = 128,
        time_embed_channels: int = 256,
        patch_size: List[int] = (1, 4),
        window_size: List[int] = (3, 9),  # must be odd
        depths: List[int] = (2, 2, 2, 2),
        num_heads: List[int] = (2, 4, 8, 16),
        dilation: List[int] = (1, 1, 1, 1),  # dinat config
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        mapping_depth: int = 2,
        positional_embedding: Literal[
            "spherical_harmonics",
            "polar_coordinates",
            "fourier_features",
            "learnable_embedding",
            None,
        ] = "learnable_embedding",
        ring: bool = True,
    ):
        super().__init__()
        self.resolution = _pair(resolution)
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        self.patch_size = _pair(patch_size)
        self.depths = depths

        coords = encoding.generate_polar_coords(*self.resolution)
        self.register_buffer("coords", coords)

        token_size = torch.tensor(self.resolution) // torch.tensor(self.patch_size)

        if positional_embedding == "learnable_embedding":
            self.spatial_pe = LearnablePositionalEmbedding(
                out_dim=base_channels,
                resolution=token_size.tolist(),
            )
        else:
            self.spatial_pe = nn.Sequential(
                AbsolutePositionalEmbedding(
                    method=positional_embedding,
                    out_dim=base_channels,
                    resolution=token_size.tolist(),
                ),
                MappingNetwork(
                    base_channels,
                    int(base_channels * mlp_ratio),
                    depth=mapping_depth,
                ),
            )

        self.timestep_pe = nn.Sequential(
            RandomFourierFeatures(time_embed_channels),
            MappingNetwork(
                time_embed_channels,
                int(time_embed_channels * mlp_ratio),
                depth=mapping_depth,
            ),
        )

        self.tokenizer = Tokenizer(
            in_channels=in_channels,
            out_channels=base_channels,
            patch_size=patch_size,
        )

        max_harmonics = (token_size / 2).int()

        self.down_levels = nn.ModuleDict()
        self.up_levels = nn.ModuleDict()
        for i, num_blocks in enumerate(depths[:-1]):
            down_blocks = nn.ModuleList()
            up_blocks = nn.ModuleList()
            for j in range(num_blocks):
                down_blocks.append(
                    Block(
                        in_dim=base_channels << i,
                        time_embed_dim=time_embed_channels,
                        num_heads=num_heads[i],
                        attn_type="local",
                        kernel_size=window_size,
                        dilation=1 if j % 2 == 0 else dilation[i],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                    )
                )
                up_blocks.append(
                    Block(
                        in_dim=base_channels << i,
                        time_embed_dim=time_embed_channels,
                        num_heads=num_heads[i],
                        attn_type="local",
                        kernel_size=window_size,
                        dilation=1 if j % 2 == 0 else dilation[i],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                    )
                )
            self.down_levels[f"level_{i}"] = down_blocks
            self.down_levels[f"merge_{i}"] = PatchMerging(base_channels << i)
            self.up_levels[f"level_{i}"] = up_blocks
            self.up_levels[f"expand_{i}"] = PatchExpanding(base_channels << (i + 1))

        self.mid_levels = nn.ModuleList()
        i = len(depths) - 1
        for j in range(depths[-1]):
            self.mid_levels.append(
                Block(
                    in_dim=base_channels << i,
                    time_embed_dim=time_embed_channels,
                    num_heads=num_heads[-1],
                    attn_type="global",
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                )
            )

        self.detokenizer = Detokenizer(
            in_channels=base_channels,
            out_channels=self.out_channels,
            patch_size=patch_size,
        )

        self.nfe = 0

    def downsample_coords(self, coords, kernel_size):
        return F.avg_pool2d(coords, kernel_size=kernel_size, stride=kernel_size)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        if len(t.shape) == 0:
            t = t[None].repeat_interleave(x.shape[0], dim=0)
        emb = self.timestep_pe(t.to(x))

        c = self.downsample_coords(self.coords, self.patch_size).to(x)
        h = self.tokenizer(x) + self.spatial_pe(c)

        stack = []
        for i in range(len(self.depths) - 1):
            for block in self.down_levels[f"level_{i}"]:
                h = block(h, c, emb)
            stack.append((h, c))
            c = self.downsample_coords(c, 2)
            h = self.down_levels[f"merge_{i}"](h)

        for block in self.mid_levels:
            h = block(h, c, emb)

        for i in reversed(range(len(self.depths) - 1)):
            h_skip, c = stack.pop()  # lifo
            h = self.up_levels[f"expand_{i}"](h, h_skip.to(h))
            for block in self.up_levels[f"level_{i}"]:
                h = block(h, c, emb)

        h = self.detokenizer(h)
        self.nfe += 1

        return h
