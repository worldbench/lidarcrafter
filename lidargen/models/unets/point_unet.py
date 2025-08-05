import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Literal

def _n_tuple(x: Iterable | int, N: int) -> tuple[int]:
    if isinstance(x, Iterable):
        assert len(x) == N
        return x
    else:
        return (x,) * N

# Point Condition Network 
class PCNet(nn.Module):
    def __init__(self, dim_in, dim_out, dim_cond):
        super(PCNet, self).__init__()
        self.fea_layer = nn.Linear(dim_in, dim_out)
        self.cond_bias = nn.Linear(dim_cond, dim_out, bias=False)
        self.cond_gate = nn.Linear(dim_cond, dim_out)

    def forward(self, fea, cond):
        gate = torch.sigmoid(self.cond_gate(cond))
        bias = self.cond_bias(cond)
        out = self.fea_layer(fea) * gate + bias
        return out

# Point Denoising Network
class PointUNet(nn.Module):

    def __init__(self, point_dim, cond_dims, residual=True):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = nn.ModuleList([
            PCNet(point_dim, 128, cond_dims+3),
            PCNet(128, 256, cond_dims+3),
            PCNet(256, 512, cond_dims+3),
            PCNet(512, 256, cond_dims+3),
            PCNet(256, 128, cond_dims+3),
            PCNet(128, point_dim, cond_dims+3)
        ])
        self.resolution = _n_tuple(1024, 1)
        self.in_channels = point_dim


    def forward(self, coords, cond_dict):
        """
        Args:
            coords:   Noise point clouds at timestep t, (B, N, 3).
            beta:     Time. (B, ).
            cond:     Condition. (B, F).
        """

        batch_size = coords.size(0)
        beta = cond_dict['time_condition']
        cond = cond_dict['other_condition']
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        cond = cond.view(batch_size, 1, -1)         # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        cond_emb = torch.cat([time_emb, cond], dim=-1)    # (B, 1, F+3)
        
        out = coords
        for i, layer in enumerate(self.layers):
            out = layer(fea=out, cond=cond_emb)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return coords + out
        else:
            return out