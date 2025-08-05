import torch
import torch.nn as nn
from functools import partial


class Identity(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, input_dict):
        return input_dict['cond']

class SpatialRescaler(nn.Module):
    def __init__(self,
                 strides=[],
                 method='bilinear',
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.strides = strides
        assert method in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area']
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method, align_corners=True)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, data_dict):
        x = data_dict['cond']
        for h_s, w_s in self.strides:
            x = self.interpolator(x, scale_factor=(1/h_s, 1/w_s))

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)