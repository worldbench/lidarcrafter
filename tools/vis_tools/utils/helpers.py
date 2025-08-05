import torch
import numpy as np


def postprocess_sincos2arctan(sincos):
    if isinstance(sincos, np.ndarray):
        assert sincos.shape[1] == 2
        return np.arctan2(sincos[0],sincos[1])
    elif isinstance(sincos,torch.Tensor):
        B, N = sincos.shape
        assert N == 2
        return torch.arctan2(sincos[:,0], sincos[:,1]).reshape(B,1)
    else:
        raise NotImplementedError