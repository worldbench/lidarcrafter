import torch

# convert dict val to cuda
def to_device(data_dict, device):
    for key in data_dict:
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].to(device)
        elif isinstance(data_dict[key], dict):
            data_dict[key] = to_device(data_dict[key], device)
    return data_dict