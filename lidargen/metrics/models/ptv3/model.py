import os
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from pointcept.engines.defaults import (
        default_argument_parser,
        default_config_parser)
    from pointcept.models import build_model
    import pointcept.utils.comm as comm
    from pointcept.datasets import build_dataset, collate_fn

except:
    pass

from collections import OrderedDict
from loguru import logger

class PTv3(nn.Module):
    def __init__(self):
        super(PTv3, self).__init__()
        args = default_argument_parser().parse_args()
        args.config_file = '../lidargen/metrics/models/ptv3/config/semseg-pt-v3m1-0-base.py'
        self.cfg = default_config_parser(args.config_file, args.options)
        self.logger = logger
        self.model = self.build_model()
        self.model.cuda()
        self.model.eval()
        self.dataset = self.build_dataset()

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")

        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, weights_only=False)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model
    
    def build_dataset(self):
        test_dataset = build_dataset(self.cfg.data.val)
        return test_dataset

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)

    def inference_one_sample(self, points):
        data_dict = dict(
            coord=points[:, :3],
            strength=points[:, 3].reshape(-1, 1))
        data_dict = self.dataset.getitem_from_outline(data_dict)
        input_dict = collate_fn([data_dict])
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.no_grad():
            pred = self.model(input_dict)["seg_logits"]  # (n, k)
            pred = pred[data_dict["inverse"]]
            pred = F.softmax(pred, -1)
            if self.cfg.empty_cache:
                torch.cuda.empty_cache()
        pred = pred.max(1)[1].data.cpu().numpy()
        colors = self.dataset.colormap[pred]
        return pred, colors
    
if __name__ == "__main__":
    ptv3_model = PTv3()
    print(ptv3_model)  # For testing purposes, prints the model structure
