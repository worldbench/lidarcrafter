import argparse
import torch

import sys
sys.path.append("../")

from lidargen.utils import inference
from lidargen.utils import common
from lidargen.utils.configs import __all__
from lidargen.dataset import __all__ as all_datasets

def main(args):
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    cfg = __all__[args.cfg]()

    # =================================================================================
    # Setup dataset
    # =================================================================================

    cfg.data.split = 'val'
    dataset = all_datasets[cfg.data.dataset](cfg.data)
    cfg.condition_model.params['vocab'] = dataset.scene_graph_assigner.vocab

    # =================================================================================
    # Load pre-trained model
    # =================================================================================
    
    cfg.resume = args.resume
    ddpm, _, _, _, _ = inference.load_model_layout_duffusion_training(cfg)
    ddpm.eval()
    ddpm.to(args.device)

    batch_dict = dataset.__getitem__(args.sample_idx)
    collate_fn = dataset.collate_fn if getattr(cfg.data, 'custom_collate_fn', False) else None
    if collate_fn is not None:
        batch_dict = collate_fn([batch_dict])
    batch_dict = common.to_device(batch_dict, args.device)

    # =================================================================================
    # Sampling (reverse diffusion)
    # =================================================================================

    xs = ddpm.sample(
        batch_dict = batch_dict,
        num_steps=args.sampling_steps,
        mode=args.mode,
        return_all=False,
    )

    # =================================================================================
    # Save as image or video
    # =================================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="resume training from ckpt",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--sample_idx", type=int, default=1)
    parser.add_argument("--mode", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--sampling_steps", type=int, default=256)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    main(args)
