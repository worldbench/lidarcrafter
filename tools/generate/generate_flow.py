import argparse
from pathlib import Path

import einops
import imageio
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

import sys
sys.path.append("../")

from lidargen.utils import inference
from lidargen.utils import render
from lidargen.utils import lidar

def main(args):
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    # =================================================================================
    # Load pre-trained model
    # =================================================================================

    flow, lidar_utils, _ = inference.setup_model_flow(args.cfg, args.ckpt, device=args.device)

    # =================================================================================
    # Sampling (reverse diffusion)
    # =================================================================================

    xs = flow.sample().clamp(-1, 1)

    # =================================================================================
    # Save as image or video
    # =================================================================================

    xs = lidar_utils.denormalize(xs)
    xs[:, [0]] = lidar_utils.revert_depth(xs[:, [0]]) / lidar_utils.max_depth

    def render_image(x, save_xyz=False):
        img = einops.rearrange(x, "B C H W -> B 1 (C H) W")
        img = render.colorize(img) / 255
        xyz = lidar_utils.to_xyz(x[:, [0]] * lidar_utils.max_depth)
        if save_xyz:
            lidar.save_points(xyz[0].reshape(3,-1).permute(1,0), "sample_points.txt")

        xyz /= lidar_utils.max_depth
        z_min, z_max = -2 / lidar_utils.max_depth, 0.5 / lidar_utils.max_depth
        z = (xyz[:, [2]] - z_min) / (z_max - z_min)
        colors = render.colorize(z.clamp(0, 1), cm.viridis) / 255
        R, t = render.make_Rt(pitch=torch.pi / 3, yaw=torch.pi / 4, z=0.8)
        bev = 1 - render.render_point_clouds(
            points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
            colors=1 - einops.rearrange(colors, "B C H W -> B (H W) C"),
            R=R.to(xyz),
            t=t.to(xyz),
        )
        return img, bev

    img, bev = render_image(xs, save_xyz=True)
    save_image(img, "samples_img.png", nrow=1)
    save_image(bev, "samples_bev.png", nrow=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--mode", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sampling_steps", type=int, default=1024)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="resume training from ckpt",
    )
    args = parser.parse_args()
    args.device = torch.device(args.device)
    main(args)
