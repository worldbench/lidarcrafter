import argparse
from pathlib import Path

import einops
import imageio
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image, save_points
from tqdm.auto import tqdm

import sys
sys.path.append("../")
from lidargen.utils.configs import __all__
from lidargen.dataset import __all__ as all_datasets
from lidargen.utils import inference
from lidargen.utils import render

def main(args):
    cfg = __all__[args.cfg]()
    cfg.resume = args.ckpt
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    # =================================================================================
    # Load pre-trained model
    # =================================================================================

    ddpm, model, lidar_utils, _, _, _ = inference.load_model_duffusion_training(cfg)
    ddpm.eval()
    lidar_utils.eval()
    ddpm.to(args.device)
    lidar_utils.to(args.device)  
    
    # =================================================================================
    # Load dataset
    # =================================================================================

    cfg.data.split = 'val'
    dataset = all_datasets[cfg.data.dataset](cfg.data)

    batch_dict = dataset.__getitem__(args.sample_idx)
    collate_fn = dataset.collate_fn if getattr(cfg.data, 'custom_collate_fn', False) else None
    if collate_fn is not None:
        batch_dict = collate_fn([batch_dict])
    for key in batch_dict:
        if isinstance(batch_dict[key], torch.Tensor):
            batch_dict[key] = batch_dict[key].to(args.device)
    
    # =================================================================================
    # Sampling (reverse diffusion)
    # =================================================================================

    def preprocess(batch):
        x = []
        if cfg.data.train_depth:
            x += [lidar_utils.convert_depth(batch["depth"])]
        if cfg.data.train_reflectance:
            x += [batch["reflectance"]]
        x = torch.cat(x, dim=1)
        x = lidar_utils.normalize(x)
        x = F.interpolate(
            x.to(args.device),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
        return x
    
    def preprocess_prev_cond(prev_cond):
        x = []
        reflectance = prev_cond[:,3,...]/255
        depth = prev_cond[:,-2,...]

        if cfg.data.train_depth:
            x += [lidar_utils.convert_depth(depth.unsqueeze(1))]
        if cfg.data.train_reflectance:
            x += [reflectance.unsqueeze(1)]
        x = torch.cat(x, dim=1)
        x = lidar_utils.normalize(x)
        x = F.interpolate(
            x.to(args.device),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
        prev_labels = prev_cond[:,4,...].long()
        one_hot = F.one_hot(prev_labels, num_classes=len(list(cfg.data.class_names))+1).permute(0, 3, 1, 2)
        x = torch.cat((x, one_hot.float()), dim=1)
        return x

    x_0 = preprocess(batch_dict)
    batch_dict['x_0'] = x_0
    if 'prev_cond' in batch_dict:
        prev_cond = preprocess_prev_cond(batch_dict['prev_cond'])
        batch_dict['cond'] = prev_cond

    xs = ddpm.sample(
        batch_dict = batch_dict,
        batch_size=1,
        num_steps=args.sampling_steps,
        mode=args.mode,
        return_all=True,
    ).clamp(-1, 1)

    # =================================================================================
    # Save as image or video
    # =================================================================================

    xs = lidar_utils.denormalize(xs)
    xs[:, :, [0]] = lidar_utils.revert_depth(xs[:, :, [0]]) / lidar_utils.max_depth

    def render_image(x, save_xyz=False):
        img = einops.rearrange(x, "B C H W -> B 1 (C H) W")
        img = render.colorize(img) / 255
        xyz = lidar_utils.to_xyz(x[:, [0]] * lidar_utils.max_depth)
        if save_xyz:
            save_points(xyz[0].reshape(3,-1).permute(1,0), "sample_points.txt")

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

    img, bev = render_image(xs[-1], save_xyz=True)
    save_image(img, "samples_img.png", nrow=1)
    save_image(bev, "samples_bev.png", nrow=4)

    x_gt = torch.cat([batch_dict['depth'], batch_dict['reflectance']], dim=1)
    x_gt[:, [0]] = x_gt[:, [0]] / lidar_utils.max_depth

    img, bev = render_image(x_gt, save_xyz=False)
    save_image(img, "gt_img.png", nrow=1)
    save_image(bev, "gt_bev.png", nrow=4)

    video = imageio.get_writer("samples.mp4", mode="I", fps=60)
    for x in tqdm(xs, desc="making video..."):
        img, bev = render_image(x)
        scale = 512 / img.shape[-1]
        img = F.interpolate(img, scale_factor=scale, mode="bilinear", antialias=True)
        scale = 512 / bev.shape[-1]
        bev = F.interpolate(bev, scale_factor=scale, mode="bilinear", antialias=True)
        img = torch.cat([img, bev], dim=2)
        img = make_grid(img, nrow=args.batch_size, pad_value=1)
        img = img.permute(1, 2, 0).mul(255).byte()
        video.append_data(img.cpu().numpy())
    video.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--sample_idx", type=int, default=1)
    parser.add_argument("--mode", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--sampling_steps", type=int, default=256)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    main(args)
