import argparse
from pathlib import Path
import pickle
from tqdm.auto import tqdm
import einops

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
sys.path.append('../')
from lidargen.utils import inference
from lidargen.metrics.eval_utils import evaluate    
from lidargen.utils.configs import __all__
from lidargen.dataset import __all__ as all_datasets
from lidargen.utils.lidar import LiDARUtility, get_linear_ray_angles

MAX_DEPTH = 63.0
MIN_DEPTH = 0.5
DATASET_MAX_DEPTH = 80.0

def resize(x, size):
    return F.interpolate(x, size=size, mode="nearest-exact")

class Samples(torch.utils.data.Dataset):
    def __init__(self, root, helper):
        self.sample_path_list = sorted(Path(root).glob("*.pth"))[:10_000]
        self.helper = helper

    def __getitem__(self, index):
        img = torch.load(self.sample_path_list[index], map_location="cpu")
        # assert img.shape[0] == 5, img.shape
        depth = img[[0]]
        mask = torch.logical_and(depth > MIN_DEPTH, depth < MAX_DEPTH).float()
        img = img * mask
        return img.float(), mask.float()

    def __len__(self):
        return len(self.sample_path_list)

def main(args):
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = __all__[args.cfg]()
    if args.batch_size is not None:
        cfg.training.batch_size_train = args.batch_size
    if args.num_workers is not None:
        cfg.training.num_workers = args.num_workers

    coords = get_linear_ray_angles(H=cfg.data.resolution[0], W=cfg.data.resolution[1], fov_up=cfg.data.fov_up, fov_down=cfg.data.fov_down)

    lidar_utils = LiDARUtility(
        resolution=cfg.data.resolution,
        depth_format=cfg.data.depth_format,
        min_depth=cfg.data.min_depth,
        max_depth=cfg.data.max_depth,
        ray_angles=coords,
    )

    lidar_utils.eval()
    lidar_utils.to(device)
    H, W = lidar_utils.resolution

    # =====================================================
    # real set
    # =====================================================

    cfg.data.split = 'all'
    cfg.data.num_sample = args.num_sample
    dataset = all_datasets[cfg.data.dataset](cfg.data)
    real_loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size_train,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        drop_last=False,
        collate_fn=dataset.collate_fn if getattr(cfg.data, 'custom_collate_fn', False) else None
    )

    cache_file_path = f"../generated_results/real_set_{cfg.data.dataset}_{cfg.data.projection}_{H}x{W}_.pkl"
    
    if Path(cache_file_path).exists():
        print(f"found cached {cache_file_path}")
        real_set = pickle.load(open(cache_file_path, "rb"))
    else:
        real_set = dict(img_feats=list(), pts_feats=list(), bev_hists=list(), points=list())
        custom_sample_num = 10000
        i = 0
        for batch in tqdm(real_loader, desc="real"):
            depth = resize(batch["depth"], (H, W)).to(device)
            xyz = resize(batch["xyz"], (H, W)).to(device)
            rflct = resize(batch["reflectance"], (H, W)).to(device)
            mask = resize(batch["mask"], (H, W)).to(device)
            mask = mask * torch.logical_and(depth > MIN_DEPTH, depth < MAX_DEPTH)
            imgs_frd = torch.cat([depth, xyz, rflct], dim=1)
            point_clouds = einops.rearrange(xyz * mask, "B C H W -> B C (H W)")
            for point_cloud in point_clouds:
                point_cloud = einops.rearrange(point_cloud, "C N -> N C")
                real_set["points"].append(point_cloud.cpu().numpy())
            if i >= custom_sample_num:
                break
            i += 1
        with open(cache_file_path, "wb") as f:
            pickle.dump(real_set, f)
    # =====================================================
    # gen set
    # =====================================================

    gen_loader = DataLoader(
        dataset=Samples(args.sample_dir, helper=lidar_utils.cpu()),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    gen_set = dict(img_feats=list(), pts_feats=list(), bev_hists=list(), points=list())
    for imgs_frd, mask in tqdm(gen_loader, desc="gen"):
        xyz = imgs_frd[:, 1:4]
        point_clouds = einops.rearrange(xyz * mask, "B C H W -> B C (H W)")
        for point_cloud in point_clouds:
            point_cloud = einops.rearrange(point_cloud, "C N -> N C")
            gen_set["points"].append(point_cloud.cpu().numpy())

    # =====================================================
    # metrics
    # =====================================================

    # mmd
    evaluate(real_set["points"], gen_set["points"], ['fpvd'], '32')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--num_sample",
        type=int,
        default=10000,
        help="Sampling number for evaluation (default: 10000)",
    )
    args = parser.parse_args()
    main(args)
