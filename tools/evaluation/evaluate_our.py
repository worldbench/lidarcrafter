import argparse
from pathlib import Path
import pickle
from tqdm.auto import tqdm
import einops
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import json
import random
from loguru import logger
from typing import Literal, Tuple
from pydantic.dataclasses import dataclass
import numpy as np
sys.path.append('../')
from lidargen.metrics.eval_utils import evaluate    
from lidargen.metrics import bev, distribution, fg_object, scene
from lidargen.metrics.extractor import pointnet, rangenet
from lidargen.utils.configs import __all__
from lidargen.dataset import __all__ as all_datasets
from lidargen.utils.lidar import LiDARUtility, get_linear_ray_angles
from lidargen.dataset.transforms_3d import common
from lidargen.dataset import utils

# from LiDARGen
# MAX_DEPTH = 80.0
# MIN_DEPTH = 1.45
# DATASET_MAX_DEPTH = 80.0
MAX_DEPTH = 63.0
MIN_DEPTH = 0.5
DATASET_MAX_DEPTH = 80.0

def find_pth_files(root_dir, endswith='.pth'):
    pth_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(endswith):
                pth_files.append(os.path.join(dirpath, fname))
    return pth_files

def resize(x, size):
    return F.interpolate(x, size=size, mode="nearest-exact")

class Samples(torch.utils.data.Dataset):
    def __init__(self, root, helper, lidar_utils=None):
        self.root = root
        self.sample_path_list = find_pth_files(Path(root), endswith='.txt' if 'dwm' in root or 'uniscene' in root else '.pth')
        self.helper = helper
        self.lidar_utils = lidar_utils
        self.lidar_utils.eval()

    # fix points num to 26000, points.shape=[N,3]
    def fix_points_num(self, points):
        if len(points) < 26000:
            points = np.pad(points, ((0, 26000 - len(points)), (0, 0)), mode='constant', constant_values=0)
        elif len(points) > 26000:
            points = points[:26000]
        return points

    def __getitem__(self, index):
        if ('dwm' in self.root):
            points = self.__getitem_points__(index)
            rotation = np.array(np.pi) / 2
            points = utils.rotate_points_along_z(points[np.newaxis, :, :], rotation.reshape(1))[0]
            points[:, 2] -= 2.0
            return torch.from_numpy(self.fix_points_num(points)).float()
        
        if ('uniscene' in self.root):
            points = self.__getitem_points__(index)
            rotation = np.array(np.pi) / 2
            points = utils.rotate_points_along_z(points[np.newaxis, :, :], rotation.reshape(1))[0]
            return torch.from_numpy(self.fix_points_num(points)).float()
        
        img = torch.load(self.sample_path_list[index], map_location="cpu")
        depth = img[[0]]
        mask = torch.logical_and(depth > MIN_DEPTH, depth < MAX_DEPTH).float()
        if img.shape[0] == 2:
            mask = mask.numpy()
            depth = self.lidar_utils.revert_depth(depth)
            points = self.lidar_utils.to_xyz(depth.unsqueeze(0)) # [1, 3, 32, 1024]
            points = torch.cat([points[0], img[[1]]], dim=0)  # [1, 4, 32, 1024]
            points = points.reshape(4,-1).permute(1,0).numpy()  # [H*W, 4]
            xyzrdm = common.load_points_as_images(points=points, scan_unfolding=False, H=32, W=1024,
                                        min_depth=1.45, max_depth=80.0, fov_up=10.0, fov_down=-30.0)
            xyzrdm = xyzrdm.transpose(2, 0, 1)
            mask = xyzrdm[[5]] * mask
            xyzrdm *= mask
            return xyzrdm[[4,0,1,2,3]], mask

        img = img * mask
        return img.float()[:5], mask.float()

    def __getitem_points__(self, index):
        points = np.loadtxt(self.sample_path_list[index], dtype=np.float32)
        return points

    def __len__(self):
        return len(self.sample_path_list)

@dataclass
class DataConfig:
    dataset: Literal["kitti_raw", "kitti_360", "nuscenes"] = "nuscenes"
    task: str = "layout_cond"
    class_names: Tuple[str, str, str, str, str, str, str, str] = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian')
    custom_collate_fn: bool = True
    data_root = "../data/nuscenes"
    pkl_path = "../data/infos/nuscenes_infos_lidargen_train.pkl"

    depth_format: Literal["log_depth", "inverse_depth", "depth"] = "log_depth"
    scan_unfolding: bool = False
    projection: Literal[
        "unfolding-2048",
        "spherical-2048",
        "unfolding-1024",
        "spherical-1024",
    ] = "spherical-1024"
    train_depth: bool = True
    train_reflectance: bool = True
    resolution: Tuple[int, int] = (32, 1024)
    min_depth = 1.45
    max_depth = 80.0
    fov_up = 10.0
    fov_down = -30.0

class EvaluationEngine:
    def __init__(self, args):
        self.args = args
        self.data_cfg = DataConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lidar_utils = LiDARUtility(
            resolution=self.data_cfg.resolution,
            depth_format=self.data_cfg.depth_format,
            min_depth=self.data_cfg.min_depth,
            max_depth=self.data_cfg.max_depth,
            ray_angles=get_linear_ray_angles(
                H=self.data_cfg.resolution[0],
                W=self.data_cfg.resolution[1],
                fov_up=self.data_cfg.fov_up,
                fov_down=self.data_cfg.fov_down
            ),
        )
        self.lidar_utils.eval()
        self.lidar_utils.to(self.device)
    
        self.extract_img_feats, self.preprocess_img = rangenet.rangenet53(
            url_or_file='../pretrained_models/evaluation/nuscenes/rangenet/rangenet.tar.gz',
            device=self.device,
            compile=True,
        )
        self.extract_pts_feats = pointnet.pretrained_pointnet(
            dataset="shapenet",
            device=self.device,
            compile=True,
        )

        self.real_set, self.real_fg_object_set = self.load_real_dataset()

    def prepare_batch(self, batch_dict):
        depth = batch_dict["depth"]
        mask = torch.logical_and(depth > MIN_DEPTH, depth < MAX_DEPTH)
        x = []
        x += [self.lidar_utils.convert_depth(depth)]
        x += [batch_dict["xyz"]]
        x += [batch_dict["reflectance"]]
        x = torch.cat(x, dim=1)
        x = F.interpolate(
            x.to(self.device),
            size=(32, 1024),
            mode="nearest-exact",
        )
        return x, (batch_dict['mask'] * mask).to(self.device)

    def prepare_gen_batch(self, imgs_frd):
        x = []
        depth = imgs_frd[:, [0]]
        x += [self.lidar_utils.convert_depth(depth)]
        x += [imgs_frd[:, [1,2,3]]]
        x += [imgs_frd[:, [4]]]
        x = torch.cat(x, dim=1)
        x = F.interpolate(
            x.to(self.device),
            size=(32, 1024),
            mode="nearest-exact",
        )
        return x

    def load_real_dataset(self):
        H, W = self.lidar_utils.resolution
        cache_file_path = f"../generated_results/ori/inference_results/real_set_{self.data_cfg.dataset}.pkl"

        if Path(cache_file_path).exists():
            logger.info(f"Found cached {cache_file_path}")
            real_set = pickle.load(open(cache_file_path, "rb"))
        else:
            real_set = defaultdict(list)
            self.data_cfg.split = 'all'
            self.data_cfg.num_sample = args.num_sample
            dataset = all_datasets[self.data_cfg.dataset](self.data_cfg)
            real_loader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                drop_last=False,
                collate_fn=dataset.collate_fn if getattr(self.data_cfg, 'custom_collate_fn', False) else None
            )

            for batch in tqdm(real_loader, desc="real"):
                imgs_frd, mask = self.prepare_batch(batch)
                with torch.inference_mode():
                    feats_img = self.extract_img_feats(
                        self.preprocess_img(imgs_frd, mask), feature="lidargen"
                    )
                real_set["img_feats"].append(feats_img.cpu())

                # points
                xyz = batch["xyz"].to(self.device)
                point_clouds = einops.rearrange(xyz * mask, "B C H W -> B C (H W)")
                for point_cloud in point_clouds:
                    point_cloud = einops.rearrange(point_cloud, "C N -> N C")
                    real_set["points"].append(point_cloud.cpu())
                    hist = bev.point_cloud_to_histogram(point_cloud)
                    real_set["bev_hists"].append(hist.cpu())

                with torch.inference_mode():
                    feats_pts = self.extract_pts_feats(point_clouds / DATASET_MAX_DEPTH)
                real_set["pts_feats"].append(feats_pts.cpu())

            real_set["img_feats"] = torch.cat(real_set["img_feats"], dim=0).numpy()
            real_set["pts_feats"] = torch.cat(real_set["pts_feats"], dim=0).numpy()
            real_set["bev_hists"] = torch.stack(real_set["bev_hists"], dim=0).numpy()

            with open(cache_file_path, "wb") as f:
                pickle.dump(real_set, f)

        fg_object_cache_file_path = f"../generated_results/ori/inference_results/real_set_fg_object_{self.data_cfg.dataset}.pkl"
        if Path(fg_object_cache_file_path).exists():
            logger.info(f"Found cached {fg_object_cache_file_path}")
            real_fg_object_set = pickle.load(open(fg_object_cache_file_path, "rb"))
        else:
            real_fg_object_set = fg_object.get_fg_object_set_feature(method_name="ori")
            with open(fg_object_cache_file_path, "wb") as f:
                pickle.dump(real_fg_object_set, f)

        logger.info(f"Real dataset loaded with {len(real_set['pts_feats'])} samples.")
        return real_set, real_fg_object_set

    def load_gen_dataset(self, gen_method, resume=False):
        cache_folder_path = f"../generated_results/{gen_method}/inference_results"
        os.makedirs(cache_folder_path, exist_ok=True)
        cache_file_path = os.path.join(cache_folder_path, f"gen_set_{self.data_cfg.dataset}.pkl")
        if Path(cache_file_path).exists() and not resume:
            logger.info(f"Found cached {cache_file_path}")
            gen_set = pickle.load(open(cache_file_path, "rb"))
        else:
            sample_dir = os.path.join(
                self.args.sample_dir, gen_method
            )
            gen_loader = DataLoader(
                dataset=Samples(sample_dir, helper=self.lidar_utils.cpu(), lidar_utils=self.lidar_utils),
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
            )

            gen_set = defaultdict(list)
            if gen_method in ["opendwm", "opendwm_dit", "uniscene"]:
                for point_clouds in tqdm(gen_loader, desc="gen"):
                    point_clouds = point_clouds.to(self.device)
                    for point_cloud in point_clouds:
                        gen_set["points"].append(point_cloud.cpu())
                        hist = bev.point_cloud_to_histogram(point_cloud)
                        gen_set["bev_hists"].append(hist.cpu())

                    with torch.inference_mode():
                        point_clouds = einops.rearrange(point_clouds, "B N C -> B C N")
                        feats_pts = self.extract_pts_feats(point_clouds / DATASET_MAX_DEPTH)
                    gen_set["pts_feats"].append(feats_pts.cpu())
            else:
                for imgs_frd, mask in tqdm(gen_loader, desc="gen"):
                    imgs_frd = self.prepare_gen_batch(imgs_frd)
                    imgs_frd, mask = imgs_frd.to(self.device), mask.to(self.device)
                    if self.data_cfg.train_reflectance:
                        with torch.inference_mode():
                            feats_img = self.extract_img_feats(
                                self.preprocess_img(imgs_frd, mask), feature="lidargen"
                            )
                        gen_set["img_feats"].append(feats_img.cpu())

                    xyz = imgs_frd[:, 1:4]
                    point_clouds = einops.rearrange(xyz * mask, "B C H W -> B C (H W)")
                    for point_cloud in point_clouds:
                        point_cloud = einops.rearrange(point_cloud, "C N -> N C")
                        gen_set["points"].append(point_cloud.cpu())
                        hist = bev.point_cloud_to_histogram(point_cloud)
                        gen_set["bev_hists"].append(hist.cpu())

                    with torch.inference_mode():
                        feats_pts = self.extract_pts_feats(point_clouds / DATASET_MAX_DEPTH)
                    gen_set["pts_feats"].append(feats_pts.cpu())

            if self.data_cfg.train_reflectance and "img_feats" in gen_set:
                gen_set["img_feats"] = torch.cat(gen_set["img_feats"], dim=0).numpy()
            gen_set["pts_feats"] = torch.cat(gen_set["pts_feats"], dim=0).numpy()
            gen_set["bev_hists"] = torch.stack(gen_set["bev_hists"], dim=0).numpy()
            with open(cache_file_path, "wb") as f:
                pickle.dump(gen_set, f)

        fg_object_cache_file_path = f"../generated_results/{gen_method}/inference_results/real_set_fg_object_{self.data_cfg.dataset}.pkl"
        if Path(fg_object_cache_file_path).exists():
            logger.info(f"Found cached {fg_object_cache_file_path}")
            gen_fg_object_set = pickle.load(open(fg_object_cache_file_path, "rb"))
        else:
            gen_fg_object_set = fg_object.get_fg_object_set_feature(method_name=gen_method)
            with open(fg_object_cache_file_path, "wb") as f:
                pickle.dump(gen_fg_object_set, f)

        return gen_set, gen_fg_object_set

    def evaluate(self, metrics, gen_method, resume=False):
        save_path = f"../generated_results/{self.data_cfg.dataset}_{gen_method}_results.json"
        if os.path.exists(save_path):
            with open(save_path, 'r', encoding='utf-8') as f:
                results = json.load(f)     
            results = defaultdict(dict, results)  # Ensure results is a defaultdict
        else:
            results = defaultdict(dict)
        if gen_method in ["lidm", "r2dm", "our", "lidargen", "opendwm", "opendwm_dit", "uniscene"]:
            gen_set, gen_fg_object_set = self.load_gen_dataset(gen_method, resume)

        # save a sample points for each set TODO: remove this in production
        # import numpy as np
        # temp_real_points = self.real_set["points"][0]
        # temp_gen_points = gen_set["points"][0]
        # np.savetxt(
        #     "../generated_results/real_sample_points.txt",
        #     temp_real_points,
        #     fmt="%.6f",
        # )
        # np.savetxt(
        #     "../generated_results/gen_sample_points.txt",
        #     temp_gen_points,
        #     fmt="%.6f",
        # )

        torch.cuda.empty_cache()
        ##################################### frd #####################################
        if 'frd' in metrics and gen_method != "ori":
            if self.data_cfg.train_reflectance:
                results["img"]["frechet_distance"] = distribution.compute_frechet_distance(
                    self.real_set["img_feats"], gen_set["img_feats"]
                )
                results["img"]["squared_mmd"] = distribution.compute_squared_mmd(
                    self.real_set["img_feats"], gen_set["img_feats"]
                )
        ##################################### fpd #####################################

        if 'fpd' in metrics and gen_method != "ori":
            results["pts"]["frechet_distance"] = distribution.compute_frechet_distance(
                self.real_set["pts_feats"], gen_set["pts_feats"]
            )
            results["pts"]["squared_mmd"] = distribution.compute_squared_mmd(
                self.real_set["pts_feats"], gen_set["pts_feats"]
            )

        ##################################### jsd #####################################

        perm = list(range(len(self.real_set["bev_hists"])))
        random.Random(0).shuffle(perm)
        perm = perm[:10_000]

        if 'jsd' in metrics and gen_method != "ori":
            results["bev"]["jsd"] = bev.compute_jsd_2d(
                torch.from_numpy(self.real_set["bev_hists"][perm]).to(self.device).float(),
                torch.from_numpy(gen_set["bev_hists"]).to(self.device).float(),
            )

        ##################################### mmd #####################################

        if 'mmd' in metrics and gen_method != "ori":
            results["bev"]["mmd"] = bev.compute_mmd_2d(
                torch.from_numpy(self.real_set["bev_hists"][perm]).to(self.device).float(),
                torch.from_numpy(gen_set["bev_hists"]).to(self.device).float(),
            )

        ############### Detection Confidence Score (DCF) ################


        if 'dcf' in metrics and gen_method != "ori":
            results['object']["dcf"] = fg_object.compute_dcf(
                method_name=gen_method,
            )

        ############### Classification-based Generation Fidelity (CGF) ################
        if 'cgf' in metrics:
            method_path = Path(f'../generated_results/{gen_method}/inference_results')
            method_path.mkdir(parents=True, exist_ok=True)
            foreground_samples_saved_path = f'../generated_results/{self.args.method}/inference_results/foreground_samples'
            if gen_method != "ori":
                assert os.path.exists(foreground_samples_saved_path), f"Foreground samples path {foreground_samples_saved_path} does not exist."
            results['object']["cgf"] = fg_object.compute_cgf(
                method_name=gen_method,
            )

        ############### Regression-based Generation Fidelity (RGF) ################
        if 'rgf' in metrics:
            foreground_samples_saved_path = f'../generated_results/{self.args.method}/inference_results/foreground_samples'
            if gen_method != "ori":
                assert os.path.exists(foreground_samples_saved_path), f"Foreground samples path {foreground_samples_saved_path} does not exist."
            results['object']["rgf"] = fg_object.compute_rgf(
                method_name=gen_method,
            )

        ############### Foreground Detection Confidence (FDC) ################
        if 'fdc' in metrics:

            # results['scene']["fdc"] = scene.compute_fdc(
            #     method_name=gen_method,
            # )
            pass

        if 'obj' in metrics:
            class_name = 'car'
            results["obj"]["frechet_distance"] = distribution.compute_frechet_distance(
                self.real_fg_object_set[class_name]["pts_feat"], gen_fg_object_set[class_name]["pts_feat"]
            )
            results["obj"]["squared_mmd"] = distribution.compute_squared_mmd(
                self.real_fg_object_set[class_name]["pts_feat"], gen_fg_object_set[class_name]["pts_feat"]
            )
            results["obj"]["jsd"] = bev.compute_jsd_2d(
                torch.from_numpy(self.real_fg_object_set[class_name]["bev_hists"]).to(self.device).float(),
                torch.from_numpy(gen_fg_object_set[class_name]["bev_hists"]).to(self.device).float(),
            )
            results["obj"]["mmd"] = bev.compute_mmd_2d(
                torch.from_numpy(self.real_fg_object_set[class_name]["bev_hists"]).to(self.device).float(),
                torch.from_numpy(gen_fg_object_set[class_name]["bev_hists"]).to(self.device).float(),
            )

        # save results to json
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Evaluation results saved to {save_path}")
        logger.info("Evaluation completed.")
        logger.info(f"Results: {results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="ori",)
    parser.add_argument("--sample_dir", type=str, default="../generated_results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--num_sample",
        type=int,
        default=10000,
        help="Sampling number for evaluation (default: 10000)",
    )
    parser.add_argument(
        "--metrics",
        nargs='*',
        default=["cgf"],
        # default=["frd"],
        help="a list of strings, e.g. --tags apple banana"
    )

    args = parser.parse_args()

    eval_egine = EvaluationEngine(args)
    eval_egine.evaluate(metrics=args.metrics, gen_method=args.method, resume=False)