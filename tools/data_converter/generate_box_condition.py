import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append('/home/alan/AlanLiang/Projects/AlanLiang/LidarGen4D')
sys.path.append('/data/yyang/workspace/projects/LidarGen4D')
sys.path.append('/data1/liangao/AlanLiang/Projects/LidarGen4D')

from lidargen.utils import inference
from lidargen.utils import common
from lidargen.utils.configs import __all__
from lidargen.dataset import __all__ as all_datasets
from lidargen.dataset.transforms_3d.common import load_points_as_images
from lidargen.utils.lidar import LiDARUtility, get_linear_ray_angles
import torch.nn.functional as F

class Object_Sampler:
    def __init__(self, cfg, ckpt_path, split='train'):
        self.cfg = __all__[cfg]()
        self.data_cfg = __all__['nuscenes-box-layout']().data
        self.data_cfg.split = split
        self.data_cfg.task = 'object_generation'
        self.cfg.resume = ckpt_path
        self.init_sampler()

    def init_sampler(self):
        dataset = all_datasets[self.data_cfg.dataset](self.data_cfg)
        self.ddpm, _, _, _, _ = inference.load_model_object_duffusion_training(self.cfg)
        self.ddpm.eval()
        self.ddpm.to('cuda')
        self.dataset = dataset

        coord = get_linear_ray_angles(H=self.data_cfg.resolution[0], W=self.data_cfg.resolution[1], fov_up=self.data_cfg.fov_up, fov_down=self.data_cfg.fov_down)
        self.lidar_utils = LiDARUtility(
            resolution=self.data_cfg.resolution,
            depth_format=self.data_cfg.depth_format,
            min_depth=self.data_cfg.min_depth,
            max_depth=self.data_cfg.max_depth,
            ray_angles=coord,
        )
        self.lidar_utils.eval()
        self.lidar_utils.to('cuda')

    def sample(self, num_steps=1024, sample_index=None, w_semantic=False):
        batch_dict = self.dataset.__getitem__(sample_index)
        collate_fn = self.dataset.collate_fn if getattr(self.cfg.data, 'custom_collate_fn', False) else None
        if collate_fn is not None:
            batch_dict = collate_fn([batch_dict])
        batch_dict = common.to_device(batch_dict, 'cuda')
        fg_encoding_box = batch_dict['fg_encoding_box'].squeeze(0)
        B, _ = fg_encoding_box.shape
        batch_dict['fg_encoding_box'] = fg_encoding_box
        batch_dict['fg_class'] = batch_dict['fg_class'].squeeze(0)
        
        generated_object_points = self.ddpm.sample(
            batch_dict = batch_dict,
            batch_size=B,
            num_steps=num_steps,
            mode='ddpm',
            return_all=False,
        )
        obj_points = self.dataset.unscaled_objs_3d(sample_index, generated_object_points.detach().cpu().numpy(), w_semantic=w_semantic)
        return obj_points
    
    def preprocess_range_feature(self, range_feature):
        x = []
        reflectance = range_feature[:,3,...]/255
        depth = range_feature[:,-2,...]

        if self.data_cfg.train_depth:
            x += [self.lidar_utils.convert_depth(depth.unsqueeze(1))]
        if self.data_cfg.train_reflectance:
            x += [reflectance.unsqueeze(1)]
        x = torch.cat(x, dim=1)
        x = self.lidar_utils.normalize(x)
        x = F.interpolate(
            x.to('cuda'),
            size=self.data_cfg.resolution,
            mode="nearest-exact",
        )
        prev_labels = range_feature[:,4,...]
        x = torch.cat((x, prev_labels.unsqueeze(1)), dim=1)
        return x[0]

    # def generate_box_condition(self):
    #     save_path = Path('../data/box_condition') / f'{self.data_cfg.split}'
    #     save_path.mkdir(parents=True, exist_ok=True)
    #     for sample_index in tqdm(range(len(self.dataset))):
    #         generated_object_points = self.sample(sample_index=sample_index, w_semantic=True)
    #         xyzrdm = load_points_as_images(points=generated_object_points, scan_unfolding=self.data_cfg.scan_unfolding, H=self.data_cfg.resolution[0], W=self.data_cfg.resolution[1],
    #                                     min_depth=self.data_cfg.min_depth, max_depth=self.data_cfg.max_depth, fov_up=self.data_cfg.fov_up, fov_down=self.data_cfg.fov_down, custom_feat_dim=1)
    #         xyzrdm = xyzrdm.transpose(2, 0, 1)
    #         xyzrdm *= xyzrdm[[-1]]
    #         xyzrdm = torch.tensor(xyzrdm, dtype=torch.float32).unsqueeze(0).to('cuda')
    #         cond = self.preprocess_range_feature(xyzrdm)
    #         sample_save_path = save_path / f'sample_{sample_index:07d}.npy'
    #         np.save(sample_save_path, cond.detach().cpu().numpy())
        
    def generate_box_condition(self, num_threads: int = 4):

        save_path = Path('../data/box_condition') / f'{self.data_cfg.split}'
        save_path.mkdir(parents=True, exist_ok=True)

        def _process(sample_index: int):
            generated_object_points = self.sample(sample_index=sample_index, w_semantic=True)
            xyzrdm = load_points_as_images(
                points=generated_object_points,
                scan_unfolding=self.data_cfg.scan_unfolding,
                H=self.data_cfg.resolution[0],
                W=self.data_cfg.resolution[1],
                min_depth=self.data_cfg.min_depth,
                max_depth=self.data_cfg.max_depth,
                fov_up=self.data_cfg.fov_up,
                fov_down=self.data_cfg.fov_down,
                custom_feat_dim=1
            )
            xyzrdm = xyzrdm.transpose(2, 0, 1)
            xyzrdm *= xyzrdm[[-1]]
            xyzrdm = torch.tensor(xyzrdm, dtype=torch.float32).unsqueeze(0).to('cuda')
            cond = self.preprocess_range_feature(xyzrdm)
            save_file = save_path / f'sample_{sample_index:07d}.npy'
            np.save(save_file, cond.detach().cpu().numpy())

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(_process, idx): idx for idx in range(len(self.dataset))}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"[{self.data_cfg.split}] Generating"):
                idx = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error at sample {idx}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate box condition for object generation")
    parser.add_argument("--cfg", type=str, default="nuscenes-object", help="Configuration file")
    parser.add_argument("--ckpt_path", type=str, default='../pretrained_models/nuscenes-object-1000000.pth', help="Path to the checkpoint file")
    args = parser.parse_args()
    # sampler = Object_Sampler(cfg=args.cfg, ckpt_path=args.ckpt_path, split='train')
    # box_condition = sampler.generate_box_condition()
    # print("Generated box condition shape:", box_condition.shape)
    # torch.cuda.empty_cache()
    sampler = Object_Sampler(cfg=args.cfg, ckpt_path=args.ckpt_path, split='val')
    box_condition = sampler.generate_box_condition()
    print("Generated box condition shape:", box_condition.shape)