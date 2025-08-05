from typing import Literal, Tuple
import numpy as np
from pydantic.dataclasses import dataclass
from dataclasses import field
from .nuscenes_dataset import NuscDataset
from .nuscenes_object_dataset import NuscObjectDataset
from .transforms_3d import common

@dataclass
class DataConfig:
    dataset: Literal["kitti_raw", "kitti_360", "nuscenes", "custom"] = "custom"
    class_names: Tuple[str, str, str, str, str, str, str, str] = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian')
    custom_collate_fn: bool = True
    data_root = None
    train_depth: bool = True
    train_reflectance: bool = True
    resolution: Tuple[int, int] = (32, 1024)
    min_depth = 1.45
    max_depth = 80.0
    fov_up = 10.0
    fov_down = -30.0
    scan_unfolding: bool = False
    split = "train"
    task = 'layout_cond'

@dataclass
class NuscObjectConfig:
    dataset: Literal["kitti_raw", "kitti_360", "nuscenes", "custom"] = "custom"
    class_names: Tuple[str, str, str, str, str, str, str, str] = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian')
    custom_collate_fn: bool = True
    data_root = None
    train_depth: bool = True
    train_reflectance: bool = True
    resolution: Tuple[int, int] = (32, 1024)
    min_depth = 1.45
    max_depth = 80.0
    fov_up = 10.0
    fov_down = -30.0
    scan_unfolding: bool = False
    split = "train"
    task = 'object_generation'

class CustomDataset(NuscDataset):
    """
    Custom dataset class that extends NuscDataset.
    This class can be used to implement specific behaviors or configurations
    for a custom dataset based on the NuScenes dataset.
    """

    def __init__(self, custom_box_infos, cfg = DataConfig()):
        self.custom_box_infos = custom_box_infos
        super().__init__(cfg)

    def prepare_data(self):
        self.data = self.custom_box_infos

    def __getitem__(self, idx, inpaint=False):
        input_dict = self.data[idx]
        if 'points' in input_dict:
            xyzrdm = common.load_points_as_images(points=input_dict['points'], scan_unfolding=self.cfg.scan_unfolding, H=self.cfg.resolution[0], W=self.cfg.resolution[1],
                                        min_depth=self.cfg.min_depth, max_depth=self.cfg.max_depth, fov_up=self.cfg.fov_up, fov_down=self.cfg.fov_down)
            xyzrdm = xyzrdm.transpose(2, 0, 1)
            xyzrdm *= xyzrdm[[5]]
            input_dict.update({
                "xyz": xyzrdm[:3],
                "reflectance": xyzrdm[[3]]/255,
                "depth": xyzrdm[[4]],
                "mask": xyzrdm[[5]],
            })
            if self.task == 'autoregressive_generation':
                    depth = input_dict['depth']
                    reflectance = input_dict['reflectance']
                    input_dict.update(
                         {
                              "autoregressive_cond": np.concatenate([depth, reflectance], axis=0)
                              })
                    if not getattr(self, 'inpaint_mode', False):
                        input_dict.pop('depth')
                        input_dict.pop('reflectance')
                        input_dict.pop('mask')
                        input_dict.pop('xyz')

        input_dict = self.pre_process(input_dict)
        if self.task == 'layout_generation':
            scenegraph_out = self.scene_graph_assigner.assign_item(idx, input_dict)
            input_dict.update({
                "custom_dict": scenegraph_out
            })
        return input_dict
    
class CustomNuscObjectDataset(NuscDataset):
    """
    Custom dataset class that extends NuscDataset.
    This class can be used to implement specific behaviors or configurations
    for a custom dataset based on the NuScenes dataset.
    """

    def __init__(self, custom_box_infos, cfg = NuscObjectConfig()):
        self.custom_box_infos = custom_box_infos
        super().__init__(cfg)

    def prepare_data(self):
        self.data = self.custom_box_infos
    
    def __getitem__(self, idx):
        input_dict = self.data[idx]
        data_dict = {}
        data_dict = self.distille_local_boxes(input_dict)
        return data_dict