"""
nuScenes Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Zheng Zhang
Please cite our work if the code is helpful to you.
"""
import torch

import os
import numpy as np
from collections.abc import Sequence
import pickle
from torch.utils.data import Dataset
from lidargen.dataset.transforms_3d import common

color_map = {
    0: [0, 0, 0],  # unlabelled
    1: [255, 0, 0],  # car
    2: [0, 255, 0],  # bus
    3: [0, 0, 255],  # truck
    4: [255, 255, 0],  # construction_vehicle
    5: [255, 0, 255],  # trailer
    6: [0, 255, 255],  # barrier
    7: [192, 192, 192],  # motorcycle
    8: [128, 0, 0],  # bicycle
    9: [128, 128, 0],  # pedestrian
    10: [128, 0, 128],  # traffic_cone
    11: [0, 128, 128],  # bus_stop
    12: [255, 128, 0],  # emergency_vehicle
    13: [128, 255, 0],  # train
    14: [128, 128, 128],  # traffic_light
    15: [255, 128, 128],  # stop_sign
    16: [128, 255, 128],  # crosswalk
}

# names = [
#     "barrier",
#     "bicycle",
#     "bus",
#     "car",
#     "construction_vehicle",
#     "motorcycle",
#     "pedestrian",
#     "traffic_cone",
#     "trailer",
#     "truck",
#     "driveable_surface",
#     "other_flat",
#     "sidewalk",
#     "terrain",
#     "manmade",
#     "vegetation",
# ]
class_names_dict = {
    0: "unlabelled",
    1: "barrier",
    2: "bicycle",
    3: "bus",
    4: "car",
    5: "construction_vehicle",
    6: "motorcycle",
    7: "pedestrian",
    8: "traffic_cone",
    9: "trailer",
    10: "truck",
    11: "driveable_surface",
    12: "other_flat",
    13: "sidewalk",
    14: "terrain",
    15: "manmade",
    16: "vegetation",
}

class NuScenesDataset(Dataset):
    def __init__(self, 
                 split, 
                 data_root='../data/nuscenes',
                 ignore_index=-1, 
                 **kwargs):
        self.sweeps = 10
        self.ignore_index = ignore_index
        self.split = split
        self.data_root = data_root
        self.learning_map = self.get_learning_map(ignore_index)
        self.class_name_dict = class_names_dict
        self.data_list = self.get_data_list()

        super().__init__()

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        if split == "train":
            return os.path.join(
                "../data/infos", f"nuscenes_infos_{self.sweeps}sweeps_train.pkl"
            )
        elif split == "val":
            return os.path.join(
                "../data/infos", f"nuscenes_infos_{self.sweeps}sweeps_val.pkl"
            )
        elif split == "test":
            return os.path.join(
                "../data/infos", f"nuscenes_infos_{self.sweeps}sweeps_test.pkl"
            )
        else:
            raise NotImplementedError

    def get_data_list(self):
        if isinstance(self.split, str):
            info_paths = [self.get_info_path(self.split)]
        elif isinstance(self.split, Sequence):
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        for info_path in info_paths:
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                data_list.extend(info)
        return data_list

    def get_data(self, idx):
        input_dict = dict()
        data = self.data_list[idx % len(self.data_list)]
        lidar_path = os.path.join(self.data_root, data["lidar_path"])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )
        points = points[:, :4]

        if "gt_segment_path" in data.keys():
            gt_segment_path = os.path.join(
                self.data_root, data["gt_segment_path"]
            )
            segment = np.fromfile(
                str(gt_segment_path), dtype=np.uint8, count=-1
            ).reshape([-1])
            segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(
                np.int64
            ) + 1
        else:
            segment = np.ones((points.shape[0],), dtype=np.int64)
        segment = segment.reshape(-1, 1)
        points = np.concatenate(
            [points, segment], axis=1
        )
        xyzrdm = common.load_points_as_images(points=points, scan_unfolding=False, H=32, W=1024,
                                       min_depth=1.45, max_depth=80.0, fov_up=10.0, fov_down=-30.0, custom_feat_dim=1)
        xyzrdm = xyzrdm.transpose(2, 0, 1)
        xyzrdm *= xyzrdm[[-1]]
        input_dict.update({
            "xyz": xyzrdm[:3],
            "reflectance": xyzrdm[[3]]/255,
            "segment": xyzrdm[[4]][0],
            "depth": xyzrdm[[-2]],
            "mask": xyzrdm[[-1]],
        })

        for key, val in input_dict.items():
            if isinstance(val, np.ndarray):
                input_dict[key] = torch.from_numpy(val)

        return input_dict

    def get_data_name(self, idx):
        # return data name for lidar seg, optimize the code when need to support detection
        return self.data_list[idx % len(self.data_list)]["lidar_token"]

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,
            1: ignore_index,
            2: 6,
            3: 6,
            4: 6,
            5: ignore_index,
            6: 6,
            7: ignore_index,
            8: ignore_index,
            9: 0,
            10: ignore_index,
            11: ignore_index,
            12: 7,
            13: ignore_index,
            14: 1,
            15: 2,
            16: 2,
            17: 3,
            18: 4,
            19: ignore_index,
            20: ignore_index,
            21: 5,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            27: 13,
            28: 14,
            29: ignore_index,
            30: 15,
            31: ignore_index,
        }
        return learning_map

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        return data_dict

    def __getitem__(self, idx):

        return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def map(label, mapdict=color_map):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]

if __name__ == "__main__":
    dataset = NuScenesDataset(split="train", data_root="../data/nuscenes")
    print(f"Number of samples in the dataset: {len(dataset)}")
    sample = dataset.get_data(0)
    print(f"Sample data: {sample}")
    print(f"Sample name: {dataset.get_data_name(0)}")
    print(f"Learning map: {dataset.learning_map}")