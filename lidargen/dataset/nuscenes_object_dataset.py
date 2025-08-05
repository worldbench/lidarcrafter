import os
import pickle
import numpy as np
from loguru import logger
import random
from .base_dataset import DatasetBase
from . import utils

class NuscObjectDataset(DatasetBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.points_range = [-80,-80,-8,80,80,8]
        self.num_samples = cfg.num_samples

    def prepare_data(self):
        assert self.pkl_path is not None
        fg_objects_file = open(self.pkl_path, 'rb')
        fg_objects_dict = pickle.load(fg_objects_file)
        self.data = []
        self.class_samples = []
        for class_idx, class_name in enumerate(self.cfg.class_names):
            fg_objects_list = fg_objects_dict[class_name]
            self.data.extend(fg_objects_list)
            self.class_samples.extend([class_idx]*len(fg_objects_list))

        combined = list(zip(self.data, self.class_samples))
        random.shuffle(combined)
        self.data, self.class_samples = zip(*combined)
        logger.info(f"Load {len(self.data)} data from {self.pkl_path}")

    def __getitem__(self, idx):

        return self.__getitem_pkl__(idx)
    
    def load_points(self, fg_path):
        fg_path = os.path.join(self.data_root, fg_path)
        fg_points = np.fromfile(fg_path, dtype=np.float32).reshape(-1,5)[:,:4]
        return fg_points

    def norm_fg_points(self, fg_points, box3d):
        rotation = -np.array([box3d[-1]])
        fg_points = utils.rotate_points_along_z(fg_points[np.newaxis, :, :], rotation)[0]
        fg_points[:,0] = 2 * fg_points[:,0] / box3d[3]
        fg_points[:,1] = 2 * fg_points[:,1] / box3d[4]
        fg_points[:,2] = 2 * fg_points[:,2] / box3d[5]
        intensity = fg_points[:,3] / 255
        fg_points[:,3] = 2*intensity - 1
        return fg_points

    def encoding_boxes_3d(self, boxes_3d):
        condtion_box = np.zeros((6), dtype=np.float32) # d, z, w, h, l, sin(yaw-alpha), cos(yaw-alpha)
        x, y, z, w, h, l, yaw = boxes_3d
        unique_alpha = yaw - np.arctan2(y, x)

        x_min, y_min, z_min, x_max, y_max, z_max = self.points_range
        x_norm = (x - x_min) / (x_max - x_min)
        y_norm = (y - y_min) / (y_max - y_min)
        z_norm = (z - z_min) / (z_max - z_min)
        condtion_box[0] = np.linalg.norm(np.array([x_norm, y_norm]), ord=2, axis=0, keepdims=True)
        condtion_box[1] = z_norm
        condtion_box[2:5] = np.log(np.array([w, h, l]) + 1e-6)
        condtion_box[5] = unique_alpha
        # condtion_box[5] = np.sin(np.array([unique_alpha]))
        # condtion_box[6] = np.cos(np.array([unique_alpha]))
        return condtion_box

    def norm_fg_points_box(self, fg_points, box3d):
        fg_points = self.norm_fg_points(fg_points, box3d)
        fg_encoding_box = self.encoding_boxes_3d(box3d)
        return fg_points, fg_encoding_box

    def sample_points(self, points):

        N = len(points)
        
        if N <= self.num_samples:
            indices = np.random.choice(N, self.num_samples, replace=True)
            return points[indices]
        
        pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
        pts_near_flag = pts_depth < 0.1
        far_idxs_choice = np.where(pts_near_flag == 0)[0]
        near_idxs = np.where(pts_near_flag == 1)[0]
        choice = []
        if self.num_samples > len(far_idxs_choice):
            near_idxs_choice = np.random.choice(near_idxs, self.num_samples - len(far_idxs_choice), replace=False)
            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
        else: 
            choice = np.arange(0, len(points), dtype=np.int32)
            choice = np.random.choice(choice, self.num_samples, replace=False)
        np.random.shuffle(choice)
        return points[choice]

    def check_valid(self, fg_info):
        if fg_info['num_points_in_gt'] < 50:
            return False
        # box range
        box3d = fg_info['box3d_lidar'][:7]
        if not (self.points_range[0] <= box3d[0] <= self.points_range[3] and
                self.points_range[1] <= box3d[1] <= self.points_range[4] and
                self.points_range[2] <= box3d[2] <= self.points_range[5]):
            return False

        return True

    def __getitem_pkl__(self, idx):
        data_dict = {}
        fg_info = self.data[idx]
        if not self.check_valid(fg_info):
            random_index = random.randint(0, self.__len__()-1)
            return self.__getitem__(random_index)
        
        fg_points_path = fg_info['path']
        fg_points = self.load_points(fg_points_path)
        box3d = fg_info['box3d_lidar'][:7]
        fg_points, fg_encoding_box = self.norm_fg_points_box(fg_points, box3d)
        fg_points = self.sample_points(fg_points)
        data_dict.update({
            'fg_encoding_box': fg_encoding_box,
            'fg_points': fg_points,
            'fg_class': np.array([self.class_samples[idx]])
        })
        return data_dict