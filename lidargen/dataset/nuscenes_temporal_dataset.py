import os
import pickle
import numpy as np
from loguru import logger

from .transforms_3d import common
from .base_dataset import DatasetBase
from . import utils
from ..ops.roiaware_pool3d import roiaware_pool3d_utils


class NuscTempDataset(DatasetBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.points_range = [-80,-80,-8,80,80,8]

    def prepare_data(self):
        assert self.pkl_path is not None
        if not isinstance(self.pkl_path, list):
            with open(self.pkl_path, 'rb') as f:
                data_infos = pickle.load(f)
            data_infos = [di for di in data_infos if di['scene_graph']['valid']]

            self.data_dicts = {}
            for data_info in data_infos:
                self.data_dicts[data_info['token']] = data_info

            self.data = [di for di in data_infos if di['prev_info']['valid']]
            self.data = [di for di in self.data if self.data_dicts.get(di['prev_info']['token'], None) is not None]

            logger.info(f"Load {len(self.data)} / {len(data_infos)} data from {self.pkl_path}")

        else:
            data_infos = []
            for pkl_path in self.pkl_path:
                with open(self.pkl_path, 'rb') as f:
                    sub_data_infos = pickle.load(f)
                data_infos.extend([di for di in sub_data_infos if di['scene_graph']['valid']])

            self.data_dicts = {}
            for data_info in data_infos:
                self.data_dicts[data_info['token']] = data_info
                
            self.data = [di for di in data_infos if di['prev_info']['valid']]
            self.data = [di for di in self.data if self.data_dicts.get(di['prev_info']['token'], None) is not None]
            logger.info(f"Load {len(self.data)} / {len(data_infos)} data from {self.pkl_path}")


    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.cfg.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.cfg.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.cfg.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.cfg.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()

        cls_infos_new = {name: [] for name in self.cfg.class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.cfg.class_names:
                    cls_infos_new[name].append(info)

        return sampled_infos

    def __getitem__(self, idx):

        return self.__getitem_pkl__(idx)

    def __getitem_ori__(self, idx):
        data_path = self.data[idx]
        # lidar point cloud
        xyzrdm = common.load_points_as_images(data_path, scan_unfolding=self.cfg.scan_unfolding, H=self.cfg.resolution[0], W=self.cfg.resolution[1],
                                       min_depth=self.cfg.min_depth, max_depth=self.cfg.max_depth, fov_up=self.cfg.fov_up, fov_down=self.cfg.fov_down)
        xyzrdm = xyzrdm.transpose(2, 0, 1)
        xyzrdm *= xyzrdm[[5]]
        return {
            "xyz": xyzrdm[:3],
            "reflectance": xyzrdm[[3]]/255,
            "depth": xyzrdm[[4]],
            "mask": xyzrdm[[5]],
        }
    
    def scale_boxes_3d(self, boxes_3d):
        box_num = boxes_3d.shape[0]
        scaled_boxes = np.zeros([box_num, boxes_3d.shape[-1]+1])
        x_min, y_min, z_min, x_max, y_max, z_max = self.points_range
        boxes_3d[:,0] = (boxes_3d[:,0] - 0) / (0 - x_min)
        boxes_3d[:,1] = (boxes_3d[:,1] - 0) / (0 - y_min)
        boxes_3d[:,2] = (boxes_3d[:,2] - 0) / (0 - z_min)
        boxes_3d[:,3:6] = np.log(boxes_3d[:,3:6]+1e-6)
        scaled_boxes[:,:6] = boxes_3d[:,:6]
        scaled_boxes[:,6] = np.sin(boxes_3d[:,6])
        scaled_boxes[:,7] = np.cos(boxes_3d[:,6])
        if boxes_3d.shape[-1] > 7:
            scaled_boxes[:,8:] = boxes_3d[:, 7:]
        return scaled_boxes
    
    def unscale_boxes_3d(self, boxes_3d_traj):
        box_num = boxes_3d_traj.shape[0]
        scaled_boxes = boxes_3d_traj[:, :8].detach().cpu().numpy()
        unscaled_boxes = np.zeros([box_num, 7])
        boxes_trajs = boxes_3d_traj[:, 8:].detach().cpu().numpy()

        x_min, y_min, z_min, x_max, y_max, z_max = self.points_range
        unscaled_boxes[:,0] = scaled_boxes[:,0] * (0 - x_min)
        unscaled_boxes[:,1] = scaled_boxes[:,1] * (0 - y_min)
        unscaled_boxes[:,2] = scaled_boxes[:,2] * (0 - z_min)
        unscaled_boxes[:,3:6] = np.exp(scaled_boxes[:,3:6])
        unscaled_boxes[:,6] = np.arctan2(scaled_boxes[:,6],  scaled_boxes[:,7])
        unscaled_boxes[0, :] = 0
        return unscaled_boxes, boxes_trajs.reshape(box_num, 6, 2)

    def allign_box_num(self, bbox_3d, bbox_2d, expet_box_num=13):
        # convert box to set number 13
        box_num = bbox_3d.shape[0]
        if box_num > expet_box_num:
            new_gt_boxes_3d = bbox_3d[:expet_box_num,:]
            new_gt_boxes_2d = bbox_2d[:expet_box_num,:]
            is_valid_obj = np.ones([expet_box_num,])
        else:
            new_gt_boxes_3d = np.zeros([expet_box_num, bbox_3d.shape[-1]])
            new_gt_boxes_2d = np.zeros([expet_box_num, bbox_2d.shape[-1]])
            new_gt_boxes_3d[:box_num, :] = bbox_3d
            new_gt_boxes_2d[:box_num, :] = bbox_2d
            is_valid_obj = np.zeros([expet_box_num,])
            is_valid_obj[:box_num] = 1
        return new_gt_boxes_3d, new_gt_boxes_2d, is_valid_obj

    def pre_process(self, data_dict):
        class_names = ['ego'] + list(self.cfg.class_names)
        gt_classes = np.array([class_names.index(n) for n in data_dict['gt_names']], dtype=np.int32)
        gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
        data_dict['gt_boxes'] = gt_boxes

        # 3D --> 2D
        gt_boxes_2d = common.convert_boxes_to_2d(
            boxes_3d=gt_boxes,
            H=self.cfg.resolution[0],
            W=self.cfg.resolution[1],
            min_depth=self.cfg.min_depth,
            max_depth=self.cfg.max_depth,
            fov_up=self.cfg.fov_up, 
            fov_down=self.cfg.fov_down
        )

        scaled_gt_boxes_3d = self.scale_boxes_3d(gt_boxes.copy())
        data_dict['gt_boxes_2d'] = gt_boxes_2d
        data_dict['scaled_gt_boxes'] = scaled_gt_boxes_3d
        # TODO: For box condition
        # input_boxes_3d, input_boxes_2d, is_valid_obj = self.allign_box_num(scaled_gt_boxes_3d, gt_boxes_2d)
        # data_dict['scaled_gt_boxes'] = input_boxes_3d
        # data_dict['gt_boxes_2d'] = input_boxes_2d
        # data_dict['is_valid_obj'] = is_valid_obj

        return data_dict

    def get_prev_points(self, prev_data_info, prev_info):
        pre_lidar_path = os.path.join(self.data_root, prev_data_info['lidar_path'])
        prev_points = np.fromfile(pre_lidar_path, dtype=np.float32).reshape(-1, 5)[:,:4]
        prev_points_labels = np.zeros([prev_points.shape[0], 1], dtype=np.float32)
        # split foreground and background points
        class_names = list(self.cfg.class_names)
        prev_gt_names = prev_data_info['scene_graph']['keep_box_names'][1:]
        prev_gt_boxes = prev_data_info['scene_graph']['keep_box'][1:, :7]
        prev_gt_classes = np.array([class_names.index(n) for n in prev_gt_names], dtype=np.int32) + 1

        boxes3d, is_numpy = utils.check_numpy_to_torch(prev_gt_boxes)
        points, is_numpy = utils.check_numpy_to_torch(prev_points)
        point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes3d) # MxN

        for box_id, gt_class in enumerate(prev_gt_classes):
            prev_points_labels[point_masks[box_id, :] > 0, 0] = gt_class

        prev_points = np.concatenate((prev_points, prev_points_labels), axis=1)

        lidar2sensor = np.eye(4)
        rot = prev_info['sensor2lidar_rotation']
        trans = prev_info['sensor2lidar_translation']
        lidar2sensor[:3, :3] = rot.T
        lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
        prev_points[:, :
                        3] = prev_points[:, :3] @ lidar2sensor[:3, :3]
        prev_points[:, :3] -= lidar2sensor[:3, 3]

        return prev_points

    def get_prev_as_condition(self, curr_data_info):
        prev_data_info = self.data_dicts[curr_data_info['prev_info']['token']]
        prev_info = curr_data_info['prev_info']
        prev_points = self.get_prev_points(prev_data_info, prev_info)
        xyzrdm = common.load_points_as_images(points=prev_points, scan_unfolding=self.cfg.scan_unfolding, H=self.cfg.resolution[0], W=self.cfg.resolution[1],
                                       min_depth=self.cfg.min_depth, max_depth=self.cfg.max_depth, fov_up=self.cfg.fov_up, fov_down=self.cfg.fov_down, custom_feat_dim=1)
        xyzrdm = xyzrdm.transpose(2, 0, 1)
        xyzrdm *= xyzrdm[[-1]]
        return xyzrdm


    def __getitem_pkl__(self, idx):
        input_dict = dict()
        data_info = self.data[idx]
        lidar_path = os.path.join(self.data_root, data_info['lidar_path'])
        # points
        xyzrdm = common.load_points_as_images(lidar_path, scan_unfolding=self.cfg.scan_unfolding, H=self.cfg.resolution[0], W=self.cfg.resolution[1],
                                       min_depth=self.cfg.min_depth, max_depth=self.cfg.max_depth, fov_up=self.cfg.fov_up, fov_down=self.cfg.fov_down)
        xyzrdm = xyzrdm.transpose(2, 0, 1)
        xyzrdm *= xyzrdm[[5]]
        input_dict.update({
            "xyz": xyzrdm[:3],
            "reflectance": xyzrdm[[3]]/255,
            "depth": xyzrdm[[4]],
            "mask": xyzrdm[[5]],
        })

        # prev points
        prev_cond = self.get_prev_as_condition(data_info)
        input_dict.update({
            "prev_cond": prev_cond
        })
        return input_dict
    