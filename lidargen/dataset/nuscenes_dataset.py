import os
import json
import pickle
import random
from pathlib import Path
import numpy as np
from loguru import logger

import torch
import clip
from pyquaternion import Quaternion

from .transforms_3d import common
from .transforms_3d.scene_graph.scene_graph import SceneGraphAssigner 
from .base_dataset import DatasetBase
from . import utils
from ..ops.roiaware_pool3d import roiaware_pool3d_utils
class NuscDataset(DatasetBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.points_range = [-80,-80,-8,80,80,8]
        self.learning_map = self.get_learning_map(-1)

        if not hasattr(self, 'scene_graph_assigner'):
            self.scene_graph_assigner = SceneGraphAssigner(
                # output_path=Path(self.data_root).parent / 'clips' / 'nuscenes',
                # output_path = Path('/dataset/shared/nuscenes') / 'clips' / 'nuscenes',
                output_path = Path('../data') / 'clips' / 'nuscenes',
                relationship_file_path='../lidargen/dataset/transforms_3d/scene_graph/relationships.txt',
                classnames_file_path='../lidargen/dataset/transforms_3d/scene_graph/classes_nuscenes.txt',
                split=self.split
            ) # TODO: For first generate scene graph

    def prepare_data(self):
        if self.pkl_path is None:
            if self.split == 'train':
                with open(os.path.join(self.data_root, 'v1.0-trainval/sample_data.json')) as f:
                    sample_data = json.load(f)
            else:
                with open(os.path.join(self.data_root, 'v1.0-mini/sample_data.json')) as f:
                    sample_data = json.load(f)

            custom_path = 'v1.0-trainval'
            file_paths = [os.path.join(self.data_root, x['filename']) 
                            for x in sample_data 
                            if 'samples/LIDAR_TOP' in x['filename']]
            self.data = sorted(file_paths)

        else:
            if isinstance(self.pkl_path, list):
                self.data = []
                for pkl_path in self.pkl_path:
                    with open(pkl_path, 'rb') as f:
                        data_infos = pickle.load(f)
                    self.data += [di for di in data_infos if di['scene_graph']['valid']]
                # shuffle
                random.shuffle(self.data)
            else:
                with open(self.pkl_path, 'rb') as f:
                    data_infos = pickle.load(f)
                self.data = [di for di in data_infos if di['scene_graph']['valid']]
            logger.info(f"Load {len(self.data)} / {len(data_infos)} data from {self.pkl_path}")

        if getattr(self.cfg, 'num_sample', None) is not None:
            self.data = self.data[:self.cfg.num_sample]

        if self.task == 'autoregressive_generation':
            self.data_dict = {}
            for info in self.data:
                token = info['token']
                self.data_dict[token] = info
            self.data = [di for di in self.data if di['prev_info']['valid']]
            self.data = [di for di in self.data if self.data_dict.get(di['prev_info']['token'], None) is not None]
            logger.info(f"Load {len(self.data)} / {len(data_infos)} data from {self.pkl_path}")

        # if self.split in ['train', 'all']:
        #     self.data = self.balanced_infos_resampling(self.data)

    def update_data_with_custom_tokens(self, custom_token_dict):
        needed_data_infos = []
        data_infos_dict = {di['token']: di for di in self.data}
        for first_token, all_tokens in custom_token_dict.items():
            if first_token in data_infos_dict:
                data_info = data_infos_dict[first_token]
                data_info.update({
                    'custom_tokens': all_tokens,
                })
                needed_data_infos.append(data_info)
        self.data = needed_data_infos

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.cfg.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.cfg.class_names}
        for info in infos:
            for name in set(info['scene_graph']['keep_box_names']):
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
            for name in set(info['scene_graph']['keep_box_names']):
                if name in self.cfg.class_names:
                    cls_infos_new[name].append(info)

        return sampled_infos

    def __getitem__(self, idx):
        if self.pkl_path is None:
            return self.__getitem_ori__(idx)
        else:
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

    def allign_box_num(self, bbox_3d, bbox_2d, fg_encoding_box, expet_box_num=13):
        # convert box to set number 13
        box_num = bbox_3d.shape[0]
        if box_num > expet_box_num:
            new_gt_boxes_3d = bbox_3d[:expet_box_num,:]
            new_gt_boxes_2d = bbox_2d[:expet_box_num,:]
            new_fg_encoding_box = fg_encoding_box[:expet_box_num, :]
            is_valid_obj = np.ones([expet_box_num,])
        else:
            new_gt_boxes_3d = np.zeros([expet_box_num, bbox_3d.shape[-1]])
            new_gt_boxes_2d = np.zeros([expet_box_num, bbox_2d.shape[-1]])
            new_fg_encoding_box = np.zeros([expet_box_num, fg_encoding_box.shape[-1]])
            new_gt_boxes_3d[:box_num, :] = bbox_3d
            new_gt_boxes_2d[:box_num, :] = bbox_2d
            new_fg_encoding_box[:box_num, :] = fg_encoding_box
            is_valid_obj = np.zeros([expet_box_num,])
            is_valid_obj[:box_num] = 1
        return new_gt_boxes_3d, new_gt_boxes_2d, new_fg_encoding_box, is_valid_obj

    def encoding_boxes_3d(self, boxes_3d, unique_mode=True):
        condtion_box = np.zeros((8), dtype=np.float32) # d, z, w, h, l, theta, sin(yaw), cos(yaw)
        x, y, z, w, h, l, yaw = boxes_3d

        x_min, y_min, z_min, x_max, y_max, z_max = self.points_range
        x_norm = (x - 0) / (0 - x_min)
        y_norm = (y - 0) / (0 - y_min)
        z_norm = (z - 0) / (0 - z_min)
        condtion_box[0] = np.linalg.norm(np.array([x_norm, y_norm]), ord=2, axis=0, keepdims=True)
        condtion_box[1] = z_norm
        condtion_box[2:5] = np.log(np.array([w, h, l]) + 1e-6)
        if unique_mode:
            unique_alpha = yaw - np.arctan2(y, x)
            condtion_box[5] = unique_alpha
            return condtion_box[:6]
        else:
            condtion_box[5] = (-np.arctan2(y, x) / np.pi + 1) / 2 % 1
            condtion_box[6] = np.sin(yaw)
            condtion_box[7] = np.cos(yaw)
            return condtion_box

    def unscaled_objs_3d(self, sample_index, custom_data_dict, generated_object_points, w_semantic=False):
        unscaled_objs_list = []
        data_info = self.data[sample_index]
        if custom_data_dict is not None:
            gt_boxes = custom_data_dict['gt_boxes'][1:, :7]
        else:
            gt_boxes = data_info['scene_graph']['keep_box'][1:, :7]
        if w_semantic:
            if custom_data_dict is not None:
                gt_names = custom_data_dict['gt_names'][1:]
            else:
                gt_names = data_info['scene_graph']['keep_box_names'][1:]
            gt_classes = np.array([self.cfg.class_names.index(n) for n in gt_names], dtype=np.int32) + 1

        assert gt_boxes.shape[0] == generated_object_points.shape[0]
        for box_id, box3d in enumerate(gt_boxes):
            obj_points = generated_object_points[box_id]
            obj_points[:, 0] = obj_points[:, 0] * box3d[3] / 2.0
            obj_points[:, 1] = obj_points[:, 1] * box3d[4] / 2.0
            obj_points[:, 2] = obj_points[:, 2] * box3d[5] / 2.0
            obj_points[:, 3] = 255*(obj_points[:, 3]+1) / 2
            yaw = box3d[6].reshape(1)
            obj_points = utils.rotate_points_along_z(obj_points.reshape(1,-1,4), yaw)[0]
            obj_points[:,:3] = obj_points[:,:3] + box3d[:3].reshape(1, 3)
            if w_semantic:
                obj_points = np.hstack((obj_points, np.full((obj_points.shape[0], 1), gt_classes[box_id])))
            unscaled_objs_list.append(obj_points)
        unscaled_objs = np.vstack(unscaled_objs_list)
        return unscaled_objs


    def distille_local_boxes(self, data_dict, unique_mode=True):
        fg_boxes = data_dict['gt_boxes'][1:]
        fg_names = data_dict['gt_names'][1:]
        fg_class = np.array([self.cfg.class_names.index(name) for name in fg_names])
        # points = data_dict['points']
        # box_points_list = utils.distille_local_boxes(gt_boxes, points)
        encoding_boxes_list = []
        for box_id, box in enumerate(fg_boxes):
            encoding_boxes_list.append(self.encoding_boxes_3d(box[:7], unique_mode))
        data_dict['fg_encoding_box'] = np.stack(encoding_boxes_list, axis=0)
        data_dict['fg_class'] = fg_class
        return data_dict

    def delete_ground(self, points, gt_segment_path):
        gt_segment_path = os.path.join(self.data_root, gt_segment_path)
        segment = np.fromfile(
            str(gt_segment_path), dtype=np.uint8, count=-1
        ).reshape([-1])
        segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(
            np.int64
        )
        mask = np.logical_and(segment != 10, segment != 12)  # Remove ground points
        points = points[mask]
        return points

    def remove_ego_points(self, points, center_radius=1.0):
        mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
        return points[mask]  

    def get_prev_frame_condition(self, curr_frame_token, prev_info, prev_num=1):
        prev_frame_token = prev_info['token']
        prev_data_dict = self.data_dict[prev_frame_token]
        for _ in range(prev_num-1):
            prev_info = prev_data_dict['prev_info']
            if prev_info['valid']:
                prev_frame_token = prev_info['token']
                if prev_frame_token in self.data_dict:
                    prev_data_dict = self.data_dict[prev_frame_token]
                else:
                    break
            else:
                break
        # split points to background and foreground
        prev_lidar_path = os.path.join(self.data_root, prev_data_dict['lidar_path'])
        prev_points = np.fromfile(prev_lidar_path, dtype=np.float32).reshape(-1, 5)[:,:4]
        prev_points = self.remove_ego_points(prev_points, center_radius=2.0)
        prev_gt_boxes_3d = prev_data_dict['scene_graph']['keep_box'][1:, :7]
        prev_gt_instance_inds = prev_data_dict['scene_graph']['keep_agent_instance_inds'][1:].tolist()
        curr_gt_instance_inds = self.data_dict[curr_frame_token]['scene_graph']['keep_agent_instance_inds'][1:].tolist()
        curr_gt_boxes_3d = self.data_dict[curr_frame_token]['scene_graph']['keep_box'][1:, :7]

        alligned_prev_gt_boxes_3d = []
        keeped_prev_gt_boxes_mask = np.zeros((len(prev_gt_instance_inds),), dtype=np.bool_)
        for prev_box_id, global_instance_id in enumerate(prev_gt_instance_inds):
            if global_instance_id in curr_gt_instance_inds:
                curr_box_id = curr_gt_instance_inds.index(global_instance_id)
                alligned_prev_gt_boxes_3d.append(curr_gt_boxes_3d[curr_box_id])
                keeped_prev_gt_boxes_mask[prev_box_id] = True
        
        if keeped_prev_gt_boxes_mask.sum() > 0:
            keeped_prev_gt_boxes_3d = prev_gt_boxes_3d[keeped_prev_gt_boxes_mask]
            boxes3d_tensor, _ = utils.check_numpy_to_torch(keeped_prev_gt_boxes_3d)
            points_tensor, _ = utils.check_numpy_to_torch(prev_points)
            point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points_tensor[:, 0:3], boxes3d_tensor) # MxN
            background_mask = point_masks.sum(axis=0) == 0
            prev_background_points = prev_points[background_mask]

            align_prev_obj_points = []
            align_prev_obj_intensity = []

            # extract each obj points
            for obj_id, obj_box_3d in enumerate(keeped_prev_gt_boxes_3d):
                obj_points = prev_points[point_masks[obj_id, :] > 0]
                x, y, z, w, l, h, yaw = obj_box_3d
                align_prev_obj_intensity.append(obj_points[:, 3])
                obj_points = obj_points[:, :3]
                obj_points_origin = obj_points - np.array([x, y, z])  # center to origin
                rotation = -np.array(yaw)
                allign_obj_points = utils.rotate_points_along_z(obj_points_origin[np.newaxis, :, :], rotation.reshape(1))[0]
                align_prev_obj_points.append(allign_obj_points)

        else:
            prev_background_points = prev_points

        lidar2sensor = np.eye(4)
        # rot = prev_info['sensor2lidar_rotation']
        # trans = prev_info['sensor2lidar_translation']
        l2e_r_s = prev_data_dict['lidar2ego_rotation']
        l2e_t_s = prev_data_dict['lidar2ego_translation']
        e2g_r_s = prev_data_dict['ego2global_rotation']
        e2g_t_s = prev_data_dict['ego2global_translation']
        l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
        e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

        e2g_t = self.data_dict[curr_frame_token]['ego2global_translation']
        e2g_r_mat = Quaternion(self.data_dict[curr_frame_token]['ego2global_rotation']).rotation_matrix
        l2e_r_mat = Quaternion(self.data_dict[curr_frame_token]['lidar2ego_rotation']).rotation_matrix
        l2e_t = self.data_dict[curr_frame_token]['lidar2ego_translation']

        R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                    ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
        rot = R.T
        trans = T

        lidar2sensor[:3, :3] = rot.T
        lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
        prev_background_points[:, :
                        3] = prev_background_points[:, :3] @ lidar2sensor[:3, :3]
        prev_background_points[:, :3] -= lidar2sensor[:3, 3]

        if keeped_prev_gt_boxes_mask.sum() > 0:
            curr_fg_points = []
            for obj_id, obj_box_3d in enumerate(alligned_prev_gt_boxes_3d):
                x, y, z, w, l, h, yaw = obj_box_3d
                rotation = np.array(yaw)
                roatated_obj_points = utils.rotate_points_along_z(align_prev_obj_points[obj_id][np.newaxis, :, :], rotation.reshape(1))[0]
                roatated_trans_obj_points = roatated_obj_points + np.array([x, y, z])[None, :]
                roatated_trans_obj_points = np.concatenate([roatated_trans_obj_points, align_prev_obj_intensity[obj_id][:, np.newaxis]], axis=-1)  # (M, 4)
                curr_fg_points.append(roatated_trans_obj_points)
            curr_fg_points = np.concatenate(curr_fg_points, axis=0)
            fut_all_points = np.concatenate([prev_background_points, curr_fg_points], axis=0)
            return fut_all_points
        else:
            return prev_background_points

    def pre_process(self, data_dict):
        if self.task is None:
            pass
        elif self.task == 'object_generation':
            data_dict = self.distille_local_boxes(data_dict)
        
        else:
            data_dict = self.distille_local_boxes(data_dict, unique_mode=False)
            data_dict.pop('fg_class', None)
            class_names = ['ego'] + list(self.cfg.class_names)
            gt_classes = np.array([class_names.index(n) for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

            # 3D --> 2D
            gt_boxes_2d, condition_mask, scene_loss_weight_map = common.convert_boxes_to_2d(
                boxes_3d=gt_boxes,
                H=self.cfg.resolution[0],
                W=self.cfg.resolution[1],
                min_depth=self.cfg.min_depth,
                max_depth=self.cfg.max_depth,
                fov_up=self.cfg.fov_up, 
                fov_down=self.cfg.fov_down
            )

            scaled_gt_boxes_3d = self.scale_boxes_3d(gt_boxes.copy())
            if self.task == 'layout_generation':
                # TODO: For scene graph: with ego
                data_dict['gt_boxes_2d'] = gt_boxes_2d
                data_dict['scaled_gt_boxes'] = scaled_gt_boxes_3d
                data_dict['condition_mask'] = condition_mask
                data_dict['scene_loss_weight_map'] = scene_loss_weight_map

            elif self.task in ['layout_cond', 'autoregressive_generation']:
                # TODO: For box condition
                input_boxes_3d, input_boxes_2d, fg_encoding_box, is_valid_obj = self.allign_box_num(scaled_gt_boxes_3d[1:], gt_boxes_2d[1:], data_dict['fg_encoding_box'])
                data_dict['scaled_gt_boxes'] = input_boxes_3d
                data_dict['fg_encoding_box'] = fg_encoding_box
                data_dict['gt_boxes_2d'] = input_boxes_2d
                data_dict['is_valid_obj'] = is_valid_obj
                data_dict['condition_mask'] = condition_mask
                data_dict['scene_loss_weight_map'] = scene_loss_weight_map
            else:
                pass

        data_dict.pop('points', None)
        return data_dict

    def __getitem_pkl__(self, idx):
        input_dict = dict()
        data_info = self.data[idx]
        input_dict.update({'token': data_info['token']})
        if self.task == 'autoregressive_generation':
            if self.split in ['train', 'all']:
                # random int from 1-5
                prev_num = random.randint(1, 5)
            else:
                prev_num = 1
            prev_cond_points = self.get_prev_frame_condition(data_info['token'], data_info['prev_info'], prev_num=prev_num)
            xyzrdm = common.load_points_as_images(points=prev_cond_points, scan_unfolding=self.cfg.scan_unfolding, H=self.cfg.resolution[0], W=self.cfg.resolution[1],
                                        min_depth=self.cfg.min_depth, max_depth=self.cfg.max_depth, fov_up=self.cfg.fov_up, fov_down=self.cfg.fov_down)
            xyzrdm = xyzrdm.transpose(2, 0, 1)
            xyzrdm *= xyzrdm[[5]]
            reflectance = xyzrdm[[3]]/255
            depth = xyzrdm[[4]]
            # add some noise to depth
            if self.split in ['train', 'all']:
                depth += np.random.normal(0, 3, depth.shape)
                mask = depth < 0
                depth[mask] = 0
            input_dict.update({
                "autoregressive_cond": np.concatenate([depth, reflectance], axis=0)
            })

        lidar_path = os.path.join(self.data_root, data_info['lidar_path'])
        # points
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:,:4]
        input_dict.update({'points': points})

        # boxes 3d
        input_dict.update({
            'gt_names': data_info['scene_graph']['keep_box_names'],
            'gt_boxes': data_info['scene_graph']['keep_box'][:, :7],
            'gt_box_relationships': data_info['scene_graph']['keep_box_relationships'],
            'gt_fut_trajs' : data_info['scene_graph']['keep_agent_fut_trajs'],
            'gt_fut_masks' : data_info['scene_graph']['keep_agent_fut_masks'],
            'gt_fut_states' : data_info['scene_graph']['keep_agent_fut_states']
        })

        if self.data_augmentor is not None:# and self.split == 'train':
            gt_boxes_mask = np.array([True for _ in range(input_dict['gt_boxes'].shape[0])], dtype=np.bool_)
            input_dict = self.data_augmentor.forward(
                data_dict={
                    **input_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
        # points2range
        points = input_dict['points']
        if getattr(self.cfg, 'delete_ground', False):
            points = self.delete_ground(points, data_info['gt_segment_path'])
        xyzrdm = common.load_points_as_images(points=points, scan_unfolding=self.cfg.scan_unfolding, H=self.cfg.resolution[0], W=self.cfg.resolution[1],
                                       min_depth=self.cfg.min_depth, max_depth=self.cfg.max_depth, fov_up=self.cfg.fov_up, fov_down=self.cfg.fov_down)
        xyzrdm = xyzrdm.transpose(2, 0, 1)
        xyzrdm *= xyzrdm[[5]]
        input_dict.update({
            "xyz": xyzrdm[:3],
            "reflectance": xyzrdm[[3]]/255,
            "depth": xyzrdm[[4]],
            "mask": xyzrdm[[5]],
        })

        input_dict = self.pre_process(input_dict)
        if self.task == 'layout_generation':
            scenegraph_out = self.scene_graph_assigner.assign_item(idx, input_dict)
            input_dict.update({
                "custom_dict": scenegraph_out
            })

        # modify trajectories
        # gt_fut_trajs = input_dict['gt_fut_trajs']
        # N, T, _ = gt_fut_trajs.shape  # (N, T, 2)
        # gt_fut_trajs = np.concatenate([np.zeros((N,1,2)), gt_fut_trajs], axis=1)  # (N+1,2)
        # deltas = gt_fut_trajs[:,1:] - gt_fut_trajs[:,:-1]                   # (N,2)
        # # set delta < 0.1 to zero
        # deltas[np.linalg.norm(deltas, axis=-1) < 0.1] = 0
        # gt_fut_trajs = np.cumsum(deltas, axis=1)  # (N+1,2)
        # input_dict['gt_fut_trajs'] = gt_fut_trajs
        if 'custom_tokens' in data_info:
            input_dict['custom_tokens'] = data_info['custom_tokens']
        return input_dict
    

    def custom_collate_fn(self, batch):
        """
        Collate function to be used when wrapping a RIODatasetSceneGraph in a
        DataLoader. Returns a dictionary
        """

        out = {}

        out['scene_points'] = []
        out['scan_id'] = []
        out['instance_id'] = []

        out['missing_nodes'] = []
        out['added_nodes'] = []
        out['missing_nodes_decoder'] = []
        out['manipulated_subs'] = []
        out['manipulated_objs'] = []
        out['manipulated_preds'] = []
        global_node_id = 0
        global_dec_id = 0
        for i in range(len(batch)):
            if batch[i] == -1:
                return -1
            # notice only works with single batches
            out['scan_id'].append(batch[i]['scan_id'])
            # out['instance_id'].append(batch[i]['instance_id'])

            if batch[i]['manipulate']['type'] == 'addition':
                out['missing_nodes'].append(global_node_id + batch[i]['manipulate']['added_node_id'])
                out['added_nodes'].append(global_dec_id + batch[i]['manipulate']['added_node_id'])
            elif batch[i]['manipulate']['type'] == 'relationship':
                rel, (sub, pred, obj) = batch[i]['manipulate']['original_relship'] # remember that this is already changed in the initial scene graph, which means this triplet is real data.
                # which node is changed in the beginning.
                out['manipulated_subs'].append(global_node_id + sub)
                out['manipulated_objs'].append(global_node_id + obj)
                out['manipulated_preds'].append(pred) # this is the real edge

            global_node_id += len(batch[i]['encoder']['objs'])
            global_dec_id += len(batch[i]['decoder']['objs'])

        for key in ['encoder', 'decoder']:
            all_objs, all_boxes, all_triples = [], [], []
            all_obj_to_scene, all_triple_to_scene = [], []
            all_points = []
            all_sdfs = []
            all_text_feats = []
            all_rel_feats = []

            obj_offset = 0

            for i in range(len(batch)):
                if batch[i] == -1:
                    print('this should not happen')
                    continue
                (objs, triples, boxes) = batch[i][key]['objs'], batch[i][key]['triples'], batch[i][key]['boxes']

                if 'points' in batch[i][key]:
                    all_points.append(batch[i][key]['points'])
                elif 'sdfs' in batch[i][key]:
                    all_sdfs.append(batch[i][key]['sdfs'])
                if 'text_feats' in batch[i][key]:
                    all_text_feats.append(batch[i][key]['text_feats'])
                if 'rel_feats' in batch[i][key]:
                    if 'changed_id' in batch[i][key]:
                        idx = batch[i][key]['changed_id']
                        if self.scene_graph_assigner.with_CLIP:
                            text_rel = clip.tokenize(batch[i][key]['words'][idx]).to('cpu')
                            rel = self.scene_graph_assigner.cond_model_cpu.encode_text(text_rel).detach().numpy()
                            batch[i][key]['rel_feats'][idx] = torch.from_numpy(np.squeeze(rel)) # this should be a fake relation from the encoder side

                    all_rel_feats.append(batch[i][key]['rel_feats'])

                num_objs, num_triples = objs.size(0), triples.size(0)

                all_objs.append(batch[i][key]['objs'])
                # all_objs_grained.append(batch[i][key]['objs_grained'])
                all_boxes.append(boxes)

                if triples.dim() > 1:
                    triples = triples.clone()
                    triples[:, 0] += obj_offset
                    triples[:, 2] += obj_offset

                    all_triples.append(triples)
                    all_triple_to_scene.append(torch.LongTensor(num_triples).fill_(i))

                all_obj_to_scene.append(torch.LongTensor(num_objs).fill_(i))

                obj_offset += num_objs

            all_objs = torch.cat(all_objs)
            # all_objs_grained = torch.cat(all_objs_grained)
            all_boxes = torch.cat(all_boxes)

            all_obj_to_scene = torch.cat(all_obj_to_scene)

            if len(all_triples) > 0:
                all_triples = torch.cat(all_triples)
                all_triple_to_scene = torch.cat(all_triple_to_scene)
            else:
                return -1

            outputs = {'objs': all_objs,
                    #    'objs_grained': all_objs_grained,
                       'tripltes': all_triples,
                       'boxes': all_boxes,
                       'obj_to_scene': all_obj_to_scene,
                       'triple_to_scene': all_triple_to_scene}

            if len(all_sdfs) > 0:
                outputs['sdfs'] = torch.cat(all_sdfs)
            elif len(all_points) > 0:
                all_points = torch.cat(all_points)
                outputs['points'] = all_points

            if len(all_text_feats) > 0:
                all_text_feats = torch.cat(all_text_feats)
                outputs['text_feats'] = all_text_feats
            if len(all_rel_feats) > 0:
                all_rel_feats = torch.cat(all_rel_feats)
                outputs['rel_feats'] = all_rel_feats
            out[key] = outputs

        return out

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