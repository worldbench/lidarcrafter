from pathlib import Path
from sklearn.model_selection import KFold
import pickle
import copy
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler
from pcdet.utils import common_utils


np.seterr(divide='ignore',invalid='ignore')

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (N, 3 + C)
        angle: float, angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    rot_matrix = np.array([
        [cosa,  sina, 0],
        [-sina, cosa, 0],
        [0, 0, 1]
    ]).astype(np.float32)
    points_rot = np.matmul(points[:, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, 3:]), axis=-1)
    return points_rot

class Object_Uncertainty_Dataset():
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.enable_similar_type = self.dataset_cfg.get("ENABLE_SIMILAR_TYPE", False)
        self.split = 'train' if self.training else 'val'
        self.pkl_path = dataset_cfg.get('PKL_PATH', None)
        self.pkl_path = Path(self.pkl_path) if self.pkl_path is not None else None

        if 'FOLD_IDX' in self.dataset_cfg:
            # db_infos_path = self.root_path / 'kitti_dbinfos_train.pkl'
            if self.pkl_path is not None:
                db_infos_path_list = [self.pkl_path]
            else:
                db_infos_path_list = [f'../data/infos/nuscenes_object_classification_{self.split}.pkl']
            infos_list = []
            for db_infos_path in db_infos_path_list:
                with open(db_infos_path, 'rb') as f:
                    infos = pickle.load(f)
                if isinstance(infos, dict):
                    temp_infos = []
                    for key, value in infos.items():
                        if key in self.class_names:
                            temp_infos.extend(value)
                    infos = temp_infos
                infos_list.append(infos)
            splits=KFold(n_splits=10,shuffle=True,random_state=42) # random_state=42
            fold_idx = self.dataset_cfg.FOLD_IDX
            used_infos = []
            used_infos.extend(infos)
            train_idx,val_idx = [x for x in splits.split(np.arange(len(used_infos)))][fold_idx]
            self.frame_ids = val_idx
            if self.training:
                self.kitti_infos = [used_infos[idx] for idx in train_idx]
            else:
                self.kitti_infos = [used_infos[idx] for idx in val_idx]     
        else:
            # Just for training with nuscenes
            db_infos_path = f'../data/infos/nuscenes_object_classification_{self.split}.pkl'
            with open(db_infos_path, 'rb') as f:
                 self.kitti_infos = pickle.load(f)
            self.kitti_infos = [x for x in self.kitti_infos if x['name'] in self.class_names]

        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        # dense gt infos
        self.dense_gt_infos = [x for x in self.kitti_infos if x['num_points_in_gt']>1000]
        logger.info(f'Length of dense_gt_infos is {len(self.dense_gt_infos)}')

        self.linear_anneal = 0
        self.force_ratio = self.dataset_cfg.FORCE_RATIO
        self.force_num = self.dataset_cfg.FORCE_NUM

        self.enable_flip = self.dataset_cfg.get('ENABLE_FLIP', False)
        self.scale_range = self.dataset_cfg.get('RANDOM_SCALE_RANGE', [1.0, 1.0])
        self.angle_rot_max = self.dataset_cfg.get('ANGLE_ROT_MAX', 0) # 0.78539816
        self.pos_shift_max = self.dataset_cfg.get('POS_SHIFT_MAX', 0)
        logger.info(f"### Aug params: flip={self.enable_flip}, scale={self.scale_range}, rot={self.angle_rot_max},shift={self.pos_shift_max}, enable_similar_type={self.enable_similar_type}")

        self.rv_width = 512
        self.rv_height = 48

        obj_text_feat_path = '../data/clips/nuscenes/obj_text_feat.pkl'
        with open(obj_text_feat_path, 'rb') as f:
            self.obj_text_feat = pickle.load(f)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):
        data_dict = {}

        info = copy.deepcopy(self.kitti_infos[index])
        if self.pkl_path is None:
            pc_path = self.root_path / info['path'] 
            points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 5)[:, :3]  # only xyz
        else:
            points = np.fromfile(info['path'], dtype=np.float32).reshape(-1, 4)[:, :3]  # only xyz

        flip_mark = False
        noise_scale = 1.0
        if self.training:
            if self.enable_flip:
                flip_mark = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
                if flip_mark:
                    points[:, 1] = -points[:, 1]

            noise_scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            points[:, :3] *= noise_scale


        # rotate points along the z axis
        if points.shape[0] != 0:
            x_mean, y_mean, z_mean = list(points[:, :3].mean(axis=0))
        else:
            x_mean, y_mean, z_mean = [0,0,0]

        point_anchor_size = [3.9, 1.6, 1.56]
        dxa, dya, dza = point_anchor_size
        diagonal = np.sqrt(dxa ** 2 + dya ** 2)

        # 增加点云到每个平面的距离

        pos_shift = [0, 0]
        angle_rot = 0

        if self.training:
        #     # rotate + movement
        #     # range: pi/4, 1m

            angle_rot = (np.random.rand(1)[0] - 0.5) / 0.5 * self.angle_rot_max
            pos_shift = ((np.random.rand(2) - 0.5) / 0.5) * self.pos_shift_max
            
            points = rotate_points_along_z(points, angle_rot)


        points[:, 0] = (points[:, 0] - x_mean + pos_shift[0]) / diagonal
        points[:, 1] = (points[:, 1] - y_mean + pos_shift[1]) / diagonal
        points[:, 2] = (points[:, 2] - z_mean) / dza

        keep_num = 512
        if points.shape[0] != 0:
            choice = np.random.choice(points.shape[0], keep_num, replace=True)
            points = points[choice, :]
        else:
            points = np.full((keep_num, 4), 0)

        # import pdb;pdb.set_trace()
        # print('info gt_idx = ', info['gt_idx'], type(info['gt_idx']))

        data_dict['points'] = points.transpose()
        # data_dict['frame_id'] = info['image_idx']
        # data_dict['gt_id'] = info['gt_idx']

        if 'box3d_lidar' not in info:
            return data_dict


        box3d_lidar_ori = np.array(info['box3d_lidar'])[:7]
        box3d_lidar = copy.deepcopy(box3d_lidar_ori)
        if flip_mark:
            # import pdb;pdb.set_trace()
            box3d_lidar[6] = -box3d_lidar[6]
        box3d_lidar[:6] *= noise_scale

        box3d_lidar[0] = (- x_mean + pos_shift[0]) / diagonal
        box3d_lidar[1] = (- y_mean + pos_shift[1]) / diagonal
        box3d_lidar[2] = (- z_mean) / dza
        box3d_lidar[3] = np.log(box3d_lidar[3]/dxa)
        box3d_lidar[4] = np.log(box3d_lidar[4]/dya)
        box3d_lidar[5] = np.log(box3d_lidar[5]/dza)
        box3d_lidar[6] += angle_rot

        box3d_lidar_dim7 = copy.deepcopy(box3d_lidar)

        angle = box3d_lidar[6]
        box3d_lidar[6] = np.sin(angle)
        box3d_lidar = np.append(box3d_lidar, np.cos(angle))

        # points sample
        # print(f"### points.shape[0] = {points.shape[0]}")

        # import pdb;pdb.set_trace()

        data_dict['gt_boxes_input'] = box3d_lidar
        data_dict['gt_boxes'] = box3d_lidar_dim7
        obj_name = info['name']
        data_dict['text_feat'] = self.obj_text_feat[obj_name][0]
        data_dict['frame_id'] = self.frame_ids[index]
        data_dict['gt_id'] = self.frame_ids[index]
        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0) # concat [(512, 4), (512, 4)] -> [1024, 4]
                elif key in ['voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in []: 
                    max_gt = max([len(x) for x in val]) 
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = np.nan

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret


class DistributedSampler(_DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset_map = {
        'NuscObjUncertaintyDataset': Object_Uncertainty_Dataset,
    }
    dataset_class = dataset_map[dataset_cfg.get('DATASET', 'KittiGtDataset')]
    dataset = dataset_class(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset) # By default, shuffle = True
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler