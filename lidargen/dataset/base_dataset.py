import torch
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
from .augmentor.data_augmentor import DataAugmentor

class DatasetBase(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.task = getattr(cfg, 'task', None)
        self.data_root = cfg.data_root
        self.split = cfg.split
        self.pkl_path = getattr(cfg, 'pkl_path', None)
        if (self.pkl_path is not None) and 'train' in self.pkl_path:
            if self.split != 'train':
                self.pkl_path = self.pkl_path.replace("train", "val")

        if self.split == 'all':
            pkl_path_list = []
            for split in ['train', 'val']:
                pkl_path_list.append(cfg.pkl_path.replace('train', split))
            self.pkl_path = pkl_path_list

        self.data_augmentor = DataAugmentor(
            root_path=self.data_root,
            augmentor_configs=cfg.data_augmentor,
            class_names=cfg.class_names
        ) if getattr(cfg, 'data_augmentor', None) else None

        self.prepare_data()

    def __len__(self):
        return len(self.data)

    def prepare_data(self):
        raise NotImplementedError

    def collate_fn(self, batch_list, _unused=False):
        ret = {}

        if 'custom_dict' in batch_list[0].keys():
            assert hasattr(self, 'custom_collate_fn'), "Custom collate function is not defined."
            custom_data_list = [db.pop('custom_dict') for db in batch_list]
            out = self.custom_collate_fn(custom_data_list)
            ret.update({
                'scenegraph_input': out
            })

        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        batch_size_ratio = 1

        if self.task == 'layout_generation':
            disable_key = ['points', 'gt_names', 'gt_boxes', 'gt_boxes_2d', 'scaled_gt_boxes', 'gt_box_relationships', 'gt_fut_trajs', 'gt_fut_masks', 'gt_fut_states', 'token']
        else:
            disable_key = ['points', 'gt_names', 'gt_boxes', 'gt_box_relationships', 'gt_fut_trajs', 'gt_fut_masks', 'gt_fut_states', 'token', 'custom_tokens']

        for key, val in data_dict.items():
            # if key not in :
            if key not in disable_key:
                temp = np.stack(val, axis=0)
                ret[key] = torch.from_numpy(temp).float()

            if key in ['token', 'custom_tokens', 'gt_boxes', 'gt_names', 'gt_fut_trajs']:
                ret[key] = val
        ret['batch_size'] = batch_size * batch_size_ratio
        
        return ret