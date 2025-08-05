
import os
import pickle
import numpy as np
from torch.utils.data import Dataset

from ...dataset import utils 

class NuscObject(Dataset):
    def __init__(self, num_points=1024, partition='train', class_name=['car', 'truck', 'bus'], pkl_path=None): # 'train', 'val'
        self.class_name = class_name
        self.pkl_path = pkl_path
        self.num_points = num_points
        self.partition = partition
        self.data_root = '../data/nuscenes' 
        self.data_infos = self.load_data(partition)

    def load_data(self, partition):
        if self.pkl_path is not None:
            self.dataset_type = 'custom'
            self.data_root = ''
            pkl_path = self.pkl_path
            data_infos = []
            data_infos_dict = pickle.load(open(pkl_path, 'rb'))
            for key, value in data_infos_dict.items():
                if key in self.class_name:
                    data_infos.extend(value)
        else:
            self.dataset_type = 'ori'
            pkl_path = f'../data/infos/nuscenes_object_classification_{partition}.pkl'
            data_infos = pickle.load(open(pkl_path, 'rb'))
        return data_infos

    def load_points(self, fg_path):
        if self.dataset_type == 'custom':
            fg_points = np.fromfile(fg_path, dtype=np.float32).reshape(-1,4)[:,:3] # only xyz
        else:
            fg_path = os.path.join(self.data_root, fg_path)
            fg_points = np.fromfile(fg_path, dtype=np.float32).reshape(-1,5)[:,:3] # only xyz
        return fg_points

    def norm_fg_points(self, fg_points, box3d):
        rotation = -np.array([box3d[-1]])
        fg_points = utils.rotate_points_along_z(fg_points[np.newaxis, :, :], rotation)[0]
        fg_points[:,0] = fg_points[:,0] / box3d[3] + 0.5
        fg_points[:,1] = fg_points[:,1] / box3d[4] + 0.5
        fg_points[:,2] = fg_points[:,2] / box3d[5] + 0.5
        return fg_points

    def __getitem__(self, item):
        info = self.data_infos[item]
        fg_points_path = info['path']
        fg_points = self.load_points(fg_points_path)
        fg_points = self.norm_fg_points(fg_points, info['box3d_lidar'][:7])
        np.random.shuffle(fg_points)
        # set points to fixed number
        if fg_points.shape[0] < self.num_points:
            fg_points = np.pad(fg_points, ((0, self.num_points - fg_points.shape[0]), (0, 0)), mode='constant')
        else:
            fg_points = fg_points[:self.num_points]
        class_name = info['name']
        if class_name not in self.class_name:
            label = np.array(0)  # background class
        else:
            label = np.array(self.class_name.index(info['name']) + 1)
        num_points_in_gt = np.array(info['num_points_in_gt'])
        return fg_points, label, num_points_in_gt

    def __len__(self):
        return len(self.data_infos)