import os
import numpy as np
from .pcdet_nuscenes_dataset import NuScenesDataset
from pathlib import Path
from lidargen.dataset import utils
import torch

def find_pth_files(root_dir, endswith='.txt'):
    pth_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(endswith):
                pth_files.append(os.path.join(dirpath, fname))
    return pth_files

class ObjectDetectionDataset(NuScenesDataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.refine_infos()
    
    def get_selected_tokens(self, samples_files):
        self.lidar_path = dict()
        tokens = []
        for file in samples_files:
            if self.gen_name == 'uniscene':
                token = Path(file).stem.split('.')[0].split('-')[0]
            elif 'dwm' in self.gen_name:
                token = Path(file).stem.split('.')[0]
            else:
                token = Path(file).stem.split('.')[0].split('_')[-1]
            tokens.append(token)
            self.lidar_path[token] = file
        return tokens

    def refine_infos(self):
        self.gen_name = self.dataset_cfg.get('METHOD_NAME')
        assert self.gen_name in ['uniscene', 'opendwm', 'opendwm_dit', 'our'], f"Unknown generation method: {self.gen_name}"
        samples_folder_path = f'../generated_results/{self.gen_name}'
        if self.gen_name == 'opendwm':
            samples_folder_path = os.path.join(samples_folder_path, 'opendwm_lidar')
            samples_files = find_pth_files(samples_folder_path, endswith=self.dataset_cfg.get('ENDWITH', '.txt'))
        elif self.gen_name == 'opendwm_dit':
            samples_folder_path = os.path.join(samples_folder_path, 'opendwm_lidar_dit')
            samples_files = find_pth_files(samples_folder_path, endswith=self.dataset_cfg.get('ENDWITH', '.txt'))
        elif self.gen_name == 'uniscene':
            samples_folder_path = os.path.join(samples_folder_path, 'pred')
            samples_files = find_pth_files(samples_folder_path, endswith=self.dataset_cfg.get('ENDWITH', '.npy'))
        else:
            samples_files = find_pth_files(samples_folder_path, endswith=self.dataset_cfg.get('ENDWITH', '.pth'))

        selected_tokens = self.get_selected_tokens(samples_files)
        infos = [info for info in self.infos if info['token'] in selected_tokens]
        self.infos = infos

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.infos[index]
        token = info['token']
        lidar_path = self.lidar_path.get(token)
        # if .txt endwith
        if Path(lidar_path).suffix == '.txt':
            points = np.loadtxt(self.lidar_path[token], dtype=np.float32)[:, :3]
            rotation = np.array(np.pi) / 2
            points = utils.rotate_points_along_z(points[np.newaxis, :, :], rotation.reshape(1))[0]
            points[:, 2] -= 2.0

        elif Path(lidar_path).suffix == '.npy':
            points = np.load(self.lidar_path[token])[:, :3]
            rotation = np.array(np.pi) / 2
            points = utils.rotate_points_along_z(points[np.newaxis, :, :], rotation.reshape(1))[0]

        else:
            range_img = torch.load(lidar_path,  map_location="cpu")
            xyz = range_img[[1,2,3]]
            points = xyz.reshape(3, -1).permute(1, 0).numpy()

        times = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.concatenate((points, times), axis=1)
        return points

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        token = self.infos[index]['token']
        data_dict['frame_id'] = token
        return data_dict
