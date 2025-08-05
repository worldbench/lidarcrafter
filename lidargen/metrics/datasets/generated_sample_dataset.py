import os
import numpy as np
from pcdet.datasets import DatasetTemplate
import torch
from lidargen.utils.lidar import LiDARUtility, get_linear_ray_angles
from lidargen.dataset import utils

class GeneratedDataset(DatasetTemplate):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.infos = []
        self.endwith = self.dataset_cfg.get('ENDWITH', 'pth')
        self.include_generated_data()

    def include_generated_data(self):
        generated_path = self.dataset_cfg['GENERATED_SAMPLES_PATH']
        pth_files = []
        for dirpath, dirnames, filenames in os.walk(generated_path):
            for fname in filenames:
                if fname.lower().endswith(f'.{self.endwith}'):
                    pth_files.append(os.path.join(dirpath, fname))
        self.infos = pth_files

        self.lidar_utils = LiDARUtility(
            resolution=(32,1024),
            depth_format="log_depth",
            min_depth=1.45,
            max_depth=80.0,
            ray_angles=get_linear_ray_angles(
                H=32,
                W=1024,
                fov_up=10,
                fov_down=-30
            ))
        self.lidar_utils.eval()

    def __len__(self):
        return len(self.infos)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            if self.dataset_cfg.get('SHIFT_COOR', None):
                #print ("*******WARNING FOR SHIFT_COOR:", self.dataset_cfg.SHIFT_COOR)
                pred_boxes[:, 0:3] -= self.dataset_cfg.SHIFT_COOR

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            # single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def load_points(self, info_path):
        if self.endwith == 'npy':
            points = np.load(info_path, allow_pickle=True)
            rotation = np.array(np.pi) / 2
            points = utils.rotate_points_along_z(points[np.newaxis, :, :], rotation.reshape(1))[0]
            points = torch.tensor(points, dtype=torch.float32)
            timestamp = torch.zeros((points.shape[0], 1))
            points = torch.cat([points, timestamp], axis=1)
            return points
        
        if self.endwith == 'txt':
            points = np.loadtxt(info_path, dtype=np.float32)
            rotation = np.array(np.pi) / 2
            points = utils.rotate_points_along_z(points[np.newaxis, :, :], rotation.reshape(1))[0]
            points[:,2] -= 2.0
            points = torch.tensor(points, dtype=torch.float32)
            timestamp = torch.zeros((points.shape[0], 1))
            points = torch.cat([points, timestamp], axis=1)
            return points
        
        img = torch.load(info_path, map_location="cpu")
        if img.shape[0] == 2:
            # depth = self.lidar_utils.denormalize(img[0])
            depth = self.lidar_utils.revert_depth(img[0])
            points = self.lidar_utils.to_xyz(depth.unsqueeze(0).unsqueeze(0)) # [1, 3, 32, 1024]
            points = points[0].reshape(3,-1).permute(1,0)
            # timestamp
            timestamp = torch.zeros((points.shape[0], 1))
            points = torch.cat([points, timestamp], axis=1)
            return points
        else:
            xyz = img[[1,2,3]].permute(1,2,0).flatten(0, 1)  # (H, W, 3) -> (H*W, 3)
            # timestamp
            timestamp = torch.zeros((xyz.shape[0], 1))
            points = torch.cat([xyz, timestamp], axis=1)
            return points.float()

    def __getitem__(self, index):
        info_path = self.infos[index]
        # load_points
        points = self.load_points(info_path) # only xyz
        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
        input_dict = {
            'frame_id': str(index).zfill(6),
            'points': points,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

if __name__ == "__main__":
    # Example usage
    dataset_cfg = {
        'GENERATED_SAMPLES_PATH': '/path/to/generated/samples'
    }
    class_names = ['Car', 'Pedestrian', 'Cyclist']
    dataset = GeneratedDataset(dataset_cfg=dataset_cfg, class_names=class_names, training=True)
    
    for i in range(len(dataset)):
        data = dataset[i]
        print(data['points'].shape)  # Print the shape of the points tensor