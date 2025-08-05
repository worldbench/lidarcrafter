import os
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
from collections import defaultdict
sys.path.append('../')
from lidargen.dataset.utils import get_points_in_box
from lidargen.dataset import utils

keeped_class_names = ['car', 'truck', 'bus']

def include_generated_data(method_name):
    generated_path = os.path.join('../generated_results', method_name)
    points_files = []
    for dirpath, dirnames, filenames in os.walk(generated_path):
        for fname in filenames:
            if fname.lower().endswith(f'.txt'):
                points_files.append(os.path.join(dirpath, fname))
    return points_files

def extract_foreground_points(data_info, points, point_id, foreground_save_path):
    boxes_3d = data_info['gt_boxes'][:, :7]
    boxes_names = data_info['gt_names']
    rotation = np.array(np.pi) / 2
    points = utils.rotate_points_along_z(points[np.newaxis, :, :], rotation.reshape(1))[0]
    points[:, 2] -= 2.0  # Adjust height if needed
    sample_dict_list = []
    for box_id, box_3d in enumerate(boxes_3d):
        sample_name = boxes_names[box_id]
        if sample_name not in keeped_class_names:
            continue
        sample_points, _ = get_points_in_box(points, box_3d)
        if sample_points.shape[0] < 50:
            continue
        sample_points[:,:3] -= box_3d[None, :3]
        file_save_path = foreground_save_path / f"{str(point_id).zfill(6)}_{sample_name}_{box_id}.bin"
        sample_points.astype('float32').tofile(file_save_path)

        sample_dict = {
            'name': sample_name,
            'path': str(file_save_path),
            'num_points_in_gt': sample_points.shape[0],
            'box3d_lidar': box_3d.tolist(),
        }
        sample_dict_list.append(sample_dict)
    return sample_dict_list

def main(method_name):
    foreground_infos = defaultdict(list)
    pickle_path  = '../data/infos/nuscenes_infos_val.pkl'
    foreground_save_path = f'../generated_results/{method_name}/inference_results/foreground_samples'
    os.makedirs(foreground_save_path, exist_ok=True)
    data_dict = {}
    data_infos = pickle.load(open(pickle_path, 'rb'))['infos']
    for data_info in data_infos:
        token = data_info['token']
        data_dict[token] = data_info
    generated_point_files = include_generated_data(method_name)
    for point_id, point_file in enumerate(tqdm(generated_point_files, desc='Processing Generated Points', dynamic_ncols=True)):
        token = os.path.basename(point_file).split('.')[0]
        if token not in data_dict:
            print(f"Token {token} not found in data dictionary.")
            continue
        data_info = data_dict[token]
        points = np.loadtxt(point_file)
        sample_dict_list = extract_foreground_points(data_info, points, point_id, Path(foreground_save_path))
        for sample_dict in sample_dict_list:
            foreground_infos[sample_dict['name']].append(sample_dict)

    with open(Path(foreground_save_path).parent / 'foreground_samples_info.pkl', 'wb') as f:
        pickle.dump(foreground_infos, f)
        print(f'Foreground samples info saved to: {Path(foreground_save_path).parent / "foreground_samples_info.pkl"}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract foreground points from generated samples.')
    parser.add_argument('--method_name', type=str, default='opendwm', help='Name of the method to extract foreground points for.')
    args = parser.parse_args()
    main(args.method_name)