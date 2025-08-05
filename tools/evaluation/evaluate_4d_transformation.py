# import os
# import pickle
# import argparse
# import numpy as np
# import open3d as o3d
# from tqdm import tqdm
# from collections import defaultdict
# from natsort import natsorted
# from pyquaternion import Quaternion

# def estimate_transform_icp(pts_source, pts_target, threshold=1.0):
#     # Convert to Open3D point clouds
#     pcd_source = o3d.geometry.PointCloud()
#     pcd_target = o3d.geometry.PointCloud()
#     pcd_source.points = o3d.utility.Vector3dVector(pts_source)
#     pcd_target.points = o3d.utility.Vector3dVector(pts_target)

#     # Initial alignment (identity)
#     trans_init = np.eye(4)

#     # Run ICP registration
#     reg_result = o3d.pipelines.registration.registration_icp(
#         pcd_source, pcd_target, threshold, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint())

#     transformation = reg_result.transformation
#     rotation = transformation[:3, :3]
#     translation = transformation[:3, 3]

#     return transformation, rotation, translation

# def get_gt_transformation(source_frame_filename, target_frame_filename, data_infos):
#     source_token = source_frame_filename.split('_')[-1].strip('.txt')
#     target_token = target_frame_filename.split('_')[-1].strip('.txt')
#     source_info = data_infos[source_token]
#     target_info = data_infos[target_token]
#     lidar2sensor = np.eye(4)
#     l2e_r_s = source_info['lidar2ego_rotation']
#     l2e_t_s = source_info['lidar2ego_translation']
#     e2g_r_s = source_info['ego2global_rotation']
#     e2g_t_s = source_info['ego2global_translation']
#     l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
#     e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

#     e2g_t = target_info['ego2global_translation']
#     e2g_r_mat = Quaternion(target_info['ego2global_rotation']).rotation_matrix
#     l2e_r_mat = Quaternion(target_info['lidar2ego_rotation']).rotation_matrix
#     l2e_t = target_info['lidar2ego_translation']

#     R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
#         np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
#     T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
#         np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
#     T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
#                 ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
#     rot = R.T
#     trans = T

#     return rot, trans

# def calculate_single_sequence_TTCE(seq_path, data_infos):
#     seq_results = defaultdict(list)
#     frame_files = natsorted(os.listdir(seq_path))
#     frame_files = [os.path.join(seq_path, f) for f in frame_files]
#     frame_num = len(frame_files)
#     for split in [1,2,3,4]:
#         for frame_id in range(frame_num):
#             if frame_id + split >= frame_num:
#                 continue
#             pts_source = np.loadtxt(frame_files[frame_id])[:, :3]
#             pts_target = np.loadtxt(frame_files[frame_id + split])[:, :3]
#             # test gt
#             # source_file_name = os.path.basename(frame_files[frame_id])
#             # target_file_name = os.path.basename(frame_files[frame_id + split])
#             # source_token = source_file_name.split('_')[-1].strip('.txt')
#             # target_token = target_file_name.split('_')[-1].strip('.txt')
#             # source_info = data_infos[source_token]
#             # target_info = data_infos[target_token]
#             # pts_source = np.fromfile(os.path.join('../data/nuscenes', source_info['lidar_path']), dtype=np.float32).reshape(-1, 5)[:, :3]
#             # pts_target = np.fromfile(os.path.join('../data/nuscenes', target_info['lidar_path']), dtype=np.float32).reshape(-1, 5)[:, :3]

#             T_est, R_est, t_est = estimate_transform_icp(pts_source, pts_target, threshold=2.0)
#             R_gt, t_gt = get_gt_transformation(os.path.basename(frame_files[frame_id]),
#                                                os.path.basename(frame_files[frame_id + split]),
#                                                data_infos)
#             seq_results[split].append(np.mean(np.mean(np.abs(t_est - t_gt))))
            
#     return seq_results

# def calculate_TTCE(method_name):
#     method_results = defaultdict(list)
#     data_infos = pickle.load(open(f'../data/infos/nuscenes_infos_val.pkl', 'rb'))
#     data_infos_dict = {info['token']: info for info in data_infos['infos']}
#     temporal_results_folder_path = f'../generated_results/{method_name}/temporal_points'
#     seq_paths = [os.path.join(temporal_results_folder_path, seq) for seq in os.listdir(temporal_results_folder_path)]
#     for seq_id, seq_path in enumerate(tqdm(seq_paths, desc=f'Calculating TTCE for {method_name}')): 
#         seq_result = calculate_single_sequence_TTCE(seq_path, data_infos_dict)
#         for split in [1,2,3,4]:
#             method_results[split].extend(seq_result[split])
#     for split in [1,2,3,4]:
#         print(f'Method: {method_name}, Split: {split}, TTCE: {np.mean(method_results[split]):.4f} m')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Calculate TTCE for 4D transformation methods.')
#     parser.add_argument('--method_name', type=str, default='opendwm', help='Name of the method to evaluate.')
#     args = parser.parse_args()
#     calculate_TTCE(args.method_name)

import os
import pickle
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
from collections import defaultdict
from natsort import natsorted
from pyquaternion import Quaternion
import concurrent.futures

def estimate_transform_icp(pts_source, pts_target, threshold=1.0):
    # Convert to Open3D point clouds
    pcd_source = o3d.geometry.PointCloud()
    pcd_target = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(pts_source)
    pcd_target.points = o3d.utility.Vector3dVector(pts_target)

    # Initial alignment (identity)
    trans_init = np.eye(4)

    # Run ICP registration
    reg_result = o3d.pipelines.registration.registration_icp(
        pcd_source, pcd_target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    transformation = reg_result.transformation
    rotation = transformation[:3, :3]
    translation = transformation[:3, 3]

    return transformation, rotation, translation

def load_points(frame_filepath, data_infos, to_global=False):
    frame_filename = os.path.basename(frame_filepath)
    token = frame_filename.split('_')[-1].strip('.txt')
    info = data_infos[token]
    # gen
    points = np.loadtxt(frame_filepath)[:, :3]
    # gt
    # points = np.fromfile(os.path.join('../data/nuscenes', info['lidar_path']), dtype=np.float32).reshape(-1, 5)[:, :3]
    # delete points near original
    points = points[np.linalg.norm(points, axis=1) > 0.2]
    if not to_global:
        return points
    else:
        l2e_r_s_mat = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        e2g_r_s_mat = Quaternion(info['ego2global_rotation']).rotation_matrix
        l2e_t_s = info['lidar2ego_translation']
        e2g_t_s = info['ego2global_translation']
        R = e2g_r_s_mat @ l2e_r_s_mat
        T = e2g_t_s + e2g_r_s_mat @ l2e_t_s
        # R = (l2e_r_s_mat.T @ e2g_r_s_mat.T)
        # T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s)
        points = (R @ points.T).T + T
        return points

def get_gt_transformation(source_frame_filename, target_frame_filename, data_infos):
    source_token = source_frame_filename.split('_')[-1].strip('.txt')
    target_token = target_frame_filename.split('_')[-1].strip('.txt')
    source_info = data_infos[source_token]
    target_info = data_infos[target_token]

    # 构造旋转和平移矩阵
    l2e_r_s_mat = Quaternion(source_info['lidar2ego_rotation']).rotation_matrix
    e2g_r_s_mat = Quaternion(source_info['ego2global_rotation']).rotation_matrix
    l2e_t_s = source_info['lidar2ego_translation']
    e2g_t_s = source_info['ego2global_translation']

    e2g_r_mat = Quaternion(target_info['ego2global_rotation']).rotation_matrix
    l2e_r_mat = Quaternion(target_info['lidar2ego_rotation']).rotation_matrix
    e2g_t = target_info['ego2global_translation']
    l2e_t = target_info['lidar2ego_translation']

    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) \
         + l2e_t @ np.linalg.inv(l2e_r_mat).T

    return R.T, T

def calculate_single_sequence_TTCE(seq_path, data_infos):
    seq_results = defaultdict(list)
    frame_files = natsorted(os.listdir(seq_path))
    frame_files = [os.path.join(seq_path, f) for f in frame_files]
    frame_num = len(frame_files)

    for split in [3, 4]:
        for frame_id in range(frame_num - split):
            pts_source = load_points(frame_files[frame_id], data_infos, to_global=False)
            pts_target = load_points(frame_files[frame_id + split], data_infos, to_global=False)

            T_est, R_est, t_est = estimate_transform_icp(pts_source, pts_target, threshold=2.0)
            R_gt, t_gt = get_gt_transformation(
                os.path.basename(frame_files[frame_id]),
                os.path.basename(frame_files[frame_id + split]),
                data_infos
            )

            seq_results[split].append(np.mean(np.abs(t_est - t_gt)))

    return seq_results

def calculate_TTCE(method_name, num_threads=None):
    data_infos = pickle.load(open(f'../data/infos/nuscenes_infos_val.pkl', 'rb'))
    data_infos_dict = {info['token']: info for info in data_infos['infos']}

    temporal_folder = f'../generated_results/{method_name}/temporal_points'
    seq_paths = [os.path.join(temporal_folder, seq) for seq in os.listdir(temporal_folder)]

    # 默认线程数为 CPU 核心数
    max_workers = num_threads or os.cpu_count()

    method_results = defaultdict(list)

    # 并行提交每个序列的计算任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_seq = {
            executor.submit(calculate_single_sequence_TTCE, seq_path, data_infos_dict): seq_path
            for seq_path in seq_paths
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_seq),
                           total=len(future_to_seq),
                           desc=f'Calculating TTCE for {method_name}'):
            seq_result = future.result()
            for split, errors in seq_result.items():
                method_results[split].extend(errors)

    # 打印最终结果
    for split in [3, 4]:
        mean_error = np.mean(method_results[split]) if method_results[split] else float('nan')
        print(f'Method: {method_name}, Split: {split}, TTCE: {mean_error:.4f} m')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calculate TTCE for 4D transformation methods (multi-threaded).'
    )
    parser.add_argument(
        '--method_name', type=str, default='our',
        help='Name of the method to evaluate.'
    )
    parser.add_argument(
        '--num_threads', type=int, default=4,
        help='Number of threads to use (default: all CPU cores).'
    )
    args = parser.parse_args()
    calculate_TTCE(args.method_name, args.num_threads)
