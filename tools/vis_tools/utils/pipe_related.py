import numpy as np
import torch
from collections import defaultdict

from lidargen.dataset import utils
from lidargen.ops.roiaware_pool3d import roiaware_pool3d_utils
from lidargen.dataset.custom_dataset import CustomDataset, CustomNuscObjectDataset
from lidargen.metrics.models.ptv3.model import PTv3
from . import common

def remove_ego_points(points, center_radius=2.0):
    mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
    return points[mask]  

def build_point_segmenter():
    """
    Build the point segmenter model based on the provided configuration.
    
    Args:
        cfg: Configuration object containing model parameters.
    
    Returns:
        An instance of the PTv3 model.
    """
    point_segmenter = PTv3()
    return point_segmenter

def get_temporal_boxes_3d(first_frame_data_dict, M=None):
    agent_fut_trajs = first_frame_data_dict['gt_fut_trajs']#[:,:3]
    agent_fut_trajs = np.insert(agent_fut_trajs, 0, 0, axis=1)
    acc_agent_fut_trajs = np.cumsum(agent_fut_trajs, axis=1)  # cumulative sum to get future trajectories
    if M is not None:
        acc_agent_fut_trajs = interp_trajs_numpy(acc_agent_fut_trajs, M=M)
    agent_fut_trajs = acc_agent_fut_trajs[:,1:] - acc_agent_fut_trajs[:,:-1]  # get future trajectories

    ego_future_trajs = agent_fut_trajs[0]  # (N, 2)
    ego_future_trajs = np.cumsum(ego_future_trajs, axis=0)

    T = ego_future_trajs.shape[0]
    object_future_trajs = agent_fut_trajs[1:]
    object_future_trajs = np.cumsum(object_future_trajs, axis=1)

    # current lidar point cloud and boxes
    curr_points = first_frame_data_dict['xyz'] # (3, H, W)
    curr_intensities = first_frame_data_dict['reflectance'] * 255 # (1, H, W)
    curr_points = np.stack([curr_points[0], curr_points[1], curr_points[2], curr_intensities[0]], axis=-1).reshape(-1, 4) # (H, W, 4)
    curr_points = remove_ego_points(curr_points)  # remove ego points
    curr_boxes_3d = first_frame_data_dict['gt_boxes'][1:, :7] # not with ego

    # split points to foreground and background
    # foreground
    boxes3d_tensor, _ = utils.check_numpy_to_torch(curr_boxes_3d)
    points_tensor, _ = utils.check_numpy_to_torch(curr_points)
    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points_tensor[:, 0:3], boxes3d_tensor) # MxN
    assert object_future_trajs.shape[0] == boxes3d_tensor.shape[0]

    align_obj_points = []
    align_obj_intensity = []
    # extract each obj points
    for obj_id, obj_box_3d in enumerate(curr_boxes_3d):
        obj_points = curr_points[point_masks[obj_id, :] > 0]
        x, y, z, w, l, h, yaw = obj_box_3d
        align_obj_intensity.append(obj_points[:, 3])
        obj_points = obj_points[:, :3]
        obj_points_origin = obj_points - np.array([x, y, z])  # center to origin
        rotation = -np.array(yaw)
        allign_obj_points = utils.rotate_points_along_z(obj_points_origin[np.newaxis, :, :], rotation.reshape(1))[0]
        align_obj_points.append(allign_obj_points)

    # background
    condition_mask = ~(first_frame_data_dict['condition_mask'][0]>0)[None, ...]
    xyz = first_frame_data_dict['xyz'] * condition_mask  # apply mask to xyz
    intensity = first_frame_data_dict['reflectance'] * 255 * condition_mask  # apply mask to intensity
    curr_background_points = np.stack([xyz[0], xyz[1], xyz[2], intensity[0]], axis=-1).reshape(-1, 4)
    # detelt points distance < 1e-2
    curr_background_points = curr_background_points[np.linalg.norm(curr_background_points[:, :3], axis=1) > 1e-2]  # remove points with very small distance

    # future box info

    fut_boxes_3d = common.warp_boxes_future(
        boxes0=curr_boxes_3d,
        traj_obj=object_future_trajs,
        traj_ego=ego_future_trajs,
        z_e=0.0
    )

    fut_background_points = common.warp_lidar_future(
        P = curr_background_points,
        future_xy = ego_future_trajs,
        z0 = 0.0
    )

    Ts = common.compute_inter_frame_transforms(future_xy=ego_future_trajs, z0=0.0)

    return curr_background_points, fut_background_points, curr_boxes_3d, fut_boxes_3d, Ts, align_obj_points, align_obj_intensity


def get_temporal_points(first_frame_data_dict):
    agent_fut_trajs = first_frame_data_dict['gt_fut_trajs'][:,:3]
    agent_fut_trajs = np.insert(agent_fut_trajs, 0, 0, axis=1)
    acc_agent_fut_trajs = np.cumsum(agent_fut_trajs, axis=1)  # cumulative sum to get future trajectories
    acc_agent_fut_trajs = interp_trajs_numpy(acc_agent_fut_trajs, M=25)
    agent_fut_trajs = acc_agent_fut_trajs[:,1:] - acc_agent_fut_trajs[:,:-1]  # get future trajectories

    ego_future_trajs = agent_fut_trajs[0]  # (N, 2)
    ego_future_trajs = np.cumsum(ego_future_trajs, axis=0)

    T = ego_future_trajs.shape[0]
    object_future_trajs = agent_fut_trajs[1:]
    object_future_trajs = np.cumsum(object_future_trajs, axis=1)

    # current lidar point cloud and boxes
    curr_points = first_frame_data_dict['xyz'] # (3, H, W)
    curr_intensities = first_frame_data_dict['reflectance'] * 255 # (1, H, W)
    curr_points = np.stack([curr_points[0], curr_points[1], curr_points[2], curr_intensities[0]], axis=-1).reshape(-1, 4) # (H, W, 4)
    curr_boxes_3d = first_frame_data_dict['gt_boxes'][1:, :7] # not with ego

    # split points to foreground and background
    boxes3d_tensor, _ = utils.check_numpy_to_torch(curr_boxes_3d)
    points_tensor, _ = utils.check_numpy_to_torch(curr_points)
    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points_tensor[:, 0:3], boxes3d_tensor) # MxN
    background_mask = point_masks.sum(axis=0) == 0
    background_points = curr_points[background_mask]
    assert object_future_trajs.shape[0] == boxes3d_tensor.shape[0]

    align_obj_points = []
    align_obj_intensity = []

    # extract each obj points
    for obj_id, obj_box_3d in enumerate(curr_boxes_3d):
        obj_points = curr_points[point_masks[obj_id, :] > 0]
        x, y, z, w, l, h, yaw = obj_box_3d
        align_obj_intensity.append(obj_points[:, 3])
        obj_points = obj_points[:, :3]
        obj_points_origin = obj_points - np.array([x, y, z])  # center to origin
        rotation = -np.array(yaw)
        allign_obj_points = utils.rotate_points_along_z(obj_points_origin[np.newaxis, :, :], rotation.reshape(1))[0]
        align_obj_points.append(allign_obj_points)

    # future box info

    fut_boxes_3d = common.warp_boxes_future(
        boxes0=curr_boxes_3d,
        traj_obj=object_future_trajs,
        traj_ego=ego_future_trajs,
        z_e=0.0
    )

    # future points
    fut_background_points = common.warp_lidar_future(
        P = background_points,
        future_xy = ego_future_trajs,
        z0 = 0.0
    )

    # paste obj points to future
    fut_all_points = defaultdict()
    for t_id in range(T):
        bg_points = fut_background_points[t_id]
        fg_points = []
        fut_boxes_3d_t = fut_boxes_3d[:, t_id]
        for obj_id, obj_box_3d in enumerate(fut_boxes_3d_t):
            x, y, z, w, l, h, yaw = obj_box_3d
            rotation = np.array(yaw)
            roatated_obj_points = utils.rotate_points_along_z(align_obj_points[obj_id][np.newaxis, :, :], rotation.reshape(1))[0]
            roatated_trans_obj_points = roatated_obj_points + np.array([x, y, z])[None, :]
            roatated_trans_obj_points = np.concatenate([roatated_trans_obj_points, align_obj_intensity[obj_id][:, np.newaxis]], axis=-1)  # (M, 4)

            fg_points.append(roatated_trans_obj_points)
        fg_points = np.concatenate(fg_points, axis=0)
        fut_all_points[t_id] = np.concatenate([bg_points, fg_points], axis=0)
        
    # save points to txt
    # for points_id in range(len(fut_all_points)):
    #     save_path = f'../tools/ALTest/temp/{points_id}.txt'
    #     np.savetxt(save_path, fut_all_points[points_id], fmt='%.6f', delimiter=' ')
    # save_path = f'../tools/ALTest/temp/original.txt'
    # np.savetxt(save_path, curr_points, fmt='%.6f', delimiter=' ')

    cond_mask_dict_list = []
    for t_id in range(T):
        points = fut_all_points[t_id]
        gt_boxes = fut_boxes_3d[:, t_id]
        ego_box = np.zeros((1, 7), dtype=np.float32)  # ego box is not used
        gt_boxes = np.concatenate([ego_box, gt_boxes], axis=0)
        cond_mask_dict = {
            'points': points,
            'gt_boxes': gt_boxes,
            'gt_names': first_frame_data_dict['gt_names']
        }
        cond_mask_dict_list.append(cond_mask_dict)

    return cond_mask_dict_list

def get_mask_cond(cond_mask_dict_list):

    dataset = CustomDataset(custom_box_infos=cond_mask_dict_list)
    return dataset.__getitem__(0)

def conduct_obj_data_dict(unscaled_boxes_list):
    dataset = CustomNuscObjectDataset(custom_box_infos=unscaled_boxes_list)
    return dataset.__getitem__(0)

def temporal_inpaint_cond(cond_mask_dict_list):
    history_keep_mask = [data_dict.pop('history_keep_mask', None) for data_dict in cond_mask_dict_list][0] # torch.Tensor [1, 32, 1024]
    xs = cond_mask_dict_list[0].pop('inpaint_cond', None)  # torch.Tensor
    dataset = CustomDataset(custom_box_infos=cond_mask_dict_list)
    data_dict = dataset.__getitem__(0)
    depth = data_dict['depth']
    # curr_keep_mask = (~(data_dict['condition_mask'][0]>0) + depth) > 0
    # keep_mask = torch.logical_and(history_keep_mask.detach().cpu(), torch.from_numpy(curr_keep_mask))
    curr_keep_mask = ~(data_dict['condition_mask'][0]>0)
    keep_mask = torch.logical_and(history_keep_mask.detach().cpu(), torch.from_numpy(curr_keep_mask))
    inpaint_cond_dict = dict(
        inpaint_cond = xs.detach().cpu(),
        mask = keep_mask
    )
    return data_dict, inpaint_cond_dict

def get_mask_cond_single(cond_mask_dict_list, temporal=False, inpaint=False):

    dataset = CustomDataset(custom_box_infos=cond_mask_dict_list)
    if temporal:
        setattr(dataset, 'task', 'autoregressive_generation')
    if inpaint:
        setattr(dataset, 'inpaint_mode', True)
    return dataset.__getitem__(0)

def interp_trajs_numpy(trajs: np.ndarray, M: int) -> np.ndarray:

    K, N, D = trajs.shape
    assert D == 2
    t_orig = np.linspace(0.0, 1.0, N)
    t_new  = np.linspace(0.0, 1.0, M)

    new_trajs = np.zeros((K, M, 2), dtype=trajs.dtype)
    for k in range(K):
        new_trajs[k, :, 0] = np.interp(t_new, t_orig, trajs[k, :, 0])
        new_trajs[k, :, 1] = np.interp(t_new, t_orig, trajs[k, :, 1])

    return new_trajs

def get_next_frame_points(curr_background_points, align_obj_points, align_obj_intensity, fut_boxes_3d, fut_boxes_names, Ts):
    curr_background_points_intensity = curr_background_points[:, 3]
    curr_background_points = curr_background_points[:, :3]
    ones = np.ones((curr_background_points.shape[0],1))
    homo_pts = np.hstack((curr_background_points, ones))   # (N,4)
    fut_background_points = (Ts @ homo_pts.T).T         # (N,4)
    fut_background_points[:,3] = curr_background_points_intensity

    cond_mask_dict = {
        'points': fut_background_points,
        'gt_boxes': np.concatenate([np.zeros((1, 7)), fut_boxes_3d]),
        'gt_names': fut_boxes_names
    }
    fut_background_points = refine_next_frame_points([cond_mask_dict])

    fut_fg_points = []
    for obj_id, obj_box_3d in enumerate(fut_boxes_3d):
        x, y, z, w, l, h, yaw = obj_box_3d
        rotation = np.array(yaw)
        roatated_obj_points = utils.rotate_points_along_z(align_obj_points[obj_id][np.newaxis, :, :], rotation.reshape(1))[0]
        roatated_trans_obj_points = roatated_obj_points + np.array([x, y, z])[None, :]
        roatated_trans_obj_points = np.concatenate([roatated_trans_obj_points, align_obj_intensity[obj_id][:, np.newaxis]], axis=-1)  # (M, 4)

        fut_fg_points.append(roatated_trans_obj_points)
    fut_fg_points = np.concatenate(fut_fg_points, axis=0)
    fut_all_points = np.concatenate([fut_background_points, fut_fg_points], axis=0)
    return fut_all_points

def refine_next_frame_points(cond_mask_dict_list):
    dataset = CustomDataset(custom_box_infos=cond_mask_dict_list)
    data_dict = dataset.__getitem__(0)
    condition_mask = ~(data_dict['condition_mask'][0]>0)[None, ...]
    xyz = data_dict['xyz'] * condition_mask  # apply mask to xyz
    intensity = data_dict['reflectance'] * 255 * condition_mask  # apply mask to intensity
    points = np.stack([xyz[0], xyz[1], xyz[2], intensity[0]], axis=-1).reshape(-1, 4)
    # detelt points distance < 1e-2
    points = points[np.linalg.norm(points[:, :3], axis=1) > 1e-2]  # remove points with very small distance
    return points

def delete_fg_points(points, boxed_3d):
    boxes3d_tensor, _ = utils.check_numpy_to_torch(boxed_3d)
    points_tensor, _ = utils.check_numpy_to_torch(points)
    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points_tensor[:, 0:3], boxes3d_tensor) # MxN
    background_mask = point_masks.sum(axis=0) == 0
    curr_background_points = points[background_mask]
    return curr_background_points

def get_mask_cond_single_timestep(points, boxes_3d):
    pass