# import numba
import numpy as np
from scipy.special import softmax

# @numba.jit(nopython=True, parallel=False)
# def scatter(array, index, value):
#     for (h, w), v in zip(index, value):
#         array[h, w] = v
#     return array

def scatter(array, index, value):
    h_idx, w_idx = index.T
    array[h_idx, w_idx] = value
    return array

def mask_points_with_distance(
    points: np.ndarray,
    min_depth: float = 1.45,
    max_depth: float = 80.0,):

    xyz = points[:, :3]  # xyz
    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    mask = (depth >= min_depth) & (depth <= max_depth)
    return mask[:,0].astype(np.bool_)

def load_points_as_images(
    point_path: str = None,
    points: np.ndarray = None,
    scan_unfolding: bool = True,
    H: int = 64,
    W: int = 2048,
    min_depth: float = 1.45,
    max_depth: float = 80.0,
    fov_up: float = 10.0,
    fov_down: float = -30.0,
    custom_feat_dim: int = 0
):
    assert point_path is not None or points is not None, "Either point_path or points must be provided."
    if point_path is not None:
        # load xyz & intensity and add depth & mask
        points = np.fromfile(point_path, dtype=np.float32).reshape(-1, 5)[:,:4]
    xyz = points[:, :3]  # xyz
    x = xyz[:, [0]]
    y = xyz[:, [1]]
    z = xyz[:, [2]]
    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    mask = (depth >= min_depth) & (depth <= max_depth)
    points = np.concatenate([points, depth, mask], axis=1)

    if scan_unfolding:
        # the i-th quadrant
        # suppose the points are ordered counterclockwise
        quads = np.zeros_like(x, dtype=np.int32)
        quads[(x >= 0) & (y >= 0)] = 0  # 1st
        quads[(x < 0) & (y >= 0)] = 1  # 2nd
        quads[(x < 0) & (y < 0)] = 2  # 3rd
        quads[(x >= 0) & (y < 0)] = 3  # 4th

        # split between the 3rd and 1st quadrants
        diff = np.roll(quads, shift=1, axis=0) - quads
        delim_inds, _ = np.where(diff == 3)  # number of lines
        inds = list(delim_inds) + [len(points)]  # add the last index

        # vertical grid
        grid_h = np.zeros_like(x, dtype=np.int32)
        cur_ring_idx = H - 1  # ...0
        for i in reversed(range(len(delim_inds))):
            grid_h[inds[i] : inds[i + 1]] = cur_ring_idx
            if cur_ring_idx >= 0:
                cur_ring_idx -= 1
            else:
                break
    else:
        h_up, h_down = np.deg2rad(fov_up), np.deg2rad(fov_down)
        elevation = np.arcsin(z / (depth+1e-6)) + abs(h_down)
        grid_h = 1 - elevation / (h_up - h_down)
        grid_h = np.floor(grid_h * H).clip(0, H - 1).astype(np.int32)

    # horizontal grid
    azimuth = -np.arctan2(y, x)  # [-pi,pi]
    grid_w = (azimuth / np.pi + 1) / 2 % 1  # [0,1]
    grid_w = np.floor(grid_w * W).clip(0, W - 1).astype(np.int32)

    grid = np.concatenate((grid_h, grid_w), axis=1)

    # projection
    order = np.argsort(-depth.squeeze(1))
    proj_points = np.zeros((H, W, 4 + custom_feat_dim + 2), dtype=points.dtype)
    proj_points = scatter(proj_points, grid[order], points[order])

    return proj_points.astype(np.float32)

def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def convert_boxes_to_2d(
    boxes_3d: np.ndarray,
    H: int = 64,
    W: int = 2048,
    min_depth: float = 1.45,
    max_depth: float = 80.0,
    fov_up: float = 10.0,
    fov_down: float = -30.0
):
    condition_mask = np.zeros([2, H, W], dtype=np.float32)
    num_box = boxes_3d.shape[0]
    scene_loss_weight_map = np.zeros([H, W, num_box], dtype=np.float32)
    boxes_vec_points = np.zeros([num_box, 3, 8])
    l,w,h = boxes_3d[:,3], boxes_3d[:,4], boxes_3d[:,5]
    c_xyz = boxes_3d[:,:3][:,:, None]
    c_depth = np.linalg.norm(c_xyz, ord=2, axis=1, keepdims=True) + 1e-6  # avoid division by zero

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    boxes_vec_points[:, 0, :] = np.transpose(np.stack(x_corners))
    boxes_vec_points[:, 1, :] = np.transpose(np.stack(y_corners))
    boxes_vec_points[:, 2, :] = np.transpose(np.stack(z_corners))

    rotzs = []
    for box in boxes_3d:
        rotzs.append(rotz(box[6]))
    rotzs = np.stack(rotzs)

    corners_3d = rotzs @ boxes_vec_points # N, 3, 8
    corners_3d += c_xyz
    corners_3d = np.transpose(corners_3d, (0,2,1)).reshape(-1, 3)

    corners_2d = convert_points_to_2d(
                                      points=corners_3d, 
                                      H=H, 
                                      W=W, 
                                      min_depth=min_depth,
                                      max_depth=max_depth,
                                      fov_up=fov_up, 
                                      fov_down=fov_down)
    corners_2d = corners_2d.reshape(num_box, 8, 2)
    corners_2d = [np.min(corners_2d[...,0], axis=1), np.min(corners_2d[...,1], axis=1), 
                  np.max(corners_2d[...,0], axis=1), np.max(corners_2d[...,1], axis=1)]
    corners_2d = np.stack(corners_2d, axis=0)
    object_areas_list = []
    for corner_id, corner in enumerate(np.transpose(corners_2d, (1,0))):
        x1, y1, x2, y2 = corner
        x1, x2 = int(x1 * W), int(x2 * W)
        y1, y2 = int(y1 * H), int(y2 * H)

        # corner case
        if (x2 - x1) / W > 0.6:
            # sementation mask
            condition_mask[0, y1:y2, 0:x1] = boxes_3d[corner_id, 7]
            condition_mask[0, y1:y2, x2:W] = boxes_3d[corner_id, 7]
            # depth mask
            condition_mask[1, y1:y2, 0:x1] = c_depth[corner_id, 0, 0]
            condition_mask[1, y1:y2, x2:W] = c_depth[corner_id, 0, 0]
            # area
            area = (W - x2 + x1) * (y2 - y1)
            object_areas_list.append(area)
            scene_loss_weight_map[y1:y2, 0:x1, corner_id] = 1.0
            scene_loss_weight_map[y1:y2, x2:W, corner_id] = 1.0

        else:
            # sementation mask
            condition_mask[0, y1:y2, x1:x2] = boxes_3d[corner_id, 7]
            # depth mask
            condition_mask[1, y1:y2, x1:x2] = c_depth[corner_id, 0, 0]
            # area
            area = (x2 - x1) * (y2 - y1)
            object_areas_list.append(area)
            scene_loss_weight_map[y1:y2, x1:x2, corner_id] = 1.0

    object_areas = np.array(object_areas_list, dtype=np.float32)
    object_weights = (3 - (object_areas) / np.max(object_areas))[np.newaxis, np.newaxis, :]
    scene_loss_weight_map = scene_loss_weight_map * object_weights
    scene_loss_weight_map = np.sum(scene_loss_weight_map, axis=-1)
    scene_loss_weight_map = np.exp(scene_loss_weight_map)  # [H, W]

    return corners_2d.transpose(1,0), condition_mask, scene_loss_weight_map


def convert_points_to_2d(
    points: np.ndarray,
    H: int = 64,
    W: int = 2048,
    min_depth: float = 1.45,
    max_depth: float = 80.0,
    fov_up: float = 10.0,
    fov_down: float = -30.0

):
    xyz = points[:, :3]  # xyz
    x = xyz[:, [0]]
    y = xyz[:, [1]]
    z = xyz[:, [2]]
    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True) +1e-6  # avoid division by zero
    # mask = (depth >= min_depth) & (depth <= max_depth)
    # points = np.concatenate([points, depth, mask], axis=1)

    h_up, h_down = np.deg2rad(fov_up), np.deg2rad(fov_down)
    elevation = np.arcsin(z / depth) + abs(h_down)
    grid_h = 1 - elevation / (h_up - h_down)
    grid_h = np.floor(grid_h * H).clip(0, H - 1)
    grid_h = grid_h / H

    # horizontal grid
    azimuth = -np.arctan2(y, x)  # [-pi,pi]
    grid_w = (azimuth / np.pi + 1) / 2 % 1  # [0,1]
    grid_w = np.floor(grid_w * W).clip(0, W - 1)
    grid_w = grid_w / W

    grid = np.concatenate((grid_w, grid_h), axis=1)

    return grid