import numpy as np

OBJECT_SIZE_DICT = {
    'ego': np.array([0, 0, 0, 0], dtype=np.float32),
    'car': np.array([-1, 4.67, 1.95, 1.74], dtype=np.float32),
    'truck': np.array([-0.34, 7.12, 2.54, 2.90], dtype=np.float32),
    'construction_vehicle': np.array([-0.14, 6.58, 2.75, 3.22], dtype=np.float32),
    'bus': np.array([-0.02, 11.23, 2.94, 3.49], dtype=np.float32),
    'trailer': np.array([0.24, 12.02, 2.91, 3.84], dtype=np.float32),
    'motorcycle': np.array([-1.11, 2.07, 0.77, 1.44], dtype=np.float32),
    'bicycle': np.array([-1.10, 1.73, 0.62, 1.32], dtype=np.float32),
    'pedestrian': np.array([-0.81, 0.77, 0.69, 1.78], dtype=np.float32)
}

POINT_RANGE = np.array([-80,-80,-8,80,80,8], dtype=np.float32)


def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def box2coord3d(boxes_3d):
    num_box = boxes_3d.shape[0]
    boxes_vec_points = np.zeros([num_box, 3, 8])
    l,w,h = boxes_3d[:,3], boxes_3d[:,4], boxes_3d[:,5]
    c_xyz = boxes_3d[:,:3][:,:, None]

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

    return corners_3d

def normlize_ndarray(array):
    """Normalize a numpy array to the range [0, 1]."""
    if array.ndim == 0:
        return array
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val - min_val == 0:
        return np.zeros_like(array)  # Avoid division by zero
    return (array - min_val) / (max_val - min_val)

def warp_lidar_future(P: np.ndarray,
                      future_xy: np.ndarray,
                      z0: float = 0.0) -> np.ndarray:
    """
    将当前帧 LiDAR 点云 P（世界系）转换到未来 N 个时刻的 LiDAR_i 系下。

    Args:
      P:        (M,4) array，当前帧点云，[x,y,z,intensity]
      future_xy:(N,2) array，未来 N 时刻 (x,y) 轨迹（相对于当前帧系）
      z0:       float，高度（保持不变）

    Returns:
      warped:   (N, M, 4) array，第 i 时刻在 LiDAR_i 系下的 [x,y,z,intensity]
    """
    M = P.shape[0]
    N = future_xy.shape[0]

    # 预分配输出
    warped = np.zeros((N, M, 4), dtype=P.dtype)

    # 1) 计算每个时刻的航向 yaw
    #    offsets[0] = future_xy[0] (相对于原点的位移)
    #    offsets[i>=1] = future_xy[i] - future_xy[i-1]
    offsets = np.vstack((
        future_xy[0:1],
        future_xy[1:] - future_xy[:-1]
    ))  # (N,2)
    yaws = np.arctan2(offsets[:, 1], offsets[:, 0]) - np.pi/2  # (N,)
    yaws[np.linalg.norm(offsets, axis=1) < 1e-1] = 0.0 

    # 2) 对每个时刻做变换
    xyz = P[:, :3]  # (M,3)
    intensity = P[:, 3]  # (M,)
    for i in range(N):
        xi, yi = future_xy[i]
        yaw = yaws[i]
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        # 构造 R_z(yaw)
        R = np.array([
            [ cos_y, -sin_y, 0.0],
            [ sin_y,  cos_y, 0.0],
            [  0.0,    0.0,  1.0]
        ], dtype=P.dtype)

        # 平移：将点云移动到 LiDAR_i 原点
        translated = xyz - np.array([xi, yi, z0], dtype=P.dtype)  # (M,3)

        # 旋转：将世界系坐标旋转到 LiDAR_i 系
        rotated = translated.dot(R)  # (M,3)

        # 填回输出
        warped[i, :, :3] = rotated
        warped[i, :, 3] = intensity

    return warped


def warp_boxes_future(
    boxes0: np.ndarray,   # (K,7): [x0,y0,z0,w,h,l,yaw0]
    traj_obj: np.ndarray, # (K,N,2): object 相对初始中心的 (dx,dy)
    traj_ego: np.ndarray, # (N,2): ego 相对初始 LiDAR 坐标系的 (x,y)
    z_e: float            # ego height（保持 z 不变）
) -> np.ndarray:
    """
    返回: warped_boxes (K, N, 7)，每个物体在每个时刻 LiDAR_i 系的框 [x,y,z,w,h,l,yaw]
    """
    K, N = traj_obj.shape[0], traj_obj.shape[1]
    warped = np.zeros((K, N, 7), dtype=boxes0.dtype)

    #—— 1) 计算 ego yaw ——#
    ego_offsets = np.vstack((traj_ego[0:1], traj_ego[1:] - traj_ego[:-1]))  # (N,2)
    yaw_ego = np.arctan2(ego_offsets[:,1], ego_offsets[:,0]) - np.pi/2                # (N,)
    yaw_ego[np.linalg.norm(ego_offsets, axis=1) < 1e-1] = 0.0 
    for k in range(K):
        x0, y0, z0, w, h, l, yaw0 = boxes0[k]
        #—— 2) 计算 object yaw 序列 ——#
        obj_offsets = np.vstack(( [0,0], traj_obj[k,1:] - traj_obj[k,:-1] ))  # (N,2)
        yaw_obj = np.empty(N, dtype=boxes0.dtype)
        yaw_obj[0] = yaw0
        temp_yaw_obj = np.arctan2(obj_offsets[1:,1], obj_offsets[1:,0])
        offsets_keep = np.linalg.norm(obj_offsets[1:], axis=1) < 1e-3
        for i in range(1, N):
            if offsets_keep[i-1]:
                yaw_obj[i] = yaw_obj[i-1]
            else:
                yaw_obj[i] = temp_yaw_obj[i-1]

        #—— 3) 对每个时刻做世界->LiDAR_i 系变换 ——#
        for i in range(N):
            # 世界系下的物体中心
            C_world = np.array([x0, y0, z0]) + np.array([traj_obj[k,i,0],
                                                        traj_obj[k,i,1],
                                                        0.0], dtype=boxes0.dtype)
            # 平移到 LiDAR_i 原点
            C_trans = C_world - np.array([traj_ego[i,0], traj_ego[i,1], z_e],
                                         dtype=boxes0.dtype)
            # 旋转矩阵 Rz(-yaw_ego[i])
            cos_e, sin_e = np.cos(yaw_ego[i]), np.sin(yaw_ego[i])
            R = np.array([[ cos_e, sin_e, 0],
                          [-sin_e, cos_e, 0],
                          [   0,     0,   1]], dtype=boxes0.dtype)

            # 应用旋转
            C_lidar = R.dot(C_trans)  # (3,)

            # 修正 yaw
            yaw_lidar = yaw_obj[i] - yaw_ego[i]

            # 填入输出
            warped[k, i, :3] = C_lidar
            warped[k, i, 3:6] = [w, h, l]
            warped[k, i, 6] = yaw_lidar

    return warped

def compute_inter_frame_transforms(future_xy: np.ndarray,
                                   z0: float = 0.0):
    """
    计算从 LiDAR_i 系到 LiDAR_{i+1} 系的同质变换矩阵序列。

    Args:
      future_xy: (T+1,2) array，时刻 0..T 在初始 LiDAR 系下的 (x,y) 轨迹
      z0:        float，LiDAR 传感器高度（假设不变）

    Returns:
      Ms: (T,4,4) array，第 i 个矩阵 M_i 把时刻 i 下的点映射到时刻 i+1。
    """
    # 时间点数
    # add [0,0] at the beginning to make it (T+1,2)
    # future_xy = np.vstack((np.array([0.0, 0.0], dtype=future_xy.dtype), future_xy))

    T1 = future_xy.shape[0]

    # 1) 估计每个时刻的 yaw（航向）
    # offsets[0] = future_xy[0]；offsets[i] = future_xy[i] - future_xy[i-1]
    offsets = np.vstack((future_xy[0:1], future_xy[1:] - future_xy[:-1]))
    yaws = np.arctan2(offsets[:,1], offsets[:,0]) - np.pi/2   # (T+1,)
    yaws[np.linalg.norm(offsets, axis=1) < 0.1] = 0.0  # 小于阈值的航向设为 0
    # 2) 构造每帧的位姿
    poses = []
    poses.append(np.eye(4, dtype=float))  # 初始帧的位姿为单位矩阵
    for i in range(T1):
        x, y = future_xy[i]
        yaw = yaws[i]
        R = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw),  np.cos(yaw), 0.0],
            [0.0,          0.0,         1.0]
        ], dtype=float)
        t = np.array([x, y, z0], dtype=float)
        # 同质位姿矩阵
        P = np.eye(4, dtype=float)
        P[:3,:3] = R
        P[:3, 3] = t
        poses.append(P)
    
    # 3) 计算相邻帧变换 M_i = inv(P_{i+1}) @ P_i
    Ms = np.zeros((T1, 4, 4), dtype=float)
    for i in range(T1):
        P_i = poses[i]
        P_ip1 = poses[i+1]
        Ms[i] = np.linalg.inv(P_ip1) @ P_i

    return Ms