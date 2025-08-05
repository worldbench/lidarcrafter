import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from pcdet.utils.box_utils import mask_boxes_outside_range_numpy
# DATA_ROOT = '/data1/liangao/AlanLiang/Projects/LidarGen4D/data/nuscenes'
DATA_ROOT = 'data/nuscenes'

KEEP_NAMES = ['car','truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']
RELATIONSHIPS = [
    'left',
    'right',
    'front',
    'behind',
    'close by',
    'bigger than',
    'smaller than',
    'taller than',
    'shorter than',
]

def cal_box_relationships(subject_box, object_box, ego=False):
    pair_relationships = []
    s_y = subject_box[1]
    o_y = object_box[1]
    # front and behind
    if s_y >= o_y:
        pair_relationships.append('front')
    else:
        pair_relationships.append('behind')

    s_x = subject_box[0]
    o_x = object_box[0]
    # right and left
    if s_x >= o_x:
        pair_relationships.append('right')
    else:
        pair_relationships.append('left')

    # distance
    distance = np.sqrt((s_y-o_y)**2 + (s_x-o_x)**2)
    if distance < 4:
        pair_relationships.append('close by')

    # big and small
    if not ego:
        v_s = subject_box[3] * subject_box[4] * subject_box[5]
        v_o = object_box[3] * object_box[4] * object_box[5]
        if v_s >= v_o:
            pair_relationships.append('bigger than')
        else:
            pair_relationships.append('smaller than')

    # taller and shorter
    if not ego:
        height_s = subject_box[2] + subject_box[5]  / 2
        height_o = object_box[2] + object_box[5]  / 2
    else:
        height_s = subject_box[2]
        height_o = 0
        
    if height_s >= height_o:
        pair_relationships.append('taller than')
    else:
        pair_relationships.append('shorter than')
        
    return pair_relationships

def save_data_info_for_scene_graph(data_infos):
    scene_graph_infos = []
    for i in tqdm(range(len(data_infos))):
        info = data_infos[i]
        # filter objects
        box_range_mask = mask_boxes_outside_range_numpy(
            info['gt_boxes'], [-80,-80,-8,80,80,8]
        )
        gt_names = info['gt_names'][box_range_mask]
        num_lidar_pts = info['num_lidar_pts'][box_range_mask]
        gt_boxes = info['gt_boxes'][box_range_mask]
        gt_box_fut_trajs = info['gt_agent_fut_trajs'][box_range_mask]
        gt_agent_fut_masks = info['gt_agent_fut_masks'][box_range_mask]
        gt_agent_fut_states = np.array(info['gt_agent_fut_states'])[box_range_mask]
        instance_inds = np.array(info['instance_inds'])[box_range_mask]

        gt_names_mask = [cat in KEEP_NAMES for cat in gt_names]
        pts_mask = [num > 30 for num in num_lidar_pts]
        mask = [x and y for x, y in zip(gt_names_mask, pts_mask)]
        if np.bool_(mask).sum()==0:
            info.update({
                'scene_graph': {
                    'valid': False
                }
            })
            scene_graph_infos.append(info)
            continue
        keep_box_names = gt_names[np.bool_(mask)]
        keep_box = gt_boxes[np.bool_(mask)]
        keep_box_fut_trajs = gt_box_fut_trajs[np.bool_(mask)]
        keep_agent_fut_masks = gt_agent_fut_masks[np.bool_(mask)]
        keep_agent_fut_states = gt_agent_fut_states[np.bool_(mask)]
        keep_agent_instance_inds = instance_inds[np.bool_(mask)]

        matrix_mask = [
        [(i != j) for j in range(keep_box_names.shape[0])]
        for i in range(keep_box_names.shape[0])
        ]
        matrix_mask = np.bool_(matrix_mask)
        keep_box_relationships = []
        for i in range(matrix_mask.shape[0]):
            for j in range(matrix_mask.shape[0]):
                if matrix_mask[i,j]:
                    subject_box = keep_box[i]
                    object_box = keep_box[j]
                    relationships = cal_box_relationships(subject_box, object_box)
                    for rela in relationships:
                        keep_box_relationships.append([i+1, RELATIONSHIPS.index(rela), j+1])
                    matrix_mask[j,i] = False
                else:
                    continue
        # ego
        ego_box = np.zeros([9])
        for i in range(matrix_mask.shape[0]):
            subject_box = keep_box[i]
            relationships = cal_box_relationships(subject_box, ego_box, ego=True)
            for rela in relationships:
                keep_box_relationships.append([i+1, RELATIONSHIPS.index(rela), 0])
        ego_fut_trajs = info['gt_ego_fut_trajs']
        ego_fut_masks = info['gt_ego_fut_masks']
        ego_fut_state = info['gt_ego_fut_state']

        info.update({
            'scene_graph': {
                'valid': True,
                'lidar_path': info['lidar_path'],
                'keep_box_names': np.insert(keep_box_names, 0, 'ego'),
                'keep_box': np.vstack((ego_box[np.newaxis, :7], keep_box)),
                'keep_box_relationships': keep_box_relationships,
                'keep_agent_fut_trajs': np.concatenate([ego_fut_trajs[np.newaxis,:,:], keep_box_fut_trajs], axis=0),
                'keep_agent_fut_masks': np.vstack((ego_fut_masks[np.newaxis, :], keep_agent_fut_masks)),
                # 'keep_agent_fut_states': np.insert(keep_agent_fut_states, 0, ego_fut_state)
                'keep_agent_fut_states': np.array([ego_fut_state] + list(keep_agent_fut_states)), # TODO: Maybe "Unkonwn State" is better?
                'keep_agent_instance_inds': np.insert(keep_agent_instance_inds, 0, -1), # -1 for ego
            }
        })
        scene_graph_infos.append(info)
        # print(info['scene_graph']['keep_agent_fut_states'])
    return scene_graph_infos

def generate_scene_graph(split):
    # load train .pkl 
    pkl_file_path = Path(DATA_ROOT).parent / 'infos' / f'nuscenes_infos_{split}.pkl'
    with open(pkl_file_path, 'rb') as f:
        data_infos = pickle.load(f)['infos']
    scene_data_infos = save_data_info_for_scene_graph(data_infos)
    with open(Path(DATA_ROOT).parent / 'infos' / f'nuscenes_infos_lidargen_{split}.pkl', 'wb') as f:
        pickle.dump(scene_data_infos, f)


if __name__ == '__main__':
    # train
    generate_scene_graph('train')
    # val
    generate_scene_graph('val')