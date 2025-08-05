import sys
from tqdm import tqdm
sys.path.append('/data/yyang/workspace/projects/LidarGen4D')
sys.path.append('/data1/liangao/AlanLiang/Projects/LidarGen4D')
from lidargen.utils.configs import __all__
from lidargen.dataset import __all__ as all_datasets

if __name__ == "__main__":
    cfg = __all__['nuscenes-box-layout']()
    cfg.data.split = 'train'
    cfg.data.data_root = "../data/nuscenes"
    cfg.data.pkl_path = "../data/infos/nuscenes_infos_lidargen_train.pkl"
    dataset = all_datasets[cfg.data.dataset](cfg.data)

    for idx in tqdm(range(len(dataset)), 'Generating scene graph features...'):
        dataset.__getitem__(idx)

    cfg.data.split = 'val'
    cfg.data.pkl_path = "../data/infos/nuscenes_infos_lidargen_val.pkl"
    dataset = all_datasets[cfg.data.dataset](cfg.data)

    for idx in tqdm(range(len(dataset)), 'Generating scene graph features...'):
        dataset.__getitem__(idx)