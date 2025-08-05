from nuscenes.nuscenes import NuScenes
import pickle
from tqdm import tqdm

nusc = NuScenes(version='v1.0-trainval', dataroot='../data/nuscenes', verbose=True)

def main(split = ['train', 'val']):

    for s in split:
        pkl_path = f'../data/infos/nuscenes_infos_lidargen_{s}.pkl'
        with open(pkl_path, 'rb') as f:
            data_infos = pickle.load(f)

        for df in tqdm(data_infos, desc="Updating ground truth segment paths"):
            sample_token = df['token']
            sample = nusc.get('sample', sample_token)
            sd_token = sample['data']['LIDAR_TOP']
            sd = nusc.get('sample_data', sd_token)
            file_name = sd['filename'] 
            gt_segment_path = nusc.get("lidarseg", sd_token)["filename"]
            df['gt_segment_path'] = gt_segment_path

        pickle.dump(data_infos, open(pkl_path, 'wb'))
        # Save the updated data_infos back to the pkl file
        print(f"Updated {pkl_path} with gt_segment_path.")
        # This script updates the nuscenes data infos with the ground truth segment path for each sample.

if __name__ == "__main__":
    main(split=['val'])
    # You can specify the split you want to update, e.g., ['train'], ['val'], or ['train', 'val'].
    # This will update the pkl files with the ground truth segment paths for the specified splits.