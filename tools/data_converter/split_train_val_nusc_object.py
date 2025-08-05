import pickle
import random

class_names = ['car', 'truck', 'bus']

def main(pkl_path, sample_num=10000):
    with open(pkl_path, 'rb') as fg_objects_file:
        fg_objects_dict = pickle.load(fg_objects_file)

    train_data = []
    val_data = []

    for class_idx, class_name in enumerate(class_names):
        fg_objects_list = [sample for sample in fg_objects_dict[class_name] if sample['num_points_in_gt'] > 50]
        random.shuffle(fg_objects_list)
        if len(fg_objects_list) > sample_num:
            fg_objects_list = fg_objects_list[:sample_num]
        split_idx = int(len(fg_objects_list) * 0.8)  # 80% for training
        train_data.extend(fg_objects_list[:split_idx])
        val_data.extend(fg_objects_list[split_idx:])

    all_class_names = list(fg_objects_dict.keys())
    negative_samples_per_class = sample_num // (len(all_class_names) - len(class_name))
    for class_idx, class_name in enumerate(all_class_names):
        if class_name in class_names:
            continue
        fg_objects_list = [sample for sample in fg_objects_dict[class_name] if sample['num_points_in_gt'] > 50]
        random.shuffle(fg_objects_list)
        if len(fg_objects_list) > negative_samples_per_class:
            fg_objects_list = fg_objects_list[:negative_samples_per_class]
        split_idx = int(len(fg_objects_list) * 0.8)
        train_data.extend(fg_objects_list[:split_idx])
        val_data.extend(fg_objects_list[split_idx:])

    # Shuffle the data
    random.shuffle(train_data)
    random.shuffle(val_data)
    # save as pkl
    with open('../data/infos/nuscenes_object_classification_train.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('../data/infos/nuscenes_object_classification_val.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    print(f"Train data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")

if __name__ == "__main__":
    main(pkl_path='../data/infos/nuscenes_dbinfos_10sweeps_withvelo.pkl')