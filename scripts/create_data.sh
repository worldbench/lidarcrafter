export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python setup.py develop

python tools/data_converter/nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes \
    --canbus ./data/nuscenes \
    --out-dir ./data/infos/ \
    --extra-tag nuscenes \
    --version v1.0-mini

python tools/data_converter/nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes \
    --canbus ./data/nuscenes \
    --out-dir ./data/infos/ \
    --extra-tag nuscenes \
    --version v1.0

python tools/data_converter/prepare_nusc_layout_dataset.py
python tools/data_converter/generate_nusc_obj_text_feature.py --save_path data/clips/nuscenes/obj_text_feat.pkl

cd scripts
python ../tools/data_converter/prepare_scene_graph_feat.py

python data_converter/generate_box_condition.py

cd ../tools/
python data_converter/split_train_val_nusc_object.py