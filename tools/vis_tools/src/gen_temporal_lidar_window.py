import os
import copy
import socket
import cv2
import pickle
import numpy as np
from pathlib import Path
from natsort import natsorted
from loguru import logger
from tqdm import tqdm
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QHBoxLayout, QVBoxLayout, QWidget, QGridLayout,\
                            QComboBox, QPushButton, QLabel, QFileDialog, QLineEdit, QTextEdit, QListWidget
import pyqtgraph.opengl as opengl

import torch

import sys
sys.path.append('../')
sys.path.append('./vis_tools')

from utils import gl_engine as gl
from lidargen.utils.lidar import LiDARUtility, get_linear_ray_angles
from lidargen.dataset import utils


def find_pth_files(root_dir, endwith='.pth'):
    pth_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(endwith):
                pth_files.append(os.path.join(dirpath, fname))
    return pth_files

class GenTemporalLidarWindow(QWidget):

    def __init__(self) -> None:
        super(GenTemporalLidarWindow, self).__init__()
        host_name = socket.gethostname()
        if host_name == 'Liang':
            self.monitor = QDesktopWidget().screenGeometry(0)
            self.monitor.setHeight(int(self.monitor.height()*0.8))
            self.monitor.setWidth(int(self.monitor.width()*0.6))
        else:
            self.monitor = QDesktopWidget().screenGeometry(0)
            self.monitor.setHeight(int(self.monitor.height()))
            self.monitor.setWidth(int(self.monitor.width()))

        self.setGeometry(self.monitor)
        self.setAcceptDrops(True)
        self.logger = logger.bind(name='cond_lidar_window')
        self.sample_index = 0
        self.current_timestamp = 0
        self.viewer_items = []
        self.gt_viewer_items = []
        # load infos
        data_infos = []
        data_info_path_list = [
            '../data/infos/nuscenes_infos_train.pkl',
            '../data/infos/nuscenes_infos_val.pkl'
        ]
        for info_path in data_info_path_list:
            with open(info_path, 'rb') as f:
                data_infos.extend(pickle.load(f)['infos'])
        self.logger.info(f'Loaded {len(data_infos)} data infos')
        self.data_infos_dict = {}
        for info in data_infos:
            self.data_infos_dict[info['token']] = info

        # lidar_utils
        self.lidar_utils = LiDARUtility(
            resolution=(32,1024),
            depth_format="log_depth",
            min_depth=1.45,
            max_depth=80.0,
            ray_angles=get_linear_ray_angles(
                H=32,
                W=1024,
                fov_up=10,
                fov_down=-30
            ))
        self.lidar_utils.eval()

        self.init_window()

    def init_window(self):
        main_layout = QHBoxLayout()
        self.init_load_file_window()
        self.init_display_window()
        self.init_temporal_window()
        main_layout.addLayout(self.load_file_layout)
        main_layout.addLayout(self.display_layout)
        main_layout.addLayout(self.temporal_layout)
        main_layout.setStretch(0, 2)
        main_layout.setStretch(1, 8)
        main_layout.setStretch(2, 2)
        self.setLayout(main_layout)

    def init_load_file_window(self):
        self.load_file_layout = QVBoxLayout()
        self.method_select_cbox = QComboBox()
        self.method_select_cbox.addItems(['our', 'opendwm', 'opendwm_dit', 'uniscene'])
        self.method_select_cbox.activated.connect(self.method_selected)
        self.load_file_layout.addWidget(self.method_select_cbox)
        self.frame_list_widget = QListWidget()
        self.frame_list_widget.currentRowChanged.connect(self.on_sequence_selected)
        self.load_file_layout.addWidget(self.frame_list_widget)
        temp_layout = QHBoxLayout()
        self.load_frames_btn = QPushButton('Load')
        self.load_frames_btn.clicked.connect(self.load_sequences)
        temp_layout.addWidget(self.load_frames_btn)
        # prev view
        self.prev_view_button = QPushButton('<<<')
        temp_layout.addWidget(self.prev_view_button)
        self.prev_view_button.clicked.connect(self.decrement_index)

        # Qlabel
        # show sample index
        self.sample_index_info = QLabel("")
        self.sample_index_info.setAlignment(Qt.AlignCenter)
        temp_layout.addWidget(self.sample_index_info)

        # Button
        # next view
        self.next_view_button = QPushButton('>>>')
        temp_layout.addWidget(self.next_view_button)
        self.next_view_button.clicked.connect(self.increment_index)
        self.load_file_layout.addLayout(temp_layout)

    def init_display_window(self):
        self.display_layout = QVBoxLayout()
        self.viewer = gl.AL_viewer()
        self.display_layout.addWidget(self.viewer)
        self.gt_viewer = gl.AL_viewer()
        self.display_layout.addWidget(self.gt_viewer)
        self.display_layout.setStretch(0, 5)
        self.display_layout.setStretch(1, 5)

    def init_temporal_window(self):
        self.temporal_layout = QVBoxLayout()
        self.temporal_frame_list_widget = QListWidget()
        self.temporal_frame_list_widget.currentRowChanged.connect(self.on_temporal_frame_selected)
        self.temporal_layout.addWidget(self.temporal_frame_list_widget)
        self.save_sqeuence_btn = QPushButton('Save Sequence')
        self.save_sqeuence_btn.clicked.connect(self.save_sequence)
        self.temporal_layout.addWidget(self.save_sqeuence_btn)

    def method_selected(self, index):
        selected_method = self.method_select_cbox.itemText(index)
        self.method_name = selected_method

    def load_sequences(self):
        generated_sequence_path = os.path.join('../generated_results', f'{self.method_name}', 'temporal_points')
        sequence_names_list = os.listdir(generated_sequence_path)
        self.sequence_path_list = [os.path.join(generated_sequence_path, seq_name) for seq_name in sequence_names_list]
        self.frame_list_widget.clear()
        self.frame_list_widget.addItems([Path(f).name for f in self.sequence_path_list])
        self.on_sequence_selected(self.sample_index)
        self.show_sample()

    def decrement_index(self) -> None:

        self.sample_index -= 1
        self.check_index_overflow()
        self.show_sample()

    def increment_index(self) -> None:

        self.sample_index += 1
        self.check_index_overflow()
        self.show_sample()

    def check_index_overflow(self) -> None:

        if self.sample_index == -1:
            self.sample_index = len(self.sequence_path_list) - 1

        if self.sample_index >= len(self.sequence_path_list):
            self.sample_index = 0

    def on_sequence_selected(self, index):
        self.sample_index = index
        frame_names = os.listdir(self.sequence_path_list[self.sample_index])
        self.frame_path_list = [os.path.join(self.sequence_path_list[self.sample_index], f) for f in frame_names]
        # order the path list
        self.frame_path_list = natsorted(self.frame_path_list)
        self.temporal_frame_list_widget.clear()
        self.temporal_frame_list_widget.addItems([Path(f).name for f in self.frame_path_list])
        self.show_sample()
        
    def on_temporal_frame_selected(self, row):
        self.current_timestamp = row
        self.show_sample()

    def load_data_dict(self):
        frame_path = self.frame_path_list[self.current_timestamp]
        points = np.loadtxt(frame_path)[:, :3]
        token = Path(frame_path).stem.split('_')[-1] 
        return {
            'points': points,
            'token': token,
        }

    def show_points(self):
        points = self.data_dict['points'].astype(np.float32)
        mesh = gl.get_points_mesh(points[:, :3], 5)
        self.viewer.addItem(mesh)                     # ✅ 立刻 addItem
        self.viewer_items.append(mesh)
        self.viewer.add_coordinate_system()

    def show_gt_points(self):
        token = self.data_dict['token']
        data_info = self.data_infos_dict[token]
        lidar_path = os.path.join('../data/nuscenes', data_info['lidar_path'])
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
        mesh = gl.get_points_mesh(points[:, :3], 5)
        self.gt_viewer.addItem(mesh)                  # ✅ 立刻 addItem
        self.gt_viewer_items.append(mesh)
        self.gt_viewer.add_coordinate_system()

    def add_boxes_to_viewer(self, box_info, custom_viewer=None, item_cache=None):
        use_viewer = self.viewer if custom_viewer is None else custom_viewer
        use_cache = self.viewer_items if item_cache is None else item_cache

        for box_item, l1_item, l2_item in zip(box_info['box_items'], box_info['l1_items'], box_info['l2_items']):
            use_viewer.addItem(box_item)
            use_viewer.addItem(l1_item)
            use_viewer.addItem(l2_item)
            use_cache.extend([box_item, l1_item, l2_item])

        use_cache.extend(box_info.get('score_items', []))
        use_cache.extend(box_info.get('text_items', []))

        for item in box_info.get('score_items', []):
            use_viewer.addItem(item)
        for item in box_info.get('text_items', []):
            use_viewer.addItem(item)


    def show_boxes(self):
        needed_class = np.array(['car', 'truck', 'bus'])
        token = self.data_dict['token']
        if token == '':
            return
        data_info = self.data_infos_dict[token]
        boxes_3d = data_info['gt_boxes'][:, :7]
        boxes_names = data_info['gt_names']
        mask = np.isin(boxes_names, needed_class)
        boxes_3d = boxes_3d[mask]
        boxes_names = boxes_names[mask]
        box_info = gl.create_boxes(bboxes_3d=boxes_3d, box_texts=boxes_names)
        # self.add_boxes_to_viewer(box_info)
        self.add_boxes_to_viewer(box_info, custom_viewer=self.gt_viewer, item_cache=self.gt_viewer_items)

    def clear_viewers(self):
        self.logger.info(f"Clearing {len(self.viewer_items)} items from viewer")
        for item in self.viewer_items:
            self.viewer.removeItem(item)
        for item in self.gt_viewer_items:
            self.gt_viewer.removeItem(item)
        self.viewer_items.clear()
        self.gt_viewer_items.clear()
        self.viewer.clear()
        self.gt_viewer.clear()

    def save_sequence(self):
        save_folder = 'ALTest/seq_example/example/experiments_seq'
        os.makedirs(save_folder, exist_ok=True)
        seq_num = len(os.listdir(save_folder))
        seq_save_path = os.path.join(save_folder, f'seq_{seq_num:04d}')
        os.makedirs(seq_save_path, exist_ok=True)
        self.logger.info(f'Saving sequence to {seq_save_path}')

        points_list = []
        save_points_path = []
        boxes = dict()
        frame_name = self.sequence_path_list[self.sample_index]
        first_token = Path(frame_name).stem.split('_')[-1]

        # generated points
        for method in ['our', 'opendwm', 'opendwm_dit', 'uniscene']:
            method_frame_path = f'../generated_results/{method}/temporal_points'
            if first_token in os.listdir(method_frame_path):
                frame_name = os.path.join(method_frame_path, first_token)
                points_path = os.listdir(frame_name)
                points_path = natsorted(points_path)
                for point_id, point_path in enumerate(tqdm(points_path, desc=f'Processing {method} points')):
                    token = Path(point_path).stem.split('_')[-1]
                    point_file = os.path.join(frame_name, point_path)
                    points_list.append(np.loadtxt(point_file)[:, :3])
                    save_path = os.path.join(seq_save_path, f'{method}_{point_id}_{token}.txt')
                    save_points_path.append(save_path)

            else:
                self.logger.warning(f'Frame {frame_name} not found in method {method}, skipping...')
                return
        
        # gt points
        for point_id, point_path in enumerate(tqdm(points_path, desc=f'Processing {method} points')):
            token = Path(point_path).stem.split('_')[-1]
            data_info = self.data_infos_dict[token]
            lidar_path = os.path.join('../data/nuscenes', data_info['lidar_path'])
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
            points_list.append(points)

            # save gt boxes
            needed_class = np.array(['car', 'truck', 'bus'])
            boxes_3d = data_info['gt_boxes'][:, :7]
            boxes_names = data_info['gt_names']
            mask = np.isin(boxes_names, needed_class)
            boxes_3d = boxes_3d[mask]
            boxes_names = boxes_names[mask]
            boxes[point_id] = {
                'boxes': boxes_3d,
                'names': boxes_names
            }
            save_path = os.path.join(seq_save_path, f'gt_{point_id}_{token}.txt')
            save_points_path.append(save_path)
        
        # save points
        for point_id, points in enumerate(tqdm(points_list, desc='Saving points')):
            save_path = save_points_path[point_id]
            np.savetxt(save_path, points, fmt='%.6f')
        # save boxes
        boxes_save_path = os.path.join(seq_save_path, 'boxes.pkl')
        with open(boxes_save_path, 'wb') as f:
            pickle.dump(boxes, f)
        self.logger.info(f'Saved sequence to {seq_save_path}')

    def show_sample(self):
        self.clear_viewers()
        self.data_dict = self.load_data_dict()
        self.show_points()
        self.show_gt_points()
        self.show_boxes()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = GenTemporalLidarWindow()
    viewer.show()
    sys.exit(app.exec_())
