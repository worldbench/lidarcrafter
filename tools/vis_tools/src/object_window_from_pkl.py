import os
import copy
import socket
import cv2
import pickle
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QHBoxLayout, QVBoxLayout, QWidget, QGridLayout,\
                            QComboBox, QPushButton, QLabel, QFileDialog, QLineEdit, QTextEdit, QListWidget

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

class ObjectLidarWindowPKL(QWidget):

    def __init__(self) -> None:
        super(ObjectLidarWindowPKL, self).__init__()
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
        main_layout.addLayout(self.load_file_layout)
        main_layout.addLayout(self.display_layout)
        main_layout.setStretch(0, 2)
        main_layout.setStretch(1, 8)
        self.setLayout(main_layout)

    def init_load_file_window(self):
        self.load_file_layout = QVBoxLayout()
        self.method_select_cbox = QComboBox()
        self.method_select_cbox.addItems(['our', 'opendwm', 'opendwm_dit', 'uniscene',
                                          'lidargen', 'lidm', 'r2dm'])
        self.method_select_cbox.activated.connect(self.method_selected)
        self.load_file_layout.addWidget(self.method_select_cbox)
        self.frame_list_widget = QListWidget()
        self.frame_list_widget.currentRowChanged.connect(self.on_frame_selected)
        self.load_file_layout.addWidget(self.frame_list_widget)
        temp_layout = QHBoxLayout()
        self.load_frames_btn = QPushButton('Load')
        self.load_frames_btn.clicked.connect(self.load_frames)
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

        self.save_screen_image_button = QPushButton('Save Screenshot')
        self.load_file_layout.addWidget(self.save_screen_image_button)
        self.save_screen_image_button.clicked.connect(self.save_screenshot)

        self.save_screen_video_button = QPushButton('Save Video')
        self.load_file_layout.addWidget(self.save_screen_video_button)
        self.save_screen_video_button.clicked.connect(self.save_video)

    def init_display_window(self):
        self.display_layout = QVBoxLayout()
        self.viewer = gl.AL_viewer()
        self.display_layout.addWidget(self.viewer)

    def method_selected(self, index):
        selected_method = self.method_select_cbox.itemText(index)
        self.method_name = selected_method

    def load_frames(self):
        generated_file_path = os.path.join('../generated_results', f'{self.method_name}', 'inference_results/selected_foreground_samples_info.pkl')
        sample_infos_dict = pickle.load(open(generated_file_path, 'rb'))
        sample_infos = []
        for ket, value in sample_infos_dict.items():
            sample_infos.extend(value)
        self.sample_infos = sample_infos

        self.frame_list_widget.clear()
        self.frame_list_widget.addItems([f"{info['name']}:{info['score']}" for info in sample_infos])

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
            self.sample_index = len(self.sample_infos) - 1

        if self.sample_index >= len(self.sample_infos):
            self.sample_index = 0

    def on_frame_selected(self, index):
        self.sample_index = index
        self.show_sample()
        
    def load_data_dict(self):
        sample_info = self.sample_infos[self.sample_index]
        lidar_path = sample_info['path']
        points = np.fromfile(lidar_path, dtype=np.float32)
        if points.size % 4 != 0:
            points = points.reshape(-1, 3)  # fallback to 3 columns if not divisible by 4
        else:
            points = points.reshape(-1, 4)
        box_3d = np.array(sample_info['box3d_lidar'])
        box_3d[:3] = 0
        return {
            'points': points,
            'box_3d': box_3d,
        }

    def show_points(self):
        points = self.data_dict['points']
        mesh = gl.get_points_mesh(points[:,:3], 7)
        self.current_mesh = mesh
        self.viewer.addItem(mesh)

    def add_boxes_to_viewer(self, box_info, custom_viewer=None):
        # keep points
        # self.reset_viewer(only_viewer=True)
        # self.viewer.addItem(self.current_mesh)
        use_viewer = self.viewer if custom_viewer is None else custom_viewer

        for box_item, l1_item, l2_item in zip(box_info['box_items'], box_info['l1_items'],\
                                                        box_info['l2_items']):
            use_viewer.addItem(box_item)
            use_viewer.addItem(l1_item)
            use_viewer.addItem(l2_item)
            
        if len(box_info['score_items']) > 0:
            for score_item in box_info['score_items']:
                use_viewer.addItem(score_item)

        if len(box_info['text_items']) > 0:
            for text_item in box_info['text_items']:
                use_viewer.addItem(text_item)

    def show_boxes(self):
        boxes_3d = self.data_dict['box_3d']
        box_info = gl.create_boxes(bboxes_3d=boxes_3d[None, :])
        self.add_boxes_to_viewer(box_info)

    def show_sample(self):
        self.viewer.clear()
        self.data_dict = self.load_data_dict()
        self.show_points()
        self.show_boxes()

    def save_screenshot(self):
        save_folder_path = 'ALTest/USER/good_example/nuscenes/screen_shot'
        img_num = len(os.listdir(save_folder_path))
        save_path = os.path.join(save_folder_path, f'screenshot_{img_num:04d}.png')
        img = self.viewer.grabFramebuffer()
        img.save(save_path, "PNG")
        self.logger.info(f'Saved screenshot to {save_path}')

    def save_video(self):
        self.viewer.start_record()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ObjectLidarWindowPKL()
    viewer.show()
    sys.exit(app.exec_())
