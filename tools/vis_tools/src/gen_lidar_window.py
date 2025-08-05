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

class CondLidarWindow(QWidget):

    def __init__(self) -> None:
        super(CondLidarWindow, self).__init__()
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

        self.save_sample_button = QPushButton('Save Sample')
        self.load_file_layout.addWidget(self.save_sample_button)
        self.save_sample_button.clicked.connect(self.save_sample)

        self.save_screen_image_button = QPushButton('Save Screenshot')
        self.load_file_layout.addWidget(self.save_screen_image_button)
        self.save_screen_image_button.clicked.connect(self.save_screenshot)

    def init_display_window(self):
        self.display_layout = QVBoxLayout()
        self.viewer = gl.AL_viewer()
        self.display_layout.addWidget(self.viewer)

    def method_selected(self, index):
        selected_method = self.method_select_cbox.itemText(index)
        self.method_name = selected_method

    def load_frames(self):
        generated_file_path = os.path.join('../generated_results', f'{self.method_name}')
        if 'dwm' in self.method_name:
            # find all *.txt files
            self.endwith = '.txt'
        elif self.method_name == 'uniscene':
            self.endwith = '.npy'
        else:
            self.endwith = '.pth'

        self.frame_path_list = find_pth_files(generated_file_path, endwith=self.endwith)
        self.frame_list_widget.clear()
        self.frame_list_widget.addItems([Path(f).name for f in self.frame_path_list])
        detection_result_info_path = f'../generated_results/{self.method_name}/inference_results/result_vxrcnn.pkl'
        if os.path.exists(detection_result_info_path):
            with open(detection_result_info_path, 'rb') as f:
                self.logger.info(f'Loaded detection results from {detection_result_info_path}')
                self.pred_data_infos = pickle.load(f)
        if 'frame_id' in self.pred_data_infos[0]:
            self.pred_data_infos_dict = {}
            for info in self.pred_data_infos:
                token = info['frame_id']
                self.pred_data_infos_dict[token] = info

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
            self.sample_index = len(self.frame_path_list) - 1

        if self.sample_index >= len(self.frame_path_list):
            self.sample_index = 0

    def on_frame_selected(self, index):
        self.sample_index = index
        self.show_sample()
        
    def load_data_dict(self):
        if 'dwm' in self.method_name:
            points_path = self.frame_path_list[self.sample_index]
            points = np.loadtxt(points_path)[:, :3]
            rotation = np.array(np.pi) / 2
            points = utils.rotate_points_along_z(points[np.newaxis, :, :], rotation.reshape(1))[0]
            points[:, 2] -= 2.0
            token = Path(points_path).stem.split('.')[0]
        
        elif self.method_name == 'uniscene':
            points_path = self.frame_path_list[self.sample_index]
            points = np.load(points_path)[:, :3]
            rotation = np.array(np.pi) / 2
            points = utils.rotate_points_along_z(points[np.newaxis, :, :], rotation.reshape(1))[0]
            token = Path(points_path).stem.split('.')[0].split('-')[0]
            self.method_name == 'uniscene'
        elif self.method_name in ['lidargen', 'lidm', 'r2dm']:
            range_img = torch.load(self.frame_path_list[self.sample_index], map_location="cpu")
            depth = range_img[[0]]
            if range_img.shape[0] == 2:
                depth = self.lidar_utils.revert_depth(depth)
                points = self.lidar_utils.to_xyz(depth.unsqueeze(0)) # [1, 3, 32, 1024]
                points = points.reshape(3,-1).permute(1,0).numpy()  # [H*W, 4]
            else:
                xyz = range_img[[1,2,3]]
                points = xyz.reshape(3, -1).permute(1, 0)  # (H, W, 3) -> (H*W, 3)
            token = ''

        else:
            range_img_path = self.frame_path_list[self.sample_index]
            range_img = torch.load(range_img_path,  map_location="cpu")
            xyz = range_img[[1,2,3]]
            points = xyz.reshape(3, -1).permute(1, 0)  # (H, W, 3) -> (H*W, 3)
            token = Path(range_img_path).stem.split('.')[0].split('_')[-1]

        return {
            'points': points,
            'token': token,
        }

    def show_points(self):
        points = self.data_dict['points']
        mesh = gl.get_points_mesh(points[:,:3], 5)
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
        needed_class = np.array(['car', 'truck', 'bus'])

        # show pred
        if hasattr(self, 'pred_data_infos'):
            index = self.sample_index
            # pred_dict = self.pred_data_infos[index]
            token = self.data_dict['token']
            pred_dict = self.pred_data_infos_dict.get(token, None)
            if pred_dict is not None:
                boxes_3d = pred_dict['boxes_lidar'][:, :7]
                boxes_names = pred_dict['name']
            else:
                pred_dict = self.pred_data_infos[index]
                boxes_3d = pred_dict['boxes_lidar'][:, :7]
                boxes_names = pred_dict['name']
            score_mask = pred_dict['score'] > 0.3
            mask = np.isin(boxes_names, needed_class)
            mask = mask & score_mask
            boxes_3d = boxes_3d[mask]
            boxes_names = boxes_names[mask]
            # add category
            boxes_3d = np.concatenate([boxes_3d, np.ones((boxes_3d.shape[0], 1))], axis=1)  # add category
            box_info = gl.create_boxes(bboxes_3d=boxes_3d, box_texts=boxes_names)
            self.add_boxes_to_viewer(box_info)

        token = self.data_dict['token']
        if token == '':
            return
        # show gt
        # data_info = self.data_infos_dict[token]
        # boxes_3d = data_info['gt_boxes'][:, :7]
        # boxes_names = data_info['gt_names']
        # mask = np.isin(boxes_names, needed_class)
        # boxes_3d = boxes_3d[mask]
        # boxes_names = boxes_names[mask]
        # box_info = gl.create_boxes(bboxes_3d=boxes_3d, box_texts=boxes_names)
        # self.add_boxes_to_viewer(box_info)

    def show_sample(self):
        self.viewer.clear()
        self.data_dict = self.load_data_dict()
        self.show_points()
        self.show_boxes()

    def save_sample(self):
        save_folder_path = f'ALTest/seq_example/example/condition_generation/{self.method_name}'
        os.makedirs(save_folder_path, exist_ok=True)
        file_num = len(os.listdir(save_folder_path))
        points = self.data_dict['points'][:,:3]
        point_save_name = f'{str(file_num).zfill(4)}_{self.data_dict["token"]}.txt'
        np.savetxt(os.path.join(save_folder_path, point_save_name), points, fmt='%.4f')
        self.logger.info(f'Saved sample to {os.path.join(save_folder_path, point_save_name)}')

    def save_screenshot(self):
        save_folder_path = 'ALTest/USER/good_example/nuscenes/screen_shot'
        img_num = len(os.listdir(save_folder_path))
        save_path = os.path.join(save_folder_path, f'screenshot_{img_num:04d}.png')
        img = self.viewer.grabFramebuffer()
        img.save(save_path, "PNG")
        self.logger.info(f'Saved screenshot to {save_path}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = CondLidarWindow()
    viewer.show()
    sys.exit(app.exec_())
