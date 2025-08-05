import os
import copy
import socket
import cv2
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QHBoxLayout, QVBoxLayout, QWidget, QGridLayout,\
                            QComboBox, QPushButton, QLabel, QFileDialog, QLineEdit, QTextEdit, QListWidget

from utils import gl_engine as gl
import sys
sys.path.append("/home/alan/AlanLiang/Projects/AlanLiang/LidarGen4D")
from vis_tools.functions.lidargen_sampler import Lidargen_Sampler
from vis_tools.functions.layout_sampler import Layout_Sampler
from vis_tools.functions.object_sampler import Object_Sampler
from utils.common import box2coord3d, normlize_ndarray
from utils.generate_graph import generate_graph
import utils.pipe_related as pipe
from .custom_window import CustomWindow

class MainWindow(QWidget):

    def __init__(self) -> None:
        super(MainWindow, self).__init__()
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

        self.grid_dimensions = 10
        self.sample_index = 0
        self.logger = gl.create_logger()
        self.custom_window = CustomWindow()
        self.image = None
        self.traj_lenth = 17
        self.init_window()

    def init_window(self):
        main_layout = QVBoxLayout()
        # image label
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label)
        # display_3d_layout
        display_3d_layout = QHBoxLayout()
        self.init_display_window()
        self.init_scene_graph_window()
        self.init_generation_window()
        display_3d_layout.addLayout(self.display_layout)
        display_3d_layout.addLayout(self.scene_graph_layout)
        display_3d_layout.addLayout(self.scene_generation_layout)
        display_3d_layout.setStretch(0, 8)
        display_3d_layout.setStretch(1, 2)
        display_3d_layout.setStretch(2, 8)
        main_layout.addLayout(display_3d_layout)
        main_layout.setStretch(0, 2)
        main_layout.setStretch(1, 8)
        self.setLayout(main_layout)

    def init_display_window(self):
        self.display_layout = QVBoxLayout()
        # image
        # self.image_label = QLabel(self)
        # self.image_label.setAlignment(Qt.AlignCenter)
        # self.display_layout.addWidget(self.image_label)
        # viewer
        self.viewer = gl.AL_viewer()
        self.display_layout.addWidget(self.viewer)
        self.display_layout.setStretch(0, 2)
        self.display_layout.setStretch(1, 8)
        # load dataset and model
        temp_layout = QHBoxLayout()
        self.cfg_select_cbox = QComboBox()
        self.cfg_select_cbox.addItems(['nuscenes-unet-uncond', 'nuscenes-hdit-uncond', 
                                       'nuscenes-box-layout', 'nuscenes-layout', 'nuscenes-auto-reg', 'nuscenes-auto-reg-v2',
                                       'nuscenes-object', 'nuscenes-box-layout-v1', 'nuscenes-box-layout-v2',
                                       'nuscenes-box-layout-v3', 'nuscenes-box-layout-v5'])
        self.cfg_select_cbox.activated.connect(self.cfg_selected)
        temp_layout.addWidget(self.cfg_select_cbox)
        self.ckpt_select_cbox = QComboBox()
        ckpt_paths = os.listdir('/home/alan/AlanLiang/Projects/AlanLiang/LidarGen4D/pretrained_models')
        self.ckpt_select_cbox.addItems(ckpt_paths)
        self.ckpt_select_cbox.activated.connect(self.ckpt_select_selected)
        temp_layout.addWidget(self.ckpt_select_cbox)

        self.init_sampler_button = QPushButton('Inint Sampler')
        self.init_sampler_button.clicked.connect(self.init_sampler)
        temp_layout.addWidget(self.init_sampler_button)

        self.display_layout.addLayout(temp_layout)    
        # << *** >>
        temp_layout = QHBoxLayout()
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

        self.goto_sample_index_box = QLineEdit(self)
        temp_layout.addWidget(self.goto_sample_index_box)
        self.goto_sample_index_box.setPlaceholderText('')
        self.goto_sample_index_button = QPushButton('GoTo')
        temp_layout.addWidget(self.goto_sample_index_button)
        self.goto_sample_index_button.clicked.connect(self.goto_sample_index)

        self.show_scene_graph_button = QPushButton('Show SG')
        temp_layout.addWidget(self.show_scene_graph_button)
        self.show_scene_graph_button.clicked.connect(self.show_scene_graph)
        self.display_layout.addLayout(temp_layout)    

    def init_scene_graph_window(self):
        self.scene_graph_layout = QVBoxLayout()
        # show words
        self.scene_triples_list = QListWidget()
        self.scene_graph_layout.addWidget(self.scene_triples_list)    

        # custom window show
        self.open_custom_window_button = QPushButton('Open Custom Window')
        self.open_custom_window_button.clicked.connect(self.open_custom_window)
        self.scene_graph_layout.addWidget(self.open_custom_window_button)

    def init_generation_window(self):
        self.scene_generation_layout = QVBoxLayout()
        # image
        # self.generation_image_label = QLabel(self)
        # self.generation_image_label.setAlignment(Qt.AlignCenter)
        # self.scene_generation_layout.addWidget(self.generation_image_label)
        # viewer
        self.generation_viewer = gl.AL_viewer()
        self.scene_generation_layout.addWidget(self.generation_viewer)
        self.scene_generation_layout.setStretch(0, 2)
        self.scene_generation_layout.setStretch(1, 8)
        temp_layout = QHBoxLayout()
        # sample
        self.sample_button = QPushButton('Sample')
        temp_layout.addWidget(self.sample_button)
        self.sample_button.clicked.connect(self.sampler_sample)
        # temporal_sample
        self.train_free_temporal_sample_button = QPushButton('TF Temporal Sample')
        temp_layout.addWidget(self.train_free_temporal_sample_button)
        self.train_free_temporal_sample_button.clicked.connect(self.train_free_sampler_temporal_sample)
        # train_free_autoreg_temporal_sample
        self.train_free_autoreg_temporal_sample_button = QPushButton('TF AG Temporal Sample') # train free autoregressive temporal sample
        temp_layout.addWidget(self.train_free_autoreg_temporal_sample_button)
        self.train_free_autoreg_temporal_sample_button.clicked.connect(self.train_free_autoreg_temporal_sample)

        self.temporal_sample_button = QPushButton('Temporal Sample')
        temp_layout.addWidget(self.temporal_sample_button)
        self.temporal_sample_button.clicked.connect(self.sampler_temporal_sample)
        # save sample
        self.save_sample_button = QPushButton('Save Sample')
        temp_layout.addWidget(self.save_sample_button)
        self.save_sample_button.clicked.connect(self.save_sample)

        self.save_screen_image_button = QPushButton('Save Screenshot')
        temp_layout.addWidget(self.save_screen_image_button)
        self.save_screen_image_button.clicked.connect(self.save_screenshot)
        self.scene_generation_layout.addLayout(temp_layout)

    def cfg_selected(self, index):
        selected_cfg = self.cfg_select_cbox.itemText(index)
        if selected_cfg == 'nuscenes-unet-uncond':
            self.logger.info('nuscenes-unet-uncond selected')
            self.conditioned_sample = False
        elif selected_cfg == 'nuscenes-hdit-uncond':
            self.logger.info('nuscenes-hdit-uncond selected')
            self.conditioned_sample = False
        elif selected_cfg == 'nuscenes-box-layout':
            self.logger.info('Nuscenes-box-layout selected')
            self.conditioned_sample = True
        elif selected_cfg == 'nuscenes-box-layout-v1':
            self.logger.info('Nuscenes-box-layout-V1 selected')
            self.conditioned_sample = True
        elif selected_cfg == 'nuscenes-box-layout-v2':
            self.logger.info('Nuscenes-box-layout-V2 selected')
            self.conditioned_sample = True
        elif selected_cfg == 'nuscenes-box-layout-v3':
            self.logger.info('Nuscenes-box-layout-V3 selected')
            self.conditioned_sample = True
        elif selected_cfg == 'nuscenes-box-layout-v5':
            self.logger.info('Nuscenes-box-layout-V5 selected')
            self.conditioned_sample = True
        elif selected_cfg == 'nuscenes-layout':
            self.logger.info('nuscenes-layout selected')
            self.conditioned_sample = True
        elif selected_cfg == 'nuscenes-object':
            self.logger.info('nuscenes-object selected')
            self.conditioned_sample = True
            self.conditioned_sample = True
        elif selected_cfg == 'nuscenes-auto-reg':
            self.logger.info('nuscenes-auto-reg')
            self.conditioned_sample = True
        elif selected_cfg == 'nuscenes-auto-reg-v2':
            self.logger.info('nuscenes-auto-reg-v2')
            self.conditioned_sample = True
        else:
            self.logger.error(f'Unknown platform: {selected_cfg}')
        self.cfg = selected_cfg

    def ckpt_select_selected(self, index):
        selected_ckpt = self.ckpt_select_cbox.itemText(index)
        self.ckpt = os.path.join('/home/alan/AlanLiang/Projects/AlanLiang/LidarGen4D/pretrained_models', selected_ckpt)
        self.logger.info(f'Checkpoint selected: {self.ckpt}')

    def build_nusc_object_sampler(self):
        self.obj_sampler = Object_Sampler('nuscenes-object', '../pretrained_models/nuscenes-object-1000000.pth')
        self.logger.info('Nuscenes-object sampler initialized.')

    def check_index_overflow(self) -> None:

        if self.sample_index == -1:
            self.sample_index = self.sampler.dataset.__len__() - 1

        if self.sample_index >= self.sampler.dataset.__len__():
            self.sample_index = 0

    def decrement_index(self) -> None:

        self.sample_index -= 1
        self.check_index_overflow()
        self.show_sample()

    def increment_index(self) -> None:

        self.sample_index += 1
        self.check_index_overflow()
        self.show_sample()

    def goto_sample_index(self) -> None:
        self.sample_index = int(self.goto_sample_index_box.text())
        self.show_sample()

    def init_sampler(self):
        assert hasattr(self, 'cfg'), 'Please select a cfg first'
        assert hasattr(self, 'ckpt'), 'Please select a checkpoint first'
        if self.cfg == 'nuscenes-layout':
            self.sampler = Layout_Sampler(self.cfg, self.ckpt)
        elif self.cfg == 'nuscenes-object':
            self.sampler = Object_Sampler(self.cfg, self.ckpt)
        else:
            self.sampler = Lidargen_Sampler(self.cfg, self.ckpt, self.conditioned_sample)
        if self.sampler.dataset is not None:
            self.show_sample()
        
    def reset_viewer(self, only_viewer=False):

        self.viewer.items = []
        if not only_viewer:
            self.sample_index_info.setText("")
        self.viewer.add_coordinate_system()


    def reset_generation_viewer(self):
        self.generation_viewer.items = []
        self.generation_viewer.add_coordinate_system()

    def open_custom_window(self):
        self.custom_window.show(self)

    def show_points(self):
        points = self.data_dict['xyz'].reshape(3, -1).transpose(1,0)
        mesh = gl.get_points_mesh(points[:,:3], 5)
        self.current_mesh = mesh
        self.viewer.addItem(mesh)

    def extrac_range_img_from_datadict(self):
        if 'depth' in self.data_dict:
            range_img = copy.deepcopy(self.data_dict['depth'][0])
            range_img = (range_img - np.min(range_img)) / (np.max(range_img) - np.min(range_img)) * 255
            range_img = range_img.astype(np.uint8)
            H,W = range_img.shape
            range_img = cv2.resize(range_img, (W, H*2), interpolation=cv2.INTER_LINEAR)
            self.image = range_img

            if 'scene_loss_weight_map' in self.data_dict:
                scene_loss_weight_map = copy.deepcopy(self.data_dict['scene_loss_weight_map'])
                scene_loss_weight_map = normlize_ndarray(scene_loss_weight_map)
                scene_loss_weight_map *= 255
                scene_loss_weight_map = scene_loss_weight_map.astype(np.uint8)
                H,W = scene_loss_weight_map.shape
                scene_loss_weight_map = cv2.resize(scene_loss_weight_map, (W, H*2), interpolation=cv2.INTER_LINEAR)
                self.image = np.concatenate([self.image, scene_loss_weight_map], axis=0)

    def extrac_range_img_from_object_condition(self):
        obj_cond_path = Path('../data/box_condition/val') / f'sample_{self.sample_index:07d}.npy'
        if obj_cond_path.exists():
            obj_cond = np.load(obj_cond_path)
            obj_cond_img = ((obj_cond[0] + 1)*255).astype(np.uint8)
            H,W = obj_cond_img.shape
            obj_cond_img = cv2.resize(obj_cond_img, (W, H*2), interpolation=cv2.INTER_LINEAR)

        else:
            obj_cond_img = None
        return obj_cond_img

    def extrac_range_img_from_generated_points(self, range_img):
        range_img = range_img.squeeze().detach().cpu().numpy()
        range_img = (range_img - np.min(range_img)) / (np.max(range_img) - np.min(range_img)) * 255
        range_img = range_img.astype(np.uint8)
        H,W = range_img.shape
        range_img = cv2.resize(range_img, (W, H*2), interpolation=cv2.INTER_LINEAR)
        return range_img

    def show_range_img(self, custom_image=None):
        if self.image is None and custom_image is None:
            return
        
        if self.image is not None and custom_image is None:
            self.current_pixmap = gl.ndarray_to_pixmap(self.image)
            self.update_image()
        
        if custom_image is not None:
            if getattr(self, 'image', None) is None:
                self.image = custom_image
            _, W_ori = self.image.shape
            H, W = custom_image.shape
            assert W_ori == W, "Custom image size does not match the original image size."
            self.image = np.concatenate([self.image, custom_image], axis=0)
            self.current_pixmap = gl.ndarray_to_pixmap(self.image)
            self.update_image()

    def update_image(self):
        if self.current_pixmap:
            pixmap = self.current_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)

    def update_generation_image(self):
        if self.current_generation_pixmap:
            pixmap = self.current_generation_pixmap.scaled(self.generation_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.generation_image_label.setPixmap(pixmap)

    def show_boxes_3d(self):
        if 'gt_boxes' not in self.data_dict:
            self.logger.info('No ground truth boxes found in the data dictionary.')
            return
        raw_boxes = self.data_dict['gt_boxes']
        box_texts = []
        boxes_name = self.sampler.dataset.scene_graph_assigner.get_unique_names(list(self.data_dict['gt_names']))
        for i in range(raw_boxes.shape[0]):
            box_name = boxes_name[i]
            motion_state = self.data_dict['gt_fut_states']
            box_texts.append(f"{box_name} ({motion_state[i]})")
        box_info = gl.create_boxes(bboxes_3d=raw_boxes, box_texts=box_texts)

        self.add_boxes_to_viewer(box_info)
        # add boxex corners
        boxes_corners = box2coord3d(raw_boxes)
        mesh = gl.get_points_mesh(boxes_corners[:,:3], 10)
        self.viewer.addItem(mesh)

    def show_trajs(self, custom_viewer=None):
        if 'gt_boxes' not in self.data_dict or 'gt_fut_trajs' not in self.data_dict:
            self.logger.info('No ground truth boxes or future trajectories found in the data dictionary.')
            return
        use_viewer = self.viewer if custom_viewer is None else custom_viewer
        # agents
        agents = self.data_dict['gt_boxes']
        agent_fut_trajs = self.data_dict['gt_fut_trajs']
        # insert 0, agent_fut_trajs [N,T,2] -> [N,T+1,2]
        agent_fut_trajs = np.insert(agent_fut_trajs, 0, 0, axis=1)
        acc_agent_fut_trajs = np.cumsum(agent_fut_trajs, axis=1)  # cumulative sum to get future trajectories
        acc_agent_fut_trajs = pipe.interp_trajs_numpy(acc_agent_fut_trajs, M=self.traj_lenth)
        agent_fut_trajs = acc_agent_fut_trajs[:,1:] - acc_agent_fut_trajs[:,:-1]  # get future trajectories
        agent_fut_states = self.data_dict['gt_fut_states']
        self.add_trajs_to_viewer(agents[:agent_fut_trajs.shape[0]], agent_fut_trajs, agent_fut_states)
            
    def add_trajs_to_viewer(self, agents, agent_fut_trajs, agent_fut_states=None, custom_viewer=None):
        use_viewer = self.viewer if custom_viewer is None else custom_viewer
        agents_items = gl.get_trajs_mesh(agents, agent_fut_trajs, agent_fut_states)
        for item in agents_items:
            use_viewer.addItem(item)

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

    def show_triples(self):
        self.scene_triples_list.clear()
        words = self.sampler.dataset.scene_graph_assigner.get_words(self.data_dict['gt_names'] ,self.data_dict['gt_box_relationships'])
        for i in range(len(words)):
            word = words[i]
            self.scene_triples_list.addItem(word)

    def show_scene_graph(self):
        generate_graph(self.sampler.dataset.scene_graph_assigner, self.data_dict)

    def show_sample(self):
        self.reset_viewer()
        self.data_dict = self.sampler.dataset.__getitem__(self.sample_index)
        self.extrac_range_img_from_datadict()
        self.sample_index_info.setText(f"{self.sample_index}/{self.sampler.dataset.__len__()}")
        # self.show_points()
        self.show_range_img(self.extrac_range_img_from_object_condition())
        self.show_boxes_3d()
        self.show_trajs()
        if 'gt_box_relationships' in self.data_dict:
            self.show_triples()
    
    def save_sample(self):
        dataset = self.sampler.cfg.data.dataset
        split = self.sampler.dataset.split
        selected_cfg = self.cfg
        output_dir = Path(f'./ALTest/USER/good_example/{dataset}/{split}/{selected_cfg}')
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.conditioned_sample:
            output_dir = output_dir / f'sample_{self.sample_index:07d}_conditioned'

        output_dir.mkdir(parents=True, exist_ok=True)
        points_num = len(os.listdir(output_dir))
        # save self.generated_points
        np.savetxt(output_dir / f"points_{points_num:04d}_{self.data_dict['token']}.txt", self.generated_points)
        self.logger.info(f'Saved generated points to {output_dir / f"points_{points_num:04d}.txt"}')

        # save custom data dict
        if hasattr(self, 'custom_data_dict'):
            custom_data_dict = self.custom_data_dict
            # save as pkl
            with open(output_dir / f'custom_data_dict_{points_num:04d}.pkl', 'wb') as f:
                pickle.dump(custom_data_dict, f)

    def sampler_sample(self):
        self.reset_generation_viewer()
        assert hasattr(self, 'sampler'), 'Please init the sampler first'
        if self.cfg == 'nuscenes-layout':
            boxes_name, unscaled_boxes, boxes_trajs = self.sampler.sample(num_steps=256, sample_index=self.sample_index)
            box_info = gl.create_boxes(bboxes_3d=unscaled_boxes)
            self.add_boxes_to_viewer(box_info, custom_viewer=self.generation_viewer)

            agent_fut_trajs = boxes_trajs
            # insert 0, agent_fut_trajs [N,T,2] -> [N,T+1,2]
            agent_fut_trajs = np.insert(agent_fut_trajs, 0, 0, axis=1)
            acc_agent_fut_trajs = np.cumsum(agent_fut_trajs, axis=1)  # cumulative sum to get future trajectories
            acc_agent_fut_trajs = pipe.interp_trajs_numpy(acc_agent_fut_trajs, M=self.traj_lenth)
            agent_fut_trajs = acc_agent_fut_trajs[:,1:] - acc_agent_fut_trajs[:,:-1]  # get future trajectories
            self.add_trajs_to_viewer(unscaled_boxes, agent_fut_trajs, agent_fut_states=self.data_dict['gt_fut_states'],custom_viewer=self.generation_viewer)

            if not hasattr(self, 'obj_sampler'):
                self.build_nusc_object_sampler()

            obj_points = self.obj_sampler.sample(num_steps=1024, sample_index=self.sample_index, custom_data_dict = pipe.conduct_obj_data_dict([{
                "gt_boxes": unscaled_boxes,
                "gt_names": boxes_name,
            }]))
            mesh = gl.get_points_mesh(obj_points[:,:3], 5)
            self.generation_viewer.addItem(mesh)

            raw_boxes = copy.deepcopy(self.data_dict['gt_boxes'])
            raw_boxes[:, :7] = unscaled_boxes
            boxes_name = self.data_dict['gt_names']
            boxes_name = self.sampler.dataset.scene_graph_assigner.get_unique_names(list(boxes_name))
            box_texts = []
            for i in range(raw_boxes.shape[0]):
                box_name = boxes_name[i]
                motion_state = self.data_dict['gt_fut_states']
                box_texts.append(f"{box_name} ({motion_state[i]})")
            box_info = gl.create_boxes(bboxes_3d=raw_boxes, box_texts=box_texts)
            self.add_boxes_to_viewer(box_info, custom_viewer=self.generation_viewer)

        elif self.cfg == 'nuscenes-object':
            obj_points = self.sampler.sample(num_steps=1024, sample_index=self.sample_index)
            mesh = gl.get_points_mesh(obj_points[:,:3], 5)
            self.generation_viewer.addItem(mesh)
            self.show_range_img(self.extrac_range_img_from_object_condition())

        else:
            points, range_image, _ = self.sampler.sample(num_steps=256, sample_index=self.sample_index, batch_dict=getattr(self, 'data_dict', None))
            mesh = gl.get_points_mesh(points[:,:3], 5)
            self.generation_viewer.addItem(mesh)
            self.show_range_img(self.extrac_range_img_from_generated_points(range_image))
            self.generated_points = points

    def sampler_temporal_sample(self, custom_data_dict=None, num_steps=256):
        if not custom_data_dict:
            custom_data_dict = copy.deepcopy(self.data_dict)
        self.reset_generation_viewer()
        assert 'nuscenes-auto-reg' in self.cfg, 'Please select a cfg that supports temporal sampling'
        curr_background_points, fut_background_points, _, fut_boxes_3d, Ts, align_obj_points, align_obj_intensity = pipe.get_temporal_boxes_3d(custom_data_dict, M=self.traj_lenth)
        T = fut_boxes_3d.shape[1]
        dynamic_points_list = []
        for t_id in tqdm(range(T)):
            # conduct cond data dict
            next_frame_points = pipe.get_next_frame_points(curr_background_points, align_obj_points, align_obj_intensity, fut_boxes_3d[:, t_id], custom_data_dict['gt_names'],Ts[t_id])
            ego_box = np.zeros((1, 7), dtype=np.float32)
            gt_boxes = np.concatenate([ego_box, fut_boxes_3d[:, t_id]], axis=0)
            temporal_data_dict = {
                'points': next_frame_points,
                'gt_boxes': gt_boxes,
                'gt_names': custom_data_dict['gt_names']
            }
            temporal_data_dict = pipe.get_mask_cond_single([temporal_data_dict], temporal=True)
            self.sample_from_outline(custom_data_dict=temporal_data_dict,
                                                        num_steps=num_steps)
            
            dynamic_points_list.append(self.generated_points)
            curr_background_points = fut_background_points[t_id]
            curr_background_new_points = pipe.delete_fg_points(self.generated_points, fut_boxes_3d[:, t_id])
            curr_background_points = np.concatenate([curr_background_points, curr_background_new_points], axis=0)
            # curr_background_points = curr_background_new_points

        dynamic_points = np.stack(dynamic_points_list, axis=0)
        np.save('../tools/ALTest/temp/dynamic_points.npy', dynamic_points)

    def train_free_sampler_temporal_sample(self, custom_data_dict=None, num_steps=256):
        if not custom_data_dict:
            custom_data_dict = copy.deepcopy(self.data_dict)
        self.reset_generation_viewer()
        assert hasattr(self, 'sampler'), 'Please init the sampler first'
        if not hasattr(self, 'point_segmentor'):
            self.point_segmentor = pipe.build_point_segmenter()
            self.logger.info('Point segmenter initialized.')
        # first try one timestamp
        curr_background_points, fut_background_points, _, fut_boxes_3d, Ts, align_obj_points, align_obj_intensity = pipe.get_temporal_boxes_3d(custom_data_dict, M=self.traj_lenth)
        T = fut_boxes_3d.shape[1]
        dynamic_points_list = []
        for t_id in tqdm(range(T)):
            # conduct cond data dict
            next_frame_points = pipe.get_next_frame_points(curr_background_points, align_obj_points, align_obj_intensity, fut_boxes_3d[:, t_id], custom_data_dict['gt_names'],Ts[t_id])
            ego_box = np.zeros((1, 7), dtype=np.float32)
            gt_boxes = np.concatenate([ego_box, fut_boxes_3d[:, t_id]], axis=0)
            cond_mask_dict = {
                'points': next_frame_points,
                'gt_boxes': gt_boxes,
                'gt_names': custom_data_dict['gt_names']
            }
            cond_mask_dict = pipe.get_mask_cond_single([cond_mask_dict])
            generated_points = self.inpaint_from_outline(custom_data_dict=cond_mask_dict,
                                        inpaint_cond_dict=cond_mask_dict,
                                        num_steps=num_steps)
            dynamic_points_list.append(generated_points)
            # V1
            curr_background_points = fut_background_points[t_id]
            self.reset_generation_viewer()
            mesh = gl.get_points_mesh(generated_points[:,:3], 5)
            self.generation_viewer.addItem(mesh)

        dynamic_points = np.stack(dynamic_points_list, axis=0)
        np.save('../tools/ALTest/temp/dynamic_points.npy', dynamic_points)

    def train_free_autoreg_temporal_sample(self, custom_data_dict=None, num_steps=32):
        if not custom_data_dict:
            custom_data_dict = copy.deepcopy(self.data_dict)
        self.reset_generation_viewer()
        assert hasattr(self, 'sampler'), 'Please init the sampler first'
        # first try one timestamp
        curr_background_points, fut_background_points, _, fut_boxes_3d, Ts, align_obj_points, align_obj_intensity = pipe.get_temporal_boxes_3d(custom_data_dict, M=self.traj_lenth)
        T = fut_boxes_3d.shape[1]
        dynamic_points_list = []
        for t_id in tqdm(range(T)):
            # conduct cond data dict
            next_frame_points = pipe.get_next_frame_points(curr_background_points, align_obj_points, align_obj_intensity, fut_boxes_3d[:, t_id], custom_data_dict['gt_names'],Ts[t_id])
            ego_box = np.zeros((1, 7), dtype=np.float32)
            gt_boxes = np.concatenate([ego_box, fut_boxes_3d[:, t_id]], axis=0)
            cond_mask_dict = {
                'points': next_frame_points,
                'gt_boxes': gt_boxes,
                'gt_names': custom_data_dict['gt_names']
            }
            cond_mask_dict = pipe.get_mask_cond_single([cond_mask_dict], temporal=True, inpaint=True)
            generated_points = self.inpaint_from_outline(custom_data_dict=cond_mask_dict,
                                        inpaint_cond_dict=cond_mask_dict,
                                        num_steps=num_steps)
            dynamic_points_list.append(generated_points)
            # V1
            curr_background_points = fut_background_points[t_id]
            self.reset_generation_viewer()
            mesh = gl.get_points_mesh(generated_points[:,:3], 5)
            self.generation_viewer.addItem(mesh)

        dynamic_points = np.stack(dynamic_points_list, axis=0)
        np.save('../tools/ALTest/temp/dynamic_points.npy', dynamic_points)

    # sample from outline
    def sample_from_outline(self, custom_data_dict, num_steps=32):
        self.reset_generation_viewer()
        points, range_image, self.inpaint_cond_dict = self.sampler.sample(num_steps=num_steps, batch_dict=custom_data_dict)
        mesh = gl.get_points_mesh(points[:,:3], 5)
        self.generation_viewer.addItem(mesh)
        self.show_range_img(self.extrac_range_img_from_generated_points(range_image))
        self.generated_points = points

    # sample from outline
    def inpaint_from_outline(self, custom_data_dict, inpaint_cond_dict, num_steps=32):
        self.reset_generation_viewer()
        points, range_image, self.inpaint_cond_dict = self.sampler.inpaint(num_steps=num_steps, 
                                                                           batch_dict=custom_data_dict,
                                                                           inpaint_cond_dict=inpaint_cond_dict)
        mesh = gl.get_points_mesh(points[:,:3], 5)
        self.generation_viewer.addItem(mesh)
        self.show_range_img(self.extrac_range_img_from_generated_points(range_image))
        self.generated_points = points
        return points
    
    def sampler_temporal_inpaint(self, custom_data_dict, inpaint_cond_dict, num_steps=32, traj_lenth=32):
        self.reset_generation_viewer()
        history_keep_mask = ~(inpaint_cond_dict['condition_mask']) # [1, 32, 1024]
        inpaint_cond = inpaint_cond_dict['inpaint_cond'] # [1, 2, 32, 1024]
        curr_background_points, fut_background_points, _, fut_boxes_3d, Ts, align_obj_points, align_obj_intensity = pipe.get_temporal_boxes_3d(custom_data_dict, M=traj_lenth)
        T = fut_boxes_3d.shape[1]
        dynamic_points_list = []
        for t_id in tqdm(range(T)):
            # conduct cond data dict
            next_frame_points = pipe.get_next_frame_points(curr_background_points, align_obj_points, align_obj_intensity, fut_boxes_3d[:, t_id], custom_data_dict['gt_names'],Ts[t_id])
            ego_box = np.zeros((1, 7), dtype=np.float32)
            gt_boxes = np.concatenate([ego_box, fut_boxes_3d[:, t_id]], axis=0)
            temporal_data_dict = {
                'points': next_frame_points,
                'gt_boxes': gt_boxes,
                'gt_names': custom_data_dict['gt_names'],
                'history_keep_mask': history_keep_mask,
                'inpaint_cond': inpaint_cond,
            }
            custom_data_dict, inpaint_cond_dict = pipe.temporal_inpaint_cond([temporal_data_dict])
            self.inpaint_from_outline(custom_data_dict, inpaint_cond_dict, num_steps=num_steps)
            
            history_keep_mask = ~self.inpaint_cond_dict['condition_mask']
            inpaint_cond = self.inpaint_cond_dict['inpaint_cond']

            dynamic_points_list.append(self.generated_points)
            curr_background_points = fut_background_points[t_id]
        dynamic_points = np.stack(dynamic_points_list, axis=0)
        folder_path = 'ALTest/seq_example/example/editing_seq'
        seq_num = len(os.listdir(folder_path))
        save_path = os.path.join(folder_path, f'{str(seq_num).zfill(4)}')
        os.makedirs(save_path, exist_ok=True)
        for point_id, point in enumerate(dynamic_points):
            point_save_path = os.path.join(save_path, f'points_{point_id:04d}.txt')
            np.savetxt(point_save_path, point)
        self.logger.info(f'Saved dynamic points to {save_path}')
        np.save('ALTest/temp/dynamic_points.npy', dynamic_points)

    def save_screenshot(self):
        save_folder_path = 'ALTest/USER/good_example/nuscenes/screen_shot'
        img_num = len(os.listdir(save_folder_path))
        save_path = os.path.join(save_folder_path, f'screenshot_{img_num:04d}.png')
        img = self.viewer.grabFramebuffer()
        img.save(save_path, "PNG")

        save_path = os.path.join(save_folder_path, f'screenshot_{(img_num+1):04d}.png')
        img = self.generation_viewer.grabFramebuffer()
        img.save(save_path, "PNG")
        self.logger.info(f'Saved screenshot to {save_path}')