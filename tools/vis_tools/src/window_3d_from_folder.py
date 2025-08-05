import sys
import os
import copy
import numpy as np
from PyQt5 import QtWidgets, QtCore
from pathlib import Path
from natsort import natsorted
import torch
import matplotlib.cm as cm
import sys
sys.path.append('../')
sys.path.append('./vis_tools')
from lidargen.utils import render

from utils import gl_engine as gl

def find_pth_files(root_dir, endswith='.txt'):
    pth_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(endswith):
                pth_files.append(os.path.join(dirpath, fname))
    return pth_files

def get_points_color(z_vals):
    z_vals /= 80
    z_min, z_max = -2 / 80, 0.5 / 80
    z = (z_vals - z_min) / (z_max - z_min)
    z = torch.from_numpy(z).reshape(1,1,-1,1)
    colors = render.colorize(z.clamp(0, 1), cm.plasma) / 255
    colors = colors.squeeze().permute(1,0).numpy()
    optical = np.ones_like(colors[:, :1])  # Create an optical channel
    colors = np.concatenate((colors[:, :3], optical), axis=1)  # Add optical channel
    return colors

class BinPointCloudViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Bin PointCloud Viewer')
        self.resize(1200, 800)

        # data attributes
        self.folder_path = ''
        self.bin_files = []
        self.current_file_index = -1

        # UI components
        # Button to select folder containing .bin files
        self.btn_select_folder = QtWidgets.QPushButton('Select Folder')
        self.btn_select_folder.clicked.connect(self.select_folder)

        # List widget to display .bin filenames
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_file_selected)

        # OpenGL view for point cloud rendering
        self.view = gl.AL_viewer()
        self.view.opts['distance'] = 40

        # Navigation controls to switch between samples
        self.btn_prev = QtWidgets.QPushButton('Previous')
        self.btn_next = QtWidgets.QPushButton('Next')
        self.edit_page = QtWidgets.QLineEdit('0/0')
        self.save_img_btn = QtWidgets.QPushButton('Save Image')
        self.edit_page.setReadOnly(True)
        self.btn_prev.clicked.connect(self.prev_sample)
        self.btn_next.clicked.connect(self.next_sample)
        self.save_img_btn.clicked.connect(self.save_screenshot)
        # Layout setup
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self.btn_select_folder)
        left_layout.addWidget(self.list_widget)

        nav_layout = QtWidgets.QHBoxLayout()
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(self.edit_page)
        nav_layout.addWidget(self.save_img_btn)


        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.view)
        right_layout.addLayout(nav_layout)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 4)
        self.setLayout(main_layout)

    def select_folder(self):
        # Open dialog to select folder and refresh file list
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder Containing .bin Files', os.getcwd())
        if path:
            self.folder_path = path
            self.refresh_file_list()

    def refresh_file_list(self):
        # List all .bin files in folder
        self.bin_files = find_pth_files(self.folder_path, endswith='.txt')
        self.bin_files = natsorted(self.bin_files)  # Sort files naturally
        self.list_widget.clear()
        self.list_widget.addItems(self.bin_files)
        # reset index and page display
        if self.bin_files:
            self.current_file_index = 0
            self.list_widget.setCurrentRow(0)
            self.update_page_display()

    def on_file_selected(self, index):
        # Load the selected .bin file
        if index < 0 or index >= len(self.bin_files):
            return
        self.current_file_index = index
        filepath = os.path.join(self.folder_path, self.bin_files[index])
        self.load_bin_file(filepath)
        self.update_page_display()

    def load_bin_file(self, filepath):
        self.view.clear()  # Clear previous items
        endwith = Path(filepath).suffix
        if endwith == '.txt':
            xyz = np.loadtxt(filepath)[:,:3]
        else:
            raise ValueError(f"Unsupported file format: {endwith}")
        
        colors = get_points_color(copy.deepcopy(xyz[:, 2]))
        mesh = gl.get_points_mesh(xyz, size=4, colors=colors)
        # mesh = gl.get_points_mesh(xyz, size=4)

        self.view.addItem(mesh)

    def update_page_display(self):
        # Show current sample index and total count
        total = len(self.bin_files)
        current = self.current_file_index + 1 if self.current_file_index >= 0 else 0
        self.edit_page.setText(f"{current}/{total}")

    def prev_sample(self):
        # Navigate to previous sample
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.list_widget.setCurrentRow(self.current_file_index)

    def next_sample(self):
        # Navigate to next sample
        if self.current_file_index < len(self.bin_files) - 1:
            self.current_file_index += 1
            self.list_widget.setCurrentRow(self.current_file_index)

    def save_screenshot(self):
        save_folder_path = 'ALTest/USER/good_example/nuscenes/screen_shot'
        img_num = len(os.listdir(save_folder_path))
        save_path = os.path.join(save_folder_path, f'screenshot_{img_num:04d}.png')
        img = self.view.grabFramebuffer()           # 返回 QImage
        img.save(save_path, "PNG")     # 保存为 PNG

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    viewer = BinPointCloudViewer()
    viewer.show()
    sys.exit(app.exec_())
