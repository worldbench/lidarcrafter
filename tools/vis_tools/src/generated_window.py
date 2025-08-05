import sys
import os
import numpy as np
import pickle
from PyQt5 import QtWidgets, QtCore
import pyqtgraph.opengl as gl
import sys
sys.path.append('../')
from tools.vis_tools.utils import gl_engine
from lidargen.dataset import utils

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
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 40
        self.scatter = gl.GLScatterPlotItem()
        self.view.addItem(self.scatter)

        # Navigation controls to switch between samples
        self.btn_prev = QtWidgets.QPushButton('Previous')
        self.btn_next = QtWidgets.QPushButton('Next')
        self.edit_page = QtWidgets.QLineEdit('0/0')
        self.edit_page.setReadOnly(True)
        self.btn_prev.clicked.connect(self.prev_sample)
        self.btn_next.clicked.connect(self.next_sample)

        # Layout setup
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self.btn_select_folder)
        left_layout.addWidget(self.list_widget)

        nav_layout = QtWidgets.QHBoxLayout()
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.edit_page)
        nav_layout.addWidget(self.btn_next)

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
        if 'dwm' in str(self.folder_path):
            self.endwith = 'txt'
            self.with_box = True
            self.bin_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.txt')])
            data_infos = pickle.load(open('../data/infos/nuscenes_infos_val.pkl', 'rb'))['infos']
            self.data_dict = {}
            for info in data_infos:
                self.data_dict[info['token']] = info

        else:
            self.endwith = 'pth'
            self.with_box = False
            self.bin_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.bin')])
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
        self.view.clear()  # Clear previous items in the viewer
        # Read N x 4 binary data and extract XYZ
        if self.endwith == 'txt':
            xyz = np.loadtxt(filepath, dtype=np.float32)
            current_token = os.path.basename(filepath).split('/')[-1].split('.')[0]
            rotation = np.array(np.pi) / 2
            xyz = utils.rotate_points_along_z(xyz[np.newaxis, :, :], rotation.reshape(1))[0]
            xyz[:, 2] -= 2.0  # Adjust height if needed
        else:
            data = np.fromfile(filepath, dtype=np.float32)
            if data.size % 4 != 0:
                pts = data.reshape(-1, 3)  # fallback to 3 columns if not divisible by 4
            else:
                pts = data.reshape(-1, 4)
            xyz = pts[:, :3]
            # Update scatter plot
        mesh = gl_engine.get_points_mesh(xyz[:,:3], 5)
        self.view.addItem(mesh)
        needed_class = np.array(['car', 'truck', 'bus'])
        if self.with_box:
            data_info = self.data_dict[current_token]
            boxes_3d = data_info['gt_boxes'][:, :7]
            boxes_names = data_info['gt_names']
            # mask objects
            mask = np.isin(boxes_names, needed_class)
            boxes_3d = boxes_3d[mask]
            boxes_names = boxes_names[mask]
            box_info = gl_engine.create_boxes(bboxes_3d=boxes_3d, box_texts=boxes_names)
            self.add_boxes_to_viewer(box_info)

    def add_boxes_to_viewer(self, box_info, custom_viewer=None):
        # keep points
        # self.reset_viewer(only_viewer=True)
        # self.viewer.addItem(self.current_mesh)
        use_viewer = self.view if custom_viewer is None else custom_viewer

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

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    viewer = BinPointCloudViewer()
    viewer.show()
    sys.exit(app.exec_())
