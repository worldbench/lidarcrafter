import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph.opengl as gl

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
        # Read N x 4 binary data and extract XYZ
        data = np.fromfile(filepath, dtype=np.float32)
        if data.size % 4 != 0:
            pts = data.reshape(-1, 3)  # fallback to 3 columns if not divisible by 4
        else:
            pts = data.reshape(-1, 4)
        xyz = pts[:, :3]
        # Update scatter plot
        self.scatter.setData(pos=xyz, size=0.2, pxMode=False)

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
