from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton,
    QTextEdit, QFrame, QVBoxLayout, QHBoxLayout, QMenu, QLabel, QToolBar, QAction, QLineEdit
)
from PyQt5.QtCore import Qt, QMimeData, pyqtSignal, QPoint, QPointF
from PyQt5.QtGui import QDrag, QPainter, QPen, QColor
import os
import sys
import numpy as np
from scipy.interpolate import interp1d

from lidargen.dataset.custom_dataset import CustomDataset
from utils import gl_engine as gl
from utils import common

class DraggableButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)

    def mouseMoveEvent(self, e):
        if e.buttons() != Qt.LeftButton:
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setText(self.text())
        drag.setMimeData(mime)
        drag.exec_(Qt.MoveAction)

class DraggableItem(QPushButton):
    moved = pyqtSignal(QPushButton, QPoint)
    deleted = pyqtSignal(QPushButton)
    editTrajectory = pyqtSignal(QPushButton)

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self._drag_offset = QPoint()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_offset = e.pos()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if e.buttons() != Qt.LeftButton:
            return
        new_pos = self.mapToParent(e.pos() - self._drag_offset)
        self.move(new_pos)
        self.moved.emit(self, new_pos)

    def showContextMenu(self, pos):
        menu = QMenu(self)
        edit_action = menu.addAction("Edit Trajectory")
        delete_action = menu.addAction("Delete")
        action = menu.exec_(self.mapToGlobal(pos))
        if action == delete_action:
            self.deleted.emit(self)
            self.deleteLater()
        elif action == edit_action:
            self.editTrajectory.emit(self)

class DropArea(QFrame):
    itemDropped = pyqtSignal(str, QPoint)
    pointAdded = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.Sunken | QFrame.StyledPanel)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("background-color: #f9f9f9; border: 2px dashed #888;")
        self.forbidden_radius = 1
        self.trajectories = {}  # item -> np.ndarray of shape (N,2)
        self.N = 20  # default interpolation points
        self.editing_item = None

    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        # forbidden zone
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)
        center = self.rect().center()
        painter.drawEllipse(center, self.forbidden_radius, self.forbidden_radius)
        painter.drawLine(center.x()-10, center.y(), center.x()+10, center.y())
        painter.drawLine(center.x(), center.y()-10, center.x(), center.y()+10)
        # draw trajectories
        traj_pen = QPen(QColor(0, 0, 255), 1)
        point_pen = QPen(QColor(0, 150, 0), 1)
        for item, traj in self.trajectories.items():
            if len(traj) > 1:
                pts = [QPointF(x, y) for x, y in traj]
                painter.setPen(traj_pen)
                painter.drawPolyline(*pts)
            painter.setPen(point_pen)
            for x, y in traj:
                painter.drawEllipse(QPointF(x, y), 3, 3)

    def mousePressEvent(self, e):
        if self.editing_item and e.button() == Qt.LeftButton:
            pos = e.pos()
            self.pointAdded.emit(pos)
            return
        super().mousePressEvent(e)

    def dragEnterEvent(self, e):
        if e.mimeData().hasText(): e.acceptProposedAction()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

    def dropEvent(self, e):
        pos = e.pos()
        center = self.rect().center()
        dx, dy = pos.x()-center.x(), pos.y()-center.y()
        if dx*dx + dy*dy <= self.forbidden_radius**2:
            return
        self.itemDropped.emit(e.mimeData().text(), pos)

class CustomWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.paddle_up_sample = 2
        self.setWindowTitle("Layout Designer")
        self.resize(800, 600)
        # toolbar
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        finish_act = QAction("Finish Edit", self)
        finish_act.triggered.connect(self.finishEditing)
        toolbar.addAction(finish_act)
        # left button area
        left_widget = QWidget()
        left_widget.setFixedWidth(150)
        left_layout = QVBoxLayout(left_widget)
        for obj in [
            'ego', 'car', 'truck', 'construction_vehicle', 'bus',
            'trailer', 'motorcycle', 'bicycle', 'pedestrian'
        ]:
            btn = DraggableButton(obj)
            left_layout.addWidget(btn)
        left_layout.addStretch()

        # right drop area
        right_layout = QVBoxLayout()

        self.drop_area = DropArea()
        self.drop_area.itemDropped.connect(self.handleDrop)
        self.drop_area.pointAdded.connect(self.addTrajectoryPoint)
        right_layout.addWidget(self.drop_area)

        # clear all button
        temp_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clearAll)
        temp_layout.addWidget(self.clear_btn)
        # ==>
        self.show_3d_bbox_btn = QPushButton("-->>")
        self.show_3d_bbox_btn.clicked.connect(self.show_3d)
        temp_layout.addWidget(self.show_3d_bbox_btn)
        # Generate
        self.generate_scene_btn = QPushButton("Generate")

        self.generate_scene_btn.clicked.connect(self.generate_scene)
        temp_layout.addWidget(self.generate_scene_btn)
        # Inpaint
        self.inpaint_scene_btn = QPushButton("Inpaint")
        self.inpaint_scene_btn.clicked.connect(self.inpaint_scene)
        temp_layout.addWidget(self.inpaint_scene_btn)


        self.sample_step_box = QLineEdit(self)
        temp_layout.addWidget(self.sample_step_box)
        self.sample_step_box.setPlaceholderText('32')
        right_layout.addLayout(temp_layout)

        # log area
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        right_layout.addWidget(self.log)

        # temporal related
        temp_layout = QHBoxLayout()
        self.temporal_inpaint_button = QPushButton('Temporal Inpaint')
        temp_layout.addWidget(self.temporal_inpaint_button)
        self.temporal_inpaint_button.clicked.connect(self.inpaint_temporal_scene)
        # Training free Temporal Generate
        self.training_free_temporal_sample_button = QPushButton('TF Temporal Generate')
        temp_layout.addWidget(self.training_free_temporal_sample_button)
        self.training_free_temporal_sample_button.clicked.connect(self.training_free_generate_temporal_scene)

        # Temporal Generate
        self.temporal_sample_button = QPushButton('Temporal Generate')
        temp_layout.addWidget(self.temporal_sample_button)
        self.temporal_sample_button.clicked.connect(self.generate_temporal_scene)

        self.save_sample_button = QPushButton('Save Sample')
        temp_layout.addWidget(self.save_sample_button)
        self.save_sample_button.clicked.connect(self.save_sample)
        
        right_layout.addLayout(temp_layout)

        # 3D display area
        display_layout = QVBoxLayout()
        self.viewer = gl.AL_viewer()
        display_layout.addWidget(self.viewer)

        # main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.addWidget(left_widget)
        main_layout.addLayout(right_layout)
        main_layout.addLayout(display_layout)
        main_layout.setStretch(1, 5)
        main_layout.setStretch(2, 5)
        self.setCentralWidget(main_widget)

        self.items = []

        self.updateLog()

    def show(self, main_window):
        self.main_window = main_window
        super().show()

    def handleDrop(self, name, pos):
        item = DraggableItem(name, self.drop_area)
        item.move(pos); item.show()
        item.moved.connect(self.updateStaticTrajectory)
        item.deleted.connect(self.deleteItem)
        item.editTrajectory.connect(self.startEditing)
        self.items.append(item)
        # default static trajectory
        pts = np.tile([pos.x(), pos.y()], (self.drop_area.N, 1))
        self.drop_area.trajectories[item] = pts
        self.updateLog(); self.drop_area.update()

    def updateStaticTrajectory(self, item, pos):
        pts = np.tile([pos.x(), pos.y()], (self.drop_area.N, 1))
        self.drop_area.trajectories[item] = pts
        self.updateLog(); self.drop_area.update()

    def deleteItem(self, item):
        if item in self.items: self.items.remove(item)
        if item in self.drop_area.trajectories: del self.drop_area.trajectories[item]
        if self.drop_area.editing_item == item: self.drop_area.editing_item = None
        self.updateLog(); self.drop_area.update()

    def startEditing(self, item):
        self.drop_area.editing_item = item
        self.drop_area.trajectories[item] = np.empty((0,2), float)
        self.log.append(f"Editing trajectory for {item.text()}")

    def addTrajectoryPoint(self, pos):
        item = self.drop_area.editing_item
        if not item: return
        arr = self.drop_area.trajectories[item]
        arr = np.vstack([arr, [pos.x(), pos.y()]])
        self.drop_area.trajectories[item] = arr
        self.updateLog(); self.drop_area.update()

    def finishEditing(self):
        item = self.drop_area.editing_item
        if item:
            pts = self.drop_area.trajectories[item]
            n = self.drop_area.N
            if pts.shape[0] >= 2:
                # parameterize by cumulative distance
                dists = np.sqrt(((np.diff(pts, axis=0))**2).sum(axis=1))
                u = np.concatenate(([0], np.cumsum(dists)))
                if u[-1] == 0:
                    u = np.linspace(0, 1, len(pts))
                else:
                    u = u / u[-1]
                t_new = np.linspace(0, 1, n)
                fx = interp1d(u, pts[:,0], kind='cubic')
                fy = interp1d(u, pts[:,1], kind='cubic')
                new_pts = np.vstack([fx(t_new), fy(t_new)]).T
                self.drop_area.trajectories[item] = new_pts
            self.log.append(f"Finished editing {item.text()}")
        self.drop_area.editing_item = None
        self.updateLog(); self.drop_area.update()

    def onItemMoved(self, item, pos):
        self.updateLog()

    def onItemDeleted(self, item):
        if item in self.items:
            self.items.remove(item)
        self.updateLog()

    def clearAll(self):
        for it in list(self.items): it.deleteLater()
        self.items.clear(); self.drop_area.trajectories.clear()
        self.drop_area.editing_item = None
        self.updateLog(); self.drop_area.update()
        self.viewer.clear()

    def updateLog(self):
        self.log.clear()
        w,h = self.drop_area.width(), self.drop_area.height()
        self.log.append(f"DropArea size: {w} x {h}")
        for it, traj in self.drop_area.trajectories.items():
            self.log.append(f"{it.text()} trajectory: {traj.tolist()}")

    def add_points_to_viewer(self, points):
        # keep points
        # self.reset_viewer(only_viewer=True)
        mesh = gl.get_points_mesh(points, 10)
        self.viewer.addItem(mesh)

    def add_points_traj_to_viewer(self, points):
        line_items = gl.get_points_traj_mesh(points)
        for item in line_items:
            self.viewer.addItem(item)

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

    def show_3d(self):
        self.viewer.clear()
        self.viewer.add_coordinate_system()
        self.show_3d_bbox()
        self.show_3d_trajectory()

    def show_3d_bbox(self):
        object_size_dict = common.OBJECT_SIZE_DICT
        point_cloud_range = common.POINT_RANGE
        size = self.drop_area.size()
        w, h = size.width(), size.height()
        objects_3d = np.zeros((len(self.items), 7), dtype=np.float32)
        box_texts = ['ego']
        for i, item in enumerate(self.items[1:]):
            object_name = item.text()
            box_texts.append(object_name)
            pos = item.pos()
            objects_3d[i+1,0] = (pos.x() - w/2) * ((point_cloud_range[3] - point_cloud_range[0]) / w / self.paddle_up_sample)
            objects_3d[i+1,1] = (pos.y() - h/2) * ((point_cloud_range[4] - point_cloud_range[1]) / h / self.paddle_up_sample)
            objects_3d[i+1,1] = -objects_3d[i+1,1]  # Invert y-axis for correct orientation
            objects_3d[i+1,2] = object_size_dict[object_name][0]
            objects_3d[i+1,3:6] = object_size_dict[object_name][1:]
            objects_3d[i+1,6] = np.pi / 2

        self.custom_box_infos = dict(
            gt_boxes=objects_3d,
            gt_names=box_texts
        )

        box_info = gl.create_boxes(bboxes_3d=objects_3d[1:,:], box_texts=box_texts[1:])
        self.add_boxes_to_viewer(box_info)

    def show_3d_trajectory(self):
        object_size_dict = common.OBJECT_SIZE_DICT
        point_cloud_range = common.POINT_RANGE
        size = self.drop_area.size()
        w, h = size.width(), size.height()
        custom_traj = np.zeros([len(self.drop_area.trajectories), self.drop_area.N-1, 2], dtype=np.float32)
        for box_id, (it, traj) in enumerate(self.drop_area.trajectories.items()):
            # self.log.append(f"{it.text()} trajectory: {traj.tolist()}")
            obj_center = self.custom_box_infos['gt_boxes'][box_id, :3].reshape(1, 3)
            object_name = it.text()
            object_traj = np.array(traj, dtype=np.float32)
            object_traj_points = np.zeros([object_traj.shape[0]-1,3])
            object_traj = object_traj - object_traj[0]
            object_traj[:,1] *= -1 
            object_traj[:,0] *= ((point_cloud_range[3] - point_cloud_range[0]) / w / self.paddle_up_sample)
            object_traj[:,1] *= ((point_cloud_range[4] - point_cloud_range[1]) / h / self.paddle_up_sample)
            object_traj_points[:,:2] = object_traj[1:]
            object_traj_points[:,2] = object_size_dict[object_name][0]
            custom_traj[box_id] = np.vstack((object_traj_points[0,:2], object_traj_points[1:,:2] - object_traj_points[:-1,:2]))
            object_traj_points += obj_center
            self.add_points_traj_to_viewer(object_traj_points)

        self.custom_box_infos.update({
            'gt_fut_trajs': custom_traj
        })

    def generate_scene(self):
        dataset = CustomDataset(custom_box_infos=[self.custom_box_infos])
        data_dict = dataset.__getitem__(0)
        self.main_window.sample_from_outline(data_dict, num_steps=int(self.sample_step_box.text()))

    def inpaint_scene(self):
        assert hasattr(self.main_window, 'inpaint_cond_dict'), \
            "Main window does not have inpaint_scene method."
        dataset = CustomDataset(custom_box_infos=[self.custom_box_infos])
        data_dict = dataset.__getitem__(0)
        inpaint_cond_dict = self.main_window.inpaint_cond_dict
        self.main_window.inpaint_from_outline(data_dict, inpaint_cond_dict)

    def inpaint_temporal_scene(self):
        assert self.custom_box_infos.get('gt_fut_trajs') is not None, \
            "Custom box infos does not have 'gt_fut_trajs'."
        points = self.main_window.generated_points
        points[:,3] *= 255
        self.custom_box_infos.update({
            'points': points
        })
        # trajectories
        dataset = CustomDataset(custom_box_infos=[self.custom_box_infos])
        data_dict = dataset.__getitem__(0)
        inpaint_cond_dict = self.main_window.inpaint_cond_dict
        self.main_window.sampler_temporal_inpaint(custom_data_dict=data_dict, inpaint_cond_dict=inpaint_cond_dict, num_steps=32)

    def training_free_generate_temporal_scene(self):
        assert hasattr(self.main_window, 'generated_points'), \
            "Main window does not have inpaint_scene method."
        points = self.main_window.generated_points
        points[:,3] *= 255
        self.custom_box_infos.update({
            'points': points
        })
        # trajectories
        dataset = CustomDataset(custom_box_infos=[self.custom_box_infos])
        data_dict = dataset.__getitem__(0)
        self.main_window.sampler_temporal_sample(custom_data_dict=data_dict, num_steps=int(self.sample_step_box.text()))

    def generate_temporal_scene(self):
        assert hasattr(self.custom_box_infos, 'gt_fut_trajs'), \
            "Custom box infos does not have 'gt_fut_trajs'."

    def save_sample(self):
        self.main_window.custom_data_dict = self.custom_box_infos
        self.main_window.save_sample()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CustomWindow()
    window.show()
    sys.exit(app.exec_())
