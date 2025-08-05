import pyqtgraph.opengl as gl
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QImage, QVector3D, QFont, QPainter, QPen, QColor
from PyQt5.QtWidgets import QFrame, QMenu, QPushButton
from PyQt5 import QtWidgets, QtCore, QtGui
from qimage2ndarray import rgb_view

import os
import cv2
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
from PIL import Image
import logging
import imageio


PRED_COLORS = [
    (  0, 255,   0, 255),  # Crimson
    (233, 150,  70, 255),  # Darksalmon
    (220,  20,  60, 255),  # Crimson
    (255,  61,  99, 255),  # Red
    (0,     0, 230, 255),  # Blue
    (255,  61,  99, 255),  # Red
    (0,     0, 230, 255),  # Blue
    (47,   79,  79, 255),  # Darkslategrey
    (112,  128, 144, 255),  # Slategrey
    (  0, 255,   0, 255),  # cars in green
    (255,   0,   0, 255),  # pedestrian in red
    (255, 255,   0, 255),  # cyclists in yellow
    (255, 127,  80, 255),  # Coral
    (233, 150,  70, 255),  # Darksalmon
        ]

def draw_polyline_with_arrows(pts: np.ndarray, state=None, arrow_length=1.0, arrow_angle=np.pi/8):
    """
    在 GLViewWidget 中绘制带箭头的多段线：
    - pts: N×3 的 numpy 数组，表示折线顶点序列
    - arrow_length: 箭头线段的长度
    - arrow_angle: 箭头开口角度（弧度）
    """
    items = []
    # a) 用 GLLinePlotItem 把折线连起来
    #    color=(1,1,0,1) 表示黄色不透明
    line = gl.GLLinePlotItem(
        pos=pts,
        color=(1.0, 1.0, 0.0, 1.0),
        width=2.0,
        antialias=True,
        mode='line_strip'  # 折线模式
    )
    items.append(line)
    dir_unit_list = [np.ones([3])*1e-6]
    # b) 对于每一段 (i -> i+1)，在 i+1 端添加箭头
    for i in range(len(pts) - 1):
        p_start = pts[i]
        p_end   = pts[i + 1]
        vec = p_end - p_start       # 方向向量 (未归一化)
        norm = np.linalg.norm(vec)
        if norm < 0.1:
            dir_unit = dir_unit_list[i-1]
        else:
            dir_unit = vec / norm      # 单位方向向量
        dir_unit_list.append(dir_unit)

        # 为了画箭头，我们需要任选一个与 dir_unit 不共线的“参考向量”来生成垂直面
        # 这里用 global up = (0,0,1)（世界坐标系），如果方向向量几乎与 up 平行，就换用 (0,1,0)
        up = np.array([0, 0, 1], dtype=float)
        if abs(np.dot(dir_unit, up)) > 0.99:
            up = np.array([0, 1, 0], dtype=float)

        # 计算出一个垂直于 dir_unit 的基向量 perp1
        perp1 = np.cross(dir_unit, up)
        perp1 /= np.linalg.norm(perp1)

        # perp2 再与 dir_unit 和 perp1 构成三维正交基
        perp2 = np.cross(dir_unit, perp1)
        perp2 /= np.linalg.norm(perp2)

        # 箭头两条分支线的端点：
        #   箭头从 p_end 往后退 arrow_length，然后在垂直面内偏转一个角度 arrow_angle
        L = arrow_length
        θ = arrow_angle

        # 分支 1 在 perp1 平面：p_end -> p_end - L·dir_unit + (L·tanθ)·perp1
        arrow_pt1 = p_end - L * dir_unit + (L * np.tan(θ)) * perp1
        # 分支 2 在 perp1 反方向：p_end -> p_end - L·dir_unit - (L·tanθ)·perp1
        arrow_pt2 = p_end - L * dir_unit - (L * np.tan(θ)) * perp1

        # 将箭头分支这两条线连成一个 GLLinePlotItem
        pts_arrow1 = np.vstack([p_end, arrow_pt1])
        pts_arrow2 = np.vstack([p_end, arrow_pt2])

        arrow_line1 = gl.GLLinePlotItem(
            pos=pts_arrow1,
            color=(1.0, 0.0, 0.0, 1.0),  # 红色箭头分支1
            width=2.0,
            antialias=True,
            mode='line_strip'
        )
        arrow_line2 = gl.GLLinePlotItem(
            pos=pts_arrow2,
            color=(1.0, 0.0, 0.0, 1.0),  # 红色箭头分支2
            width=2.0,
            antialias=True,
            mode='line_strip'
        )

        items.append(arrow_line1)
        items.append(arrow_line2)

        # （可选）如果要让箭头更立体，可以再加一条在 perp2 平面上的分支：
        arrow_pt3 = p_end - L * dir_unit + (L * np.tan(θ)) * perp2
        arrow_pt4 = p_end - L * dir_unit - (L * np.tan(θ)) * perp2

        arrow_line3 = gl.GLLinePlotItem(
            pos=np.vstack([p_end, arrow_pt3]),
            color=(1.0, 0.0, 0.0, 1.0),
            width=2.0,
            antialias=True,
            mode='line_strip'
        )
        arrow_line4 = gl.GLLinePlotItem(
            pos=np.vstack([p_end, arrow_pt4]),
            color=(1.0, 0.0, 0.0, 1.0),
            width=2.0,
            antialias=True,
            mode='line_strip'
        )
        items.append(arrow_line3)
        items.append(arrow_line4)
    # if state is None:
    #     state = 'unknown'
    # text_item = gl.GLTextItem(pos=[0,0,0], text=state, color=(255, 255, 255, 255), font=QFont('Helvetica', 8))
    # text_item.translate(pts[1,0], pts[1,1], pts[1,2])
    # items.append(text_item)

    return items

class DropArea(QFrame):
    itemDropped = pyqtSignal(str, QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.Sunken | QFrame.StyledPanel)
        self.setMinimumSize(400, 400)
        self.setStyleSheet(
            "background-color: #f9f9f9; border: 2px dashed #888;"
        )
        self.forbidden_radius = 30  # pixels

    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        pen = QPen(QColor(255, 0, 0), 2, Qt.SolidLine)
        painter.setPen(pen)
        center = self.rect().center()
        # draw forbidden circle at center
        painter.drawEllipse(center, self.forbidden_radius, self.forbidden_radius)
        # draw crosshair
        painter.drawLine(center.x() - 10, center.y(), center.x() + 10, center.y())
        painter.drawLine(center.x(), center.y() - 10, center.x(), center.y() + 10)

    def dragEnterEvent(self, e):
        if e.mimeData().hasText():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

    def dropEvent(self, e):
        pos = e.pos()
        center = self.rect().center()
        # check forbidden region
        dx = pos.x() - center.x()
        dy = pos.y() - center.y()
        if dx*dx + dy*dy <= self.forbidden_radius**2:
            # ignore drop inside forbidden circle
            return
        name = e.mimeData().text()
        self.itemDropped.emit(name, pos)


class DraggableItem(QPushButton):
    moved = pyqtSignal(QPushButton, QPoint)
    deleted = pyqtSignal(QPushButton)

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
        delete_action = menu.addAction("Delete")
        action = menu.exec_(self.mapToGlobal(pos))
        if action == delete_action:
            self.deleted.emit(self)
            self.deleteLater()

def qimage_to_ndarray(qimg: QtGui.QImage) -> np.ndarray:
    qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
    return arr[..., :3]

def pixmap_to_rgb(pixmap: QtGui.QPixmap) -> np.ndarray:
    img = pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    ptr = img.bits()
    ptr.setsize(h * w * 4)
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
    return arr[...,:3]  # drop alpha
# start_record
class AL_viewer(gl.GLViewWidget):
    
    def __init__(self):
        super().__init__()

        self.noRepeatKeys = [Qt.Key.Key_W, Qt.Key.Key_S, Qt.Key.Key_A, Qt.Key.Key_D, Qt.Key.Key_Q, Qt.Key.Key_E,
            Qt.Key.Key_Right, Qt.Key.Key_Left, Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_PageUp, Qt.Key.Key_PageDown]
        
        self.speed = 5
        self.fps = 20
        self.recording = False
        self.record_timer = QtCore.QTimer()

    def evalKeyState(self):
        vel_speed = 10 * self.speed 
        if len(self.keysPressed) > 0:
            for key in self.keysPressed:
                if key == Qt.Key.Key_Right:
                    self.orbit(azim=-self.speed, elev=0)
                elif key == Qt.Key.Key_Left:
                    self.orbit(azim=self.speed, elev=0)
                elif key == Qt.Key.Key_Up:
                    self.orbit(azim=0, elev=-self.speed)
                elif key == Qt.Key.Key_Down:
                    self.orbit(azim=0, elev=self.speed)
                elif key == Qt.Key.Key_A:
                    self.pan(vel_speed * self.speed, 0, 0, 'view-upright')  # 修正: vel_speed 拼写错误
                elif key == Qt.Key.Key_D:
                    self.pan(-vel_speed, 0, 0, 'view-upright')
                elif key == Qt.Key.Key_W:
                    self.pan(0, vel_speed, 0, 'view-upright')
                elif key == Qt.Key.Key_S:
                    self.pan(0, -vel_speed, 0, 'view-upright')
                elif key == Qt.Key.Key_Q:
                    self.pan(0, 0, vel_speed, 'view-upright')
                elif key == Qt.Key.Key_E:
                    self.pan(0, 0, -vel_speed, 'view-upright')
                elif key == Qt.Key.Key_PageUp:
                    pass
                elif key == Qt.Key.Key_PageDown:
                    pass
            self.keyTimer.start(16)
        else:
            self.keyTimer.stop()

    def start_record(self, duration_sec=5, out_video='ALTest/USER/good_example/save_videos/output.mp4', out_dir='ALTest/USER/good_example/save_videos/record_frames'):
        if self.recording:
            return
        
        os.makedirs(out_dir, exist_ok=True)
        self.recording = True
        self.frames = []
        self.frame_count = 0
        self.total_frames = int(duration_sec * self.fps)

        # 定时器控制录制
        self.record_timer.timeout.connect(lambda: self._record_frame(out_dir))
        self.record_timer.start(int(1000 / self.fps))
        
        self.out_video = out_video

    def _record_frame(self, out_dir):
        # 旋转相机
        self.orbit(azim=-self.speed, elev=0)

        img = self.grabFramebuffer()
        img_path = os.path.join(out_dir, f'frame_{self.frame_count:04d}.png')
        img.save(img_path)

        # 保存到视频帧
        self.frames.append(imageio.imread(img_path))
        
        self.frame_count += 1
        if self.frame_count >= self.total_frames:
            self._finish_record()

    def _finish_record(self):
        self.record_timer.stop()
        imageio.mimsave(self.out_video, self.frames, fps=self.fps)
        self.recording = False
        print(f"录制完成，视频保存在 {self.out_video}")

    def add_coordinate_system(self):
        x_axis = np.array([[0, 0, 0], [1, 0, 0]])
        y_axis = np.array([[0, 0, 0], [0, 1, 0]])
        z_axis = np.array([[0, 0, 0], [0, 0, 1]])

        x_axis_item = gl.GLLinePlotItem(pos=x_axis, color=(1, 0, 0, 1), width=2)
        y_axis_item = gl.GLLinePlotItem(pos=y_axis, color=(0, 1, 0, 1), width=2)
        z_axis_item = gl.GLLinePlotItem(pos=z_axis, color=(0, 0, 1, 1), width=2)

        self.addItem(x_axis_item)
        self.addItem(y_axis_item)
        self.addItem(z_axis_item)


class QHLine(QFrame):

    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

def ndarray_to_pixmap(ndarray):
 
    if len(ndarray.shape) == 2:
        height, width = ndarray.shape
        bytes_per_line = width
        qimage = QImage(ndarray.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    elif len(ndarray.shape) == 3:
        height, width, channels = ndarray.shape
        bytes_per_line = 3 * width
        qimage = QImage(ndarray.data, width, height, bytes_per_line, QImage.Format_BGR888)
    else:
        raise ValueError("ndarray must be 3D or 2D ndarry")
    
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, m, centroid

def get_points_mesh(points, size, colors = None):

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    if colors is None:
        # feature = normalize_feature(points[:,2])
        feature = points[:,2]
        norm = mpl.colors.Normalize(vmin=-2.5, vmax=1.5)
        # norm = mpl.colors.Normalize(vmin=feature.min()+0.5, vmax=feature.max()-0.5)
        cmap = cm.jet
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        colors = m.to_rgba(feature)
        colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
        colors[:, 3] = 0.5

    else:
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()

    mesh = gl.GLScatterPlotItem(pos=np.asarray(points[:, 0:3]), size=size, color=colors, pxMode=True)
    # mesh.setGLOptions('translucent')

    return mesh

def resize_img(img: np.ndarray, target_width, target_height):
    image = Image.fromarray(img)
    image_resized = image.resize((target_width, target_height))
    return np.array(image_resized)

def save_temp_ndarry(ndarry):
    cv2.imwrite('../vis_tools/qt_windows/tmp/temp.jpg', ndarry*255)

def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    return logger

def get_points_traj_mesh(points):
    line_items = []
    for i in range(points.shape[0]-1):
        p1 = [points[i, 0], points[i, 1], points[i, 2]]
        p2 = [points[i+1, 0], points[i+1, 1], points[i+1, 2]]
        pts = np.array([p1, p2])
        l1 = gl.GLLinePlotItem(pos=pts, width=2 / 3, color=(255, 255, 255, 255), antialias=True, mode='lines')
        line_items.append(l1)
    return line_items

def create_boxes(bboxes_3d, scores=None, colors=None, box_texts=None):
    boxes = {}
    box_items = []
    l1_items = []
    l2_items = []
    score_items = []
    text_items = []

    box_width = 2

    # create annotation boxes
    for i in range(bboxes_3d.shape[0]):

        annotation = bboxes_3d[i]
        if annotation.shape[0] == 8:
            x, y, z, w, l, h, rotation, category = annotation
        else:
            x, y, z, w, l, h, rotation = annotation[:7]
            category = 0

        rotation = np.rad2deg(rotation) + 90
        if colors is None:
            try:
                color = PRED_COLORS[int(category)]
            except IndexError:
                color = (255, 255, 255, 255)
        else:
            color = colors[i]

        box = gl.GLBoxItem(QVector3D(1, 1, 1), color=color)
        box.setSize(l, w, h)
        box.translate(-l / 2, -w / 2, -h / 2)
        box.rotate(angle=rotation, x=0, y=0, z=1)
        box.translate(x, y, z)
        box_items.append(box)

        #################
        # heading lines #
        #################

        p1 = [-l / 2, -w / 2, -h / 2]
        p2 = [l / 2, -w / 2, h / 2]

        pts = np.array([p1, p2])

        l1 = gl.GLLinePlotItem(pos=pts, width=2 / 3, color=color, antialias=True, mode='lines')
        l1.rotate(angle=rotation, x=0, y=0, z=1)
        l1.translate(x, y, z)

        l1_items.append(l1)

        p3 = [-l / 2, -w / 2, h / 2]
        p4 = [l / 2, -w / 2, -h / 2]

        pts = np.array([p3, p4])

        l2 = gl.GLLinePlotItem(pos=pts, width=2 / 3, color=color, antialias=True, mode='lines')
        l2.rotate(angle=rotation, x=0, y=0, z=1)
        l2.translate(x, y, z)

        l2_items.append(l2)

        distance = np.linalg.norm([x, y, z], axis=0)
        boxes[distance] = (box, l1, l2)

        if scores is not None:
            round_score = np.round(scores[i],2)
            text_item = gl.GLTextItem(pos=[0,0,0], text=str(round_score), color=color, font=QFont('Helvetica', 8))
            text_item.translate(x, y, z)
            score_items.append(text_item)
        # else:
        #     round_score = np.round(annotation[6],2)
        #     text_item = gl.GLTextItem(pos=[0,0,1], text=str(round_score), color=color, font=QFont('Helvetica', 8))
        #     text_item.translate(x, y, z)
        #     score_items.append(text_item)
        if box_texts is not None:
            text_item = gl.GLTextItem(pos=[0,0,0], text=str(box_texts[i]), color=color, font=QFont('Helvetica', 18))
            text_item.translate(x, y, z)
            text_items.append(text_item)


    box_info = {
        'boxes' : boxes,
        'box_items' : box_items,
        'l1_items' : l1_items,
        'l2_items' : l2_items,
        'score_items': score_items,
        'text_items': text_items
    }

    return box_info

class M3ED_BOX:
    def __init__(self, numpy_box):
        self.classIdx = numpy_box[-1]
        self.w = numpy_box[4]
        self.l = numpy_box[3]
        self.h = numpy_box[5]
        self.ry = numpy_box[-2]
        self.t = (numpy_box[0], numpy_box[1], numpy_box[2])

def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    return pts_hom

def get_fov_flag(points, extristric, K, D, image_shape):
    extend_points = cart_to_hom(points[:,:3])
    points_cam = extend_points @ extristric.T
    # points_mask = points_cam[:,2] > 0
    # points_cam = points_cam[points_mask]

    rvecs = np.zeros((3,1))
    tvecs = np.zeros((3,1))

    pts_img, _ = cv2.projectPoints(points_cam[:,:3].astype(np.float32), rvecs, tvecs,
            K, D)

    imgpts = pts_img[:,0,:]

    imgpts[:, 1] = np.clip(imgpts[:, 1], 0, image_shape[1] - 1)
    imgpts[:, 0] = np.clip(imgpts[:, 0], 0, image_shape[0] - 1)
    return imgpts

def get_trajs_mesh(agents, agent_fut_trajs, agent_fut_states=None):
    trajs_items = []
    timestamps = agent_fut_trajs.shape[1]
    agents_traj_points = np.zeros([agents.shape[0], timestamps+1, 3])
    agents_traj_points[:,1:,:2] = agent_fut_trajs
    agents_traj_points = np.cumsum(agents_traj_points, axis=1)
    agents_traj_points[:,:,-1] = agents[:,2][:, np.newaxis]  # z axis is height, so we add half of the height to the z axis
    agents_traj_points[:,:,:2] += agents[:,:2][:, np.newaxis, :]
    for traj_id, traj in enumerate(agents_traj_points):
        items = draw_polyline_with_arrows(traj, state=agent_fut_states[traj_id])
        if items is not None:
            trajs_items.extend(items)

    return trajs_items

class DynamicPointCloudPlayer(QtWidgets.QWidget):
    def __init__(self, frame_items, interval_ms=50):

        super().__init__()
        self.frame_items = frame_items
        self.n_frames = len(frame_items)
        self.idx = 0

        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 20
        self.view.setWindowTitle('Dynamic PointCloud Player')
        self.view.setGeometry(0, 0, 800, 600)

        self.sp = frame_items[0]
        self.view.addItem(self.sp)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(interval_ms)

    def update_frame(self):
        self.idx = (self.idx + 1) % self.n_frames
        self.view.clear()
        self.sp = self.frame_items[self.idx]
        self.view.addItem(self.sp)
