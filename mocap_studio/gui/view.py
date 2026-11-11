"""Виджеты: холст видео (разметка пола, поза, оверлей) и 3D-взгляд."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..core.preview import project_world_to_image

MODE_VIEW = "view"
MODE_PLANE = "plane"
MODE_POSE = "pose"


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    """Кадр OpenCV (BGR) -> QImage (копия, чтобы не держать буфер numpy)."""
    if frame is None:
        return QImage()
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    h, w = rgb.shape[:2]
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return img.copy()


class VideoView(QWidget):
    """Холст видео: показывает кадр, разметку пола, позу и найденный скелет."""

    planePointClicked = Signal(float, float)
    poseJointMoved = Signal()
    zoomChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self.frame: np.ndarray | None = None
        self.frame_size: tuple[int, int] = (0, 0)
        self.mode = MODE_VIEW
        self.plane_points: list[tuple[float, float]] = []
        self.pose_px: dict[str, np.ndarray] = {}
        self.rig = None
        self.plane = None
        self.solution = None
        self.detection2d: dict[str, np.ndarray] = {}
        self.show_detection = False
        self.zoom = 1.0
        self._drag_joint: str | None = None
        self._offset = QPointF(0, 0)
        self._scale = 1.0

    # -- содержимое -----------------------------------------------------

    def set_frame(self, frame: np.ndarray | None) -> None:
        self.frame = frame
        if frame is not None:
            self.frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        self.update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.setCursor(Qt.CrossCursor if mode != MODE_VIEW else Qt.ArrowCursor)
        self.update()

    def set_plane_points(self, points) -> None:
        self.plane_points = [tuple(map(float, p)) for p in points]
        self.update()

    def set_pose(self, pose_px: dict[str, np.ndarray]) -> None:
        self.pose_px = dict(pose_px)
        self.update()

    def set_solution(self, solution, plane=None, rig=None) -> None:
        self.solution = solution
        if plane is not None:
            self.plane = plane
        if rig is not None:
            self.rig = rig
        self.update()

    # -- координаты -----------------------------------------------------

    def _fit(self) -> tuple[float, float, float, float]:
        """Возвращает (масштаб, ox, oy) перевода кадра в виджет."""
        w, h = self.frame_size
        if w <= 0 or h <= 0:
            return 1.0, 0.0, 0.0
        s = min(self.width() / w, self.height() / h) * self.zoom
        ox = (self.width() - w * s) / 2.0
        oy = (self.height() - h * s) / 2.0
        return s, ox, oy

    def _to_widget(self, x: float, y: float) -> QPointF:
        s, ox, oy = self._fit()
        return QPointF(ox + x * s, oy + y * s)

    def _to_frame(self, x: float, y: float) -> tuple[float, float]:
        s, ox, oy = self._fit()
        if s <= 0:
            return 0.0, 0.0
        return (x - ox) / s, (y - oy) / s

    # -- отрисовка ------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(18, 20, 24))
        if self.frame is None:
            p.setPen(QColor(150, 150, 160))
            p.drawText(self.rect(), Qt.AlignCenter, "Откройте видео")
            return

        img = bgr_to_qimage(self.frame)
        s, ox, oy = self._fit()
        p.drawImage(QRectF(ox, oy, self.frame_size[0] * s, self.frame_size[1] * s), img)

        if self.show_detection:
            self._draw_detection(p)
        self._draw_plane(p)
        self._draw_solution(p)
        self._draw_pose(p)

    def _draw_detection(self, p: QPainter) -> None:
        p.setPen(QPen(QColor(90, 200, 255, 160), 1))
        for pt in self.detection2d.values():
            q = self._to_widget(float(pt[0]), float(pt[1]))
            p.drawEllipse(q, 2.5, 2.5)

    def _draw_plane(self, p: QPainter) -> None:
        if not self.plane_points:
            return
        poly = QPolygonF([self._to_widget(x, y) for x, y in self.plane_points])
        color = QColor(255, 190, 60) if self.mode == MODE_PLANE else QColor(255, 190, 60, 140)
        p.setPen(QPen(color, 2, Qt.DashLine))
        p.setBrush(QColor(255, 190, 60, 35))
        p.drawPolygon(poly)
        for i, (x, y) in enumerate(self.plane_points):
            q = self._to_widget(x, y)
            p.setBrush(color)
            p.setPen(QPen(QColor(0, 0, 0), 1))
            p.drawEllipse(q, 5, 5)
            p.setPen(color)
            p.drawText(QRectF(q.x() + 7, q.y() - 14, 60, 16), Qt.AlignLeft, str(i + 1))

    def _draw_solution(self, p: QPainter) -> None:
        sol = self.solution
        if sol is None or self.plane is None or self.rig is None:
            return
        w, h = self.frame_size
        p.setPen(QPen(QColor(80, 255, 120), 2))
        for bone in self.rig.bones:
            head = sol.joints.get(bone.head)
            tail = sol.joints.get(bone.tail)
            if head is None or tail is None:
                continue
            a = project_world_to_image(self.plane, head, w, h)
            b = project_world_to_image(self.plane, tail, w, h)
            if a is None or b is None:
                continue
            p.drawLine(self._to_widget(*a), self._to_widget(*b))
        p.setBrush(QColor(80, 255, 120))
        p.setPen(Qt.NoPen)
        for pos in sol.joints.values():
            q = self._to_widget(float(pos[0]), float(pos[1]))
            p.drawEllipse(q, 2.0, 2.0)
        for side, contact in (sol.contacts or {}).items():
            if not contact:
                continue
            pos = sol.joints.get(f"ankle_{side}")
            if pos is None:
                continue
            q = project_world_to_image(self.plane, pos, w, h)
            if q is None:
                continue
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(255, 120, 40), 2))
            p.drawEllipse(self._to_widget(*q), 8, 8)

    def _draw_pose(self, p: QPainter) -> None:
        if not self.pose_px or self.rig is None:
            return
        active = self.mode == MODE_POSE
        p.setPen(QPen(QColor(120, 200, 255, 200) if active else QColor(120, 200, 255, 90), 1))
        for bone in self.rig.bones:
            h = self.pose_px.get(bone.head)
            t = self.pose_px.get(bone.tail)
            if h is None or t is None:
                continue
            p.drawLine(self._to_widget(float(h[0]), float(h[1])), self._to_widget(float(t[0]), float(t[1])))
        for name, pt in self.pose_px.items():
            q = self._to_widget(float(pt[0]), float(pt[1]))
            p.setBrush(QColor(120, 200, 255) if active else QColor(120, 200, 255, 90))
            p.setPen(QPen(QColor(20, 30, 40), 1))
            p.drawEllipse(q, 4.5 if active else 3, 4.5 if active else 3)

    # -- мышь ------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.frame is None:
            return
        pos = event.position()
        fx, fy = self._to_frame(pos.x(), pos.y())
        if self.mode == MODE_PLANE:
            self.planePointClicked.emit(fx, fy)
            return
        if self.mode == MODE_POSE and self.pose_px:
            best = None
            best_d = 14.0
            for name, pt in self.pose_px.items():
                d = float(np.hypot(pt[0] - fx, pt[1] - fy))
                if d < best_d:
                    best_d = d
                    best = name
            if best is not None:
                self._drag_joint = best
                self.pose_px[best] = np.array([fx, fy])
                self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_joint is None:
            return
        pos = event.position()
        fx, fy = self._to_frame(pos.x(), pos.y())
        self.pose_px[self._drag_joint] = np.array([fx, fy])
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_joint is not None:
            self._drag_joint = None
            self.poseJointMoved.emit()

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta:
            self.zoom = float(np.clip(self.zoom * (1.1 if delta > 0 else 0.9), 0.3, 6.0))
            self.zoomChanged.emit(self.zoom)
            self.update()


class Glance3D(QWidget):
    """Лёгкий 3D-взгляд на скелет: перспективная проекция линий."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.rig = None
        self.solution = None
        self.azimuth = -35.0
        self.elevation = 18.0
        self.distance = 4.2
        self.setMouseTracking(True)
        self._drag = False
        self._last = None

    def set_solution(self, solution, rig=None) -> None:
        self.solution = solution
        if rig is not None:
            self.rig = rig
        self.update()

    def _project(self, pts: np.ndarray) -> np.ndarray:
        a = np.radians(self.azimuth)
        e = np.radians(self.elevation)
        Ry = np.array(
            [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
        )
        Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
        p = (pts - pts.mean(axis=0)) @ (Rx @ Ry).T
        z = self.distance - p[:, 2]
        z = np.where(np.abs(z) < 1e-3, 1e-3, z)
        k = 520.0 / z
        x = p[:, 0] * k + self.width() / 2.0
        y = -p[:, 1] * k + self.height() / 2.0
        return np.column_stack([x, y])

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(24, 26, 32))
        sol = self.solution
        if sol is None or self.rig is None:
            p.setPen(QColor(140, 140, 150))
            p.drawText(self.rect(), Qt.AlignCenter, "3D-взгляд появится после расчёта")
            return
        names = list(sol.joints.keys())
        pts = np.array([sol.joints[n] for n in names], dtype=np.float64)
        proj = self._project(pts)
        idx = {n: i for i, n in enumerate(names)}
        p.setPen(QPen(QColor(90, 220, 140), 2))
        for bone in self.rig.bones:
            if bone.head in idx and bone.tail in idx:
                a = proj[idx[bone.head]]
                b = proj[idx[bone.tail]]
                p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))
        p.setBrush(QColor(230, 230, 120))
        p.setPen(Qt.NoPen)
        for a in proj:
            p.drawEllipse(QPointF(a[0], a[1]), 2.5, 2.5)
        # ось пола
        p.setPen(QPen(QColor(90, 110, 140), 1))
        p.drawLine(QPointF(30, self.height() - 30), QPointF(self.width() - 30, self.height() - 30))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._drag = True
        self._last = event.position()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._drag or self._last is None:
            return
        d = event.position() - self._last
        self.azimuth += d.x() * 0.4
        self.elevation = float(np.clip(self.elevation - d.y() * 0.4, -80, 80))
        self._last = event.position()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag = False
