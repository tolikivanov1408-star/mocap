"""3D-вьюпорт: скелет рига и (если загружена) скелетная модель пользователя.

Рендер программный: трансформация вершин numpy + заливка треугольников через
OpenCV (алгоритм художника). GPU не требуется — работает везде, включая
виртуальные машины без драйверов.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..core.mesh import MeshModel


def _look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Матрица world→camera (3x3) и позиция камеры."""
    f = target - eye
    f = f / max(1e-9, np.linalg.norm(f))
    u = np.asarray(up, dtype=np.float64)
    r = np.cross(f, u)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, np.array([0.0, 0.0, 1.0]))
    r = r / max(1e-9, np.linalg.norm(r))
    u2 = np.cross(r, f)
    R = np.column_stack([r, u2, -f])  # оси камеры в мире
    return R


class Model3DView(QWidget):
    """Показывает скелет рига и, при загруженной модели, её скелетную сетку."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self.model: MeshModel | None = None
        self.skin = None  # MeshSkin
        self.solution = None
        self.rig = None
        self.show_mesh = True
        self.show_bones = True

        self.azimuth = -35.0
        self.elevation = 16.0
        self.zoom = 1.0
        self._drag = False
        self._last = None
        self._cache: dict = {}

    # -- содержимое -----------------------------------------------------

    def set_model(self, model: MeshModel | None, skin=None) -> None:
        self.model = model
        self.skin = skin
        self._cache.clear()
        self.update()

    def set_solution(self, solution, rig=None) -> None:
        self.solution = solution
        if rig is not None:
            self.rig = rig
        self.update()

    # -- камера ---------------------------------------------------------

    def _scene_bounds(self):
        pts = []
        # границы берём из «одетой» модели — она в системе координат рига
        skinned = self._skinned_vertices() if (self.skin is not None and self.solution is not None) else None
        if skinned is not None and len(skinned):
            pts.append(np.asarray(skinned, dtype=np.float64))
        elif self.model is not None and len(self.model.vertices):
            pts.append(np.asarray(self.model.vertices, dtype=np.float64))
        if self.solution is not None and self.solution.joints:
            j = np.array(list(self.solution.joints.values()), dtype=np.float64)
            if j.ndim == 2 and len(j):
                pts.append(j)
        if not pts:
            return np.array([-1.0, 0.0, -1.0]), np.array([1.0, 1.8, 1.0])
        allp = np.concatenate(pts, axis=0)
        return allp.min(axis=0), allp.max(axis=0)

    def _camera(self, lo: np.ndarray, hi: np.ndarray):
        center = (lo + hi) * 0.5
        radius = float(np.max(hi - lo)) * 0.5 + 1e-3
        fov = np.radians(50.0)
        dist = radius / np.tan(fov * 0.5) / max(0.2, self.zoom)
        a = np.radians(self.azimuth)
        e = np.radians(self.elevation)
        direction = np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
        eye = center + direction * dist
        R = _look_at(eye, center)
        return R, eye, center, radius

    # -- отрисовка ------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        import cv2

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), Qt.black)
        w, h = self.width(), self.height()
        lo, hi = self._scene_bounds()
        R, eye, center, radius = self._camera(lo, hi)

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (26, 28, 34)  # BGR

        if self.show_mesh and self.model is not None and self.skin is not None and self.solution is not None:
            verts = self._skinned_vertices()
            if verts is not None:
                self._draw_mesh(canvas, verts, self.model, R, eye, w, h)
        if self.show_bones and self.rig is not None and self.solution is not None:
            self._draw_bones(canvas, self.solution.joints, self.rig, R, eye, w, h)

        # осевая линия пола
        floor_y = lo[1] - 0.02
        p_end, _ = self._project(np.array([center[0], floor_y, center[2]]), R, eye, w, h)
        for dx in (-radius, radius):
            a, _ = self._project(np.array([center[0] + dx, floor_y, center[2]]), R, eye, w, h)
            p.setPen(QPen(Qt.darkGray, 1))
            p.drawLine(QPointF(*p_end[0]), QPointF(*a[0]))

        img = QImage(canvas.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        p.drawImage(0, 0, img)
        if self.model is None:
            p.setPen(Qt.lightGray)
            p.drawText(
                self.rect(),
                Qt.AlignCenter,
                "Загрузите свою модель со скелетом\n(вкладка «Модель»): .glb, .gltf, .fbx",
            )

    # -- геометрия ------------------------------------------------------

    def _skinned_vertices(self) -> np.ndarray | None:
        key = id(self.solution)
        if self._cache.get("key") == key:
            return self._cache.get("verts")
        try:
            verts = self.skin.skin(self.solution)
        except Exception:  # noqa: BLE001
            return None
        self._cache = {"key": key, "verts": verts}
        return verts

    @staticmethod
    def _project(pts: np.ndarray, R: np.ndarray, eye: np.ndarray, w: int, h: int):
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        cam = (pts - eye) @ R  # R — оси камеры в мире (строки); камера смотрит вдоль -Z
        z = -cam[:, 2]  # расстояние вперёд от камеры
        f = (h * 0.5) / np.tan(np.radians(50.0) * 0.5)
        safe = np.where(np.abs(z) < 1e-4, 1e-4, z)
        x = f * cam[:, 0] / safe + w * 0.5
        y = -f * cam[:, 1] / safe + h * 0.5
        return np.column_stack([x, y]), z

    def _draw_mesh(self, canvas, verts, model, R, eye, w, h) -> None:
        import cv2

        if not len(verts) or not len(model.indices):
            return
        proj, depth = self._project(verts, R, eye, w, h)
        tris = model.indices
        if tris.max(initial=0) >= len(proj):
            return
        a = proj[tris[:, 0]]
        b = proj[tris[:, 1]]
        c = proj[tris[:, 2]]
        zmean = depth[tris].mean(axis=1)
        # нормали граней и отсечение тыльных (по направлению на камеру)
        v0 = verts[tris[:, 1]] - verts[tris[:, 0]]
        v1 = verts[tris[:, 2]] - verts[tris[:, 0]]
        nrm = np.cross(v0, v1)
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = nrm / np.maximum(ln, 1e-9)
        centroid = verts[tris].mean(axis=1)
        to_eye = eye[None, :] - centroid
        front = (nrm * to_eye).sum(axis=1) > 0
        # освещение: лампа сверху-слева + полусферический ambient
        light = np.array([-0.4, 0.75, 0.5])
        light = light / np.linalg.norm(light)
        shade = np.abs(nrm @ light)
        intensity = 0.45 + 0.55 * np.clip(shade, 0.0, 1.0)
        order = np.argsort(-zmean)  # дальние первыми
        for i in order:
            if not front[i]:
                continue
            if zmean[i] <= 0.05:
                continue
            poly = np.array([a[i], b[i], c[i]], dtype=np.int32)
            g = float(intensity[i])
            color = (
                int(np.clip(238 * g, 0, 255)),
                int(np.clip(222 * g, 0, 255)),
                int(np.clip(205 * g, 0, 255)),
            )
            cv2.fillConvexPoly(canvas, poly, color, lineType=cv2.LINE_AA)
        # рёбра — только для негустых моделей, иначе каша
        if len(order) <= 4000:
            for i in order:
                if not front[i]:
                    continue
                poly = np.array([a[i], b[i], c[i]], dtype=np.int32)
                cv2.polylines(canvas, [poly], True, (70, 70, 78), 1, cv2.LINE_AA)

    def _draw_bones(self, canvas, joints, rig, R, eye, w, h) -> None:
        import cv2

        pts = []
        for bone in rig.bones:
            head = joints.get(bone.head)
            tail = joints.get(bone.tail)
            if head is None or tail is None:
                continue
            pts.append((head, tail))
        if not pts:
            return
        allp = np.array([p for pair in pts for p in pair])
        proj, _ = self._project(allp, R, eye, w, h)
        for i, (head, tail) in enumerate(pts):
            p1 = tuple(np.round(proj[2 * i]).astype(int))
            p2 = tuple(np.round(proj[2 * i + 1]).astype(int))
            cv2.line(canvas, p1, p2, (90, 230, 140), 2, cv2.LINE_AA)

    # -- мышь -----------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._drag = True
        self._last = event.position()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._drag or self._last is None:
            return
        d = event.position() - self._last
        self.azimuth += d.x() * 0.4
        self.elevation = float(np.clip(self.elevation - d.y() * 0.4, -85, 85))
        self._last = event.position()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag = False

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta:
            self.zoom = float(np.clip(self.zoom * (1.1 if delta > 0 else 0.9), 0.2, 6.0))
            self.update()
