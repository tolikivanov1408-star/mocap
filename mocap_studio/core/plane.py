"""Плоскость пола: из 2D-полигона в кадре — 3D-плоскость в пространстве сцены.

Идея: пользователь рисует на кадре прямоугольник на полу и задаёт его реальный
размер. По четырём точкам и матрице камеры вычисляется гомография, а из неё —
положение плоскости в системе координат камеры. Дальше луч камеры пересекается
с этой плоскостью, и мы получаем мировую точку «на полу» для любого пикселя.
Именно это даёт глубину: монокамера сама по себе глубину не определяет.

Система координат мира: X — вдоль первой стороны прямоугольника, Y — вверх
(нормаль к полу), Z — вдоль второй стороны. Начало — в первой вершине.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .math3d import safe_unit


@dataclass
class Camera:
    """Простая пинхоль-камера: вертикальный FOV, квадратные пиксели."""

    width: int
    height: int
    fov_y_deg: float = 50.0

    @property
    def focal(self) -> float:
        return (self.height / 2.0) / np.tan(np.radians(self.fov_y_deg) / 2.0)

    @property
    def matrix(self) -> np.ndarray:
        f = self.focal
        return np.array(
            [
                [f, 0.0, self.width / 2.0],
                [0.0, f, self.height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)

    def ray(self, px: float, py: float) -> np.ndarray:
        """Направление луча из камеры в пиксель (в системе координат камеры)."""
        p = self.inverse @ np.array([px, py, 1.0])
        return safe_unit(p)


@dataclass
class FloorPlane:
    """Плоскость пола, заданная 4 точками изображения и реальным размером."""

    camera: Camera
    points_px: np.ndarray  # (4,2) — вершины прямоугольника на полу
    size_m: tuple[float, float] = (1.0, 1.0)  # (длина вдоль u, длина вдоль v)
    up_sign: int = 1
    _r1: np.ndarray = field(default=None, repr=False)
    _r2: np.ndarray = field(default=None, repr=False)
    _n: np.ndarray = field(default=None, repr=False)
    _r2w: np.ndarray = field(default=None, repr=False)
    _t: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.points_px = np.asarray(self.points_px, dtype=np.float64).reshape(4, 2)
        self._compute()

    # -- построение -----------------------------------------------------

    def _compute(self) -> None:
        K = self.camera.matrix
        Kinv = np.linalg.inv(K)
        u_len, v_len = self.size_m

        # Целевые координаты вершин прямоугольника на плоскости (метры).
        dst = np.array(
            [
                [0.0, 0.0],
                [u_len, 0.0],
                [u_len, v_len],
                [0.0, v_len],
            ]
        )
        H_plane = cv2_homography(self.points_px, dst)  # изображение -> плоскость
        # Разложение делаем для гомографии плоскость -> изображение:
        # H_img = K [r1 r2 t], откуда r1 = normalize(K^-1 h1) и т.д.
        try:
            H_img = np.linalg.inv(H_plane)
        except np.linalg.LinAlgError:
            H_img = np.eye(3)
        h1 = Kinv @ H_img[:, 0]
        h2 = Kinv @ H_img[:, 1]
        h3 = Kinv @ H_img[:, 2]
        l1 = float(np.linalg.norm(h1))
        l2 = float(np.linalg.norm(h2))
        if l1 < 1e-12 or l2 < 1e-12:
            raise ValueError("Вырожденная плоскость: проверьте точки и FOV")
        lam = 1.0 / max(1e-9, (l1 + l2) / 2.0)
        r1 = h1 * lam
        r2 = h2 * lam
        t = h3 * lam
        # Гомография определена с точностью до знака: если плоскость оказалась
        # за камерой — переворачиваем весь базис целиком.
        if float(t[2]) < 0:
            r1, r2, t = -r1, -r2, -t
        # ортогонализация (Грама-Шмидт) — убирает накопленную ошибку
        r1 = safe_unit(r1, fallback=(1.0, 0.0, 0.0))
        r2 = r2 - float(np.dot(r2, r1)) * r1
        r2 = safe_unit(r2, fallback=(0.0, 0.0, 1.0))
        n = np.cross(r1, r2)
        n = safe_unit(n, fallback=(0.0, -1.0, 0.0))
        # Нормаль пола смотрит вверх: камера должна быть над плоскостью,
        # то есть n·t < 0.
        if float(np.dot(n, t)) > 0:
            n = -n
            r2 = -r2
        if self.up_sign < 0:
            n = -n
            r2 = -r2
        # Мировой базис должен быть правым: X = r1, Y = вверх, Z = X x Y.
        r2w = np.cross(r1, n)
        r2w = safe_unit(r2w, fallback=(0.0, 0.0, 1.0))
        self._r1, self._r2, self._n, self._r2w, self._t = r1, r2, n, r2w, t

    # -- преобразования --------------------------------------------------

    @property
    def normal(self) -> np.ndarray:
        """Нормаль пола в системе координат камеры (смотрит вверх)."""
        return self._n

    @property
    def camera_to_world(self) -> np.ndarray:
        """Матрица перехода из системы камеры в мир (поворот, без сдвига).

        Мир: X — вдоль первой стороны прямоугольника, Y — вверх,
        Z — вдоль второй стороны (правый базис).
        """
        return np.column_stack([self._r1, self._n, self._r2w]).T

    def camera_from_world(self, p_world: np.ndarray) -> np.ndarray:
        """Мировые координаты (X — вдоль r1, Y — вверх, Z — вдоль r2) -> камера."""
        p = np.asarray(p_world, dtype=np.float64)
        return self._t + p[0] * self._r1 + p[1] * self._n + p[2] * self._r2w

    def world_from_camera(self, p_cam: np.ndarray) -> np.ndarray:
        """Точка в системе камеры -> мировые координаты (Y — высота над полом)."""
        p = np.asarray(p_cam, dtype=np.float64) - self._t
        return np.array(
            [
                float(np.dot(p, self._r1)),
                float(np.dot(p, self._n)),
                float(np.dot(p, self._r2w)),
            ]
        )

    def project_to_floor(self, px: float, py: float) -> np.ndarray | None:
        """Пересечение луча камеры (через пиксель) с плоскостью пола."""
        d = self.camera.ray(px, py)
        denom = float(np.dot(self._n, d))
        if abs(denom) < 1e-9:
            return None
        s = float(np.dot(self._n, self._t)) / denom
        if s <= 0:
            return None
        return self.world_from_camera(s * d)

    def project_to_plane_at_height(self, px: float, py: float, height: float) -> np.ndarray | None:
        """Точка луча камеры на заданной высоте над полом (в мировых координатах)."""
        d = self.camera.ray(px, py)
        # плоскость, параллельная полу, на высоте height
        point_cam = self._t + height * self._n
        denom = float(np.dot(self._n, d))
        if abs(denom) < 1e-9:
            return None
        s = (float(np.dot(self._n, point_cam))) / denom
        if s <= 0:
            return None
        return self.world_from_camera(s * d)

    def height_above_floor(self, p_cam: np.ndarray) -> float:
        """Высота точки (в координатах камеры) над плоскостью пола."""
        p = np.asarray(p_cam, dtype=np.float64) - self._t
        return float(np.dot(p, self._n))

    def is_inside(self, p_world: np.ndarray, margin: float = 0.0) -> bool:
        """Внутри ли прямоугольника пола точка (с запасом).

        Прямоугольник отмечен пользователем в порядке «ближняя левая,
        ближняя правая, дальняя правая, дальняя левая», поэтому его вторая
        сторона направлена вдоль -Z мира.
        """
        p = np.asarray(p_world, dtype=np.float64)
        u_len, v_len = self.size_m
        return (
            -margin <= p[0] <= u_len + margin
            and -v_len - margin <= p[2] <= margin
        )

    # -- автокалибровка масштаба ----------------------------------------

    def scale_to_person(self, rig, joints2d_list) -> float:
        """Подгоняет метрику сцены под рост фигуры.

        Пользователь отмечает на полу прямоугольник и указывает его размер;
        если размер назван «на глаз», сцена получается не той. Тогда по
        кадру можно оценить истинный масштаб: рост фигуры в пикселях известен,
        а её рост в риге — тоже. Из этого получаем нужную глубину, а значит
        и масштаб прямоугольника на полу.

        Пересобирается именно размер прямоугольника (пиксели отметки не
        меняются), поэтому разметка пола остаётся на месте.

        Возвращает применённый коэффициент.
        """
        if not joints2d_list:
            return 1.0
        heights: list[float] = []
        feet: list[np.ndarray] = []
        for j2 in joints2d_list:
            head = j2.get("head")
            foot = j2.get("ankle_l")
            if foot is None:
                foot = j2.get("ankle_r")
            if head is None or foot is None:
                continue
            d = float(np.linalg.norm(head[:2] - foot[:2]))
            if d > 20.0:
                heights.append(d)
                feet.append(np.asarray(foot[:2], dtype=np.float64))
        if not heights:
            return 1.0
        h_px = float(np.median(heights))
        # рост рига: от щиколотки до головы
        try:
            h_rig = float(
                rig.rest_positions["head"][1] - rig.rest_positions["ankle_l"][1]
            )
        except KeyError:
            return 1.0
        if h_rig <= 1e-3:
            return 1.0
        depth_target = h_rig * self.camera.focal / h_px

        foot = feet[int(np.argmax(heights))]
        point = self.project_to_floor(float(foot[0]), float(foot[1]))
        if point is None:
            return 1.0
        p_cam = self.camera_from_world(point)
        depth_now = float(p_cam[2])
        if depth_now <= 1e-6:
            return 1.0
        k = depth_target / depth_now
        if not (0.15 <= k <= 8.0):
            return 1.0
        self.size_m = (self.size_m[0] * k, self.size_m[1] * k)
        self._compute()
        return k

    # -- сериализация ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "points_px": self.points_px.tolist(),
            "size_m": list(self.size_m),
            "up_sign": self.up_sign,
            "fov_y_deg": self.camera.fov_y_deg,
            "width": self.camera.width,
            "height": self.camera.height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FloorPlane":
        cam = Camera(
            width=int(data.get("width", 1920)),
            height=int(data.get("height", 1080)),
            fov_y_deg=float(data.get("fov_y_deg", 50.0)),
        )
        return cls(
            camera=cam,
            points_px=np.array(data["points_px"], dtype=np.float64),
            size_m=tuple(data.get("size_m", (1.0, 1.0))),
            up_sign=int(data.get("up_sign", 1)),
        )


def cv2_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Гомография src->dst для четырёх точек (через OpenCV)."""
    import cv2

    src = np.asarray(src, dtype=np.float32).reshape(4, 2)
    dst = np.asarray(dst, dtype=np.float32).reshape(4, 2)
    return np.asarray(cv2.getPerspectiveTransform(src, dst), dtype=np.float64)
