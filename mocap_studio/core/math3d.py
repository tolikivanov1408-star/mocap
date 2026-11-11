"""Векторная и кватернионная математика для решателя скелета.

Соглашения:
  * Мировая система координат приложения: X — вправо, Y — вверх, Z — назад.
    Персонаж смотрит в -Z, левая сторона персонажа = +X (совпадает с FBX/Maya).
  * Кватернионы: (x, y, z, w), как в scipy.spatial.transform.
  * Локальный фрейм кости: X — вдоль кости, Y — ось «twist» (полюс), Z = X x Y.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

EPS = 1e-9


def v3(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    return np.array([float(x), float(y), float(z)], dtype=np.float64)


def norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.array([0.0, 0.0, 0.0])
    return v / n


def safe_unit(v: np.ndarray, fallback=(1.0, 0.0, 0.0)) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < EPS:
        return v3(*fallback)
    return v / n


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Произведение кватернионов a*b (применение b первым, затем a)."""
    return (R.from_quat(a) * R.from_quat(b)).as_quat()


def q_inv(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).inv().as_quat()


def q_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return R.from_quat(q).apply(v)


def q_identity() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0])


def q_from_matrix(m: np.ndarray) -> np.ndarray:
    return R.from_matrix(m).as_quat()


def q_to_matrix(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_matrix()


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = norm(np.asarray(axis, dtype=np.float64))
    if float(np.linalg.norm(axis)) < EPS:
        return q_identity()
    return R.from_rotvec(axis * float(angle)).as_quat()


def quat_angle(q: np.ndarray) -> float:
    """Угол поворота кватерниона в радианах."""
    return float(R.from_quat(q).magnitude())


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return R.from_quat(a).slerp(float(t), R.from_quat(b)).as_quat()


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Минимальный поворот, переводящий направление a в направление b."""
    a = safe_unit(a)
    b = safe_unit(b)
    dot = float(np.dot(a, b))
    if dot > 1.0 - 1e-9:
        return q_identity()
    if dot < -1.0 + 1e-9:
        # antipodal: выбираем любую перпендикулярную ось
        axis = np.cross(a, v3(1.0, 0.0, 0.0))
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.cross(a, v3(0.0, 1.0, 0.0))
        axis = norm(axis)
        return quat_from_axis_angle(axis, np.pi)
    axis = np.cross(a, b)
    s = np.sqrt((1.0 + dot) * 2.0)
    return np.array([axis[0] / s, axis[1] / s, axis[2] / s, s * 0.5])


def build_frame(direction: np.ndarray, pole: np.ndarray | None) -> np.ndarray:
    """Матрица фрейма кости: X вдоль direction, Y вдоль pole (ортогонализованного).

    Если pole вырожден (параллелен direction), подбирается запасная ось.
    """
    x = safe_unit(direction)
    y = None
    if pole is not None:
        p = np.asarray(pole, dtype=np.float64)
        p = p - float(np.dot(p, x)) * x
        if float(np.linalg.norm(p)) > 1e-6:
            y = safe_unit(p)
    if y is None:
        # запасная ось: любая, не параллельная x
        helper = v3(0.0, 0.0, 1.0) if abs(x[2]) < 0.9 else v3(1.0, 0.0, 0.0)
        y = safe_unit(np.cross(helper, x))
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def aim_quat(direction: np.ndarray, pole: np.ndarray | None) -> np.ndarray:
    """Кватернион ориентации кости: X — вдоль direction, Y — вдоль pole."""
    m = build_frame(direction, pole)
    return q_from_matrix(m)


def project_perpendicular(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Компонент v, перпендикулярный axis (не нормализуется)."""
    a = np.asarray(axis, dtype=np.float64)
    return np.asarray(v, dtype=np.float64) - float(np.dot(v, a)) * a


def euler_xyz_degrees(q: np.ndarray) -> np.ndarray:
    """Углы Эйлера XYZ в градусах — формат, который ожидает FBX.

    Используется внешняя (extrinsic) последовательность XYZ, как в Blender,
    3ds Max и Unreal: матрица поворота = Rz @ Ry @ Rx.
    """
    return R.from_quat(q).as_euler("xyz", degrees=True)


def euler_zxy_degrees(q: np.ndarray) -> np.ndarray:
    """Углы Эйлера в порядке ZXY (в градусах) — порядок каналов BVH.

    Матрица поворота = Rx @ Ry @ Rz.
    """
    m = q_to_matrix(q)
    cb = float(np.clip(-m[2, 0], -1.0, 1.0))
    b = np.arcsin(cb)
    if abs(cb) < 0.9999:
        c = np.arctan2(m[1, 0], m[0, 0])
        a = np.arctan2(m[2, 1], m[2, 2])
    else:  # вырожденный случай (спина у горизонта)
        c = np.arctan2(-m[0, 1], m[1, 1])
        a = 0.0
    return np.degrees(np.array([a, b, c]))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    """Плавная ступень 0..1 ( quintic smootherstep )."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def weighted_angle_axis(q: np.ndarray) -> tuple[np.ndarray, float]:
    rot = R.from_quat(q)
    return rot.as_rotvec(), float(rot.magnitude())


def pose_matrix_to_quat(m: np.ndarray) -> np.ndarray:
    return q_from_matrix(m)


def rotation_difference(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """q = q_a^-1 * q_b."""
    return q_mul(q_inv(q_a), q_b)
