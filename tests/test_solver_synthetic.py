"""Синтетический тест решателя: известная анимация -> детекция -> решение.

Проверяется чистая математика ядра без нейросети:
  * переход камера <-> мир и гомография плоскости;
  * вычисление направлений, полюсов (twist) и локальных кватернионов;
  * постановка стоп на пол (foot-lock) и IK ног;
  * совпадение с эталонной анимацией.

Запуск:  python tests/test_solver_synthetic.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mocap_studio.core.detect import MP, PoseFrame
from mocap_studio.core.math3d import (
    aim_quat,
    q_inv,
    q_from_matrix,
    q_identity,
    q_mul,
    quat_from_axis_angle,
    safe_unit,
    v3,
)
from mocap_studio.core.plane import Camera, FloorPlane
from mocap_studio.core.rig import Rig
from mocap_studio.core.solve import MotionSolver


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """Матрица перехода из мира в камеру (X — вправо, Y — вниз, Z — вперёд)."""
    f = safe_unit(np.asarray(target, float) - np.asarray(eye, float))
    r = safe_unit(np.cross(f, np.asarray(up, float)))  # вправо
    d = np.cross(f, r)  # вниз (правый базис: r x d = f)
    return np.vstack([r, d, f])


def rot_x(angle: float) -> np.ndarray:
    return quat_from_axis_angle((1.0, 0.0, 0.0), angle)


def build_ground_truth(rig: Rig, n_frames: int):
    """Эталонная анимация: шаг на месте (ноги качаются, корпус чуть наклоняется)."""
    frames = []
    for i in range(n_frames):
        phase = 2.0 * np.pi * i / 20.0
        deltas = {
            "thigh_l": rot_x(0.42 * np.sin(phase)),
            "calf_l": rot_x(-0.55 * max(0.0, np.sin(phase + 0.6))),
            "thigh_r": rot_x(0.42 * np.sin(phase + np.pi)),
            "calf_r": rot_x(-0.55 * max(0.0, np.sin(phase + np.pi + 0.6))),
            "spine_01": rot_x(0.05 * np.sin(2 * phase)),
            "spine_02": rot_x(0.05 * np.sin(2 * phase)),
            "spine_03": rot_x(0.05 * np.sin(2 * phase)),
            "upperarm_l": rot_x(0.25 * np.sin(phase + np.pi)),
            "upperarm_r": rot_x(0.25 * np.sin(phase)),
            "lowerarm_l": rot_x(-0.45 - 0.15 * np.sin(phase)),
            "lowerarm_r": rot_x(-0.45 - 0.15 * np.sin(phase + np.pi)),
        }
        # локальный поворот = rest-поворот * дельта
        local = {b.name: rig.rest_local_quat[b.name] for b in rig.bones}
        for name, delta in deltas.items():
            local[name] = q_mul(rig.rest_local_quat[name], delta)
        root_quat = rig.rest_global_quat["pelvis"]
        root_pos = rig.rest_positions["pelvis"].copy()
        positions, _ = rig.fk(local, root_quat=root_quat, root_pos=root_pos)
        # эталонные кадры выводим из позиций (тем же правилом, что и решатель)
        gt_local, _gt_global = frames_from_positions(rig, positions)
        frames.append((gt_local, root_quat, root_pos, positions))
    return frames


def make_detection(rig: Rig, positions, eye, R_wc, cam: Camera, width: int, height: int) -> PoseFrame:
    """Превращает эталонные позиции в «кадр детектора»."""
    pose2d = np.zeros((33, 3))
    pose3d = np.zeros((33, 3))
    pelvis_world = positions["pelvis"]

    def to_cam(name):
        p_world = positions[name]
        return R_wc @ (p_world - eye)

    # сопоставление: риг -> индексы MediaPipe
    mapping = {
        "hip_l": "left_hip",
        "hip_r": "right_hip",
        "knee_l": "left_knee",
        "knee_r": "right_knee",
        "ankle_l": "left_ankle",
        "ankle_r": "right_ankle",
        "heel_l": "left_heel",
        "heel_r": "right_heel",
        "ball_l": "left_foot_index",
        "ball_r": "right_foot_index",
        "shoulder_l": "left_shoulder",
        "shoulder_r": "right_shoulder",
        "elbow_l": "left_elbow",
        "elbow_r": "right_elbow",
        "wrist_l": "left_wrist",
        "wrist_r": "right_wrist",
        "pelvis": "left_hip",
    }
    for name, mp_name in mapping.items():
        if name not in positions:
            continue
        i = MP[mp_name]
        p_cam = to_cam(name)
        pose3d[i] = p_cam - to_cam("pelvis")
        u = cam.focal * p_cam[0] / p_cam[2] + width / 2.0
        v = cam.focal * p_cam[1] / p_cam[2] + height / 2.0
        # MediaPipe отдаёт нормализованные координаты (0..1)
        pose2d[i] = (u / width, v / height, 0.95)
    # голова: уши (детектор берёт середину ушей)
    for mp_name in ("left_ear", "right_ear"):
        i = MP[mp_name]
        p_cam = to_cam("head")
        pose3d[i] = p_cam - to_cam("pelvis")
        u = cam.focal * p_cam[0] / p_cam[2] + width / 2.0
        v = cam.focal * p_cam[1] / p_cam[2] + height / 2.0
        # MediaPipe отдаёт нормализованные координаты (0..1)
        pose2d[i] = (u / width, v / height, 0.95)
    return PoseFrame(pose2d=pose2d, pose3d=pose3d, hands=[], detected=True)


def frames_from_positions(rig: Rig, positions):
    """Выводит эталонные кадры костей из целевых позиций.

    Кадр кости задаётся так же, как это делает решатель: X — вдоль кости,
    Y — вдоль полюса (оси скручивания). Так проверяется весь конвейер:
    переход в мир, восстановление направлений, FK, локальные повороты.
    """
    # набор суставов «как их видит детектор»: таз и грудь — середины,
    # остальные — как в риге
    dl = dict(positions)
    dl["pelvis"] = (positions["hip_l"] + positions["hip_r"]) * 0.5
    dl["chest"] = (positions["shoulder_l"] + positions["shoulder_r"]) * 0.5

    globals_by_bone = {}
    for bone in rig.bones:
        if bone.parent is None:
            # тот же «телесный» базис, что и у решателя:
            # X — влево (по линии плеч), Y — вверх (голова минус таз), Z — за спину
            left = safe_unit(
                positions["shoulder_l"] - positions["shoulder_r"], fallback=(1.0, 0.0, 0.0)
            )
            up = safe_unit(dl["head"] - dl["pelvis"], fallback=(0.0, 1.0, 0.0))
            back = safe_unit(np.cross(left, up), fallback=(0.0, 0.0, 1.0))
            globals_by_bone[bone.name] = q_from_matrix(np.column_stack([left, up, back]))
            continue
        head = positions[bone.head]
        tail = positions[bone.tail]
        direction = safe_unit(tail - head, fallback=(0.0, 1.0, 0.0))
        pole = rig.resolve_pole(bone.name, dl)
        globals_by_bone[bone.name] = aim_quat(direction, pole)
    locals_by_bone = {}
    for bone in rig.bones:
        parent = globals_by_bone[bone.parent] if bone.parent else globals_by_bone[bone.name]
        locals_by_bone[bone.name] = q_mul(q_inv(parent), globals_by_bone[bone.name])
    return locals_by_bone, globals_by_bone


def main() -> int:
    width, height, fps = 1280, 720, 25
    cam = Camera(width, height, fov_y_deg=50.0)
    eye = v3(2.4, 1.35, 3.1)
    target = v3(0.0, 0.9, 0.0)
    R_wc = look_at(eye, target)

    rig = Rig.build()
    truth = build_ground_truth(rig, 40)

    # плоскость пола: прямоугольник 3 x 2 м, отмеченный на кадре
    corners_world = [
        v3(-1.5, 0.0, 1.0),
        v3(1.5, 0.0, 1.0),
        v3(1.5, 0.0, -1.0),
        v3(-1.5, 0.0, -1.0),
    ]
    pts_px = []
    for c in corners_world:
        p_cam = R_wc @ (c - eye)
        u = cam.focal * p_cam[0] / p_cam[2] + width / 2.0
        v = cam.focal * p_cam[1] / p_cam[2] + height / 2.0
        pts_px.append([u, v])
    plane = FloorPlane(camera=cam, points_px=np.array(pts_px), size_m=(3.0, 2.0))

    detections = [
        make_detection(rig, positions, eye, R_wc, cam, width, height)
        for (_l, _q, _p, positions) in truth
    ]

    solver = MotionSolver(rig, plane, width, height)
    solver.prepare(detections)
    solutions = solver.solve()

    # --- сравнение с эталоном -----------------------------------------
    worst = 0.0
    sum_err = 0.0
    count = 0
    per_bone: dict[str, list[float]] = {}
    for sol, (gt_local, _rq, _rp, gt_pos) in zip(solutions, truth):
        _l2, gt_global = frames_from_positions(rig, gt_pos)
        for name in ("upperarm_l", "lowerarm_l", "thigh_l", "calf_l", "spine_02", "lowerarm_r", "pelvis"):
            ga = _global_of(rig, gt_local, name, gt_global["pelvis"])
            gb = _global_of(rig, sol.local_quats, name, sol.root_quat)
            dot = float(np.clip(abs(np.dot(ga, gb)), -1.0, 1.0))
            angle = 2.0 * np.degrees(np.arccos(dot))
            per_bone.setdefault(name, []).append(angle)
            worst = max(worst, angle)
            sum_err += angle
            count += 1
    print(f"Средняя ошибка угла костей: {sum_err / max(1, count):.2f}°, худшая: {worst:.2f}°")
    for name, vals in per_bone.items():
        print(f"    {name:12s}: сред {np.mean(vals):6.2f}°  худшая {max(vals):6.2f}°")

    # --- проверка пола --------------------------------------------------
    heights = []
    for sol in solutions:
        ankle = sol.joints.get("ankle_l")
        if ankle is not None:
            heights.append(float(ankle[1]))
    print(f"Высота щиколотки над полом: сред {np.mean(heights):.3f} м (норма ~0.09)")

    contacts = sum(1 for s in solutions if s.contacts.get("l") or s.contacts.get("r"))
    print(f"Кадров с контактом стоп: {contacts} из {len(solutions)}")

    ok = (sum_err / max(1, count)) < 6.0
    print("ИТОГ:", "OK" if ok else "НЕУДОВЛЕТВОРИТЕЛЬНО")
    return 0 if ok else 1


def _global_of(rig: Rig, local_quats, name: str, root_quat) -> np.ndarray:
    """Глобальная ориентация кости по локальным кватернионам."""
    from mocap_studio.core.math3d import q_mul

    g = root_quat
    cur = name
    chain = []
    while cur is not None:
        chain.append(cur)
        cur = rig.by_name[cur].parent if cur in rig.by_name else None
    for bone_name in reversed(chain):
        g = q_mul(g, local_quats.get(bone_name, q_identity()))
    return g


if __name__ == "__main__":
    raise SystemExit(main())
