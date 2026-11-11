"""Превью: проекция решённого скелета обратно в кадр видео.

Нужна и для проверки качества, и как пользовательская функция — записать
ролик «скелет поверх видео», чтобы оценить результат до экспорта в DCC.
"""
from __future__ import annotations

import numpy as np

from .rig import Rig


def project_world_to_image(plane, point_world: np.ndarray, width: int, height: int):
    """Мировая точка -> пиксель кадра (None, если за камерой)."""
    p_cam = plane.camera_from_world(np.asarray(point_world, dtype=np.float64))
    if p_cam[2] <= 1e-4:
        return None
    k = plane.camera.focal
    u = k * p_cam[0] / p_cam[2] + width / 2.0
    v = k * p_cam[1] / p_cam[2] + height / 2.0
    return float(u), float(v)


def draw_skeleton(
    image,
    rig: Rig,
    solution,
    plane,
    width: int,
    height: int,
    color=(0, 255, 80),
    thickness: int = 2,
    show_contacts: bool = True,
):
    """Рисует кости решения поверх кадра (OpenCV BGR)."""
    import cv2

    joints = solution.joints
    for bone in rig.bones:
        head = joints.get(bone.head)
        tail = joints.get(bone.tail)
        if head is None or tail is None:
            continue
        p1 = project_world_to_image(plane, head, width, height)
        p2 = project_world_to_image(plane, tail, width, height)
        if p1 is None or p2 is None:
            continue
        cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)
        cv2.circle(image, (int(p1[0]), int(p1[1])), 3, color, -1)

    if show_contacts:
        for side, contact in (solution.contacts or {}).items():
            if not contact:
                continue
            joint = joints.get(f"ankle_{side}")
            if joint is None:
                continue
            p = project_world_to_image(plane, joint, width, height)
            if p is not None:
                cv2.circle(image, (int(p[0]), int(p[1])), 7, (0, 140, 255), 2)
    return image


def render_preview(
    video,
    solutions,
    rig: Rig,
    plane,
    out_path: str,
    start: int = 0,
    end: int | None = None,
    show_video: bool = True,
    fps: float | None = None,
) -> str:
    """Записывает mp4/gif со скелетом поверх видео."""
    import cv2

    info = video.info
    width, height = info.width, info.height
    last = end if end is not None else min(info.frame_count, start + len(solutions))
    fps = fps or info.fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    try:
        for idx, frame in video.iter_frames(start, last):
            sol = solutions[idx - start] if 0 <= idx - start < len(solutions) else None
            canvas = frame.copy() if show_video else np.zeros_like(frame)
            if sol is not None and sol.ok:
                draw_skeleton(canvas, rig, sol, plane, width, height)
            writer.write(canvas)
    finally:
        writer.release()
    return out_path
