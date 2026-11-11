"""Командный интерфейс: видео -> анимация -> FBX/BVH (без графики).

Пример:
    python -m mocap_studio.cli --video walk.mp4 \
        --plane 40,980 300,980 300,860 40,860 --plane-size 1.2 1.2 \
        --out walk.fbx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .core.corrections import CorrectionSet
from .core.detect import Detector
from .core.export.fbx_ascii import ExportOptions, export_bvh, export_fbx_ascii
from .core.plane import Camera, FloorPlane
from .core.rig import Rig
from .core.solve import MotionSolver, SolveOptions, calibrate_lengths
from .core.video import VideoSource


def parse_points(text: str) -> np.ndarray:
    """Парсит 4 точки: принимает форматы "x1,y1 x2,y2 ..." и "x1 y1 x2 y2 ..."."""
    import re

    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if len(nums) != 8:
        raise SystemExit("Нужно ровно 4 точки плоскости: x1,y1 x2,y2 x3,y3 x4,y4")
    return np.array(nums, dtype=np.float64).reshape(4, 2)


def detect_all(video: VideoSource, detector: Detector, start: int, end: int | None, step: int = 1):
    frames = []
    indices = []
    total = end - start if end is not None else (video.frame_count - start)
    t0 = time.time()
    last = end if end is not None else video.frame_count
    for i, frame in video.iter_frames(start, last, step):
        ts = int(round(i * 1000.0 / video.fps))
        pf = detector.detect(frame, ts)
        frames.append(pf)
        indices.append(i)
        if len(frames) % 10 == 0:
            done = len(indices) / max(1, total)
            sys.stdout.write(
                f"\r  детекция {len(indices)}/{total} кадров ({done * 100:.0f}%), "
                f"{(time.time() - t0) / max(1, len(indices)):.3f} с/кадр"
            )
            sys.stdout.flush()
    sys.stdout.write("\n")
    return indices, frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MoCap Studio — захват движения по видео")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None, help="путь к .fbx")
    ap.add_argument("--bvh", default=None, help="путь к .bvh")
    ap.add_argument("--plane", default=None, help="4 точки пола в пикселях: x1,y1 x2,y2 x3,y3 x4,y4")
    ap.add_argument("--plane-size", default="1.0,1.0", help="реальный размер прямоугольника пола, м")
    ap.add_argument(
        "--no-auto-scale",
        dest="auto_scale",
        action="store_false",
        help="не подгонять масштаб сцены под рост человека",
    )
    ap.set_defaults(auto_scale=True)
    ap.add_argument("--fov", type=float, default=50.0, help="вертикальный FOV камеры, градусы")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--fps", type=float, default=None, help="fps анимации (по умолчанию — fps видео)")
    ap.add_argument("--names", "--preset", dest="names", default="ue5",
                    choices=["ue5", "mixamo", "max_biped"], help="пресет имён костей")
    ap.add_argument("--preview", default=None, help="ролик со скелетом поверх видео (mp4)")
    ap.add_argument("--user-pose", default=None, help="JSON с позой первого кадра (длины костей)")
    ap.add_argument("--corrections", default=None, help="JSON с ручными правками")
    ap.add_argument("--report", default=None, help="куда сохранить JSON-отчёт")
    args = ap.parse_args(argv)

    if not os.path.exists(args.video):
        raise SystemExit(f"Видео не найдено: {args.video}")

    video = VideoSource(args.video)
    info = video.info
    print(f"Видео: {info.width}x{info.height} @ {info.fps:.3f} fps, кадров {info.frame_count}")

    detector = Detector()
    indices, frames = detect_all(video, detector, args.start, args.end, args.step)

    end = args.end if args.end is not None else video.frame_count
    if not frames:
        raise SystemExit("Не удалось прочитать кадры")

    cam = Camera(width=info.width, height=info.height, fov_y_deg=args.fov)
    if args.plane:
        size = tuple(float(x) for x in args.plane_size.split(","))
        plane = FloorPlane(camera=cam, points_px=parse_points(args.plane), size_m=size)
    else:
        raise SystemExit("Нужен --plane (4 точки прямоугольника пола в кадре)")

    rig = Rig.build()
    if args.user_pose:
        with open(args.user_pose, "r", encoding="utf-8") as fh:
            user = json.load(fh)
        lengths = calibrate_lengths(rig, frames[0], {k: np.array(v) for k, v in user.items()}, info.width, info.height)
        rig.apply_lengths(lengths)

    corrections = CorrectionSet()
    if args.corrections:
        with open(args.corrections, "r", encoding="utf-8") as fh:
            corrections = CorrectionSet.from_dict(json.load(fh))

    solver = MotionSolver(
        rig, plane, info.width, info.height,
        options=SolveOptions(), corrections=corrections,
    )
    solver.prepare(frames)
    if args.auto_scale:
        k = plane.scale_to_person(rig, solver._joints2d)
        print(f"Масштаб сцены подобран по фигуре: x{k:.3f}")
    solutions = solver.solve()

    print(f"Решено кадров: {len(solutions)}")
    contacts_l = sum(1 for s in solutions if s.contacts.get("l"))
    contacts_r = sum(1 for s in solutions if s.contacts.get("r"))
    print(f"Контакт стопы: левая {contacts_l} кадров, правая {contacts_r} кадров")

    fps = args.fps or info.fps
    opts = ExportOptions(fps=fps, name_preset=args.names)
    if args.out:
        export_fbx_ascii(args.out, rig, solutions, opts)
        print(f"FBX: {args.out} ({os.path.getsize(args.out) / 1024:.0f} КБ)")
    if args.bvh:
        export_bvh(args.bvh, rig, solutions, fps=fps, name_preset=args.names)
        print(f"BVH: {args.bvh} ({os.path.getsize(args.bvh) / 1024:.0f} КБ)")

    if args.preview:
        from .core.preview import render_preview

        render_preview(video, solutions, rig, plane, args.preview, fps=fps)
        print(f"Превью: {args.preview}")

    if args.report:
        report = {
            "frames": len(solutions),
            "fps": fps,
            "plane": plane.to_dict(),
            "contacts_l": contacts_l,
            "contacts_r": contacts_r,
            "root": [list(map(float, s.root_pos)) for s in solutions],
            "confidence": [float(s.confidence) for s in solutions],
        }
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"Отчёт: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
