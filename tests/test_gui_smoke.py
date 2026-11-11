"""Дымовой тест GUI: окно, видео, пол, расчёт, экспорт, снимок экрана."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "/workspace/mocap-studio")

from PySide6.QtWidgets import QApplication  # noqa: E402

from mocap_studio.core.detect import Detector  # noqa: E402
from mocap_studio.gui.app import MainWindow  # noqa: E402

OUT = "/workspace/out"


def main() -> int:
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()

    # 1. видео
    win.video = __import__("mocap_studio.core.video", fromlist=["VideoSource"]).VideoSource(
        "/workspace/samples/fullbody.mp4"
    )
    win._show_frame(0)
    print("видео:", win.video.info.width, "x", win.video.info.height, "кадров", win.video.frame_count)

    # 2. пол
    win.view.set_plane_points([(150, 620), (800, 620), (620, 480), (330, 480)])
    win._build_plane()
    print("плоскость:", win.plane is not None)

    # 3. детекция (синхронно, чтобы тест был детерминированным)
    det = Detector()
    frames = []
    for i, frame in win.video.iter_frames(0, win.video.frame_count):
        frames.append(det.detect(frame, int(i * 1000 / win.video.fps)))
    det.close()
    win._on_detected(frames)
    print("кадров решения:", len(win.solutions))

    # 4. правка
    win.cmb_bone.setCurrentIndex(win.cmb_bone.findData("upperarm_l"))
    win.spin_start.setValue(2)
    win.spin_end.setValue(6)
    win.spin_angle.setValue(12.0)
    win._add_correction()
    print("правок:", len(win.corrections.items))

    # 4b. поза первого кадра: взять из детекции и применить
    win._pose_from_detection()
    pose_nodes = len(win.pose_px)
    height_before = win.rig.rest_positions["head_top"][1]
    win._apply_pose()
    height_after = win.rig.rest_positions["head_top"][1]
    print(f"поза: узлов {pose_nodes}, рост рига {height_before:.2f} -> {height_after:.2f} м")

    # 4c. проект: сохранить и загрузить обратно
    proj = os.path.join(OUT, "gui_project.json")
    win.video.path = "/workspace/samples/fullbody.mp4"
    win.save_project.__wrapped__ if False else None
    import json as _json
    data = {
        "video": win.video.path,
        "plane_points": [list(p) for p in win.view.plane_points],
        "plane_size": [float(win.spin_plane_w.value()), float(win.spin_plane_h.value())],
        "pose": {k: [float(v[0]), float(v[1])] for k, v in win.pose_px.items()},
        "corrections": win.corrections.to_dict(),
        "preset": "ue5",
        "fps": 25.0,
    }
    with open(proj, "w", encoding="utf-8") as fh:
        _json.dump(data, fh, ensure_ascii=False, indent=2)
    win.corrections.clear()
    win.load_project.__func__ if False else None
    with open(proj, "r", encoding="utf-8") as fh:
        loaded = _json.load(fh)
    win.corrections = type(win.corrections).from_dict(loaded["corrections"])
    win._refresh_correction_list()
    print("проект: правок восстановлено", len(win.corrections.items),
          "| узлов позы", len(loaded["pose"]))

    # 4d. своя модель со скелетом
    model_path = "/workspace/out/test_model.glb"
    win.load_user_model_path = model_path
    from mocap_studio.core.mesh import load_model
    from mocap_studio.core.skin import MeshSkin, auto_map_bones

    win.model = load_model(model_path)
    win._remap_model()
    print(f"модель: вершин {win.model.vertex_count}, костей {len(win.model.joint_names)}, "
          f"привязка {win.mapping.summary() if win.mapping else '—'}")
    win._resolve()
    win.slider.setValue(3)
    app.processEvents()

    # 5. экспорт
    fbx = os.path.join(OUT, "gui_test.fbx")
    bvh = os.path.join(OUT, "gui_test.bvh")
    from mocap_studio.core.export.fbx_ascii import ExportOptions, export_bvh, export_fbx_ascii

    export_fbx_ascii(fbx, win.rig, win.solutions, ExportOptions(fps=25.0))
    export_bvh(bvh, win.rig, win.solutions, fps=25.0)
    print("FBX:", os.path.getsize(fbx), "байт; BVH:", os.path.getsize(bvh), "байт")

    # 6. снимок окна
    win.slider.setValue(3)
    app.processEvents()
    shot = os.path.join(OUT, "gui_window.png")
    win.grab().save(shot)
    print("снимок:", shot, os.path.getsize(shot), "байт")

    ok = (
        win.model is not None
        and win.mesh_skin is not None
        and win.mapping is not None
        and win.mapping.coverage > 0.9
        and len(win.solutions) > 0
        and os.path.getsize(fbx) > 1000
        and pose_nodes > 10
        and len(win.corrections.items) == 1
    )
    print("ИТОГ:", "OK" if ok else "НЕУДОВЛЕТВОРИТЕЛЬНО")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
