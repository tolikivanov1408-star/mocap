"""Главное окно MoCap Studio."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QTableWidget,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QSpinBox,
    QStatusBar,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.corrections import Correction, CorrectionSet
from ..core.detect import MP, PoseFrame
from ..core.export.fbx_ascii import ExportOptions, export_bvh, export_fbx_ascii
from ..core.math3d import euler_xyz_degrees, quat_from_axis_angle
from ..core.mesh import load_model
from ..core.skin import MeshSkin, auto_map_bones
from ..core.plane import Camera, FloorPlane
from ..core.rig import Rig
from ..core.solve import MotionSolver, SolveOptions, build_joints, calibrate_lengths
from ..core.video import VideoSource
from .glview import Model3DView
from .view import MODE_PLANE, MODE_POSE, MODE_VIEW, VideoView
from .worker import DetectWorker

APP_TITLE = "MoCap Studio — захват движения по видео"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1440, 900)

        # состояние
        self.video: VideoSource | None = None
        self.frames: list[PoseFrame] = []
        self.solutions = []
        self.plane: FloorPlane | None = None
        self.rig = Rig.build()
        self.corrections = CorrectionSet()
        self.pose_px: dict[str, np.ndarray] = {}
        self.model = None
        self.mapping = None
        self.mesh_skin = None
        self._worker: DetectWorker | None = None
        self._playing = False
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(33)
        self._play_timer.timeout.connect(self._play_step)

        self._build_ui()
        self._fill_bone_combo()
        self._update_actions()

    # ------------------------------------------------------------------
    # интерфейс
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.view = VideoView()
        self.view.planePointClicked.connect(self._on_plane_click)
        self.view.poseJointMoved.connect(self._on_pose_moved)
        self.glance = Model3DView()

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_plane_tab(), "Пол")
        self.tabs.addTab(self._build_pose_tab(), "Поза")
        self.tabs.addTab(self._build_corrections_tab(), "Правки")
        self.tabs.addTab(self._build_model_tab(), "Модель")
        self.tabs.addTab(self._build_export_tab(), "Экспорт")

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 6, 6, 6)
        rv.addWidget(self.tabs)
        rv.addWidget(self.glance, 1)
        right.setMinimumWidth(420)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.view)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split, 1)
        lay.addWidget(self._build_timeline())
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        self._build_menu()
        self._build_toolbar()

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("Файл")
        act_open = QAction("Открыть видео…", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self.open_video)
        m.addAction(act_open)
        act_save_proj = QAction("Сохранить проект…", self)
        act_save_proj.triggered.connect(self.save_project)
        m.addAction(act_save_proj)
        act_load_proj = QAction("Открыть проект…", self)
        act_load_proj.triggered.connect(self.load_project)
        m.addAction(act_load_proj)
        m.addSeparator()
        act_quit = QAction("Выход", self)
        act_quit.triggered.connect(self.close)
        m.addAction(act_quit)

        m2 = self.menuBar().addMenu("Действия")
        act_plane = QAction("Разметить пол", self)
        act_plane.setShortcut("P")
        act_plane.triggered.connect(lambda: self._set_mode(MODE_PLANE))
        m2.addAction(act_plane)
        act_pose = QAction("Править позу", self)
        act_pose.setShortcut("O")
        act_pose.triggered.connect(lambda: self._set_mode(MODE_POSE))
        m2.addAction(act_pose)
        m2.addSeparator()
        act_solve = QAction("Считать движение", self)
        act_solve.setShortcut("Ctrl+R")
        act_solve.triggered.connect(self.solve_motion)
        m2.addAction(act_solve)

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Основное")
        self.act_open = QAction("Открыть видео", self)
        self.act_open.triggered.connect(self.open_video)
        tb.addAction(self.act_open)
        tb.addSeparator()
        self.act_plane = QAction("Пол", self)
        self.act_plane.triggered.connect(lambda: self._set_mode(MODE_PLANE))
        tb.addAction(self.act_plane)
        self.act_pose = QAction("Поза", self)
        self.act_pose.triggered.connect(lambda: self._set_mode(MODE_POSE))
        tb.addAction(self.act_pose)
        tb.addSeparator()
        self.act_solve = QAction("Считать движение", self)
        self.act_solve.triggered.connect(self.solve_motion)
        tb.addAction(self.act_solve)
        self.act_cancel = QAction("Стоп", self)
        self.act_cancel.setEnabled(False)
        self.act_cancel.triggered.connect(self._cancel_detection)
        tb.addAction(self.act_cancel)
        tb.addSeparator()
        self.act_fbx = QAction("Экспорт FBX", self)
        self.act_fbx.triggered.connect(self.export_fbx)
        tb.addAction(self.act_fbx)

    def _build_plane_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        hint = QLabel(
            "Включите режим «Пол» и щёлкните по четырём точкам на полу —\n"
            "ближняя левая, ближняя правая, дальняя правая, дальняя левая.\n"
            "Эти точки задают плоскость: по ней считается глубина сцены."
        )
        hint.setWordWrap(True)
        f.addRow(hint)
        self.spin_plane_w = QDoubleSpinBox()
        self.spin_plane_w.setRange(0.1, 100.0)
        self.spin_plane_w.setDecimals(2)
        self.spin_plane_w.setValue(1.0)
        self.spin_plane_w.setSuffix(" м")
        f.addRow("Размер по первой стороне:", self.spin_plane_w)
        self.spin_plane_h = QDoubleSpinBox()
        self.spin_plane_h.setRange(0.1, 100.0)
        self.spin_plane_h.setDecimals(2)
        self.spin_plane_h.setValue(1.0)
        self.spin_plane_h.setSuffix(" м")
        f.addRow("Размер по второй стороне:", self.spin_plane_h)
        btn_reset = QPushButton("Сбросить разметку пола")
        btn_reset.clicked.connect(self._reset_plane)
        f.addRow(btn_reset)
        self.lbl_plane_info = QLabel("Пол не размечен")
        self.lbl_plane_info.setWordWrap(True)
        f.addRow(self.lbl_plane_info)
        self.chk_autoscale = QCheckBox("Подогнать масштаб сцены под рост фигуры")
        self.chk_autoscale.setChecked(True)
        f.addRow(self.chk_autoscale)
        return w

    def _build_pose_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        txt = QLabel(
            "Поза ставится на первом кадре: включите режим «Поза» и\n"
            "перетащите узлы так, чтобы скелет совпал с человеком.\n"
            "После этого длины костей рига подстраиваются под позу."
        )
        txt.setWordWrap(True)
        v.addWidget(txt)
        btn_init = QPushButton("Взять позу из детекции")
        btn_init.clicked.connect(self._pose_from_detection)
        v.addWidget(btn_init)
        btn_apply = QPushButton("Применить позу (длины костей)")
        btn_apply.clicked.connect(self._apply_pose)
        v.addWidget(btn_apply)
        self.lbl_pose_info = QLabel("Поза не задана")
        self.lbl_pose_info.setWordWrap(True)
        v.addWidget(self.lbl_pose_info)
        v.addStretch(1)
        return w

    def _build_corrections_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        txt = QLabel(
            "Правка — это поворот кости на выбранном отрезке кадров.\n"
            "После применения движение пересчитывается целиком:\n"
            "стопы, прижатые к полу, остаются на месте."
        )
        txt.setWordWrap(True)
        v.addWidget(txt)
        self.cmb_bone = QComboBox()
        v.addWidget(QLabel("Кость:"))
        v.addWidget(self.cmb_bone)
        row = QHBoxLayout()
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, 100000)
        self.spin_end = QSpinBox()
        self.spin_end.setRange(0, 100000)
        row.addWidget(QLabel("кадры:"))
        row.addWidget(self.spin_start)
        row.addWidget(QLabel("—"))
        row.addWidget(self.spin_end)
        v.addLayout(row)
        row2 = QHBoxLayout()
        self.cmb_axis = QComboBox()
        for name, title in (("X", "ось X (вдоль кости)"), ("Y", "ось Y"), ("Z", "ось Z")):
            self.cmb_axis.addItem(title, name)
        row2.addWidget(self.cmb_axis)
        self.spin_angle = QDoubleSpinBox()
        self.spin_angle.setRange(-180.0, 180.0)
        self.spin_angle.setDecimals(1)
        self.spin_angle.setSuffix(" °")
        self.spin_angle.setValue(10.0)
        row2.addWidget(self.spin_angle)
        v.addLayout(row2)
        btn_add = QPushButton("Добавить правку и пересчитать")
        btn_add.clicked.connect(self._add_correction)
        v.addWidget(btn_add)
        self.list_corrections = QListWidget()
        v.addWidget(self.list_corrections, 1)
        row3 = QHBoxLayout()
        btn_del = QPushButton("Удалить выбранную")
        btn_del.clicked.connect(self._remove_correction)
        row3.addWidget(btn_del)
        btn_clear = QPushButton("Очистить все")
        btn_clear.clicked.connect(self._clear_corrections)
        row3.addWidget(btn_clear)
        v.addLayout(row3)
        return w

    def _build_model_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        txt = QLabel(
            "Своя модель со скелетом: загрузите .glb/.gltf/.fbx — кости модели\n"
            "сопоставятся с костями рига, и вы сразу увидите анимацию."
        )
        txt.setWordWrap(True)
        v.addWidget(txt)
        self.cmb_model_preset = QComboBox()
        for key, title in (
            ("auto", "Авто (определить по именам костей)"),
            ("ue5", "Unreal Engine (mannequin)"),
            ("mixamo", "Mixamo"),
            ("max_biped", "3ds Max Biped"),
        ):
            self.cmb_model_preset.addItem(title, key)
        self.cmb_model_preset.currentIndexChanged.connect(self._on_preset_changed)
        v.addWidget(QLabel("Пресет имён костей модели:"))
        v.addWidget(self.cmb_model_preset)
        btn_load = QPushButton("Загрузить модель…")
        btn_load.clicked.connect(self.load_user_model)
        v.addWidget(btn_load)
        self.lbl_model_info = QLabel("Модель не загружена")
        self.lbl_model_info.setWordWrap(True)
        v.addWidget(self.lbl_model_info)
        self.lbl_map_info = QLabel("")
        self.lbl_map_info.setWordWrap(True)
        v.addWidget(self.lbl_map_info)
        self.chk_show_mesh = QCheckBox("Показывать модель")
        self.chk_show_mesh.setChecked(True)
        self.chk_show_mesh.toggled.connect(self._on_view_options)
        v.addWidget(self.chk_show_mesh)
        self.chk_show_bones = QCheckBox("Показывать кости рига")
        self.chk_show_bones.setChecked(True)
        self.chk_show_bones.toggled.connect(self._on_view_options)
        v.addWidget(self.chk_show_bones)
        v.addWidget(QLabel("Ручная привязка костей (непривязанные кости рига):"))
        self.tbl_map = QTableWidget(0, 2)
        self.tbl_map.setHorizontalHeaderLabels(["Кость рига", "Кость модели"])
        self.tbl_map.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.tbl_map, 1)
        btn_remap = QPushButton("Перепривязать автоматически")
        btn_remap.clicked.connect(self._remap_model)
        v.addWidget(btn_remap)
        return w

    def _build_export_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.cmb_preset = QComboBox()
        for key, title in (
            ("ue5", "Unreal Engine (mannequin)"),
            ("mixamo", "Mixamo"),
            ("max_biped", "3ds Max Biped"),
        ):
            self.cmb_preset.addItem(title, key)
        f.addRow("Имена костей:", self.cmb_preset)
        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(1.0, 240.0)
        self.spin_fps.setValue(30.0)
        f.addRow("Кадров в секунду:", self.spin_fps)
        btn_fbx = QPushButton("Сохранить FBX…")
        btn_fbx.clicked.connect(self.export_fbx)
        f.addRow(btn_fbx)
        btn_bvh = QPushButton("Сохранить BVH…")
        btn_bvh.clicked.connect(self.export_bvh)
        f.addRow(btn_bvh)
        btn_json = QPushButton("Сохранить отчёт (JSON)…")
        btn_json.clicked.connect(self.export_report)
        f.addRow(btn_json)
        self.lbl_export_info = QLabel("Считать движение, чтобы включить экспорт.")
        self.lbl_export_info.setWordWrap(True)
        f.addRow(self.lbl_export_info)
        return w

    def _build_timeline(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(8, 4, 8, 6)
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(40)
        self.btn_play.clicked.connect(self._toggle_play)
        h.addWidget(self.btn_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderReleased.connect(self._on_slider_release)
        self.slider.valueChanged.connect(self._on_slider_moved)
        h.addWidget(self.slider, 1)
        self.lbl_frame = QLabel("кадр 0")
        self.lbl_frame.setMinimumWidth(160)
        h.addWidget(self.lbl_frame)
        return w

    # ------------------------------------------------------------------
    # видео
    # ------------------------------------------------------------------

    def open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть видео", "", "Видео (*.mp4 *.mov *.mkv *.avi *.webm);;Все файлы (*)"
        )
        if not path:
            return
        try:
            self.video = VideoSource(path)
        except IOError as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return
        self.frames = []
        self.solutions = []
        self.plane = None
        self.pose_px = {}
        self.corrections.clear()
        self.view.set_plane_points([])
        self.view.set_pose({})
        self.view.set_solution(None)
        self.glance.set_solution(None)
        self.slider.setEnabled(False)
        self.slider.setRange(0, 0)
        self.lbl_plane_info.setText("Пол не размечен")
        self.lbl_pose_info.setText("Поза не задана")
        self._refresh_correction_list()
        self._show_frame(0)
        self.statusBar().showMessage(f"Загружено: {os.path.basename(path)} — {self.video.info.width}×{self.video.info.height}")
        self._update_actions()

    def _show_frame(self, index: int) -> None:
        if self.video is None:
            return
        idx = max(0, min(index, self.video.frame_count - 1))
        frame = self.video.read_at(idx)
        if frame is None:
            return
        self.view.set_frame(frame)
        if 0 <= idx < len(self.solutions):
            sol = self.solutions[idx]
            self.view.set_solution(sol, self.plane, self.rig)
            self.glance.set_solution(sol, self.rig)
        else:
            self.view.set_solution(None)
            self.glance.set_solution(None)
        self.lbl_frame.setText(f"кадр {idx} / {max(0, self.video.frame_count - 1)}")
        if 0 <= idx < len(self.frames):
            self._show_detection(idx)

    def _show_detection(self, index: int) -> None:
        frame = self.frames[index]
        j2 = {}
        for name, i in MP.items():
            if frame.pose2d is not None and i < len(frame.pose2d):
                j2[name] = np.array(
                    [
                        frame.pose2d[i, 0] * self.video.info.width,
                        frame.pose2d[i, 1] * self.video.info.height,
                    ]
                )
        self.view.detection2d = j2
        self.view.update()

    # ------------------------------------------------------------------
    # пол
    # ------------------------------------------------------------------

    def _set_mode(self, mode: str) -> None:
        self.view.set_mode(mode)
        if mode == MODE_PLANE:
            self.statusBar().showMessage(
                "Щёлкните четыре точки на полу: ближняя левая, ближняя правая, дальняя правая, дальняя левая"
            )
        elif mode == MODE_POSE:
            self.statusBar().showMessage("Перетащите узлы скелета, чтобы совпасть с позой человека")
        else:
            self.statusBar().clearMessage()

    def _on_plane_click(self, x: float, y: float) -> None:
        if self.video is None:
            return
        pts = list(self.view.plane_points)
        pts.append((float(x), float(y)))
        self.view.set_plane_points(pts)
        if len(pts) == 4:
            self._build_plane()
            self.view.set_mode(MODE_VIEW)
        else:
            self.statusBar().showMessage(f"Точка {len(pts)}/4. Нажмите Esc, чтобы сбросить.")

    def _build_plane(self) -> None:
        if self.video is None or len(self.view.plane_points) != 4:
            return
        info = self.video.info
        cam = Camera(info.width, info.height, 50.0)
        try:
            self.plane = FloorPlane(
                camera=cam,
                points_px=np.array(self.view.plane_points, dtype=np.float64),
                size_m=(float(self.spin_plane_w.value()), float(self.spin_plane_h.value())),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Плоскость", str(exc))
            self.plane = None
            return
        self.lbl_plane_info.setText(
            f"Плоскость задана ({len(self.plane.points_px)} точки). "
            "При смене размера прямоугольника нажмите «Считать движение» заново."
        )
        self.statusBar().showMessage("Пол размечен. Теперь можно задать позу и посчитать движение.")

    def _reset_plane(self) -> None:
        self.view.set_plane_points([])
        self.plane = None
        self.lbl_plane_info.setText("Пол не размечен")

    # ------------------------------------------------------------------
    # поза
    # ------------------------------------------------------------------

    def _pose_from_detection(self) -> None:
        if self.video is None:
            return
        if not self.frames:
            try:
                from ..core.detect import Detector

                det = Detector()
                frame = self.video.read_at(0)
                pf = det.detect(frame, 0)
                det.close()
                self.frames = [pf]
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "Поза", f"Не удалось получить детекцию: {exc}")
                return
        j3, j2 = build_joints(self.frames[0], self.rig, self.video.info.width, self.video.info.height)
        pose: dict[str, np.ndarray] = {}
        for name, pt in j2.items():
            if pt[2] is not None and float(pt[2]) > 0.2:
                pose[name] = np.array([float(pt[0]), float(pt[1])])
        if not pose:
            QMessageBox.warning(self, "Поза", "Детектор не нашёл суставы в первом кадре")
            return
        self.pose_px = pose
        self.view.set_pose(pose)
        self._set_mode(MODE_POSE)
        self.lbl_pose_info.setText(f"Задано узлов: {len(pose)}. Подправьте их и нажмите «Применить позу».")

    def _on_pose_moved(self) -> None:
        self.lbl_pose_info.setText(f"Узлов: {len(self.pose_px)}. Нажмите «Применить позу».")

    def _apply_pose(self) -> None:
        if not self.pose_px or not self.frames:
            QMessageBox.information(self, "Поза", "Сначала задайте позу на первом кадре")
            return
        lengths = calibrate_lengths(
            self.rig, self.frames[0], self.pose_px, self.video.info.width, self.video.info.height
        )
        if not lengths:
            QMessageBox.information(self, "Поза", "Не удалось вычислить длины костей")
            return
        self.rig = Rig.build(lengths)
        self.lbl_pose_info.setText(
            f"Поза применена: костей подогнано {len(lengths)}. Рост персонажа ≈ "
            f"{self.rig.rest_positions['head_top'][1]:.2f} м"
        )
        self._fill_bone_combo()
        # длины костей изменились — движение пересчитываем сразу (детекция уже в памяти)
        if self.solutions and self.plane is not None:
            self._resolve()
        self._update_actions()

    # ------------------------------------------------------------------
    # расчёт
    # ------------------------------------------------------------------

    def solve_motion(self) -> None:
        if self.video is None:
            QMessageBox.information(self, "Расчёт", "Откройте видео")
            return
        if self.plane is None:
            QMessageBox.information(self, "Расчёт", "Сначала разметьте пол (вкладка «Пол»)")
            return
        if self._worker is not None and self._worker.isRunning():
            return
        self.act_solve.setEnabled(False)
        self.act_cancel.setEnabled(True)
        self.statusBar().showMessage("Детекция…")
        self._worker = DetectWorker(self.video, 0, self.video.frame_count, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_detected)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._worker_cleanup)
        self._worker.start()

    def _cancel_detection(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def _worker_cleanup(self) -> None:
        self.act_cancel.setEnabled(False)
        self._update_actions()

    def _on_progress(self, done: int, total: int, per: float) -> None:
        self.statusBar().showMessage(f"Детекция {done}/{total} кадров ({per:.2f} с/кадр)")

    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка детекции", message)
        self.statusBar().clearMessage()

    def _on_detected(self, frames) -> None:
        self.frames = frames
        self.statusBar().showMessage("Детекция завершена, считаем движение…")
        self._resolve()

    def _resolve(self) -> None:
        """Пересчитывает движение из кэша детекции (быстро)."""
        if not self.frames or self.plane is None:
            return
        # плоскость пересобирается, если пользователь поменял размер
        info = self.video.info
        cam = Camera(info.width, info.height, 50.0)
        self.plane = FloorPlane(
            camera=cam,
            points_px=np.array(self.view.plane_points, dtype=np.float64),
            size_m=(float(self.spin_plane_w.value()), float(self.spin_plane_h.value())),
        )
        solver = MotionSolver(self.rig, self.plane, info.width, info.height, SolveOptions())
        solver.prepare(self.frames)
        if self.chk_autoscale.isChecked():
            self.plane.scale_to_person(self.rig, solver._joints2d)
        self.solutions = solver.solve()
        self.slider.setEnabled(True)
        self.slider.setRange(0, max(0, len(self.solutions) - 1))
        self.slider.setValue(0)
        self._show_frame(0)
        contacts = sum(1 for s in self.solutions if any(s.contacts.values()))
        self.lbl_export_info.setText(
            f"Готово: кадров {len(self.solutions)}, с контактом стоп {contacts}."
        )
        self.statusBar().showMessage(
            f"Движение посчитано: {len(self.solutions)} кадров, контакт стоп в {contacts} кадрах"
        )
        self._update_actions()

    # ------------------------------------------------------------------
    # правки
    # ------------------------------------------------------------------

    def _fill_bone_combo(self) -> None:
        current = self.cmb_bone.currentData()
        self.cmb_bone.blockSignals(True)
        self.cmb_bone.clear()
        for bone in self.rig.bones:
            self.cmb_bone.addItem(bone.name, bone.name)
        idx = self.cmb_bone.findData(current)
        if idx >= 0:
            self.cmb_bone.setCurrentIndex(idx)
        self.cmb_bone.blockSignals(False)

    def _add_correction(self) -> None:
        bone = self.cmb_bone.currentData()
        if not bone:
            return
        start = int(self.spin_start.value())
        end = int(self.spin_end.value())
        if end < start:
            start, end = end, start
        axis = self.cmb_axis.currentData()
        angle = float(self.spin_angle.value())
        axis_vec = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}[axis]
        quat = quat_from_axis_angle(axis_vec, angle)
        self.corrections.add(Correction(bone=bone, start=start, end=end, quat=quat))
        self._refresh_correction_list()
        self._resolve()
        self.statusBar().showMessage(f"Правка «{bone}» применена, движение пересчитано")

    def _remove_correction(self) -> None:
        row = self.list_corrections.currentRow()
        if row < 0:
            return
        self.corrections.remove(row)
        self._refresh_correction_list()
        self._resolve()

    def _clear_corrections(self) -> None:
        self.corrections.clear()
        self._refresh_correction_list()
        self._resolve()

    def _refresh_correction_list(self) -> None:
        self.list_corrections.clear()
        for c in self.corrections.items:
            e = euler_xyz_degrees(c.quat)
            ang = float(np.degrees(np.arccos(np.clip(float(c.quat[3]), -1, 1))))
            self.list_corrections.addItem(
                QListWidgetItem(f"{c.bone}  кадры {c.start}–{c.end}  угол {ang:.1f}°")
            )


    # ------------------------------------------------------------------
    # своя модель
    # ------------------------------------------------------------------

    def load_user_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить модель со скелетом",
            "",
            "3D-модель (*.glb *.gltf *.fbx);;Все файлы (*)",
        )
        if not path:
            return
        try:
            model = load_model(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Модель", f"Не удалось прочитать файл:\n{exc}")
            return
        if not len(model.vertices):
            QMessageBox.warning(self, "Модель", "В файле нет геометрии")
            return
        self.model = model
        self._update_model_info()
        self._remap_model()
        self.statusBar().showMessage(f"Загружена модель: {os.path.basename(path)}")

    def _update_model_info(self) -> None:
        if self.model is None:
            self.lbl_model_info.setText("Модель не загружена")
            return
        model = self.model
        self.lbl_model_info.setText(
            f"{model.name}\nформат: {model.source_format}, вершин: {model.vertex_count}, "
            f"треугольников: {model.triangle_count}, костей: {len(model.joint_names)}"
            + ("" if model.has_skin else "\nВНИМАНИЕ: у модели нет весов скиннинга")
        )

    def _on_preset_changed(self) -> None:
        if self.model is not None:
            self._remap_model()

    def _remap_model(self) -> None:
        if self.model is None:
            return
        self._update_model_info()
        preset = self.cmb_model_preset.currentData()
        self.mapping = auto_map_bones(
            self.rig, self.model, None if preset in (None, "auto") else preset
        )
        self._rebuild_skin()
        self._fill_mapping_table()

    def _rebuild_skin(self) -> None:
        if self.model is None or self.mapping is None:
            self.mesh_skin = None
        else:
            self.mesh_skin = MeshSkin(self.rig, self.model, self.mapping)
        self.glance.set_model(self.model, self.mesh_skin)
        if self.mapping is not None:
            self.lbl_map_info.setText(
                f"Привязка: {self.mapping.summary()}. "
                f"Не привязано костей модели: {len(self.mapping.unmatched_model)}."
            )

    def _fill_mapping_table(self) -> None:
        """Таблица ручной привязки: для непривязанных костей рига — выбор кости модели."""
        self.tbl_map.setRowCount(0)
        if self.model is None or self.mapping is None:
            return
        rig = self.rig
        mapping = self.mapping
        rows = []
        for ri, bone in enumerate(rig.bones):
            if bone.name in mapping.rig_to_model:
                continue
            rows.append((ri, bone.name))
        if not rows:
            note = QTableWidgetItem("(все кости рига привязаны)")
            self.tbl_map.setRowCount(1)
            self.tbl_map.setItem(0, 0, note)
            self.tbl_map.setItem(0, 1, QTableWidgetItem(""))
            return
        self.tbl_map.setRowCount(len(rows))
        for row, (ri, bone_name) in enumerate(rows):
            item = QTableWidgetItem(bone_name)
            item.setData(Qt.UserRole, ri)
            self.tbl_map.setItem(row, 0, item)
            combo = QComboBox()
            combo.addItem("— не использовать —", -1)
            for j, mname in enumerate(self.model.joint_names):
                combo.addItem(mname, j)
            current = -1
            if bone_name in mapping.rig_to_model:
                current = mapping.rig_to_model[bone_name]
            idx = combo.findData(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.currentIndexChanged.connect(self._on_manual_mapping)
            self.tbl_map.setCellWidget(row, 1, combo)

    def _on_manual_mapping(self) -> None:
        """Пользователь поправил привязку вручную."""
        if self.model is None or self.mapping is None:
            return
        mapping = self.mapping
        mapping.model_to_rig.clear()
        mapping.rig_to_model.clear()
        mapping.matched_by_name = 0
        mapping.matched_by_position = 0
        for row in range(self.tbl_map.rowCount()):
            item = self.tbl_map.item(row, 0)
            widget = self.tbl_map.cellWidget(row, 1)
            if item is None or not isinstance(widget, QComboBox):
                continue
            ri = int(item.data(Qt.UserRole))
            bone_name = item.text()
            j = widget.currentData()
            if j is None or int(j) < 0:
                continue
            mapping.model_to_rig[int(j)] = bone_name
            mapping.rig_to_model[bone_name] = int(j)
        mapping.unmatched_model = [
            self.model.joint_names[j]
            for j in range(len(self.model.joint_names))
            if j not in mapping.model_to_rig
        ]
        mapping.unmatched_rig = [
            b.name for b in self.rig.bones if b.name not in mapping.rig_to_model
        ]
        self._rebuild_skin()
        self.lbl_map_info.setText(
            f"Привязка вручную: костей {len(mapping.rig_to_model)}. "
            f"Не привязано костей модели: {len(mapping.unmatched_model)}."
        )

    def _on_view_options(self) -> None:
        self.glance.show_mesh = self.chk_show_mesh.isChecked()
        self.glance.show_bones = self.chk_show_bones.isChecked()
        self.glance.update()

    # ------------------------------------------------------------------
    # экспорт
    # ------------------------------------------------------------------

    def export_fbx(self) -> None:
        if not self.solutions:
            QMessageBox.information(self, "Экспорт", "Сначала посчитайте движение")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить FBX", "mocap.fbx", "FBX (*.fbx)")
        if not path:
            return
        opts = ExportOptions(
            fps=float(self.spin_fps.value()),
            name_preset=self.cmb_preset.currentData(),
        )
        try:
            export_fbx_ascii(path, self.rig, self.solutions, opts)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Экспорт", f"Не удалось сохранить FBX: {exc}")
            return
        self.statusBar().showMessage(f"FBX сохранён: {path}")

    def export_bvh(self) -> None:
        if not self.solutions:
            QMessageBox.information(self, "Экспорт", "Сначала посчитайте движение")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить BVH", "mocap.bvh", "BVH (*.bvh)")
        if not path:
            return
        try:
            export_bvh(
                path,
                self.rig,
                self.solutions,
                fps=float(self.spin_fps.value()),
                name_preset=self.cmb_preset.currentData(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Экспорт", f"Не удалось сохранить BVH: {exc}")
            return
        self.statusBar().showMessage(f"BVH сохранён: {path}")

    def export_report(self) -> None:
        if not self.solutions:
            QMessageBox.information(self, "Экспорт", "Сначала посчитайте движение")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить отчёт", "mocap.json", "JSON (*.json)")
        if not path:
            return
        data = {
            "frames": len(self.solutions),
            "fps": float(self.spin_fps.value()),
            "preset": self.cmb_preset.currentData(),
            "plane": self.plane.to_dict() if self.plane else None,
            "corrections": self.corrections.to_dict(),
            "rig": {name: float(np.linalg.norm(self.rig.rest_positions[b.tail] - self.rig.rest_positions[b.head]))
                    for b in self.rig.bones for name, b in [(b.name, b)]},
            "root_positions": [[float(x) for x in s.root_pos] for s in self.solutions],
            "contacts": [{k: bool(v) for k, v in s.contacts.items()} for s in self.solutions],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        self.statusBar().showMessage(f"Отчёт сохранён: {path}")

    # ------------------------------------------------------------------
    # проект
    # ------------------------------------------------------------------

    def save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", "mocap_project.json", "JSON (*.json)")
        if not path:
            return
        data = {
            "video": self.video.path if self.video else None,
            "plane_points": [list(p) for p in self.view.plane_points],
            "plane_size": [float(self.spin_plane_w.value()), float(self.spin_plane_h.value())],
            "pose": {k: [float(v[0]), float(v[1])] for k, v in self.pose_px.items()},
            "corrections": self.corrections.to_dict(),
            "preset": self.cmb_preset.currentData(),
            "fps": float(self.spin_fps.value()),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        self.statusBar().showMessage(f"Проект сохранён: {path}")

    def load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", "", "JSON (*.json)")
        if not path:
            return
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        video_path = data.get("video")
        if video_path and os.path.exists(video_path):
            self.video = VideoSource(video_path)
            self._show_frame(0)
        pts = data.get("plane_points") or []
        self.view.set_plane_points([tuple(p) for p in pts])
        size = data.get("plane_size") or [1.0, 1.0]
        self.spin_plane_w.setValue(float(size[0]))
        self.spin_plane_h.setValue(float(size[1]))
        if len(pts) == 4:
            self._build_plane()
        self.pose_px = {k: np.array(v, dtype=np.float64) for k, v in (data.get("pose") or {}).items()}
        self.view.set_pose(self.pose_px)
        self.corrections = CorrectionSet.from_dict(data.get("corrections") or [])
        self._refresh_correction_list()
        idx = self.cmb_preset.findData(data.get("preset", "ue5"))
        if idx >= 0:
            self.cmb_preset.setCurrentIndex(idx)
        self.spin_fps.setValue(float(data.get("fps", 30.0)))
        self.statusBar().showMessage(f"Проект загружен: {path}")

    # ------------------------------------------------------------------
    # таймлайн и служебное
    # ------------------------------------------------------------------

    def _on_slider_moved(self, value: int) -> None:
        self._show_frame(int(value))

    def _on_slider_release(self) -> None:
        self._show_frame(int(self.slider.value()))

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        self.btn_play.setText("❚❚" if self._playing else "▶")
        if self._playing:
            if self.video is None:
                self._playing = False
                self.btn_play.setText("▶")
                return
            self._play_timer.start()
        else:
            self._play_timer.stop()

    def _play_step(self) -> None:
        if self.video is None:
            return
        nxt = int(self.slider.value()) + 1
        if nxt > self.slider.maximum():
            nxt = 0
        self.slider.setValue(nxt)

    def _update_actions(self) -> None:
        has_video = self.video is not None
        has_sol = bool(self.solutions)
        self.act_solve.setEnabled(has_video and self.plane is not None)
        self.act_fbx.setEnabled(has_sol)
        self.act_pose.setEnabled(has_video)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            if self.view.mode == MODE_PLANE:
                self.view.set_plane_points([])
                self.view.set_mode(MODE_VIEW)
                self.statusBar().clearMessage()
            else:
                self.view.set_mode(MODE_VIEW)
            return
        super().keyPressEvent(event)


def _install_crash_handler(app) -> None:
    """Показывает ошибку пользователю и пишет лог (в .exe нет консоли)."""
    import traceback

    def hook(exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
        log_path = os.path.join(log_dir, "mocap_studio_error.log")
        try:
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError:
            log_path = "(не удалось записать лог)"
        try:
            QMessageBox.critical(
                None,
                "Ошибка",
                f"Произошла непредвиденная ошибка:\\n\\n{exc}\\n\\n"
                f"Подробности записаны в файл:\\n{log_path}",
            )
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = hook


def main() -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("MoCap Studio")
    app.setApplicationVersion("1.0.0")
    _install_crash_handler(app)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
