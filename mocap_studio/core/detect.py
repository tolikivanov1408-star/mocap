"""Обёртка над MediaPipe: поза тела и кисти рук из видеокадра."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
HANDS_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Индексы ключевых точек MediaPipe Pose (33 точки)
MP = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


@dataclass
class HandFrame:
    """Кисть: 21 точка в пикселях и в метрах (система координат камеры)."""

    side: str  # "l" | "r"
    points2d: np.ndarray = field(default_factory=lambda: np.zeros((21, 3)))
    points3d: np.ndarray = field(default_factory=lambda: np.zeros((21, 3)))
    score: float = 0.0


@dataclass
class PoseFrame:
    """Результат детекции одного кадра."""

    pose2d: np.ndarray  # (33, 3): x_px, y_px, visibility
    pose3d: np.ndarray  # (33, 3): метры, начало — между бёдрами
    hands: list[HandFrame] = field(default_factory=list)
    detected: bool = False

    def vis(self, name: str) -> float:
        return float(self.pose2d[MP[name], 2])


def _model_dir() -> str:
    """Каталог с моделями MediaPipe (в собранном .exe — рядом с ним)."""
    import sys

    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        return os.path.join(base, "models")
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "models")


def model_dir() -> str:
    """Публичный доступ к каталогу моделей (для скрипта установки)."""
    return _model_dir()


def model_path(name: str) -> str:
    return os.path.join(_model_dir(), name)


def ensure_models(progress=None) -> tuple[str, str]:
    """Скачивает модели MediaPipe, если их нет (≈38 МБ)."""
    import shutil
    import urllib.request

    os.makedirs(_model_dir(), exist_ok=True)
    out = []
    for url, fname in ((POSE_URL, "pose_landmarker_heavy.task"), (HANDS_URL, "hand_landmarker.task")):
        path = model_path(fname)
        if not os.path.exists(path) or os.path.getsize(path) < 1024:
            if progress:
                progress(f"Скачивание модели {fname}…")
            tmp = path + ".part"
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
                        shutil.copyfileobj(resp, fh)
                    break
                except Exception as exc:  # сеть может оборваться
                    if attempt == 3:
                        raise
                    time.sleep(2.0 * (attempt + 1))
            os.replace(tmp, path)
        out.append(path)
    return out[0], out[1]


class Detector:
    """Детектор позы и кистей. Работает в VIDEO-режиме (быстрее пакетного)."""

    def __init__(
        self,
        pose_model: str | None = None,
        hand_model: str | None = None,
        delegate: str = "CPU",
        num_hands: int = 2,
    ):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._mp = mp
        pose_model, hand_model = ensure_models()
        delegate_enum = (
            BaseOptions.Delegate.GPU if delegate.upper() == "GPU" else BaseOptions.Delegate.CPU
        )
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=pose_model, delegate=delegate_enum),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
        )
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_model, delegate=delegate_enum),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=num_hands,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
        )
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseFrame:
        mp = self._mp
        rgb = frame_bgr[:, :, ::-1].copy()
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(timestamp_ms)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        pose2d = np.zeros((33, 3))
        pose3d = np.zeros((33, 3))
        detected = False
        res = self.pose.detect_for_video(image, ts)
        if res.pose_landmarks:
            lm = res.pose_landmarks[0]
            for i, p in enumerate(lm):
                vis = getattr(p, "visibility", None)
                pose2d[i] = (p.x, p.y, 0.0 if vis is None else float(vis))
            detected = True
        if res.pose_world_landmarks:
            wl = res.pose_world_landmarks[0]
            for i, p in enumerate(wl):
                pose3d[i] = (p.x, p.y, p.z)

        hres = self.hands.detect_for_video(image, ts)
        hand_frames: list[HandFrame] = []
        for k, hand in enumerate(hres.hand_landmarks):
            pts2 = np.array(
                [
                    [
                        p.x,
                        p.y,
                        0.0 if getattr(p, "visibility", None) is None else float(p.visibility),
                    ]
                    for p in hand
                ]
            )
            if hres.hand_world_landmarks:
                wl = hres.hand_world_landmarks[k]
                pts3 = np.array([[p.x, p.y, p.z] for p in wl])
            else:
                pts3 = np.zeros((21, 3))
            score = 1.0
            if hres.handedness:
                cat = hres.handedness[k][0]
                score = float(cat.score)
            side = self._assign_side(pts2, pose2d, hres.handedness[k][0].category_name if hres.handedness else "Right")
            hand_frames.append(HandFrame(side=side, points2d=pts2, points3d=pts3, score=score))

        return PoseFrame(pose2d=pose2d, pose3d=pose3d, hands=hand_frames, detected=detected)

    @staticmethod
    def _assign_side(pts2: np.ndarray, pose2d: np.ndarray, label: str) -> str:
        """Определяет сторону кисти.

        MediaPipe считает, что картинка зеркальная (селфи-режим), поэтому его
        handedness нужно переворачивать. Надёжнее — сопоставить кисть с
        запястьем позы по расстоянию в кадре.
        """
        if pose2d[MP["left_wrist"]].any():
            lw = pose2d[MP["left_wrist"], :2]
            rw = pose2d[MP["right_wrist"], :2]
            wrist = pts2[0, :2]
            if lw.any() and rw.any():
                dl = float(np.linalg.norm(lw - wrist))
                dr = float(np.linalg.norm(rw - wrist))
                if min(dl, dr) > 1e-6:
                    return "l" if dl <= dr else "r"
        return "l" if label.lower().startswith("l") else "r"

    def close(self) -> None:
        try:
            self.pose.close()
        except Exception:
            pass
        try:
            self.hands.close()
        except Exception:
            pass
