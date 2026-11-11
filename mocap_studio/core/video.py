"""Чтение видео: покадровый доступ, метаданные, декодирование через OpenCV."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float

    @property
    def aspect(self) -> float:
        return self.width / max(1, self.height)


class VideoSource:
    """Обёртка над cv2.VideoCapture с кэшем позиционирования."""

    def __init__(self, path: str):
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {path}")
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(self._fps) or self._fps <= 1e-3:
            self._fps = 25.0
        count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._count = count
        self._cursor = 0  # индекс кадра, на котором стоит каптира
        if width and height:
            self._w, self._h = width, height
        else:
            self._w = self._h = 0

    # -- метаданные -----------------------------------------------------

    @property
    def info(self) -> VideoInfo:
        return VideoInfo(
            path=self.path,
            width=self._w,
            height=self._h,
            fps=self._fps,
            frame_count=self._count,
            duration=(self._count / self._fps) if self._count else 0.0,
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._count

    # -- чтение ---------------------------------------------------------

    def read_at(self, index: int) -> np.ndarray | None:
        """Кадр по индексу (BGR). None — конец файла."""
        index = max(0, int(index))
        if self._count and index >= self._count:
            return None
        # Перемотка назад всегда, вперёд — последовательно (быстрее и надёжнее).
        if index < self._cursor:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._cursor = index
        while self._cursor < index:
            ok, _ = self._cap.read()
            if not ok:
                return None
            self._cursor += 1
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._cursor = index + 1
        if not self._w or not self._h:
            self._h, self._w = frame.shape[:2]
        return frame

    def iter_frames(self, start: int = 0, end: int | None = None, step: int = 1):
        """Итератор кадров [start, end) с шагом step."""
        last = self._count if end is None else min(end, self._count or end)
        idx = start
        while idx < last:
            frame = self.read_at(idx)
            if frame is None:
                return
            yield idx, frame
            idx += step

    def release(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
