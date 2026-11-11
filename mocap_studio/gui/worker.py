"""Фоновые задачи: детекция видео в отдельном потоке."""
from __future__ import annotations

import time

from PySide6.QtCore import QThread, Signal

from ..core.detect import Detector
from ..core.video import VideoSource


class DetectWorker(QThread):
    """Считает позу по всем кадрам видео, не блокируя интерфейс."""

    progress = Signal(int, int, float)  # сделано, всего, секунд на кадр
    finished_ok = Signal(object)  # список PoseFrame
    failed = Signal(str)

    def __init__(self, video: VideoSource, start: int = 0, end: int | None = None, parent=None):
        super().__init__(parent)
        self.video = video
        self.start_idx = start
        self.end_idx = end if end is not None else video.frame_count
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D102
        try:
            detector = Detector()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Не удалось запустить детектор: {exc}")
            return
        frames = []
        t0 = time.time()
        total = max(1, self.end_idx - self.start_idx)
        try:
            for i, frame in self.video.iter_frames(self.start_idx, self.end_idx):
                if self._cancel:
                    break
                ts = int(round(i * 1000.0 / max(1e-3, self.video.fps)))
                frames.append(detector.detect(frame, ts))
                done = len(frames)
                per = (time.time() - t0) / max(1, done)
                if done % 5 == 0 or done == total:
                    self.progress.emit(done, total, per)
            detector.close()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Ошибка детекции: {exc}")
            return
        self.finished_ok.emit(frames)
