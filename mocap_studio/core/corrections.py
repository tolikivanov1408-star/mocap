"""Слой ручных правок поверх автоматически посчитанного движения.

Правка задаётся как мировое вращение кости на диапазоне кадров. Хранится
не «готовая» анимация, а ограничение, поэтому после правки движение
пересчитывается целиком: родители двигают потомков (FK), а ноги с
прижатой к полу стопой решаются через IK — стопа не отрывается от пола.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .math3d import q_identity, q_mul, smoothstep


@dataclass
class Correction:
    """Мировое вращение кости на отрезке кадров."""

    bone: str
    start: int
    end: int  # включительно
    quat: np.ndarray = field(default_factory=q_identity)
    blend: int = 4  # длина зоны сглаживания за границами отрезка

    def weight_at(self, index: int) -> float:
        if index < self.start - self.blend or index > self.end + self.blend:
            return 0.0
        if self.start <= index <= self.end:
            return 1.0
        if index < self.start:
            t = (index - (self.start - self.blend)) / max(1, self.blend)
            return smoothstep(t)
        t = ((self.end + self.blend) - index) / max(1, self.blend)
        return smoothstep(t)


class CorrectionSet:
    """Набор правок. Применяется к мировым ориентациям костей."""

    def __init__(self, items: list[Correction] | None = None):
        self.items: list[Correction] = list(items or [])
        self._cache: dict[int, dict[str, np.ndarray]] = {}

    def add(self, correction: Correction) -> None:
        self.items.append(correction)
        self._cache.clear()

    def remove(self, index: int) -> None:
        self.items.pop(index)
        self._cache.clear()

    def clear(self) -> None:
        self.items.clear()
        self._cache.clear()

    def delta_for(self, bone: str, index: int) -> np.ndarray:
        """Суммарная дельта для кости на кадре (с учётом веса зон сглаживания)."""
        result = q_identity()
        for c in self.items:
            if c.bone != bone:
                continue
            w = c.weight_at(index)
            if w <= 0.0:
                continue
            q = c.quat
            if w < 1.0:
                # интерполяция кватернионов от identity к целевому
                from .math3d import quat_slerp

                q = quat_slerp(q_identity(), c.quat, w)
            result = q_mul(result, q)
        return result

    def affected_frames(self) -> tuple[int, int] | None:
        if not self.items:
            return None
        starts = [c.start - c.blend for c in self.items]
        ends = [c.end + c.blend for c in self.items]
        return min(starts), max(ends)

    # -- сериализация ----------------------------------------------------

    def to_dict(self) -> list[dict]:
        return [
            {
                "bone": c.bone,
                "start": int(c.start),
                "end": int(c.end),
                "quat": [float(x) for x in c.quat],
                "blend": int(c.blend),
            }
            for c in self.items
        ]

    @classmethod
    def from_dict(cls, data: list[dict]) -> "CorrectionSet":
        items = [
            Correction(
                bone=d["bone"],
                start=int(d["start"]),
                end=int(d["end"]),
                quat=np.array(d.get("quat", [0, 0, 0, 1]), dtype=np.float64),
                blend=int(d.get("blend", 4)),
            )
            for d in data
        ]
        return cls(items)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CorrectionSet":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
