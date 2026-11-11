"""Привязка чужой скелетной модели к нашему ригу и скининг.

Идея: у модели свои кости и своя bind-поза. Мы сопоставляем кости модели с
костями рига, а для каждой кадрa считаем «поправку», которая переносит
bind-позу модели в текущую позу рига:

    M_j(кадр) = R_j(риг, кадр) · (M_j(bind) · R_j(bind)⁻¹)

После этого вершины считаются обычным линейным скинингом с весами модели.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .rig import Rig

_PREFIX_RE = re.compile(r"^(mixamorig[:_]|bip\d*[:_ ]?|bone[:_]|def-|joint[:_]|b_|bn_)", re.IGNORECASE)


def normalize_name(name: str) -> str:
    """Приводит имя кости к сравнимому виду: без префиксов и разделителей."""
    n = name.strip()
    n = _PREFIX_RE.sub("", n)
    n = re.sub(r"[^0-9a-zA-Z]+", "", n)
    return n.lower()


@dataclass
class BoneMapping:
    """Соответствие «кость модели → кость рига»."""

    model_to_rig: dict[int, str] = field(default_factory=dict)  # индекс кости модели → имя кости рига
    rig_to_model: dict[str, int] = field(default_factory=dict)  # имя кости рига → индекс кости модели
    preset: str = ""
    matched_by_name: int = 0
    matched_by_position: int = 0
    unmatched_model: list[str] = field(default_factory=list)
    unmatched_rig: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        total = len(self.rig_to_model)
        return (len(self.rig_to_model) / total) if total else 0.0

    def summary(self) -> str:
        return (
            f"пресет {self.preset or '—'}: по именам {self.matched_by_name}, "
            f"по позе {self.matched_by_position}, всего {len(self.rig_to_model)}"
        )


def auto_map_bones(rig: Rig, model, preset: str | None = None) -> BoneMapping:
    """Подбирает соответствие костей: сначала по именам, потом по позе."""
    result = BoneMapping()
    if not model.joint_names:
        return result

    rig_names = [b.name for b in rig.bones]
    model_norm = {j: normalize_name(nm) for j, nm in enumerate(model.joint_names)}

    presets = [preset] if preset else ["ue5", "mixamo", "max_biped"]
    best: tuple[int, str, dict[int, int]] = (0, "ue5", {})
    for pre in presets:
        names = rig.export_names(pre)
        norm_rig = {n: normalize_name(names[n]) for n in rig_names}
        mapping: dict[int, str] = {}
        for j, mn in model_norm.items():
            if not mn:
                continue
            for ri, rn in norm_rig.items():
                if rn and rn == mn:
                    mapping[j] = ri
                    break
        if len(mapping) > best[0]:
            best = (len(mapping), pre, mapping)
    result.preset = best[1]
    result.matched_by_name = len(best[2])
    for j, ri in best[2].items():
        result.model_to_rig[j] = ri
        result.rig_to_model[ri] = j

    # оставшиеся кости рига ищем по позе (ближайшая по bind-центру)
    if model.joint_world.size:
        mpos = np.array([m[:3, 3] for m in model.joint_world])
        mlo, mhi = mpos.min(axis=0), mpos.max(axis=0)
        mspan = np.maximum(mhi - mlo, 1e-6)
        mnorm = (mpos - mlo) / mspan
        rpos = np.array([rig.rest_positions[b.head] for b in rig.bones])
        rlo, rhi = rpos.min(axis=0), rpos.max(axis=0)
        rspan = np.maximum(rhi - rlo, 1e-6)
        rnorm = (rpos - rlo) / rspan
        used_models = set(result.model_to_rig)
        for ri, bone in enumerate(rig.bones):
            if bone.name in result.rig_to_model:
                continue
            d = np.linalg.norm(mnorm - rnorm[ri][None, :], axis=1)
            for j in np.argsort(d):
                j = int(j)
                if j in used_models:
                    continue
                if d[j] < 0.22:  # близко по позе — считаем тем же суставом
                    result.model_to_rig[j] = bone.name
                    result.rig_to_model[bone.name] = j
                    used_models.add(j)
                    result.matched_by_position += 1
                break
    result.unmatched_model = [
        model.joint_names[j] for j in range(len(model.joint_names)) if j not in result.model_to_rig
    ]
    result.unmatched_rig = [b.name for b in rig.bones if b.name not in result.rig_to_model]
    return result


class MeshSkin:
    """Готовая привязка модели к ригу: считает позу модели по кадрам."""

    def __init__(self, rig: Rig, model, mapping: BoneMapping):
        self.rig = rig
        self.model = model
        self.mapping = mapping
        n_joints = len(model.joint_names)
        # поправка bind-позы: M_bind · R_bind⁻¹
        self.correction = np.tile(np.eye(4), (n_joints, 1, 1))
        self.bound = np.zeros(n_joints, dtype=bool)
        if model.joint_world.size and rig.rest_global_quat:
            for j, bone_name in mapping.model_to_rig.items():
                if j >= n_joints:
                    continue
                bone = rig.by_name.get(bone_name)
                if bone is None:
                    continue
                rq = rig.rest_global_quat[bone.name]
                rmat = np.eye(4)
                rmat[:3, :3] = _quat_to_mat(rq)
                rmat[:3, 3] = rig.rest_positions[bone.head]
                self.correction[j] = model.joint_world[j] @ np.linalg.inv(rmat)
                self.bound[j] = True
        self._last_solution = None
        self._rig_index = {b.name: i for i, b in enumerate(rig.bones)}

    # -- анимация -------------------------------------------------------

    def rig_matrices(self, solution) -> np.ndarray:
        """Мировые матрицы костей рига для кадра: (K,4,4) в порядке rig.bones."""
        positions, quats = self.rig.fk(
            solution.local_quats,
            root_quat=solution.root_quat,
            root_pos=solution.root_pos,
        )
        out = np.tile(np.eye(4), (len(self.rig.bones), 1, 1))
        for i, bone in enumerate(self.rig.bones):
            out[i, :3, :3] = _quat_to_mat(quats[bone.name])
            pos = positions.get(bone.head)
            if pos is not None:
                out[i, :3, 3] = pos
        return out

    def joint_matrices(self, solution) -> np.ndarray:
        """Новые мировые матрицы костей модели: (J,4,4)."""
        n = len(self.model.joint_names)
        mats = np.tile(np.eye(4), (n, 1, 1))
        if self.model.joint_world.size:
            mats = self.model.joint_world.copy()
        rigm = self.rig_matrices(solution)
        for j, bone_name in self.mapping.model_to_rig.items():
            if j >= n:
                continue
            ri = self._rig_index.get(bone_name)
            if ri is None or ri >= len(rigm):
                continue
            mats[j] = rigm[ri] @ self.correction[j]
        return mats

    def skin(self, solution) -> np.ndarray:
        """Позиции вершин модели в текущей позе: (N,3)."""
        model = self.model
        if not len(model.vertices) or not model.has_skin:
            return model.vertices
        mats = self.joint_matrices(solution)
        # взвешенные матрицы: M_j · IBM_j (по одной 4x4 на кость)
        weighted = np.matmul(mats, model.ibm)  # (J,4,4)
        # (N,4) веса и (N,4) индексы костей
        idx = np.clip(model.joints, 0, max(0, len(mats) - 1))
        picked = weighted[idx]  # (N,4,4,4)
        w = model.weights[..., None, None]
        mat = np.sum(picked * w, axis=1)  # (N,4,4)
        verts = np.concatenate([model.vertices, np.ones((len(model.vertices), 1))], axis=1)
        out = np.einsum("nab,nb->na", mat, verts)
        return out[:, :3]


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
