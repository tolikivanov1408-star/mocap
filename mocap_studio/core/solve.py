"""Решатель: из кадров видео — анимация скелета.

Принцип: длины костей задаёт РИГ (пропорции выставляет пользователь), а
детектор даёт только НАПРАВЛЕНИЯ (куда смотрит сустав) и положение стоп на
плоскости пола. Благодаря этому метрика нейросети не влияет на размер
персонажа, а пол задаёт глубину сцены.

Логика кадра:
  1. Ориентация тела в мире (система пола) из 3D-направлений детектора.
  2. Контакт стопы с полом: луч камеры пересекает плоскость пола в точке
     стопы; если эта точка не движется — стопа стоит и «прибивается» к полу.
  3. Направления костей + полюса (twist) -> глобальные ориентации ->
     локальные кватернионы.
  4. Положение таза: якорь по стоящей ноге минус смещение щиколотки.
  5. Ноги с прижатой стопой решаются 2-костной IK; корень подтягивается,
     чтобы нога не вытягивалась.
  6. Поверх накладываются ручные правки (слой ограничений).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .corrections import CorrectionSet
from .detect import MP, PoseFrame
from .math3d import (
    aim_quat,
    q_from_matrix,
    q_identity,
    q_inv,
    q_mul,
    safe_unit,
    v3,
)
from .rig import Rig


# ---------------------------------------------------------------------------
# Опции
# ---------------------------------------------------------------------------


@dataclass
class SolveOptions:
    contact_speed: float = 0.06  # м/кадр — допустимый дрейф точки стопы
    contact_release: float = 0.12  # м/кадр — порог выхода из контакта
    leg_straight_on: float = 150.0  # угол в колене (град): нога выпрямлена — стопа стоит
    leg_straight_off: float = 80.0  # ниже этого нога считается сложенной
    leg_folded_out: float = 60.0  # совсем сложенная нога — стопа точно в воздухе
    ankle_height: float = 0.09  # высота щиколотки над полом, м
    pelvis_height: float = 1.0  # высота таза над полом, м (для оценки без контакта)
    lock_feet: bool = True
    smooth_window: int = 5  # кадры сглаживания траекторий (0 — без)
    apply_corrections: bool = True


@dataclass
class FrameSolution:
    index: int
    root_quat: np.ndarray
    root_pos: np.ndarray
    local_quats: dict[str, np.ndarray] = field(default_factory=dict)
    joints: dict[str, np.ndarray] = field(default_factory=dict)
    contacts: dict[str, bool] = field(default_factory=dict)
    confidence: float = 0.0
    ok: bool = True


# ---------------------------------------------------------------------------
# Построение суставов рига из детекции
# ---------------------------------------------------------------------------

_HAND_FINGER_MAP = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * float(t)


def build_joints(
    frame: PoseFrame, rig: Rig, width: int, height: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Строит суставы рига в 3D (система камеры) и 2D (пиксели)."""
    j3: dict[str, np.ndarray] = {}
    j2: dict[str, np.ndarray] = {}
    p3 = frame.pose3d
    p2 = frame.pose2d

    def take(name: str, mp_name: str) -> None:
        i = MP[mp_name]
        j3[name] = p3[i].astype(np.float64)
        j2[name] = np.array([p2[i, 0] * width, p2[i, 1] * height, p2[i, 2]])

    for rig_name, mp_name in (
        ("hip_l", "left_hip"),
        ("hip_r", "right_hip"),
        ("knee_l", "left_knee"),
        ("knee_r", "right_knee"),
        ("ankle_l", "left_ankle"),
        ("ankle_r", "right_ankle"),
        ("heel_l", "left_heel"),
        ("heel_r", "right_heel"),
        ("ball_l", "left_foot_index"),
        ("ball_r", "right_foot_index"),
        ("shoulder_l", "left_shoulder"),
        ("shoulder_r", "right_shoulder"),
        ("elbow_l", "left_elbow"),
        ("elbow_r", "right_elbow"),
        ("wrist_l", "left_wrist"),
        ("wrist_r", "right_wrist"),
    ):
        take(rig_name, mp_name)

    def mid(a: str, b: str, out: str) -> None:
        if a in j3 and b in j3:
            j3[out] = (j3[a] + j3[b]) * 0.5
            j2[out] = (j2[a] + j2[b]) * 0.5

    mid("hip_l", "hip_r", "pelvis")
    mid("shoulder_l", "shoulder_r", "chest")

    # голова: середина ушей, при отсутствии — нос
    i_l, i_r = MP["left_ear"], MP["right_ear"]
    if p2[i_l, 2] > 0.2 or p2[i_r, 2] > 0.2:
        j3["head"] = (p3[i_l] + p3[i_r]) * 0.5
        j2["head"] = (
            np.array([p2[i_l, 0] * width, p2[i_l, 1] * height, p2[i_l, 2]])
            + np.array([p2[i_r, 0] * width, p2[i_r, 1] * height, p2[i_r, 2]])
        ) * 0.5
    else:
        take("head", "nose")

    rp = rig.rest_positions

    def frac(joint: str, a: str, b: str) -> float:
        va = rp[joint] - rp[a]
        vb = rp[b] - rp[a]
        denom = float(np.dot(vb, vb))
        if denom < 1e-9:
            return 0.0
        return float(np.dot(va, vb)) / denom

    # позвоночник
    if "pelvis" in j3 and "chest" in j3:
        for sj in ("spine_01", "spine_02", "spine_03"):
            t = frac(sj, "pelvis", "chest")
            j3[sj] = _lerp(j3["pelvis"], j3["chest"], t)
            j2[sj] = _lerp(j2["pelvis"], j2["chest"], t)
    # шея и затылок
    if "chest" in j3 and "head" in j3:
        t = frac("neck_01", "chest", "head")
        j3["neck_01"] = _lerp(j3["chest"], j3["head"], t)
        j2["neck_01"] = _lerp(j2["chest"], j2["head"], t)
        t2 = frac("head_top", "neck_01", "head")
        j3["head_top"] = _lerp(j3["neck_01"], j3["head"], t2)
        j2["head_top"] = _lerp(j2["neck_01"], j2["head"], t2)
    # ключицы
    for side in ("l", "r"):
        if "neck_01" in j3 and f"shoulder_{side}" in j3:
            t = frac(f"clavicle_{side}", "neck_01", f"shoulder_{side}")
            j3[f"clavicle_{side}"] = _lerp(j3["neck_01"], j3[f"shoulder_{side}"], t)
            j2[f"clavicle_{side}"] = _lerp(j2["neck_01"], j2[f"shoulder_{side}"], t)
        # носок
        if f"ball_{side}" in j3 and f"ankle_{side}" in j3:
            t = frac(f"toe_{side}", f"ankle_{side}", f"ball_{side}")
            j3[f"toe_{side}"] = _lerp(j3[f"ankle_{side}"], j3[f"ball_{side}"], t)
            j2[f"toe_{side}"] = _lerp(j2[f"ankle_{side}"], j2[f"ball_{side}"], t)

    # кисти: запястье из позы, пальцы из детектора рук
    for hand in frame.hands:
        side = hand.side
        wrist3 = j3.get(f"wrist_{side}")
        wrist2 = j2.get(f"wrist_{side}")
        if wrist3 is None:
            continue
        # Нормализуем размер кисти по ригу: детектор рук и детектор позы
        # используют разный метрический масштаб, из-за чего направления
        # пальцев могут «разъезжаться».
        hand_rel = hand.points3d - hand.points3d[0]
        rig_hand = float(
            np.linalg.norm(rig.rest_positions[f"hand_mid_{side}"] - rig.rest_positions[f"wrist_{side}"])
        )
        detected_hand = float(np.linalg.norm(hand_rel[9]))
        if detected_hand > 1e-4:
            hand_rel = hand_rel * (rig_hand / detected_hand)
        hand3 = wrist3 + hand_rel
        hand2 = np.stack(
            [hand.points2d[:, 0] * width, hand.points2d[:, 1] * height, hand.points2d[:, 2]], axis=1
        )
        # Проверяем надёжность детекции кисти: если ладонь «не смотрит» вдоль
        # предплечья или фаланги разной длины — детектор ошибся, и пальцы
        # оставляем прямыми (направление из rest-позы), а не вытягиваем мусор.
        reliable = True
        elbow3 = j3.get(f"elbow_{side}")
        if elbow3 is not None:
            palm_dir = safe_unit(hand_rel[9], fallback=(0.0, 0.0, 0.0))
            forearm_dir = safe_unit(wrist3 - elbow3, fallback=(0.0, 0.0, 0.0))
            if float(np.dot(palm_dir, forearm_dir)) < 0.45:
                reliable = False
        prox = []
        for _f, (_a, _b, _c, _d) in _HAND_FINGER_MAP.items():
            if _f == "thumb":
                continue
            prox.append(float(np.linalg.norm(hand_rel[_b] - hand_rel[_a])))
        if prox and min(prox) > 1e-5 and max(prox) / min(prox) > 2.5:
            reliable = False
        if not reliable:
            continue
        j3[f"hand_mid_{side}"] = hand3[9]
        j2[f"hand_mid_{side}"] = hand2[9]
        for finger, (a, b, c, d) in _HAND_FINGER_MAP.items():
            for suffix, idx in (("cmc", a), ("mcp", b), ("ip", c), ("tip", d)):
                key = f"{finger}_{suffix}_{side}"
                j3[key] = hand3[idx]
                j2[key] = hand2[idx]
        j3[f"wrist_{side}"] = wrist3
        j2[f"wrist_{side}"] = wrist2

    # если кисть не найдена — оцениваем длину ладони по предплечью
    for side in ("l", "r"):
        if f"hand_mid_{side}" in j3:
            continue
        w = j3.get(f"wrist_{side}")
        e = j3.get(f"elbow_{side}")
        if w is None or e is None:
            continue
        direction = safe_unit(w - e, fallback=(0.0, 1.0, 0.0))
        j3[f"hand_mid_{side}"] = w + direction * 0.09
        w2 = j2.get(f"wrist_{side}")
        e2 = j2.get(f"elbow_{side}")
        if w2 is not None and e2 is not None:
            d2 = safe_unit(w2[:2] - e2[:2], fallback=(0.0, 0.0, 0.0))
            j2[f"hand_mid_{side}"] = w2 + np.array(
                [d2[0] * 0.09 * height, d2[1] * 0.09 * height, 0.0]
            )

    return j3, j2


def calibrate_lengths(
    rig: Rig,
    frame: PoseFrame,
    user_joints_px: dict[str, np.ndarray],
    width: int,
    height: int,
) -> dict[str, float]:
    """Пересчитывает длины костей рига по позе, выставленной пользователем.

    Масштаб «пиксель -> метр» берём из детектора (3D-длина / 2D-длина),
    далее длины костей = пользовательская 2D-длина * масштаб.
    """
    j3, j2 = build_joints(frame, rig, width, height)

    core_pairs = [
        ("pelvis", "chest"),
        ("shoulder_l", "elbow_l"),
        ("elbow_l", "wrist_l"),
        ("shoulder_r", "elbow_r"),
        ("elbow_r", "wrist_r"),
        ("hip_l", "knee_l"),
        ("knee_l", "ankle_l"),
        ("hip_r", "knee_r"),
        ("knee_r", "ankle_r"),
        ("shoulder_l", "shoulder_r"),
        ("hip_l", "hip_r"),
    ]
    ratios = []
    for a, b in core_pairs:
        if a not in j3 or b not in j3 or a not in j2 or b not in j2:
            continue
        len3 = float(np.linalg.norm(j3[a] - j3[b]))
        len2 = float(np.linalg.norm(j2[a][:2] - j2[b][:2]))
        if len3 > 1e-4 and len2 > 1e-3:
            ratios.append(len3 / len2)
    m_per_px = float(np.median(ratios)) if ratios else 1.0 / max(1, height)

    lengths: dict[str, float] = {}
    for bone in rig.bones:
        h = user_joints_px.get(bone.head)
        t = user_joints_px.get(bone.tail)
        if h is None or t is None:
            continue
        d2 = float(np.linalg.norm(np.asarray(h)[:2] - np.asarray(t)[:2]))
        if d2 > 1e-3:
            lengths[bone.name] = d2 * m_per_px
    # кисти: масштабируем пальцы по длине ладони
    for side in ("l", "r"):
        w = user_joints_px.get(f"wrist_{side}")
        m = user_joints_px.get(f"hand_mid_{side}")
        if w is None or m is None:
            continue
        d2 = float(np.linalg.norm(np.asarray(w)[:2] - np.asarray(m)[:2]))
        if d2 < 1e-3:
            continue
        base = float(
            np.linalg.norm(
                rig.rest_positions[f"hand_mid_{side}"] - rig.rest_positions[f"wrist_{side}"]
            )
        )
        factor = d2 * m_per_px / max(1e-6, base)
        for bone in rig.bones:
            if bone.name.startswith(("thumb_", "index_", "middle_", "ring_", "pinky_")) and bone.name.endswith(side):
                lengths[bone.name] = rig.bone_length(bone.name) * factor
    return lengths


# ---------------------------------------------------------------------------
# Основной решатель
# ---------------------------------------------------------------------------


class MotionSolver:
    def __init__(
        self,
        rig: Rig,
        plane,
        width: int,
        height: int,
        options: SolveOptions | None = None,
        corrections: CorrectionSet | None = None,
    ):
        self.rig = rig
        self.plane = plane
        self.width = width
        self.height = height
        self.options = options or SolveOptions()
        self.corrections = corrections or CorrectionSet()
        self.scale = 1.0
        self._frames: list[PoseFrame] = []
        self._joints3d: list[dict[str, np.ndarray]] = []
        self._joints2d: list[dict[str, np.ndarray]] = []

    # -- подготовка ------------------------------------------------------

    def prepare(self, frames: list[PoseFrame]) -> None:
        """Строит траектории суставов и сглаживает их."""
        self._frames = frames
        j3_list, j2_list = [], []
        for f in frames:
            j3, j2 = build_joints(f, self.rig, self.width, self.height)
            j3_list.append(j3)
            j2_list.append(j2)
        self._joints3d = j3_list
        self._joints2d = j2_list
        self._smooth()

    def _smooth(self) -> None:
        w = int(self.options.smooth_window)
        if w < 2:
            return
        from scipy.signal import savgol_filter

        keys = set()
        for j in self._joints3d:
            keys.update(j.keys())
        for key in keys:
            series = np.array(
                [j.get(key, np.full(3, np.nan)) for j in self._joints3d], dtype=np.float64
            )
            valid = ~np.isnan(series).any(axis=1)
            if valid.sum() < w:
                continue
            idx = np.where(valid)[0]
            window = min(w if w % 2 == 1 else w + 1, len(idx) if len(idx) % 2 == 1 else len(idx) - 1)
            if window < 3:
                continue
            try:
                sm = savgol_filter(series[idx], window_length=window, polyorder=2, axis=0)
            except ValueError:
                continue
            for i, row in zip(idx, sm):
                self._joints3d[i][key] = row

    # -- решение ---------------------------------------------------------

    def solve(self) -> list[FrameSolution]:
        solutions: list[FrameSolution] = []
        lock = {"l": None, "r": None}  # «прибитая» точка стопы (только при контакте)
        seen = {"l": None, "r": None}  # последняя точка стопы на полу (всегда)
        contact_state = {"l": False, "r": False}
        prev_root = None
        prev_vel = np.zeros(3)

        for idx, frame in enumerate(self._frames):
            sol, lock, contact_state, prev_root, prev_vel, seen = self._solve_frame(
                idx, frame, lock, contact_state, prev_root, prev_vel, seen
            )
            solutions.append(sol)
        return solutions

    # -- отдельный кадр ---------------------------------------------------

    def _solve_frame(self, idx, frame, lock, contact_state, prev_root, prev_vel, seen=None):
        j3 = self._joints3d[idx] if idx < len(self._joints3d) else {}
        j2 = self._joints2d[idx] if idx < len(self._joints2d) else {}
        plane = self.plane
        opts = self.options
        rig = self.rig

        if "pelvis" not in j3 or "chest" not in j3 or "head" not in j3:
            empty = FrameSolution(
                index=idx,
                root_quat=q_identity(),
                root_pos=prev_root if prev_root is not None else np.zeros(3),
                confidence=0.0,
                ok=False,
            )
            return empty, lock, contact_state, prev_root, prev_vel, seen

        # --- 1. ориентация тела ------------------------------------------
        # Система координат детектора совпадает с системой камеры (X — вправо,
        # Y — вниз, Z — вперёд), поэтому переход в мир идёт через плоскость.
        R_cw = plane.camera_to_world
        up_body = safe_unit(j3["head"] - j3["pelvis"], fallback=(0.0, -1.0, 0.0))
        if "shoulder_l" in j3 and "shoulder_r" in j3:
            left_body = safe_unit(j3["shoulder_l"] - j3["shoulder_r"], fallback=(-1.0, 0.0, 0.0))
        else:
            left_body = v3(-1.0, 0.0, 0.0)
        back_body = safe_unit(np.cross(left_body, up_body), fallback=(0.0, 0.0, 1.0))
        R_body_cam = np.column_stack([left_body, up_body, back_body])
        R_root = q_from_matrix(R_cw @ R_body_cam)

        # --- 2. контакт стопы с полом ------------------------------------
        # Стопа стоит на полу, если её точка на плоскости практически не
        # движется (классическая «прибитая» стопа) и нога не сложена
        # пополам. Угол в колене — масштабонезависимый признак, в отличие
        # от скорости точки и от сравнения с длиной ноги рига (единицы
        # детектора и метры рига не совпадают).
        contacts: dict[str, bool] = {}
        anchors: dict[str, np.ndarray] = {}
        if seen is None:
            seen = {"l": None, "r": None}
        for side in ("l", "r"):
            anchor = self._floor_anchor(side, j2)
            straight = self._leg_straight(side, j3, j2)
            if straight is None:
                contacts[side] = contact_state[side]
                continue
            prev = lock[side]
            # дрейф считаем по последней точке стопы, а не по «прибитой»:
            # иначе стабильность нельзя определить до первого контакта
            last = seen.get(side)
            drift = (
                None
                if (last is None or anchor is None)
                else float(np.linalg.norm(anchor - last))
            )
            stable = drift is not None and drift <= opts.contact_speed
            if anchor is not None:
                seen[side] = anchor
            if contact_state[side]:
                extended = straight >= opts.leg_straight_off and (stable or straight >= opts.leg_straight_on)
            else:
                extended = straight >= opts.leg_straight_on or (
                    straight >= opts.leg_folded_out and stable
                )
            if not extended or anchor is None:
                contact_state[side] = False
                contacts[side] = False
                continue
            contact_state[side] = True
            contacts[side] = True
            if prev is None:
                lock[side] = anchor
            elif drift is not None and drift > opts.contact_release:
                # стопа явно уехала — переустанавливаем якорь
                lock[side] = anchor
            anchors[side] = lock[side]

        # --- 3. направления костей (длины задаёт риг) -------------------
        directions: dict[str, np.ndarray] = {}
        for bone in rig.bones:
            h = j3.get(bone.head)
            t = j3.get(bone.tail)
            if h is None or t is None:
                continue
            directions[bone.name] = safe_unit(R_cw @ (t - h), fallback=(0.0, 1.0, 0.0))

        # --- 4. ориентации костей ----------------------------------------
        global_quats: dict[str, np.ndarray] = {}
        for bone in rig.bones:
            if bone.parent is None:
                # Корневая кость несёт «телесный» базис (X — влево, Y — вверх,
                # Z — за спину) — так же, как в rest-позе рига.
                global_quats[bone.name] = R_root
                continue
            direction = directions.get(bone.name)
            if direction is None:
                global_quats[bone.name] = global_quats.get(bone.parent, q_identity())
                continue
            pole = rig.resolve_pole(bone.name, j3)  # в системе камеры
            pole_world = (R_cw @ pole) if pole is not None else None
            gq = aim_quat(direction, pole_world)
            if opts.apply_corrections and self.corrections.items:
                delta = self.corrections.delta_for(bone.name, idx)
                if not np.allclose(delta, q_identity()):
                    gq = q_mul(delta, gq)
            global_quats[bone.name] = gq

        local_quats = self._local_from_global(rig, global_quats, R_root)

        # --- 5. положение таза -------------------------------------------
        rel_pos, _ = rig.fk(local_quats, root_quat=R_root, root_pos=np.zeros(3))
        root_pos = self._root_position(j3, j2, anchors, prev_root, prev_vel, rel_pos)

        # --- 6. IK ног с прижатой стопой ---------------------------------
        for side in ("l", "r"):
            if not contacts.get(side):
                continue
            target = anchors.get(side)
            if target is None:
                continue
            root_pos, rel_pos, global_quats = self._solve_leg_ik(
                side, target, root_pos, rel_pos, global_quats
            )
            local_quats = self._local_from_global(rig, global_quats, R_root)

        # --- 7. финальные позиции суставов (FK) -------------------------
        positions, _ = rig.fk(local_quats, root_quat=R_root, root_pos=root_pos)

        conf = (
            float(
                np.mean(
                    [
                        frame.pose2d[MP[k], 2]
                        for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
                    ]
                )
            )
            if frame.detected
            else 0.0
        )

        sol = FrameSolution(
            index=idx,
            root_quat=R_root,
            root_pos=root_pos,
            local_quats=local_quats,
            joints=positions,
            contacts=dict(contacts),
            confidence=conf,
            ok=True,
        )
        if prev_root is not None:
            prev_vel = root_pos - prev_root
        return sol, lock, contact_state, root_pos, prev_vel, seen

    @staticmethod
    def _local_from_global(rig: Rig, global_quats: dict[str, np.ndarray], R_root: np.ndarray) -> dict[str, np.ndarray]:
        """Глобальные ориентации -> локальные (относительно родителя)."""
        local: dict[str, np.ndarray] = {}
        for bone in rig.bones:
            gq = global_quats.get(bone.name, q_identity())
            if bone.parent:
                pg = global_quats.get(bone.parent, q_identity())
            else:
                pg = R_root
            local[bone.name] = q_mul(q_inv(pg), gq)
        return local

    # -- вспомогательные --------------------------------------------------

    def _leg_straight(self, side: str, j3, j2) -> float | None:
        """Насколько нога выпрямлена: угол в колене, градусы (180 — прямая).

        Считаем по 3D-координатам детектора (масштаб не важен), а если нужные
        точки ненадёжны — по 2D-картинке.
        """
        hip = j3.get(f"hip_{side}")
        knee = j3.get(f"knee_{side}")
        ankle = j3.get(f"ankle_{side}")
        if hip is not None and knee is not None and ankle is not None:
            a = hip - knee
            b = ankle - knee
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na > 1e-5 and nb > 1e-5:
                cos = float(np.dot(a, b)) / (na * nb)
                return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        # запасной путь: сумма длин звеньев против расстояния бедро-щиколотка
        h2 = j2.get(f"hip_{side}")
        k2 = j2.get(f"knee_{side}")
        a2 = j2.get(f"ankle_{side}")
        if h2 is None or k2 is None or a2 is None:
            return None
        upper = float(np.linalg.norm(k2[:2] - h2[:2]))
        lower = float(np.linalg.norm(a2[:2] - k2[:2]))
        span = float(np.linalg.norm(a2[:2] - h2[:2]))
        if upper < 1e-3 or lower < 1e-3:
            return None
        ratio = float(np.clip(span / (upper + lower), 0.0, 1.0))
        return 180.0 * ratio

    def _floor_anchor(self, side: str, j2: dict[str, np.ndarray]) -> np.ndarray | None:
        """Мировая точка стопы на полу (пересечение луча камеры с плоскостью)."""
        px = j2.get(f"ankle_{side}")
        if px is None or px[2] < 0.25:
            px = j2.get(f"ball_{side}")
        if px is None or px[2] < 0.25:
            return None
        point = self.plane.project_to_floor(float(px[0]), float(px[1]))
        if point is None:
            return None
        # Запас большой: отмеченный прямоугольник может быть заметно меньше
        # зоны, где стоит человек, — отсекаем только совсем дикие проекции.
        margin = max(0.5, 0.5 * max(self.plane.size_m))
        if not self.plane.is_inside(point, margin=margin):
            return None
        point = np.array(point, dtype=np.float64)
        point[1] += self.options.ankle_height  # щиколотка над полом
        return point

    def _root_position(self, j3, j2, anchors, prev_root, prev_vel, rel_pos) -> np.ndarray:
        """Положение таза в мире."""
        # 1) якорь по стоящей ноге: таз = точка стопы - смещение щиколотки
        candidates = []
        for side in ("l", "r"):
            anchor = anchors.get(side)
            ankle = rel_pos.get(f"ankle_{side}")
            if anchor is None or ankle is None:
                continue
            candidates.append(anchor - ankle)
        if candidates:
            return np.mean(candidates, axis=0)

        # 2) оценка по изображению: луч камеры на высоте таза
        px = j2.get("pelvis")
        if px is not None:
            point = self.plane.project_to_plane_at_height(
                float(px[0]), float(px[1]), self.options.pelvis_height
            )
            if point is not None:
                return np.array(point, dtype=np.float64)

        # 3) экстраполяция по скорости
        if prev_root is not None:
            return prev_root + prev_vel * 0.5
        return np.zeros(3)

    def _solve_leg_ik(self, side, target, root_pos, rel_pos, global_quats):
        """2-костная IK: бедро + голень до зафиксированной стопы."""
        rig = self.rig
        thigh_name = f"thigh_{side}"
        calf_name = f"calf_{side}"
        foot_name = f"foot_{side}"
        if thigh_name not in rig.by_name:
            return root_pos, rel_pos, global_quats

        L1 = rig.bone_length(thigh_name)
        L2 = rig.bone_length(calf_name)

        hip_off = rel_pos.get(f"hip_{side}")
        if hip_off is None:
            return root_pos, rel_pos, global_quats
        hip = root_pos + hip_off

        d_vec = target - hip
        d = float(np.linalg.norm(d_vec))
        max_len = L1 + L2 - 1e-4
        if d > max_len and d > 1e-6:
            # нога не дотягивается — подтягиваем корень к стопе
            shift = d_vec * ((d - max_len) / d)
            root_pos = root_pos + shift
            hip = hip + shift
            d = max_len
            d_vec = target - hip
            d = float(np.linalg.norm(d_vec))

        if d < 1e-6:
            return root_pos, rel_pos, global_quats

        u = d_vec / d
        forward = self._body_forward(rel_pos)
        w = forward - float(np.dot(forward, u)) * u
        if float(np.linalg.norm(w)) < 1e-6:
            helper = v3(1.0, 0.0, 0.0)
            if abs(float(np.dot(helper, u))) > 0.9:
                helper = v3(0.0, 0.0, 1.0)
            w = helper - float(np.dot(helper, u)) * u
        w = safe_unit(w)

        a = (L1 * L1 - L2 * L2 + d * d) / (2.0 * d)
        h = float(np.sqrt(max(0.0, L1 * L1 - a * a)))
        knee = hip + u * a + w * h

        thigh_dir = safe_unit(knee - hip)
        calf_dir = safe_unit(target - knee)

        pole_thigh = rig.resolve_pole(thigh_name, rel_pos)
        pole_calf = rig.resolve_pole(calf_name, rel_pos)
        global_quats[thigh_name] = aim_quat(thigh_dir, pole_thigh)
        global_quats[calf_name] = aim_quat(calf_dir, pole_calf)

        # стопа остаётся на месте: направление носа берём из детекции
        ankle = rel_pos.get(f"ankle_{side}")
        ball = rel_pos.get(f"ball_{side}")
        if ankle is not None and ball is not None:
            foot_dir = safe_unit(ball - ankle)
            leg_up = safe_unit(ankle - knee)
            global_quats[foot_name] = aim_quat(foot_dir, leg_up)

        return root_pos, rel_pos, global_quats

    @staticmethod
    def _body_forward(joints) -> np.ndarray:
        """Направление «вперёд» тела (в системе, где переданные координаты)."""
        sl = joints.get("shoulder_l")
        sr = joints.get("shoulder_r")
        pelvis = joints.get("pelvis")
        chest = joints.get("chest")
        if sl is None or sr is None:
            return v3(0.0, 0.0, -1.0)
        right = safe_unit(sr - sl, fallback=(1.0, 0.0, 0.0))
        if chest is None or pelvis is None:
            up = v3(0.0, 1.0, 0.0)
        else:
            up = safe_unit(chest - pelvis, fallback=(0.0, 1.0, 0.0))
        return safe_unit(np.cross(right, up), fallback=(0.0, 0.0, -1.0))
