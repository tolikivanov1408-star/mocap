"""Канонический риг: скелет с пальцами, иерархия костей, FK и пресеты имён.

Координаты: X — вправо, Y — вверх, Z — назад (персонаж смотрит в -Z,
левая сторона персонажа = +X, как в FBX/Maya).

Риг задаётся как набор костей (name, parent, head_joint, tail_joint) плюс
«каноническая» T-поза. Пользователь может переопределить длины костей по
позе первого кадра — тогда rest-поза пересобирается с сохранением топологии.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .math3d import (
    q_from_matrix,
    aim_quat,
    q_identity,
    q_inv,
    q_mul,
    q_rotate,
    safe_unit,
    v3,
)

# ---------------------------------------------------------------------------
# Каноническая T-поза (метры, рост ~1.78 м)
# ---------------------------------------------------------------------------

REST_JOINTS: dict[str, tuple[float, float, float]] = {
    "pelvis": (0.000, 1.000, 0.000),
    "spine_01": (0.000, 1.080, 0.000),
    "spine_02": (0.000, 1.190, 0.000),
    "spine_03": (0.000, 1.310, 0.000),
    "chest": (0.000, 1.450, 0.000),
    "neck_01": (0.000, 1.545, 0.000),
    "head": (0.000, 1.625, 0.000),
    "head_top": (0.000, 1.780, 0.000),
    # левая рука (+X)
    "clavicle_l": (0.050, 1.470, 0.000),
    "shoulder_l": (0.190, 1.465, 0.000),
    "elbow_l": (0.490, 1.465, 0.000),
    "wrist_l": (0.760, 1.465, 0.000),
    "hand_mid_l": (0.860, 1.465, 0.000),
    "index_mcp_l": (0.815, 1.470, -0.021),
    "index_pip_l": (0.865, 1.470, -0.024),
    "index_dip_l": (0.905, 1.470, -0.024),
    "index_tip_l": (0.935, 1.470, -0.022),
    "middle_mcp_l": (0.820, 1.470, 0.000),
    "middle_pip_l": (0.875, 1.470, 0.001),
    "middle_dip_l": (0.918, 1.470, 0.001),
    "middle_tip_l": (0.950, 1.470, 0.000),
    "ring_mcp_l": (0.812, 1.470, 0.021),
    "ring_pip_l": (0.865, 1.470, 0.021),
    "ring_dip_l": (0.905, 1.470, 0.021),
    "ring_tip_l": (0.935, 1.470, 0.020),
    "pinky_mcp_l": (0.795, 1.470, 0.040),
    "pinky_pip_l": (0.840, 1.470, 0.040),
    "pinky_dip_l": (0.875, 1.470, 0.039),
    "pinky_tip_l": (0.903, 1.470, 0.038),
    "thumb_cmc_l": (0.778, 1.468, -0.008),
    "thumb_mcp_l": (0.812, 1.468, -0.030),
    "thumb_ip_l": (0.836, 1.467, -0.052),
    "thumb_tip_l": (0.852, 1.466, -0.072),
    # правая рука (-X)
    "clavicle_r": (-0.050, 1.455, 0.000),
    "shoulder_r": (-0.190, 1.470, 0.000),
    "elbow_r": (-0.490, 1.465, 0.000),
    "wrist_r": (-0.760, 1.465, 0.000),
    "hand_mid_r": (-0.860, 1.465, 0.000),
    "index_mcp_r": (-0.815, 1.470, -0.021),
    "index_pip_r": (-0.865, 1.470, -0.024),
    "index_dip_r": (-0.905, 1.470, -0.024),
    "index_tip_r": (-0.935, 1.470, -0.022),
    "middle_mcp_r": (-0.820, 1.470, 0.000),
    "middle_pip_r": (-0.875, 1.470, 0.001),
    "middle_dip_r": (-0.918, 1.470, 0.001),
    "middle_tip_r": (-0.950, 1.470, 0.000),
    "ring_mcp_r": (-0.812, 1.470, 0.021),
    "ring_pip_r": (-0.865, 1.470, 0.021),
    "ring_dip_r": (-0.905, 1.470, 0.021),
    "ring_tip_r": (-0.935, 1.470, 0.020),
    "pinky_mcp_r": (-0.795, 1.470, 0.040),
    "pinky_pip_r": (-0.840, 1.470, 0.040),
    "pinky_dip_r": (-0.875, 1.470, 0.039),
    "pinky_tip_r": (-0.903, 1.470, 0.038),
    "thumb_cmc_r": (-0.778, 1.468, -0.008),
    "thumb_mcp_r": (-0.812, 1.468, -0.030),
    "thumb_ip_r": (-0.836, 1.467, -0.052),
    "thumb_tip_r": (-0.852, 1.466, -0.072),
    # ноги
    "hip_l": (0.100, 0.965, 0.000),
    "knee_l": (0.100, 0.510, 0.000),
    "ankle_l": (0.100, 0.090, 0.000),
    "heel_l": (0.100, 0.065, 0.065),
    "ball_l": (0.100, 0.040, -0.140),
    "toe_l": (0.100, 0.025, -0.225),
    "hip_r": (-0.100, 0.965, 0.000),
    "knee_r": (-0.100, 0.510, 0.000),
    "ankle_r": (-0.100, 0.090, 0.000),
    "heel_r": (-0.100, 0.065, 0.065),
    "ball_r": (-0.100, 0.040, -0.140),
    "toe_r": (-0.100, 0.025, -0.225),
}


@dataclass(frozen=True)
class BoneSpec:
    name: str
    parent: str | None
    head: str
    tail: str
    group: str  # torso | arm | hand | finger | leg | root


def _arm(side: str) -> list[BoneSpec]:
    s = side
    out = [
        BoneSpec(f"clavicle_{s}", "chest", f"clavicle_{s}", f"shoulder_{s}", "torso"),
        BoneSpec(f"upperarm_{s}", f"clavicle_{s}", f"shoulder_{s}", f"elbow_{s}", "arm"),
        BoneSpec(f"lowerarm_{s}", f"upperarm_{s}", f"elbow_{s}", f"wrist_{s}", "arm"),
        BoneSpec(f"hand_{s}", f"lowerarm_{s}", f"wrist_{s}", f"hand_mid_{s}", "hand"),
    ]
    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    for f in fingers:
        prev = f"hand_{s}"
        # у большого пальца цепочка начинается с CMC, у остальных — с MCP
        chain = (
            [("cmc", "mcp"), ("mcp", "ip"), ("ip", "tip")]
            if f == "thumb"
            else [("mcp", "pip"), ("pip", "dip"), ("dip", "tip")]
        )
        for seg, (a, b) in enumerate(chain, start=1):
            name = f"{f}_{seg:02d}_{s}"
            out.append(
                BoneSpec(name, prev, f"{f}_{a}_{s}", f"{f}_{b}_{s}", "finger")
            )
            prev = name
    return out


def _leg(side: str) -> list[BoneSpec]:
    s = side
    return [
        BoneSpec(f"thigh_{s}", "pelvis", f"hip_{s}", f"knee_{s}", "leg"),
        BoneSpec(f"calf_{s}", f"thigh_{s}", f"knee_{s}", f"ankle_{s}", "leg"),
        BoneSpec(f"foot_{s}", f"calf_{s}", f"ankle_{s}", f"ball_{s}", "leg"),
        BoneSpec(f"ball_{s}", f"foot_{s}", f"ball_{s}", f"toe_{s}", "leg"),
    ]


BONE_SPECS: list[BoneSpec] = (
    [
        BoneSpec("pelvis", None, "pelvis", "spine_01", "torso"),
        BoneSpec("spine_01", "pelvis", "spine_01", "spine_02", "torso"),
        BoneSpec("spine_02", "spine_01", "spine_02", "spine_03", "torso"),
        BoneSpec("spine_03", "spine_02", "spine_03", "chest", "torso"),
        BoneSpec("chest", "spine_03", "chest", "neck_01", "torso"),
        BoneSpec("neck_01", "chest", "neck_01", "head", "torso"),
        BoneSpec("head", "neck_01", "head", "head_top", "torso"),
    ]
    + _arm("l")
    + _arm("r")
    + _leg("l")
    + _leg("r")
)

# ---------------------------------------------------------------------------
# Пресеты имён костей для экспорта
# ---------------------------------------------------------------------------

NAME_PRESETS: dict[str, dict[str, str]] = {
    "ue5": {
        "pelvis": "pelvis",
        "spine_01": "spine_01",
        "spine_02": "spine_02",
        "spine_03": "spine_03",
        "chest": "spine_04",
        "neck_01": "neck_01",
        "head": "head",
        "clavicle_l": "clavicle_l",
        "upperarm_l": "upperarm_l",
        "lowerarm_l": "lowerarm_l",
        "hand_l": "hand_l",
        "thigh_l": "thigh_l",
        "calf_l": "calf_l",
        "foot_l": "foot_l",
        "ball_l": "ball_l",
        "clavicle_r": "clavicle_r",
        "upperarm_r": "upperarm_r",
        "lowerarm_r": "lowerarm_r",
        "hand_r": "hand_r",
        "thigh_r": "thigh_r",
        "calf_r": "calf_r",
        "foot_r": "foot_r",
        "ball_r": "ball_r",
    },
    "mixamo": {
        "pelvis": "Hips",
        "spine_01": "Spine",
        "spine_02": "Spine1",
        "spine_03": "Spine2",
        "chest": "Spine3",
        "neck_01": "Neck",
        "head": "Head",
        "clavicle_l": "LeftShoulder",
        "upperarm_l": "LeftArm",
        "lowerarm_l": "LeftForeArm",
        "hand_l": "LeftHand",
        "thigh_l": "LeftUpLeg",
        "calf_l": "LeftLeg",
        "foot_l": "LeftFoot",
        "ball_l": "LeftToeBase",
        "clavicle_r": "RightShoulder",
        "upperarm_r": "RightArm",
        "lowerarm_r": "RightForeArm",
        "hand_r": "RightHand",
        "thigh_r": "RightUpLeg",
        "calf_r": "RightLeg",
        "foot_r": "RightFoot",
        "ball_r": "RightToeBase",
    },
    "max_biped": {
        "pelvis": "Pelvis",
        "spine_01": "Spine",
        "spine_02": "Spine1",
        "spine_03": "Spine2",
        "chest": "Chest",
        "neck_01": "Neck",
        "head": "Head",
        "clavicle_l": "L Clavicle",
        "upperarm_l": "L UpperArm",
        "lowerarm_l": "L Forearm",
        "hand_l": "L Hand",
        "thigh_l": "L Thigh",
        "calf_l": "L Shin",
        "foot_l": "L Foot",
        "ball_l": "L Toe",
        "clavicle_r": "R Clavicle",
        "upperarm_r": "R UpperArm",
        "lowerarm_r": "R Forearm",
        "hand_r": "R Hand",
        "thigh_r": "R Thigh",
        "calf_r": "R Shin",
        "foot_r": "R Foot",
        "ball_r": "R Toe",
    },
}

# пальцы: индексы для быстрого доступа
_FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def _build_name_preset_tables() -> dict[str, dict[str, str]]:
    """Дополняет базовые пресеты полными таблицами (включая пальцы)."""
    tables: dict[str, dict[str, str]] = {}
    for key, base in NAME_PRESETS.items():
        table = dict(base)
        for side, sp in (("l", "Left"), ("r", "Right")):
            for fi, finger in enumerate(_FINGER_ORDER):
                for seg in (1, 2, 3):
                    canon = f"{finger}_{seg:02d}_{side}"
                    if key == "ue5":
                        table[canon] = f"{finger}_{seg:02d}_{side}"
                    elif key == "mixamo":
                        table[canon] = f"{sp}Hand{finger.capitalize()}{seg}"
                    else:
                        table[canon] = f"{'L' if side == 'l' else 'R'} Finger{fi * 3 + seg:02d}"
        tables[key] = table
    return tables


NAME_PRESETS = _build_name_preset_tables()

# ---------------------------------------------------------------------------
# Правила выбора «полюса» (оси twist) для каждой кости
# ---------------------------------------------------------------------------

POLE_RULES: dict[str, str] = {
    "pelvis": "body_forward",
    "spine": "body_forward",
    "chest": "body_forward",
    "neck": "body_forward",
    "head": "body_forward",
    "clavicle": "body_up",
    "upperarm": "body_forward",
    "lowerarm": "body_forward",
    "hand": "palm",
    "finger": "finger_adjacent",
    "thigh": "body_forward",
    "calf": "body_forward",
    "foot": "leg_up",
}


def pole_rule_for(bone: str) -> str:
    if bone.startswith("finger"):
        return "finger_adjacent"
    for prefix, rule in POLE_RULES.items():
        if bone.startswith(prefix):
            return rule
    return "body_forward"


FINGER_POLE_NEIGHBOUR = {
    "thumb": "index",
    "index": "middle",
    "middle": "index",
    "ring": "middle",
    "pinky": "ring",
}


@dataclass
class Rig:
    """Риг персонажа: кости, rest-положения, FK."""

    bones: list[BoneSpec]
    rest_positions: dict[str, np.ndarray]
    by_name: dict[str, BoneSpec] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    rest_global_quat: dict[str, np.ndarray] = field(default_factory=dict)
    rest_local_quat: dict[str, np.ndarray] = field(default_factory=dict)
    rest_offset: dict[str, np.ndarray] = field(default_factory=dict)
    rest_tail_local: dict[str, np.ndarray] = field(default_factory=dict)

    # -- построение ---------------------------------------------------

    @classmethod
    def build(cls, lengths: dict[str, float] | None = None) -> "Rig":
        """Собирает риг. `lengths` — переопределение длин костей (в метрах)."""
        positions = {k: v3(*v) for k, v in REST_JOINTS.items()}
        rig = cls(bones=list(BONE_SPECS), rest_positions=positions)
        rig._index()
        if lengths:
            rig.apply_lengths(lengths)
        else:
            rig.recompute_rest()
        return rig

    def _index(self) -> None:
        self.by_name = {b.name: b for b in self.bones}
        self.children = {b.name: [] for b in self.bones}
        for b in self.bones:
            if b.parent:
                self.children[b.parent].append(b.name)
        self.order = [b.name for b in self.bones]

    # -- длины костей ---------------------------------------------------

    def bone_length(self, name: str) -> float:
        b = self.by_name[name]
        return float(np.linalg.norm(self.rest_positions[b.tail] - self.rest_positions[b.head]))

    def bone_lengths(self) -> dict[str, float]:
        return {b.name: self.bone_length(b.name) for b in self.bones}

    def scale_uniform(self, factor: float) -> None:
        """Пропорционально масштабирует риг (рост персонажа)."""
        factor = float(factor)
        if abs(factor - 1.0) < 1e-6:
            return
        lengths = {b.name: self.bone_length(b.name) * factor for b in self.bones}
        self.apply_lengths(lengths)

    def apply_lengths(self, lengths: dict[str, float]) -> None:
        """Перестраивает rest-позу под заданные длины костей (топология та же)."""
        lengths = dict(lengths)
        for b in self.bones:
            length = lengths.get(b.name)
            if not length or length <= 1e-4:
                continue
            head = self.rest_positions[b.head]
            tail = self.rest_positions[b.tail]
            direction = safe_unit(tail - head, fallback=(0.0, 1.0, 0.0))
            self.rest_positions[b.tail] = head + direction * float(length)
        self.recompute_rest()

    def recompute_rest(self) -> None:
        """Пересчитывает rest-ориентации, локальные кватернионы и оффсеты."""
        self.rest_global_quat = {}
        self.rest_local_quat = {}
        self.rest_offset = {}
        self.rest_tail_local = {}
        for b in self.bones:
            head = self.rest_positions[b.head]
            tail = self.rest_positions[b.tail]
            direction = safe_unit(tail - head, fallback=(0.0, 1.0, 0.0))
            if b.parent is None:
                # Корневая кость задаёт «телесный» базис сцены:
                # X — влево персонажа, Y — вверх, Z — за спину.
                # Именно в этом базисе авторы rest-позы.
                left = safe_unit(
                    self.rest_positions["shoulder_l"] - self.rest_positions["shoulder_r"],
                    fallback=(1.0, 0.0, 0.0),
                )
                up = safe_unit(
                    self.rest_positions["chest"] - self.rest_positions["pelvis"],
                    fallback=(0.0, 1.0, 0.0),
                )
                back = safe_unit(np.cross(left, up), fallback=(0.0, 0.0, 1.0))
                gq = q_from_matrix(np.column_stack([left, up, back]))
            else:
                pole = self.resolve_pole(b.name, self.rest_positions)
                gq = aim_quat(direction, pole)
            self.rest_global_quat[b.name] = gq
            # вектор «хвост - голова» в локальном фрейме кости (лежит на +X)
            self.rest_tail_local[b.name] = q_rotate(q_inv(gq), tail - head)
            if b.parent:
                pg = self.rest_global_quat[b.parent]
                self.rest_local_quat[b.name] = q_mul(q_inv(pg), gq)
                parent_head = self.rest_positions[self.by_name[b.parent].head]
                self.rest_offset[b.name] = q_rotate(q_inv(pg), head - parent_head)
            else:
                # корневая кость: её ориентация задаётся корнем сцены,
                # поэтому локальный поворот в rest-позе — единичный
                self.rest_local_quat[b.name] = q_identity()
                self.rest_offset[b.name] = np.zeros(3)

    # -- полюса ---------------------------------------------------------

    def resolve_pole(
        self, bone: str, joints: dict[str, np.ndarray]
    ) -> np.ndarray | None:
        """Направление, задающее ось twist кости, по текущим положениям суставов."""
        spec = self.by_name[bone]
        rule = pole_rule_for(bone)
        side = bone.rsplit("_", 1)[-1] if bone[-1] in "lr" else None

        if rule == "body_forward":
            chest = joints.get("chest")
            pelvis = joints.get("pelvis")
            if chest is None or pelvis is None:
                return None
            up = safe_unit(chest - pelvis, fallback=(0.0, 1.0, 0.0))
            sl = joints.get("shoulder_l")
            sr = joints.get("shoulder_r")
            if sl is None or sr is None:
                return None
            right = safe_unit(sr - sl, fallback=(1.0, 0.0, 0.0))
            return np.cross(right, up)

        if rule == "body_up":
            chest = joints.get("chest")
            pelvis = joints.get("pelvis")
            if chest is None or pelvis is None:
                return None
            return chest - pelvis

        if rule == "palm":
            if side not in ("l", "r"):
                return None
            a = joints.get(f"index_mcp_{side}")
            b = joints.get(f"pinky_mcp_{side}")
            if a is None or b is None:
                return None
            return a - b

        if rule == "finger_adjacent":
            if side not in ("l", "r"):
                return None
            finger = bone.split("_")[0]
            neighbour = FINGER_POLE_NEIGHBOUR.get(finger, "index")
            a = joints.get(f"{finger}_cmc_{side}")
            if a is None:
                a = joints.get(spec.head)
            b = joints.get(f"{neighbour}_mcp_{side}")
            if a is None or b is None:
                return None
            return b - a

        if rule == "leg_up":
            if side not in ("l", "r"):
                return None
            knee = joints.get(f"knee_{side}")
            ankle = joints.get(f"ankle_{side}")
            if knee is None or ankle is None:
                return None
            return knee - ankle

        return None

    # -- FK -------------------------------------------------------------

    def fk(
        self,
        local_quats: dict[str, np.ndarray],
        root_quat: np.ndarray | None = None,
        root_pos: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Прямая кинематика.

        Возвращает (мировые позиции суставов, мировые ориентации костей).
        """
        root_quat = q_identity() if root_quat is None else root_quat
        root_pos = np.zeros(3) if root_pos is None else root_pos
        world_quat: dict[str, np.ndarray] = {}
        world_pos: dict[str, np.ndarray] = {}

        for b in self.bones:
            lq = local_quats.get(b.name, self.rest_local_quat[b.name])
            if b.parent:
                pg = world_quat[b.parent]
                gq = q_mul(pg, lq)
                parent_head = world_pos[self.by_name[b.parent].head]
                offset = self.rest_offset[b.name]
                head_pos = parent_head + q_rotate(pg, offset)
            else:
                gq = q_mul(root_quat, lq)
                head_pos = root_pos + q_rotate(root_quat, self.rest_offset[b.name])
            world_quat[b.name] = gq
            world_pos[b.head] = head_pos
            world_pos[b.tail] = head_pos + q_rotate(gq, self.rest_tail_local[b.name])

        return world_pos, world_quat

    # -- утилиты --------------------------------------------------------

    def bone_chain(self, name: str) -> list[str]:
        """Цепочка костей от корня до указанной."""
        chain = []
        cur = name
        while cur in self.by_name:
            chain.append(cur)
            cur = self.by_name[cur].parent
            if cur is None:
                break
        return list(reversed(chain))

    def children_of(self, name: str) -> list[str]:
        return list(self.children.get(name, []))

    def names(self) -> list[str]:
        return [b.name for b in self.bones]

    def export_names(self, preset: str = "ue5") -> dict[str, str]:
        table = NAME_PRESETS.get(preset, NAME_PRESETS["ue5"])
        return {b.name: table.get(b.name, b.name) for b in self.bones}

    def side_of(self, name: str) -> str | None:
        if name.endswith("_l"):
            return "l"
        if name.endswith("_r"):
            return "r"
        return None
