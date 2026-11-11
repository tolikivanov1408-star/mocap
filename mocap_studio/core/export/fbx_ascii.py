"""Экспорт анимации в FBX (ASCII 7.4) без внешних SDK.

Пишется иерархия костей (LimbNode), анимационные стеки и кривые
(вращение — эйлеры XYZ в градусах, перевод — в сантиметрах, как требует FBX).
Файл читается 3ds Max, Maya, Blender и Unreal Engine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..math3d import euler_xyz_degrees, euler_zxy_degrees
from ..rig import Rig

FBX_TIME_UNIT = 46186158000  # тиков в секунду (FBX SDK)
CM = 100.0  # FBX: 1 единица = 1 см


def _fmt(v: float) -> str:
    if abs(v) < 1e-9:
        v = 0.0
    text = f"{v:.6f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-") else "0"


def _vec(values) -> str:
    return ",".join(_fmt(float(v)) for v in values)


@dataclass
class ExportOptions:
    fps: float = 30.0
    unit_scale: float = CM
    name_preset: str = "ue5"
    root_name: str = "root"
    take_name: str = "Take 001"


class _Writer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._id = 1000000

    def next_id(self) -> int:
        self._id += 1
        return self._id

    def w(self, line: str = "") -> None:
        self.lines.append(line)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def export_fbx_ascii(
    path: str,
    rig: Rig,
    solutions,
    options: ExportOptions | None = None,
) -> str:
    """Экспортирует решения решателя в FBX ASCII."""
    opts = options or ExportOptions()
    names = rig.export_names(opts.name_preset)
    w = _Writer()

    root_id = w.next_id()
    bone_ids: dict[str, int] = {}
    for bone in rig.bones:
        bone_ids[bone.name] = w.next_id()

    stack_id = w.next_id()
    layer_id = w.next_id()

    fps = max(1e-3, float(opts.fps))
    ticks_per_frame = FBX_TIME_UNIT / fps

    # кривые: для каждой кости — перенос (T) и поворот (R), каждая по три
    # компоненты (X, Y, Z). В FBX на каждую компоненту своя кривая.
    curve_node_ids: dict[tuple[str, str, str], int] = {}
    curve_ids: dict[tuple[str, str, str], int] = {}
    curve_keys = [("__root__", t, c) for t in ("T", "R") for c in "XYZ"] + [
        (b.name, t, c) for b in rig.bones for t in ("T", "R") for c in "XYZ"
    ]
    for key in curve_keys:
        curve_node_ids[key] = w.next_id()
        curve_ids[key] = w.next_id()

    # ---------------- значения кривых --------------------------------
    n_frames = len(solutions)
    key_times = [int(round(i * ticks_per_frame)) for i in range(n_frames)]
    comp_index = {"X": 0, "Y": 1, "Z": 2}

    def values_for(key: tuple[str, str, str]) -> list[float]:
        bone_name, axis, comp = key
        idx = comp_index[comp]
        out: list[float] = []
        for sol in solutions:
            if bone_name == "__root__":
                if axis == "T":
                    out.append(float(sol.root_pos[idx]) * opts.unit_scale)
                else:
                    e = euler_xyz_degrees(sol.root_quat)
                    out.append(float(e[idx]))
            else:
                if axis == "T":
                    off = rig.rest_offset.get(bone_name, np.zeros(3))
                    out.append(float(off[idx]) * opts.unit_scale)
                else:
                    q = sol.local_quats.get(bone_name)
                    e = euler_xyz_degrees(q if q is not None else np.array([0, 0, 0, 1.0]))
                    out.append(float(e[idx]))
        return out

    # ---------------- заголовок ---------------------------------------
    w.w("; FBX 7.4.0 project file")
    w.w("; ----------------------------------------------------")
    w.w("")
    w.w("FBXHeaderExtension:  {")
    w.w("\tFBXHeaderVersion: 1003")
    w.w("\tFBXVersion: 7400")
    w.w('\tCreator: "mocap-studio"')
    w.w('\tSceneInfo: "SceneInfo::GlobalInfo", "UserData" {')
    w.w("\t\tType: \"UserData\"")
    w.w("\t\tVersion: 100")
    w.w("\t\tMetaData:  {")
    w.w("\t\t\tVersion: 100")
    w.w('\t\t\tTitle: ""')
    w.w('\t\t\tSubject: ""')
    w.w('\t\t\tAuthor: ""')
    w.w('\t\t\tKeywords: ""')
    w.w('\t\t\tRevision: ""')
    w.w('\t\t\tComment: ""')
    w.w("\t\t}")
    w.w("\t\tProperties70:  {")
    w.w('\t\t\tP: "DocumentUrl", "KString", "Url", "", ""')
    w.w('\t\t\tP: "SrcDocumentUrl", "KString", "Url", "", ""')
    w.w('\t\t\tP: "Original", "KString", "", "", ""')
    w.w('\t\t\tP: "Original|ApplicationName", "KString", "", "", "mocap-studio"')
    w.w('\t\t\tP: "LastSaved", "KTime", "Time", "", 0')
    w.w("\t\t}")
    w.w("\t}")
    w.w("}")
    w.w("GlobalSettings:  {")
    w.w("\tVersion: 1000")
    w.w("\tProperties70:  {")
    w.w('\t\tP: "UpAxis", "int", "Integer", "", 1')
    w.w('\t\tP: "UpAxisSign", "int", "Integer", "", 1')
    w.w('\t\tP: "FrontAxis", "int", "Integer", "", 2')
    w.w('\t\tP: "FrontAxisSign", "int", "Integer", "", 1')
    w.w('\t\tP: "CoordAxis", "int", "Integer", "", 0')
    w.w('\t\tP: "CoordAxisSign", "int", "Integer", "", 1')
    w.w(f'\t\tP: "UnitScaleFactor", "double", "Number", "", {_fmt(opts.unit_scale)})')
    w.w(f'\t\tP: "OriginalUnitScaleFactor", "double", "Number", "", {_fmt(opts.unit_scale)})')
    w.w('\t\tP: "AmbientColor", "ColorRGB", "Color", "", 0,0,0')
    w.w('\t\tP: "DefaultCamera", "KString", "", "", "Producer Perspective"')
    w.w('\t\tP: "TimeMode", "enum", "", "", 6')
    w.w('\t\tP: "TimeSpanStart", "KTime", "Time", "", 0')
    w.w(f'\t\tP: "TimeSpanStop", "KTime", "Time", "", {key_times[-1] if key_times else 0}')
    w.w('\t\tP: "CustomFrameRate", "double", "Number", "", -1')
    w.w("\t}")
    w.w("}")
    w.w("Documents:  {")
    w.w("\tCount: 1")
    w.w('\tDocument: 1234567890, "", "Scene" {')
    w.w("\t\tProperties70:  {")
    w.w('\t\t\tP: "SourceObject", "object", "", ""')
    w.w(f'\t\t\tP: "ActiveAnimStackName", "KString", "", "", "{opts.take_name}"')
    w.w("\t\t}")
    w.w("\t\tRootNode: 0")
    w.w('\t\tTime: "00:00:00.0000000"')
    w.w("\t}")
    w.w("}")
    w.w("Definitions:  {")
    w.w("\tVersion: 100")
    w.w("\tCount: 5")
    w.w('\tObjectType: "Model" {')
    w.w(f"\t\tCount: {len(rig.bones) + 1}")
    w.w("\t}")
    w.w('\tObjectType: "AnimationStack" {')
    w.w("\t\tCount: 1")
    w.w("\t}")
    w.w('\tObjectType: "AnimationLayer" {')
    w.w("\t\tCount: 1")
    w.w("\t}")
    w.w('\tObjectType: "AnimationCurveNode" {')
    w.w(f"\t\tCount: {len(curve_node_ids)}")
    w.w("\t}")
    w.w('\tObjectType: "AnimationCurve" {')
    w.w(f"\t\tCount: {len(curve_ids)}")
    w.w("\t}")
    w.w("}")
    w.w("Objects:  {")

    # корневой узел
    w.w(f'\tModel: {root_id}, "Model::{opts.root_name}", "Null" {{')
    w.w("\t\tVersion: 232")
    w.w("\t\tProperties70:  {")
    w.w('\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A", 0,0,0')
    w.w('\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A", 0,0,0')
    w.w('\t\t\tP: "Lcl Scaling", "Lcl Scaling", "", "A", 1,1,1')
    w.w('\t\t\tP: "RotationOrder", "enum", "", "", 0')
    w.w("\t\t}")
    w.w("\t\tShading: T")
    w.w('\t\tCulling: "CullingOff"')
    w.w("\t}")

    for bone in rig.bones:
        bid = bone_ids[bone.name]
        disp = names.get(bone.name, bone.name)
        w.w(f'\tModel: {bid}, "Model::{disp}", "LimbNode" {{')
        w.w("\t\tVersion: 232")
        w.w("\t\tProperties70:  {")
        off = rig.rest_offset.get(bone.name, np.zeros(3)) * opts.unit_scale
        rest_rot = euler_xyz_degrees(rig.rest_local_quat.get(bone.name, np.array([0.0, 0.0, 0.0, 1.0])))
        w.w(f'\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A", {_vec(off)}')
        w.w(f'\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A", {_vec(rest_rot)}')
        w.w('\t\t\tP: "Lcl Scaling", "Lcl Scaling", "", "A", 1,1,1')
        w.w('\t\t\tP: "RotationOrder", "enum", "", "", 0')
        w.w("\t\t}")
        w.w("\t\tShading: T")
        w.w('\t\tCulling: "CullingOff"')
        w.w("\t}")

    w.w(f'\tAnimationStack: {stack_id}, "AnimStack::{opts.take_name}" {{')
    w.w("\t\tProperties70:  {")
    w.w(f'\t\t\tP: "LocalTimeSpan", "KTimeRange", "TimeRange", "", 0,{key_times[-1] if key_times else 0}')
    w.w("\t\t}")
    w.w("\t}")
    w.w(f'\tAnimationLayer: {layer_id}, "AnimLayer::BaseLayer" {{')
    w.w("\t}")

    for (bone_name, axis, comp), node_id in curve_node_ids.items():
        w.w(f'\tAnimationCurveNode: {node_id}, "AnimCurveNode::{axis}|{comp}" {{')
        w.w("\t\tProperties70:  {")
        w.w(f'\t\t\tP: "d|{comp}", "Number", "", "", 0')
        w.w("\t\t}")
        w.w("\t}")

    for (bone_name, axis, comp), curve_id in curve_ids.items():
        vals = values_for((bone_name, axis, comp))
        if not vals:
            vals = [0.0]
        w.w(f'\tAnimationCurve: {curve_id}, "AnimCurve::" {{')
        w.w("\t\tProperties70:  {")
        w.w(f'\t\t\tP: "KeyTime", "Time", "", "", {",".join(str(t) for t in key_times[: len(vals)])}')
        w.w(f'\t\t\tP: "KeyValueFloat", "Float", "", "", {_vec(vals)}')
        w.w('\t\t\tP: "KeyAttrFlags", "enum", "", "", 24836')
        w.w('\t\t\tP: "KeyAttrData", "Compound", "", ""')
        w.w('\t\t\tP: "KeyAttrRefCount", "int", "Integer", "", 4')
        w.w("\t\t}")
        w.w("\t\tDefault: 0")
        w.w("\t\tKeyVer: 4009")
        w.w("\t\tRefCount: 2")
        w.w("\t}")

    w.w("}")
    w.w("Connections:  {")
    w.w(f'\tC: "OO", {root_id}, 0')
    for bone in rig.bones:
        parent_id = bone_ids[bone.parent] if bone.parent else root_id
        w.w(f'\tC: "OO", {bone_ids[bone.name]}, {parent_id}')
    w.w(f'\tC: "OO", {layer_id}, {stack_id}')
    prop_names = {"T": "Lcl Translation", "R": "Lcl Rotation", "S": "Lcl Scaling"}
    for key, node_id in curve_node_ids.items():
        bone_name, axis, comp = key
        model_id = root_id if bone_name == "__root__" else bone_ids[bone_name]
        w.w(f'\tC: "OO", {curve_ids[key]}, {node_id}')
        w.w(f'\tC: "OO", {node_id}, {layer_id}')
        # OP-связь: без неё импортёр не поймёт, какое свойство анимирует кривая
        w.w(f'\tC: "OP", {node_id}, {model_id}, "{prop_names[axis]}"')
    w.w("Takes:  {")
    w.w('\tCurrent: ""')
    w.w("}")

    text = w.text()
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


def export_bvh(
    path: str,
    rig: Rig,
    solutions,
    fps: float = 30.0,
    name_preset: str = "ue5",
) -> str:
    """Экспортирует решения решателя в BVH (универсальный формат).

    Порядок каналов поворота — ZXY (так читает почти любой импортёр).
    Позиции и смещения — в сантиметрах, как принято в BVH.
    """
    names = rig.export_names(name_preset)

    def channels() -> list[str]:
        return [
            "Xposition",
            "Yposition",
            "Zposition",
            "Zrotation",
            "Xrotation",
            "Yrotation",
        ]

    # порядок обхода — как в HIERARCHY
    ordered: list[str] = []

    def collect(bone_name: str) -> None:
        for child in rig.children_of(bone_name):
            ordered.append(child)
            collect(child)

    collect("pelvis")

    lines = [
        "HIERARCHY",
        f"ROOT {names['pelvis']}",
        "{",
        "\tOFFSET 0 0 0",
        "\tCHANNELS 6 " + " ".join(channels()),
    ]

    def write_bone(bone_name: str, depth: int) -> None:
        pad = "\t" * depth
        off = rig.rest_offset.get(bone_name, np.zeros(3)) * CM
        lines.append(f"{pad}JOINT {names[bone_name]}")
        lines.append(f"{pad}{{")
        lines.append(f"{pad}\tOFFSET {off[0]:.4f} {off[1]:.4f} {off[2]:.4f}")
        lines.append(f"{pad}\tCHANNELS 6 " + " ".join(channels()))
        for child in rig.children_of(bone_name):
            write_bone(child, depth + 1)
        lines.append(f"{pad}}}")

    for child in rig.children_of("pelvis"):
        write_bone(child, 2)
    lines.append("}")
    lines.append("MOTION")
    lines.append(f"Frames: {len(solutions)}")
    lines.append(f"Frame Time: {1.0 / max(1e-6, fps):.6f}")

    for sol in solutions:
        values: list[float] = []
        # корень: позиция таза в мире (м -> см) и поворот
        values.extend(float(x) * CM for x in sol.root_pos)
        values.extend(float(x) for x in euler_zxy_degrees(sol.root_quat))
        for name in ordered:
            off = rig.rest_offset.get(name, np.zeros(3)) * CM
            values.extend(float(x) for x in off)
            q = sol.local_quats.get(name)
            e = euler_zxy_degrees(q if q is not None else np.array([0.0, 0.0, 0.0, 1.0]))
            values.extend(float(x) for x in e)
        lines.append(" ".join(f"{v:.5f}" for v in values))

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
