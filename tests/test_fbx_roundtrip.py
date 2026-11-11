"""Проверка экспорта FBX: разбор файла + round-trip через FK.

Тест не требует 3ds Max/Unreal: он читает свой же FBX ASCII тем же способом,
как это делает импортёр, восстанавливает локальные повороты из эйлеров и
прогоняет FK — позиции суставов должны совпасть с исходным решением.
"""
from __future__ import annotations

import math
import re
import sys

import numpy as np

sys.path.insert(0, "/workspace/mocap-studio")

from mocap_studio.core.export.fbx_ascii import ExportOptions, export_fbx_ascii  # noqa: E402
from mocap_studio.core.math3d import q_from_matrix  # noqa: E402
from mocap_studio.core.rig import Rig  # noqa: E402
from mocap_studio.core.solve import FrameSolution  # noqa: E402


def parse_fbx(path: str) -> dict:
    """Минимальный разбор FBX ASCII: объекты, свойства, связи, кривые."""
    models: dict[int, dict] = {}
    curves: dict[int, dict] = {}
    nodes: dict[int, str] = {}
    connections: list[tuple[str, int, int]] = []
    stack: list[str] = []
    cur_id = None
    cur_kind = None
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            m = re.match(r'^(C): "(\w+)",\s*(-?\d+),\s*(-?\d+)$', line)
            if m:
                connections.append((m.group(2), int(m.group(3)), int(m.group(4))))
                continue
            m = re.match(r'^(C): "(\w+)",\s*(-?\d+),\s*(-?\d+),\s*"([^"]*)"$', line)
            if m:
                connections.append((m.group(2), int(m.group(3)), int(m.group(4))))
                continue
            if line == "}":
                stack.pop()
                cur_id = None
                cur_kind = None
                continue
            m = re.match(r'^(\w+):\s*(-?\d+),\s*"([^"]*)"(?:,\s*"([^"]*)")?\s*\{$', line)
            if m:
                kind, oid, name = m.group(1), int(m.group(2)), m.group(3)
                sub = m.group(4) or ""
                stack.append(kind)
                cur_id, cur_kind = oid, kind
                if kind == "Model":
                    models[oid] = {"name": name, "type": sub, "props": {}}
                elif kind == "AnimationCurve":
                    curves[oid] = {"name": name, "props": {}}
                elif kind == "AnimationCurveNode":
                    nodes[oid] = name
                continue
            m = re.match(r'^P: "([^"]*)",\s*"([^"]*)",\s*"([^"]*)",\s*"([^"]*)"(.*)$', line)
            if m and cur_id is not None:
                key = m.group(1)
                rest = m.group(5).strip()
                vals = []
                if rest:
                    for token in rest.split(","):
                        token = token.strip()
                        if not token:
                            continue
                        try:
                            vals.append(float(token))
                        except ValueError:
                            vals.append(token)
                if cur_kind == "Model":
                    models[cur_id]["props"][key] = vals
                elif cur_kind == "AnimationCurve":
                    curves[cur_id]["props"][key] = vals
                continue
            if re.match(r'^[\w"]+:.*\{$', line):
                stack.append(line)
    return {
        "models": models,
        "curves": curves,
        "nodes": nodes,
        "connections": connections,
    }


def euler_to_quat(deg) -> np.ndarray:
    """Эйлеры XYZ (в градусах) -> кватернион (как их читает импортёр)."""
    x, y, z = (math.radians(float(v)) for v in deg)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    # внешняя последовательность XYZ (как в Blender/FBX): R = Rz @ Ry @ Rx
    m = np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ]
    )
    return q_from_matrix(m)


def main() -> int:
    rig = Rig.build()
    # синтетическая анимация
    from mocap_studio.core.math3d import quat_from_axis_angle

    solutions = []
    for i in range(12):
        local = {}
        for bone in rig.bones:
            if bone.parent is None:
                local[bone.name] = quat_from_axis_angle((0.0, 1.0, 0.0), 0.05 * i)
            else:
                axis = (0.3, 0.5, 0.8)
                local[bone.name] = quat_from_axis_angle(axis, 0.02 * i * (1 + bone.name.count("_")))
        root_quat = quat_from_axis_angle((0.0, 1.0, 0.0), 0.05 * i)
        root_pos = np.array([0.1 * i, 1.0, -0.05 * i])
        positions, _ = rig.fk(local, root_quat=root_quat, root_pos=root_pos)
        solutions.append(
            FrameSolution(
                index=i,
                root_quat=root_quat,
                root_pos=root_pos,
                local_quats=local,
                joints=positions,
            )
        )

    path = "/workspace/out/roundtrip.fbx"
    export_fbx_ascii(path, rig, solutions, ExportOptions(fps=25.0))
    data = parse_fbx(path)

    models = data["models"]
    curves = data["curves"]
    nodes = data["nodes"]
    connections = data["connections"]

    parent_of: dict[int, int] = {}
    for kind, child, parent in connections:
        if kind == "OO" and child in models:
            parent_of[child] = parent

    # кривые, привязанные к моделям: OO(curve_node -> model) и OO(curve -> curve_node)
    node_to_model: dict[int, int] = {}
    curve_to_node: dict[int, int] = {}
    for kind, child, parent in connections:
        if kind not in ("OO", "OP"):
            continue
        if child in nodes and parent in models:
            node_to_model[child] = parent
        if child in curves and parent in nodes:
            curve_to_node[child] = parent

    names = rig.export_names("ue5")
    id_by_name = {m["name"].replace("Model::", ""): mid for mid, m in models.items()}

    # 1) все кости на месте
    missing = [b.name for b in rig.bones if names[b.name] not in id_by_name]
    print("костей в файле:", len(models), "| отсутствуют:", missing or "нет")

    # 2) иерархия совпадает с ригом
    root_null = id_by_name["root"]
    bad_parent = []
    for bone in rig.bones:
        mid = id_by_name[names[bone.name]]
        expect = id_by_name[names[bone.parent]] if bone.parent else root_null
        if parent_of.get(mid) != expect:
            bad_parent.append(bone.name)
    print("нарушений иерархии:", bad_parent or "нет")

    # 3) round-trip: восстанавливаем позу из кривых и сравниваем суставы
    def curves_for(model_id: int, axis: str) -> list[list[float]]:
        """Значения трёх компонент (X, Y, Z) для свойства axis у модели."""
        out: dict[str, list[float]] = {}
        for node_id, model in node_to_model.items():
            if model != model_id:
                continue
            name = nodes[node_id]
            if not name.startswith(f"AnimCurveNode::{axis}|"):
                continue
            comp = name.rsplit("|", 1)[-1]
            for cid, nid in curve_to_node.items():
                if nid == node_id:
                    out[comp] = curves[cid]["props"]["KeyValueFloat"]
        return [out.get(c, []) for c in "XYZ"]

    root_id = root_null
    max_err = 0.0
    for fi, sol in enumerate(solutions):
        locals_from_file: dict[str, np.ndarray] = {}
        tx, ty, tz = curves_for(root_id, "T")
        rx, ry, rz = curves_for(root_id, "R")
        root_pos_file = np.array([tx[fi], ty[fi], tz[fi]]) / 100.0
        root_quat_file = euler_to_quat([rx[fi], ry[fi], rz[fi]])
        for bone in rig.bones:
            mid = id_by_name[names[bone.name]]
            bx, by, bz = curves_for(mid, "R")
            if not (bx and by and bz):
                continue
            locals_from_file[bone.name] = euler_to_quat([bx[fi], by[fi], bz[fi]])
        positions, _ = rig.fk(locals_from_file, root_quat=root_quat_file, root_pos=root_pos_file)
        for joint, pos in sol.joints.items():
            if joint not in positions:
                continue
            max_err = max(max_err, float(np.linalg.norm(positions[joint] - pos)))
    print(f"максимальное расхождение суставов после round-trip: {max_err * 100:.2f} см")

    ok = not missing and not bad_parent and max_err < 0.02
    print("ИТОГ:", "OK" if ok else "НЕУДОВЛЕТВОРИТЕЛЬНО")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
