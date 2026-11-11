"""Тест работы с чужой 3D-моделью: загрузка GLB/FBX, привязка костей, скининг.

Строит тестовую скелетную модель (коробки вдоль костей рига), записывает её
в GLB и в ASCII-FBX, читает обратно, проверяет привязку и то, что при повороте
кости двигаются именно «её» вершины.
"""
from __future__ import annotations

import json
import struct
import sys

import numpy as np

sys.path.insert(0, "/workspace/mocap-studio")

from mocap_studio.core.mesh import load_model  # noqa: E402
from mocap_studio.core.math3d import q_rotate  # noqa: E402
from mocap_studio.core.rig import Rig  # noqa: E402
from mocap_studio.core.skin import MeshSkin, auto_map_bones  # noqa: E402
from mocap_studio.core.solve import FrameSolution  # noqa: E402

OUT = "/workspace/out"


# ---------------------------------------------------------------------------
# сборка тестовой модели
# ---------------------------------------------------------------------------


def _box(center, half, axes):
    """Вершины и треугольники коробки, ориентированной по осям (3x3)."""
    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    verts = center[None, :] + signs @ (axes * half[None, :])
    faces = [
        [0, 1, 2], [0, 2, 3],
        [5, 4, 7], [5, 7, 6],
        [4, 0, 3], [4, 3, 7],
        [1, 5, 6], [1, 6, 2],
        [3, 2, 6], [3, 6, 7],
        [4, 5, 1], [4, 1, 0],
    ]
    return verts, np.array(faces, dtype=np.int32)


def build_test_mesh(rig: Rig):
    """Простая «кукла» из коробок: по одной на каждую кость рига."""
    verts_all, faces_all, joints_all, weights_all = [], [], [], []
    v_off = 0
    for bi, bone in enumerate(rig.bones):
        head = rig.rest_positions[bone.head]
        tail = rig.rest_positions[bone.tail]
        center = (head + tail) * 0.5
        axis = tail - head
        length = float(np.linalg.norm(axis))
        if length < 1e-6:
            continue
        zdir = axis / length
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(zdir, helper))) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        xdir = np.cross(helper, zdir)
        xdir /= np.linalg.norm(xdir)
        ydir = np.cross(zdir, xdir)
        axes = np.column_stack([xdir, ydir, zdir])
        half = np.array([0.05, 0.05, length * 0.5 + 0.02])
        v, f = _box(center, half, axes)
        verts_all.append(v)
        faces_all.append(f + v_off)
        joints_all.append(np.full((len(v), 4), bi, dtype=np.int32))
        weights_all.append(np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(v), 1)))
        v_off += len(v)
    verts = np.concatenate(verts_all)
    faces = np.concatenate(faces_all)
    joints = np.concatenate(joints_all)
    weights = np.concatenate(weights_all)
    return verts, faces, joints, weights


# ---------------------------------------------------------------------------
# запись GLB
# ---------------------------------------------------------------------------


def write_glb(path: str, rig: Rig, verts, faces, joints, weights):
    bin_parts = []
    offset = 0

    def add_buffer(data: bytes, target=None):
        nonlocal offset
        pad = (4 - len(data) % 4) % 4
        bin_parts.append(data + b"\x00" * pad)
        view = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(data),
        }
        if target:
            view["target"] = target
        offset += len(data) + pad
        return len(bin_parts) - 1, offset - len(data) - pad

    accessors = []
    buffer_views = []

    def add_accessor(data: bytes, comp_type: int, type_: str, count: int, minv=None, maxv=None):
        view_idx, view_off = add_buffer(data, 34962)
        buffer_views.append(buffer_views.pop() if False else {})
        # bufferViews собираем отдельно ниже
        return None

    # собираем буферViews корректно
    buffer_views = []
    buf = bytearray()
    views_meta = []

    def put(data: bytes, target=None):
        nonlocal offset
        pad = (4 - len(data) % 4) % 4
        idx = len(views_meta)
        views_meta.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        buf.extend(data)
        buf.extend(b"\x00" * pad)
        offset += len(data) + pad
        return idx

    def acc(data: bytes, comp_type, type_, count, target=34962, extra=None):
        view = put(data, target)
        a = {
            "bufferView": view,
            "componentType": comp_type,
            "count": count,
            "type": type_,
        }
        if extra:
            a.update(extra)
        accessors.append(a)
        return len(accessors) - 1

    a_pos = acc(verts.astype(np.float32).tobytes(), 5126, "VEC3", len(verts))
    a_nrm = acc(np.tile(np.array([0.0, 1.0, 0.0], np.float32), (len(verts), 1)).tobytes(), 5126, "VEC3", len(verts))
    a_jnt = acc(joints.astype(np.uint16).tobytes(), 5123, "VEC4", len(joints))
    a_wgt = acc(weights.astype(np.float32).tobytes(), 5126, "VEC4", len(weights))
    a_idx = acc(faces.astype(np.uint32).tobytes(), 5125, "SCALAR", faces.size, target=34963)

    # мировые матрицы костей и bind-позы
    world = []
    for bone in rig.bones:
        q = rig.rest_global_quat[bone.name]
        m = np.eye(4)
        m[:3, :3] = _quat_mat(q)
        m[:3, 3] = rig.rest_positions[bone.head]
        world.append(m)
    world = np.array(world)
    a_ibm = acc(
        np.ascontiguousarray(np.linalg.inv(world).transpose(0, 2, 1)).astype(np.float32).tobytes(),
        5126,
        "MAT4",
        len(world),
    )

    # узлы скелета
    nodes = []
    name_to_idx = {}
    for i, bone in enumerate(rig.bones):
        parent = rig.by_name[bone.parent].name if bone.parent else None
        local_q = rig.rest_local_quat[bone.name]
        offset = (
            rig.rest_positions[bone.head]
            if bone.parent is None
            else rig.rest_offset[bone.name]
        )
        node = {
            "name": bone.name,
            "translation": [float(x) for x in offset],
            "rotation": [float(x) for x in local_q],
            "scale": [1.0, 1.0, 1.0],
            "children": [],
        }
        name_to_idx[bone.name] = len(nodes)
        nodes.append(node)
    for i, bone in enumerate(rig.bones):
        parent = rig.by_name[bone.parent].name if bone.parent else None
        if parent:
            nodes[name_to_idx[parent]]["children"].append(i)

    mesh_node = len(nodes)
    nodes.append({"name": "Body", "mesh": 0, "skin": 0})
    root_nodes = [i for i, bone in enumerate(rig.bones) if not bone.parent]

    doc = {
        "asset": {"version": "2.0", "generator": "mocap-studio test"},
        "scene": 0,
        "scenes": [{"nodes": root_nodes}],
        "nodes": nodes,
        "meshes": [
            {
                "name": "Body",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": a_pos,
                            "NORMAL": a_nrm,
                            "JOINTS_0": a_jnt,
                            "WEIGHTS_0": a_wgt,
                        },
                        "indices": a_idx,
                        "mode": 4,
                    }
                ],
            }
        ],
        "skins": [
            {
                "joints": list(range(len(rig.bones))),
                "inverseBindMatrices": a_ibm,
            }
        ],
        "accessors": accessors,
        "bufferViews": views_meta,
        "buffers": [{"byteLength": len(buf)}],
    }

    json_bytes = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_bytes = bytes(buf)
    bin_bytes += b"\x00" * ((4 - len(bin_bytes) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        fh.write(json_bytes)
        fh.write(struct.pack("<II", len(bin_bytes), 0x004E4942))
        fh.write(bin_bytes)
    return path


def _quat_mat(q):
    x, y, z, w = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# ---------------------------------------------------------------------------
# запись ASCII-FBX
# ---------------------------------------------------------------------------


def write_fbx_ascii(path: str, rig: Rig, verts, faces, joints, weights):
    lines = ["FBXHeaderExtension:  {", "}", "Definitions:  {", "}", "Objects:  {"]
    ids = {}
    next_id = 100000
    for bone in rig.bones:
        next_id += 1
        ids[bone.name] = next_id
    mesh_id = next_id + 1
    geom_id = next_id + 2
    lines.append("")
    for bone in rig.bones:
        q = rig.rest_local_quat[bone.name]
        # эйлер XYZ (внешний)
        m = _quat_mat(q)
        ry = float(np.arcsin(np.clip(-m[2, 0], -1, 1)))
        rx = float(np.arctan2(m[2, 1], m[2, 2]))
        rz = float(np.arctan2(m[1, 0], m[0, 0]))
        euler = [np.degrees(rx), np.degrees(ry), np.degrees(rz)]
        off = (
            rig.rest_positions[bone.head] if bone.parent is None else rig.rest_offset[bone.name]
        ) * 100.0
        lines.append(f'\tModel: {ids[bone.name]}, "Model::{bone.name}", "LimbNode" {{')
        lines.append("\t\tVersion: 232")
        lines.append("\t\tProperties70:  {")
        lines.append(
            f'\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A", {off[0]:.6f},{off[1]:.6f},{off[2]:.6f}'
        )
        lines.append(
            f'\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A", {euler[0]:.6f},{euler[1]:.6f},{euler[2]:.6f}'
        )
        lines.append('\t\t\tP: "Lcl Scaling", "Lcl Scaling", "", "A", 1,1,1')
        lines.append("\t\t}")
        lines.append("\t}")
    # сетка
    lines.append(f'\tModel: {mesh_id}, "Model::Body", "Mesh" {{')
    lines.append("\t\tVersion: 232")
    lines.append("\t\tProperties70:  {")
    lines.append('\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A", 0,0,0')
    lines.append('\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A", 0,0,0')
    lines.append('\t\t\tP: "Lcl Scaling", "Lcl Scaling", "", "A", 1,1,1')
    lines.append("\t\t}")
    lines.append(f'\t\tGeometry: {geom_id}, "Geometry::", "Mesh" {{')
    flat = (verts * 100.0).reshape(-1)
    lines.append(f"\t\t\tVertices: *{len(flat)} {{a: " + ",".join(f"{v:.6f}" for v in flat) + "}")
    poly_idx = []
    for f in faces:
        poly_idx.extend([int(f[0]), int(f[1]), ~int(f[2])])
    lines.append(
        f"\t\t\tPolygonVertexIndex: *{len(poly_idx)} {{a: " + ",".join(str(v) for v in poly_idx) + "}"
    )
    lines.append("\t\t}")
    lines.append("\t}")
    # деформаторы: по одному на кость
    skin_ids = {}
    for i, bone in enumerate(rig.bones):
        sel = np.where(joints[:, 0] == i)[0]
        if not len(sel):
            continue
        next_id += 1
        skin_id = next_id
        next_id += 1
        cluster_id = next_id
        skin_ids[bone.name] = (skin_id, cluster_id)
        w = weights[sel, 0]
        lines.append(f'\tDeformer: {skin_id}, "Deformer::{bone.name}", "Skin" {{')
        lines.append(
            f"\t\tIndexes: *{len(sel)} {{a: " + ",".join(str(int(v)) for v in sel) + "}"
        )
        lines.append(f"\t\tWeights: *{len(sel)} {{a: " + ",".join(f"{v:.6f}" for v in w) + "}")
        lines.append(f'\t\tSubDeformer: {cluster_id}, "SubDeformer::{bone.name}", "Cluster" {{')
        lines.append(f'\t\t\tLink_DeformAcyclic: {ids[bone.name]}, "SubDeformer::{bone.name}", ""')
        lines.append("\t\t}")
        lines.append("\t}")
    lines.append("}")
    lines.append("Connections:  {")
    for bone in rig.bones:
        parent = rig.by_name[bone.parent].name if bone.parent else None
        pid = ids[parent] if parent else 0
        lines.append(f'\tC: "OO", {ids[bone.name]}, {pid}')
    lines.append(f'\tC: "OO", {mesh_id}, 0')
    lines.append(f'\tC: "OO", {geom_id}, {mesh_id}')
    for bone, (skin_id, cluster_id) in skin_ids.items():
        lines.append(f'\tC: "OO", {skin_id}, {mesh_id}')
        lines.append(f'\tC: "OO", {cluster_id}, {skin_id}')
    lines.append("}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------


def main() -> int:
    rig = Rig.build()
    verts, faces, joints, weights = build_test_mesh(rig)

    glb_path = f"{OUT}/test_model.glb"
    fbx_path = f"{OUT}/test_model.fbx"
    write_glb(glb_path, rig, verts, faces, joints, weights)
    write_fbx_ascii(fbx_path, rig, verts, faces, joints, weights)

    results = {}
    for label, path in (("GLB", glb_path), ("FBX", fbx_path)):
        model = load_model(path)
        ok_v = abs(model.vertex_count - len(verts)) <= 1
        ok_f = abs(model.triangle_count - len(faces)) <= 1
        ok_b = len(model.joint_names) == len(rig.bones)
        print(
            f"{label}: вершин {model.vertex_count} (ожидалось {len(verts)}), "
            f"треугольников {model.triangle_count} (ожидалось {len(faces)}), "
            f"костей {len(model.joint_names)}"
        )
        results[label] = (model, ok_v and ok_f and ok_b)

    # привязка костей
    model = results["GLB"][0]
    mapping = auto_map_bones(rig, model)
    print(f"привязка: {mapping.summary()}, покрытие {mapping.coverage * 100:.0f}%")

    # скининг: в покое вершины на местах, при повороте кости — двигаются
    skin = MeshSkin(rig, model, mapping)
    # поза рига в состоянии покоя: корень несёт «телесный» базис
    sol = FrameSolution(
        index=0,
        root_quat=rig.rest_global_quat["pelvis"],
        root_pos=rig.rest_positions["pelvis"].copy(),
        local_quats={b.name: rig.rest_local_quat[b.name] for b in rig.bones},
    )
    rest_verts = skin.skin(sol)
    err_rest = float(np.max(np.linalg.norm(rest_verts - model.vertices, axis=1)))
    print(f"покой: максимальное смещение вершин {err_rest * 100:.3f} см")

    # повернём предплечье
    from mocap_studio.core.math3d import quat_from_axis_angle

    q = quat_from_axis_angle((0.0, 0.0, 1.0), 0.9)
    locals2 = dict(sol.local_quats)
    from mocap_studio.core.math3d import q_mul

    locals2["lowerarm_l"] = q_mul(q, rig.rest_local_quat["lowerarm_l"])
    sol2 = FrameSolution(
        index=1,
        root_quat=sol.root_quat,
        root_pos=sol.root_pos,
        local_quats=locals2,
    )
    moved = skin.skin(sol2)
    delta = np.linalg.norm(moved - rest_verts, axis=1)
    # вершины предплечья должны сдвинуться, таз — нет
    arm_joint = model.joint_names.index("lowerarm_l")
    lower = model.joints[:, 0] == arm_joint
    moved_arm = float(delta[lower].max()) if lower.any() else 0.0
    pelvis_joint = model.joint_names.index("pelvis")
    pelvis_zone = model.joints[:, 0] == pelvis_joint
    still = float(delta[pelvis_zone].max()) if pelvis_zone.any() else 0.0
    print(f"поворот предплечья: максимальный сдвиг {moved_arm * 100:.1f} см, таз {still * 100:.2f} см")

    ok = (
        results["GLB"][1]
        and results["FBX"][1]
        and mapping.coverage > 0.9
        and err_rest < 0.02
        and moved_arm > 0.05
        and still < 0.02
    )
    print("ИТОГ:", "OK" if ok else "НЕУДОВЛЕТВОРИТЕЛЬНО")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
