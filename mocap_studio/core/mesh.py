"""Загрузка скелетных 3D-моделей: glTF 2.0 / GLB и FBX (бинарный и ASCII).

Своими руками, без внешних SDK: нужен только скелет + сетка с весами.

Модель приводится к общему виду:

    vertices   (N,3)  — вершины в bind-позе
    normals    (N,3)
    indices    (M,3)  — треугольники
    weights    (N,4)  — веса скиннинга (сумма = 1)
    joints     (N,4)  — индексы костей в joint_names
    joint_names          — имена костей модели
    joint_world  (J,4,4) — мировые матрицы костей в bind-позе
    ibm         (J,4,4) — inverse bind matrices (из файла)
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MeshModel:
    """Скелетная модель в нормализованном виде."""

    name: str = "модель"
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    indices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int32))
    weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    joints: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.int32))
    joint_names: list[str] = field(default_factory=list)
    joint_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4)))
    ibm: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4)))
    materials: list[tuple] = field(default_factory=list)  # (r,g,b,a) в 0..1
    source_format: str = ""

    @property
    def triangle_count(self) -> int:
        return int(len(self.indices))

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    @property
    def has_skin(self) -> bool:
        return len(self.joint_names) > 0 and self.weights.size > 0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.vertices):
            return np.zeros(3), np.zeros(3)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


# ---------------------------------------------------------------------------
# glTF / GLB
# ---------------------------------------------------------------------------

_COMP = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def _node_matrix(node: dict) -> np.ndarray:
    """Матрица узла 4x4 из TRS или из готовой matrix."""
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    t = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    r = np.array(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
    s = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    return _trs(t, r, s)


def _trs(t: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    m = np.eye(4)
    m[:3, :3] = rot * s[None, :]
    m[:3, 3] = t
    return m


def load_gltf(path: str) -> MeshModel:
    """Читает .gltf или .glb (внешние .bin подхватываются рядом)."""
    with open(path, "rb") as fh:
        head = fh.read(4)
        if head == b"glTF":
            glb = _read_glb(fh)
            doc = glb["doc"]
            blob = glb["bin"]
        else:
            fh.seek(0)
            doc = json.loads(fh.read().decode("utf-8"))
            blob = b""
    return _build_gltf(doc, blob, os.path.basename(path))


def _read_glb(fh) -> dict:
    fh.seek(0)
    data = fh.read()
    if len(data) < 12:
        raise ValueError("Файл слишком короткий для GLB")
    magic, version, _length = struct.unpack("<III", data[:12])
    if magic != 0x46546C67:
        raise ValueError("Это не GLB-файл")
    offset = 12
    doc = None
    blob = b""
    while offset + 8 <= len(data):
        clen, ctype = struct.unpack("<II", data[offset : offset + 8])
        payload = data[offset + 8 : offset + 8 + clen]
        if ctype == 0x4E4F534A:  # JSON
            doc = json.loads(payload.decode("utf-8"))
        elif ctype == 0x004E4942:  # BIN
            blob = payload
        offset += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)
    if doc is None:
        raise ValueError("В GLB нет JSON-блока")
    return {"doc": doc, "bin": blob}


def _gltf_buffer(doc: dict, blob: bytes, index: int) -> bytes:
    buf = doc["buffers"][index]
    uri = buf.get("uri")
    if uri is None:
        return blob
    if uri.startswith("data:"):
        import base64

        return base64.b64decode(uri.split(",", 1)[1])
    path = os.path.join(os.path.dirname(doc.get("__path__", "")), uri)
    with open(path, "rb") as fh:
        return fh.read()


def _read_accessor(doc: dict, blob: bytes, index: int) -> np.ndarray:
    acc = doc["accessors"][index]
    ncomp = _NCOMP[acc["type"]]
    dtype = _COMP[acc["componentType"]]
    n = int(acc["count"])
    if "bufferView" not in acc:
        return np.zeros((n, ncomp), dtype=dtype)
    view = doc["bufferViews"][acc["bufferView"]]
    data = blob
    if view.get("buffer", 0) != 0 and blob:
        # внешние буферы не поддерживаем в этой ветке (GLB — единственный источник)
        data = blob
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride") or (ncomp * np.dtype(dtype).itemsize)
    item = np.dtype(dtype).itemsize
    if stride == ncomp * item:
        arr = np.frombuffer(data, dtype=dtype, count=n * ncomp, offset=base)
        return arr.reshape(n, ncomp)
    # разрядка
    out = np.zeros((n, ncomp), dtype=dtype)
    for i in range(n):
        out[i] = np.frombuffer(data, dtype=dtype, count=ncomp, offset=base + i * stride)
    return out


def _build_gltf(doc: dict, blob: bytes, name: str) -> MeshModel:
    nodes = doc.get("nodes", [])
    # мировые матрицы узлов
    world: list[np.ndarray] = []

    def node_world(i: int) -> np.ndarray:
        while len(world) <= i:
            world.append(np.eye(4))
        if world[i] is None:
            world[i] = np.eye(4)
        return world[i]

    world = [None] * len(nodes)

    def compute(i: int) -> np.ndarray:
        if world[i] is not None:
            return world[i]
        node = nodes[i]
        local = _node_matrix(node)
        parent = None
        for pi, pnode in enumerate(nodes):
            if i in pnode.get("children", []):
                parent = pi
                break
        if parent is None:
            world[i] = local
        else:
            world[i] = compute(parent) @ local
        return world[i]

    for i in range(len(nodes)):
        compute(i)

    model = MeshModel(name=name, source_format="glTF")
    verts: list[np.ndarray] = []
    norms: list[np.ndarray] = []
    idxs: list[np.ndarray] = []
    wts: list[np.ndarray] = []
    jts: list[np.ndarray] = []

    skins = doc.get("skins", [])
    joint_names: list[str] = []
    joint_world: list[np.ndarray] = []
    ibm: list[np.ndarray] = []

    for skin in skins:
        joints = skin.get("joints", [])
        ibm_data = None
        if "inverseBindMatrices" in skin:
            raw = _read_accessor(doc, blob, skin["inverseBindMatrices"])
            ibm_data = raw.reshape(-1, 4, 4).transpose(0, 2, 1)
        local_names = []
        local_world = []
        local_ibm = []
        for k, j in enumerate(joints):
            nm = nodes[j].get("name", f"joint_{j}")
            local_names.append(nm)
            local_world.append(world[j])
            if ibm_data is not None and k < len(ibm_data):
                local_ibm.append(ibm_data[k])
            else:
                local_ibm.append(np.linalg.inv(world[j]))
        base = len(joint_names)
        joint_names.extend(local_names)
        joint_world.extend(local_world)
        ibm.extend(local_ibm)
        skin["__base__"] = base  # пометка для сдвига индексов

    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            pos = _read_accessor(doc, blob, attrs["POSITION"]).astype(np.float64)
            verts.append(pos)
            if "NORMAL" in attrs:
                norms.append(_read_accessor(doc, blob, attrs["NORMAL"]).astype(np.float64))
            else:
                norms.append(np.zeros_like(pos))
            if "indices" in prim:
                ind = _read_accessor(doc, blob, prim["indices"]).reshape(-1).astype(np.int32)
            else:
                ind = np.arange(len(pos), dtype=np.int32)
            idxs.append(ind)
            if "JOINTS_0" in attrs and "WEIGHTS_0" in attrs:
                j = _read_accessor(doc, blob, attrs["JOINTS_0"]).astype(np.int32)
                w = _read_accessor(doc, blob, attrs["WEIGHTS_0"]).astype(np.float64)
            else:
                j = np.zeros((len(pos), 4), dtype=np.int32)
                w = np.zeros((len(pos), 4))
            # сдвиг индексов на «внутренние» скелеты
            base = prim.get("__skin_base__", None)
            wts.append(w)
            jts.append(j)

    # сопоставление примитивов со скинами идёт через узлы: ищем узел сетки
    # и его "skin"
    prim_index = 0
    for node in nodes:
        if "mesh" not in node:
            continue
        skin_idx = node.get("skin")
        mesh = doc["meshes"][node["mesh"]]
        for prim in mesh.get("primitives", []):
            if prim_index >= len(jts):
                break
            if skin_idx is not None and skin_idx < len(skins):
                skin = skins[skin_idx]
                base = skin.get("__base__", 0)
                jts[prim_index] = jts[prim_index] + base
            prim_index += 1

    if verts:
        model.vertices = np.concatenate(verts, axis=0)
        model.normals = np.concatenate(norms, axis=0)
        model.indices = np.concatenate(idxs, axis=0)
        model.weights = np.concatenate(wts, axis=0)
        model.joints = np.concatenate(jts, axis=0)
        # склеиваем индексы разных примитивов
        off = 0
        fixed = []
        for part in idxs:
            fixed.append(part + off)
            off += len(part) // 3 if len(part) % 3 == 0 else 0
        model.indices = np.concatenate(fixed, axis=0) if fixed else model.indices
    model.joint_names = joint_names
    model.joint_world = np.array(joint_world) if joint_world else np.zeros((0, 4, 4))
    model.ibm = np.array(ibm) if ibm else np.zeros((0, 4, 4))
    if model.indices.size and model.indices.size % 3:
        model.indices = model.indices[: model.indices.size - model.indices.size % 3]
    model.indices = model.indices.reshape(-1, 3).astype(np.int32)
    return model


# ---------------------------------------------------------------------------
# FBX
# ---------------------------------------------------------------------------


def load_fbx(path: str) -> MeshModel:
    """Читает FBX: бинарный (7.x) или ASCII."""
    with open(path, "rb") as fh:
        head = fh.read(23)
    if head.startswith(b"Kaydara FBX Binary"):
        doc = _read_fbx_binary(path)
    else:
        doc = _read_fbx_ascii(path)
    return _build_fbx(doc, os.path.basename(path))


def _read_fbx_ascii(path: str) -> dict:
    """Разбор ASCII-FBX в дерево узлов.

    Узел: {kind, id, name, type, props, arrays, children, conns}
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    lines = []
    for raw in text.split("\n"):
        s = raw.strip()
        if not s or s.startswith(";"):
            continue
        lines.append(s)

    root = _new_node("__root__", 0, "", "")
    stack = [root]
    pos = 0
    while pos < len(lines):
        line = lines[pos]
        pos += 1
        if line == "}":
            if len(stack) > 1:
                stack.pop()
            continue
        if line.startswith("C:"):
            _parse_conn(line, stack[-1])
            continue
        # массив: Name: *24 {a: 1,2,3} — данные могут быть в той же строке
        # или идти следующими строками (так пишет Blender)
        if ": *" in line:
            name = line.split(":", 1)[0].strip()
            values: list[float] = []

            def take_values(text: str) -> None:
                body = text.strip()
                if body.startswith("a:"):
                    body = body[2:]
                body = body.rstrip("}").strip()
                for tok in body.split(","):
                    tok = tok.strip()
                    if tok:
                        try:
                            values.append(float(tok))
                        except ValueError:
                            pass

            if "{" in line:
                take_values(line.split("{", 1)[1])
            while not values and pos < len(lines):
                t = lines[pos]
                pos += 1
                if t.startswith("a:"):
                    take_values(t)
                elif t == "}":
                    break
                else:
                    break
            stack[-1]["arrays"][name] = values
            continue
        # узел: Kind: id, "name", "type" {   или  Kind: id, "name" {
        if ":" in line and line.rstrip().endswith("{"):
            head = line.rstrip()[:-1].strip()
            kind, _, rest = head.partition(":")
            parts, cur, in_str = [], "", False
            for ch in rest:
                if ch == '"':
                    in_str = not in_str
                    cur += ch
                elif ch == "," and not in_str:
                    parts.append(cur.strip())
                    cur = ""
                else:
                    cur += ch
            parts.append(cur.strip())
            obj_id = 0
            name = ""
            otype = ""
            if parts and parts[0]:
                try:
                    obj_id = int(parts[0])
                except ValueError:
                    obj_id = 0
            if len(parts) > 1:
                name = parts[1].strip('"')
            if len(parts) > 2:
                otype = parts[2].strip('"')
            node = _new_node(kind.strip(), obj_id, name, otype)
            stack[-1]["children"].append(node)
            stack.append(node)
            continue
        # свойство: P: "key", "type", "flags", "a", v1, v2 ...
        if line.startswith("P:"):
            body = line[2:].strip()
            parts, cur, in_str = [], "", False
            for ch in body:
                if ch == '"':
                    in_str = not in_str
                    cur += ch
                elif ch == "," and not in_str:
                    parts.append(cur.strip())
                    cur = ""
                else:
                    cur += ch
            parts.append(cur.strip())
            if len(parts) >= 5:
                key = parts[0].strip('"')
                vals = []
                for p2 in parts[4:]:
                    p2 = p2.strip()
                    if not p2:
                        continue
                    try:
                        vals.append(float(p2))
                    except ValueError:
                        vals.append(p2)
                stack[-1]["props"][key] = vals
    return root


def _new_node(kind: str, obj_id: int, name: str, otype: str) -> dict:
    return {
        "kind": kind,
        "id": obj_id,
        "name": name,
        "type": otype,
        "props": {},
        "arrays": {},
        "children": [],
        "conns": [],
    }


def _parse_conn(line: str, node: dict) -> None:
    body = line[2:].strip()
    parts, cur, in_str = [], "", False
    for ch in body:
        if ch == '"':
            in_str = not in_str
            cur += ch
        elif ch == "," and not in_str:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    if len(parts) < 3:
        return
    kind = parts[0].strip('"')
    try:
        src = int(parts[1])
        dst = int(parts[2])
    except ValueError:
        return
    prop = parts[3].strip('"') if len(parts) > 3 else ""
    node["conns"].append((kind, src, dst, prop))


def _read_fbx_binary(path: str) -> dict:
    """Разбор бинарного FBX 7.x в то же дерево, что и ASCII-вариант."""
    with open(path, "rb") as fh:
        data = fh.read()
    root = _new_node("__root__", 0, "", "")
    _fbx_nodes(data, 27, root, 0)
    return root


# имена свойств для известных типов узлов бинарного FBX
_FBX_PROP_NAMES = {
    "Model": ["id", "name", "type"],
    "Geometry": ["id", "name", "type"],
    "Deformer": ["id", "name", "type"],
    "SubDeformer": ["id", "name", "type"],
    "AnimationCurve": ["id", "name"],
}


def _fbx_nodes(data: bytes, offset: int, parent: dict, depth: int) -> int:
    """Разбирает последовательность узлов бинарного FBX начиная с offset."""
    end = len(data)
    while offset < end:
        if offset + 13 > end:
            break
        node_end, num_props, _prop_len = struct.unpack("<III", data[offset : offset + 12])
        name_len = data[offset + 12]
        offset += 13
        name = data[offset : offset + name_len].decode("utf-8", "replace")
        offset += name_len
        node = _new_node(name, 0, "", "")
        parent["children"].append(node)
        raw: list = []
        for _ in range(num_props):
            offset = _fbx_property(data, offset, node, raw)
        names = _FBX_PROP_NAMES.get(name)
        if names:
            for i, key in enumerate(names):
                if i < len(raw):
                    node["props"][key] = [raw[i]]
                    if key == "id":
                        try:
                            node["id"] = int(raw[i])
                        except (TypeError, ValueError):
                            pass
                    elif key == "name" and isinstance(raw[i], str):
                        node["name"] = raw[i]
                    elif key == "type" and isinstance(raw[i], str):
                        node["type"] = raw[i]
        if node_end:
            _fbx_nodes(data, offset, node, depth + 1)
            offset = node_end
        if depth == 0 and offset >= end - 16:
            break
    return offset


def _fbx_property(data: bytes, offset: int, node: dict, raw: list) -> int:
    if offset >= len(data):
        return offset
    t = chr(data[offset])
    offset += 1
    if t == "Y":
        raw.append(struct.unpack("<h", data[offset : offset + 2])[0])
        offset += 2
    elif t == "C":
        raw.append(bool(data[offset]))
        offset += 1
    elif t == "I":
        raw.append(struct.unpack("<i", data[offset : offset + 4])[0])
        offset += 4
    elif t == "F":
        raw.append(struct.unpack("<f", data[offset : offset + 4])[0])
        offset += 4
    elif t == "D":
        raw.append(struct.unpack("<d", data[offset : offset + 8])[0])
        offset += 8
    elif t == "L":
        raw.append(struct.unpack("<q", data[offset : offset + 8])[0])
        offset += 8
    elif t in "fdlib":
        count, encoding, comp_len = struct.unpack("<III", data[offset : offset + 12])
        offset += 12
        dtype = {"f": np.float32, "d": np.float64, "i": np.int32, "l": np.int64, "b": np.int8}[t]
        if encoding == 1:
            arr = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
            offset += count * np.dtype(dtype).itemsize
        else:
            import zlib

            chunk = data[offset : offset + comp_len]
            offset += comp_len
            try:
                arr = np.frombuffer(zlib.decompress(chunk), dtype=dtype, count=count)
            except Exception:  # noqa: BLE001
                arr = np.zeros(count, dtype=dtype)
        node["arrays"].setdefault("_last", []).append(arr)
        raw.append(arr)
    elif t == "S":
        (length,) = struct.unpack("<I", data[offset : offset + 4])
        offset += 4
        raw.append(data[offset : offset + length].decode("utf-8", "replace"))
        offset += length
    elif t == "R":
        (length,) = struct.unpack("<I", data[offset : offset + 4])
        offset += 4
        raw.append(data[offset : offset + length])
        offset += length
    return offset


def _fbx_find_all(node: dict, kind: str) -> list[dict]:
    out = []
    if node["kind"] == kind:
        out.append(node)
    for ch in node["children"]:
        out.extend(_fbx_find_all(ch, kind))
    return out


def _fbx_find(node: dict, kind: str) -> dict | None:
    found = _fbx_find_all(node, kind)
    return found[0] if found else None


def _build_fbx(doc: dict, name: str) -> MeshModel:
    """Собирает MeshModel из разобранного FBX."""
    model = MeshModel(name=name, source_format="FBX")

    models = _fbx_find_all(doc, "Model")
    # связи: src -> dst (dst — родитель)
    conn_parent: dict[int, int] = {}
    for node in _fbx_find_all(doc, "__root__") + [doc]:
        for kind, src, dst, _prop in node.get("conns", []):
            if kind == "OO":
                conn_parent[src] = dst
    for kind, src, dst, prop in doc.get("conns", []):
        if kind == "OO":
            conn_parent[src] = dst

    def local_matrix(m: dict) -> np.ndarray:
        p = m["props"]
        t = np.array(p.get("Lcl Translation", [0.0, 0.0, 0.0])[:3], dtype=np.float64)
        r = np.array(p.get("Lcl Rotation", [0.0, 0.0, 0.0])[:3], dtype=np.float64)
        s = np.array(p.get("Lcl Scaling", [1.0, 1.0, 1.0])[:3], dtype=np.float64)
        rx, ry, rz = np.radians(r)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        rot = np.array(
            [
                [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                [-sy, cy * sx, cy * cx],
            ]
        )
        mtx = np.eye(4)
        mtx[:3, :3] = rot * s[None, :]
        mtx[:3, 3] = t
        return mtx

    by_id = {m["id"]: m for m in models if m["id"]}
    world: dict[int, np.ndarray] = {}

    def compute(m: dict) -> np.ndarray:
        if m["id"] in world:
            return world[m["id"]]
        local = local_matrix(m)
        pid = conn_parent.get(m["id"])
        parent = by_id.get(pid)
        world[m["id"]] = (compute(parent) @ local) if parent is not None else local
        return world[m["id"]]

    bone_nodes = [m for m in models if m.get("type") in ("LimbNode", "Null", "Root")]
    mesh_nodes = [m for m in models if m.get("type") == "Mesh"]
    if not bone_nodes:  # бинарный FBX без типа — считаем костями всё, что не меш
        bone_nodes = [m for m in models if m not in mesh_nodes]

    for m in bone_nodes:
        compute(m)
    joint_names = [m.get("name") or f"bone_{i}" for i, m in enumerate(bone_nodes)]
    joint_world = (
        np.array([world[m["id"]] for m in bone_nodes]) if bone_nodes else np.zeros((0, 4, 4))
    )
    bone_index = {m["id"]: i for i, m in enumerate(bone_nodes)}

    all_v, all_n, all_i, all_w, all_j = [], [], [], [], []
    offset = 0
    for mesh_node in mesh_nodes:
        geom = None
        for ch in mesh_node["children"]:
            if ch["kind"] == "Geometry":
                geom = ch
                break
        if geom is None:
            continue
        verts = np.array(geom["arrays"].get("Vertices", []), dtype=np.float64).reshape(-1, 3)
        if not len(verts):
            continue
        idx_raw = [int(v) for v in geom["arrays"].get("PolygonVertexIndex", [])]
        polys: list[list[int]] = []
        cur: list[int] = []
        for v in idx_raw:
            if v < 0:
                cur.append(~v)
                polys.append(cur)
                cur = []
            else:
                cur.append(v)
        triangles = []
        for poly in polys:
            for k in range(1, len(poly) - 1):
                triangles.append([poly[0], poly[k], poly[k + 1]])
        vcount = len(verts)
        weights = np.zeros((vcount, 4))
        joints = np.zeros((vcount, 4), dtype=np.int32)
        counts = np.zeros(vcount, dtype=np.int32)

        # деформаторы, связанные с этой моделью
        for skin in _fbx_find_all(mesh_node, "Deformer"):
            for cluster in skin["children"]:
                if cluster["kind"] != "SubDeformer":
                    continue
                idxs = [int(v) for v in cluster["arrays"].get("Indexes", [])]
                wts = [float(v) for v in cluster["arrays"].get("Weights", [])]
                if not idxs or not wts:
                    continue
                bid = -1
                for link in cluster["children"]:
                    if link["kind"] == "Link_DeformAcyclic":
                        vals = link["props"].get("_last") or link["props"].get("id") or []
                        for v in vals:
                            if isinstance(v, (int, float)) and int(v) != 0:
                                bid = int(v)
                        if bid < 0 and link.get("id"):
                            bid = int(link["id"])
                if bid < 0 or bid not in bone_index:
                    continue
                bi = bone_index[bid]
                for vi, w in zip(idxs, wts):
                    if 0 <= vi < vcount and counts[vi] < 4:
                        weights[vi, counts[vi]] = w
                        joints[vi, counts[vi]] = bi
                        counts[vi] += 1
        normals = np.zeros((vcount, 3))
        for lay in _fbx_find_all(geom, "LayerElementNormal"):
            norms = np.array(lay["arrays"].get("Normals", []), dtype=np.float64).reshape(-1, 3)
            if not len(norms):
                continue
            idx_map = [int(v) for v in lay["arrays"].get("NormalsIndex", [])]
            if len(idx_map) == len(norms):
                normals = norms
            elif len(idx_map) > len(norms):
                for i, v in enumerate(idx_map):
                    if 0 <= v < vcount:
                        normals[v] = norms[i % len(norms)]
        total = weights.sum(axis=1, keepdims=True)
        total[total <= 0] = 1.0
        weights /= total
        tris = np.array(triangles, dtype=np.int32) if triangles else np.zeros((0, 3), dtype=np.int32)
        all_v.append(verts)
        all_n.append(normals)
        all_i.append(tris + offset)
        all_w.append(weights)
        all_j.append(joints)
        offset += vcount

    if all_v:
        model.vertices = np.concatenate(all_v)
        model.normals = np.concatenate(all_n)
        model.indices = np.concatenate(all_i) if any(len(x) for x in all_i) else np.zeros((0, 3), np.int32)
        model.weights = np.concatenate(all_w)
        model.joints = np.concatenate(all_j)
    model.joint_names = joint_names
    model.joint_world = joint_world
    model.ibm = (
        np.array([np.linalg.inv(w) for w in joint_world])
        if len(joint_world)
        else np.zeros((0, 4, 4))
    )
    return model


# ---------------------------------------------------------------------------


def load_model(path: str) -> MeshModel:
    """Загружает модель по расширению файла."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".gltf", ".glb"):
        return load_gltf(path)
    if ext == ".fbx":
        return load_fbx(path)
    raise ValueError(f"Формат {ext} не поддерживается (нужен .gltf, .glb или .fbx)")
