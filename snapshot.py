"""Serialize armature data into plain JSON so the node graph is self-sufficient.

Decompiling a rig used to leave the graph *referencing* the armature: delete
the armature and there was nothing left to build from. Everything here turns
live Blender data (bones, constraints, widget meshes) into JSON-safe dicts
that are stored on the nodes, and back into BoneDefs at build time.
"""

import json

import bpy
from mathutils import Vector

from .core import BoneDef, ConstraintDef, ShapeDef

# Constraint RNA properties that are never useful to replay.
_SKIP_CONSTRAINT_PROPS = {
    "rna_type", "name", "type", "is_valid", "is_override_data", "error_location",
    "error_rotation", "active", "show_expanded", "is_proxy_local",
}
_OBJECT_POINTER_PROPS = {"target", "pole_target", "space_object"}


# ---------------------------------------------------------------------------
# Bones
# ---------------------------------------------------------------------------


def bone_roll(bone):
    """Roll of a (non-edit) Bone, recovered from its rest matrix."""
    try:
        mat = bone.matrix_local.to_3x3()
        return float(bpy.types.Bone.AxisRollFromMatrix(mat)[1])
    except Exception:
        return 0.0


def serialize_constraint(con):
    """Constraint -> {'type', 'name', 'params'} with object refs as names."""
    params = {}
    for prop in con.bl_rna.properties:
        pid = prop.identifier
        if pid in _SKIP_CONSTRAINT_PROPS or prop.is_readonly:
            continue
        try:
            value = getattr(con, pid)
        except AttributeError:
            continue
        if prop.type == "POINTER":
            if pid in _OBJECT_POINTER_PROPS and value is not None:
                params[pid] = value.name
            continue
        if prop.type == "COLLECTION":
            # Armature-constraint targets: store (object, subtarget, weight).
            if pid == "targets":
                params[pid] = [
                    {
                        "target": t.target.name if t.target else "",
                        "subtarget": t.subtarget,
                        "weight": t.weight,
                    }
                    for t in value
                ]
            continue
        if prop.type in ("BOOLEAN", "INT", "FLOAT") and getattr(prop, "array_length", 0) > 0:
            params[pid] = [v for v in value]
        elif prop.type == "ENUM" and prop.is_enum_flag:
            # Sets are not JSON-safe; mark them so they round-trip as sets.
            params[pid] = {"__enum_flags__": sorted(value)}
        elif prop.type in ("BOOLEAN", "INT", "FLOAT", "STRING", "ENUM"):
            params[pid] = value
    return {"type": con.type, "name": con.name, "params": params}


def constraint_from_dict(data):
    params = dict(data.get("params", {}))
    for key, value in list(params.items()):
        if isinstance(value, dict) and "__enum_flags__" in value:
            params[key] = set(value["__enum_flags__"])
        elif isinstance(value, list) and key != "targets":
            params[key] = tuple(value)  # vector props
    return ConstraintDef(type=data["type"], name=data.get("name", ""), params=params)


def serialize_pose_bone(pbone, include_shape=True):
    """PoseBone -> dict covering rest pose, hierarchy, constraints, widget."""
    bone = pbone.bone
    data = {
        "name": bone.name,
        "head": list(bone.head_local),
        "tail": list(bone.tail_local),
        "roll": bone_roll(bone),
        "parent": bone.parent.name if bone.parent else "",
        "children": [c.name for c in bone.children],
        "use_connect": bone.use_connect,
        "use_deform": bone.use_deform,
        "envelope_distance": bone.envelope_distance,
        "envelope_weight": bone.envelope_weight,
        "constraints": [serialize_constraint(c) for c in pbone.constraints],
    }
    if include_shape and pbone.custom_shape is not None:
        data["shape"] = serialize_shape(pbone)
    return data


def serialize_shape(pbone):
    widget = pbone.custom_shape
    return {
        "widget": widget.name if widget else "",
        "scale": list(pbone.custom_shape_scale_xyz),
        "translation": list(pbone.custom_shape_translation),
        "rotation": list(pbone.custom_shape_rotation_euler),
        "wire_width": float(getattr(pbone, "custom_shape_wire_width", 1.0)),
        "scale_to_bone_length": pbone.use_custom_shape_bone_size,
        "show_wire": pbone.bone.show_wire,
        "geometry": mesh_geometry(widget) if widget else None,
    }


def bonedef_from_dict(data):
    bdef = BoneDef(
        name=data["name"],
        head=tuple(data.get("head", (0, 0, 0))),
        tail=tuple(data.get("tail", (0, 0, 1))),
        roll=float(data.get("roll", 0.0)),
        parent=data.get("parent") or None,
        use_connect=bool(data.get("use_connect", False)),
        use_deform=bool(data.get("use_deform", True)),
        envelope_distance=float(data.get("envelope_distance", 0.25)),
        envelope_weight=float(data.get("envelope_weight", 1.0)),
        constraints=[constraint_from_dict(c) for c in data.get("constraints", [])],
    )
    shape = data.get("shape")
    if shape:
        bdef.shape = ShapeDef(
            widget=shape.get("widget", ""),
            preset="NONE",
            scale=tuple(shape.get("scale", (1, 1, 1))),
            translation=tuple(shape.get("translation", (0, 0, 0))),
            rotation=tuple(shape.get("rotation", (0, 0, 0))),
            wire_width=float(shape.get("wire_width", 1.0)),
            scale_to_bone_length=bool(shape.get("scale_to_bone_length", True)),
            show_wire=bool(shape.get("show_wire", True)),
            geometry=shape.get("geometry"),
        )
    return bdef


def serialize_armature(obj):
    """Whole armature -> list of bone dicts (hierarchy order: parents first)."""
    def depth(b):
        d = 0
        while b.parent is not None:
            d += 1
            b = b.parent
        return d

    bones = sorted(obj.data.bones, key=lambda b: (depth(b), b.name))
    pose = obj.pose
    out = []
    for b in bones:
        pb = pose.bones.get(b.name) if pose else None
        if pb is not None:
            out.append(serialize_pose_bone(pb))
        else:
            out.append({
                "name": b.name, "head": list(b.head_local), "tail": list(b.tail_local),
                "roll": bone_roll(b), "parent": b.parent.name if b.parent else "",
                "use_connect": b.use_connect, "use_deform": b.use_deform,
            })
    return out


# ---------------------------------------------------------------------------
# Widget meshes
# ---------------------------------------------------------------------------


def mesh_geometry(obj):
    """Mesh object -> geometry in the mesh's own local coordinates PLUS the
    object's world matrix.

    Blender draws a custom shape from the widget's *mesh data* only and ignores
    the widget object's own location/rotation/scale, so the vertices are stored
    untouched (never baked through ``matrix_world``).  Baking would move the
    shape on the bone and apply rotation/scale twice.

    The object's placement in the 3D world (global frame) is however part of
    the rig: a widget that sat at (2, 0, 1) must be rebuilt at (2, 0, 1), not
    at the origin.  That placement is therefore captured separately as
    ``matrix`` (the 4x4 ``matrix_world``) and re-applied by
    ``widgets.ensure_widget`` when the object is recreated.
    """
    if obj is None or obj.type != "MESH":
        return None
    mesh = obj.data
    matrix = [[round(c, 6) for c in row] for row in obj.matrix_world]
    verts = [list(v.co) for v in mesh.vertices]
    edges = [list(e.vertices) for e in mesh.edges]
    if not edges and mesh.polygons:
        seen = set()
        for poly in mesh.polygons:
            for a, b in poly.edge_keys:
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    edges.append([a, b])
    faces = [list(poly.vertices) for poly in mesh.polygons]
    return {
        "verts": [[round(c, 5) for c in v] for v in verts],
        "edges": edges,
        "faces": faces,
        "matrix": matrix,
    }


def describe_geometry(geo):
    """Short human description of stored widget geometry for the node UI."""
    if not geo or not geo.get("verts"):
        return "no geometry"
    verts = geo["verts"]
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    size = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    return (
        f"{len(verts)} verts, {len(geo.get('edges', []))} edges, "
        f"{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}"
    )


# ---------------------------------------------------------------------------
# JSON helpers for StringProperty storage
# ---------------------------------------------------------------------------


def dumps(data):
    return json.dumps(data, separators=(",", ":")) if data is not None else ""


def loads(text, default=None):
    if not text:
        return default
    try:
        return json.loads(text)
    except ValueError:
        return default
