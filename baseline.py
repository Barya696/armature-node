"""The rig's own record of what it was before the graph touched it.

The Armature Input must not read the live armature. The Output writes to that
same armature, so a live read feeds the graph its own results: a stack that
sets ``use_deform`` off, or replaces a widget, sees the changed value on the
next evaluation and can never get back to the original. Worse, anything the
graph stops producing stays applied forever, because nothing remembers what it
replaced.

So the unmodified rig is serialised once and stored **on the armature object**
as a custom property. The Armature Input inherits from that, every evaluation
starts from the same base state, and removing a node genuinely undoes it.

Only rest data is stored: bones, hierarchy, deform flags, constraints and
custom shapes. Pose is deliberately excluded -- posing is what the graph does,
and a baseline that remembered it would fight the Transform nodes.
"""

import json

from .core import BoneDef, ConstraintDef, ShapeDef, bone_roll

# Custom property on the armature OBJECT holding the serialised baseline.
BASELINE_KEY = "an_baseline"
# Bumped when the stored shape changes, so an old capture is re-taken rather
# than half-read into the current BoneDef layout.
BASELINE_VERSION = 1


def _shape_to_dict(shape):
    if shape is None:
        return None
    return {
        "widget": shape.widget,
        "preset": shape.preset,
        "scale": list(shape.scale),
        "translation": list(shape.translation),
        "rotation": list(shape.rotation),
        "wire_width": shape.wire_width,
        "scale_to_bone_length": shape.scale_to_bone_length,
        "show_wire": shape.show_wire,
    }


def _shape_from_dict(data):
    if not data:
        return None
    return ShapeDef(
        widget=data.get("widget", ""),
        preset=data.get("preset", "NONE"),
        scale=tuple(data.get("scale", (1.0, 1.0, 1.0))),
        translation=tuple(data.get("translation", (0.0, 0.0, 0.0))),
        rotation=tuple(data.get("rotation", (0.0, 0.0, 0.0))),
        wire_width=data.get("wire_width", 1.0),
        scale_to_bone_length=data.get("scale_to_bone_length", True),
        show_wire=data.get("show_wire", True),
    )


def _constraint_to_dict(con):
    """Serialise a pose constraint.

    Object-valued parameters are stored by NAME and resolved at build time --
    a baseline that held live object pointers would keep deleted objects alive
    and break the moment the file is reopened.
    """
    params = {}
    for attr in ("influence", "chain_count", "iterations", "use_stretch", "pole_angle",
                 "subtarget", "head_tail", "mute"):
        if hasattr(con, attr):
            params[attr] = getattr(con, attr)
    for attr in ("target", "pole_target"):
        obj = getattr(con, attr, None)
        if obj is not None:
            params[attr] = obj.name
    return {"type": con.type, "name": con.name, "params": params}


def capture(obj):
    """Serialise ``obj``'s current rest state. Returns a list of dicts."""
    if obj is None or obj.type != "ARMATURE":
        return []
    pose = obj.pose
    bones = []
    for b in obj.data.bones:
        pbone = pose.bones.get(b.name) if pose else None
        shape = None
        if pbone is not None and pbone.custom_shape is not None:
            shape = {
                "widget": pbone.custom_shape.name,
                "preset": "NONE",
                "scale": list(pbone.custom_shape_scale_xyz),
                "translation": list(pbone.custom_shape_translation),
                "rotation": list(pbone.custom_shape_rotation_euler),
                "wire_width": float(getattr(pbone, "custom_shape_wire_width", 1.0)),
                "scale_to_bone_length": pbone.use_custom_shape_bone_size,
                "show_wire": b.show_wire,
            }
        bones.append(
            {
                "name": b.name,
                "head": list(b.head_local),
                "tail": list(b.tail_local),
                "roll": bone_roll(b),
                "parent": b.parent.name if b.parent else None,
                "use_connect": b.use_connect,
                "use_deform": b.use_deform,
                "envelope_distance": b.envelope_distance,
                "envelope_weight": b.envelope_weight,
                "shape": shape,
                "constraints": [
                    _constraint_to_dict(c) for c in (pbone.constraints if pbone else ())
                ],
            }
        )
    return bones


def store(obj, bones=None):
    """Write the baseline onto the armature object. Returns the bone count."""
    if obj is None or obj.type != "ARMATURE":
        return 0
    bones = capture(obj) if bones is None else bones
    obj[BASELINE_KEY] = json.dumps(
        {"version": BASELINE_VERSION, "bones": bones}, separators=(",", ":")
    )
    return len(bones)


def has_baseline(obj):
    return obj is not None and obj.get(BASELINE_KEY) not in (None, "")


def clear(obj):
    if obj is not None and BASELINE_KEY in obj:
        del obj[BASELINE_KEY]


def load(obj):
    """The stored baseline as a list of dicts, or [] when there is none.

    A capture from an older layout is discarded rather than half-read, so the
    caller re-takes it against the current rig.
    """
    if obj is None:
        return []
    raw = obj.get(BASELINE_KEY)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if data.get("version") != BASELINE_VERSION:
        return []
    return data.get("bones") or []


def bone_defs(obj, ensure=True):
    """The baseline as ``BoneDef`` objects, ready to put on the wire.

    With ``ensure``, an armature that has no baseline yet gets one captured
    from its current state -- that first read is the only time the live rig is
    consulted, and it is by definition unmodified because the graph has not
    run against it.
    """
    stored = load(obj)
    if not stored and ensure and obj is not None and obj.type == "ARMATURE":
        store(obj)
        stored = load(obj)
    out = []
    for d in stored:
        out.append(
            BoneDef(
                name=d["name"],
                head=tuple(d["head"]),
                tail=tuple(d["tail"]),
                roll=d.get("roll", 0.0),
                parent=d.get("parent"),
                use_connect=d.get("use_connect", False),
                use_deform=d.get("use_deform", True),
                envelope_distance=d.get("envelope_distance", 0.25),
                envelope_weight=d.get("envelope_weight", 1.0),
                shape=_shape_from_dict(d.get("shape")),
                constraints=[
                    ConstraintDef(
                        type=c["type"], name=c.get("name", ""), params=dict(c.get("params", {}))
                    )
                    for c in d.get("constraints", ())
                ],
            )
        )
    return out
