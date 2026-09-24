"""Serialisation and validation for :mod:`model.types`.

JSON is the storage format because the record lives on the armature object as
a custom property, and custom properties survive save/reload and library
linking only for plain data. Kept separate from ``types`` so the dataclasses
stay free of I/O concerns.

Round-tripping is exact: ``from_json(to_json(r)) == r`` for any record built
by ``capture``. The tests assert that, because the record is the only copy of
the original rig -- a lossy round trip is data loss.
"""

import json

from .types import (
    ArmatureDef,
    BoneDef,
    ConstraintDef,
    DisplayDef,
    MembershipDef,
    PoseDef,
    RestDef,
    RigRecord,
    TransformDef,
    SourceDef,
    WidgetLib,
)

#: Bumped whenever the stored shape changes. ``migrate`` upgrades anything
#: older; a record from a *newer* version is refused rather than half-read.
RECORD_VERSION = 2


#: Every TransformDef field, in storage order. Missing keys read back as None,
#: so a record written before the relative fields existed still loads.
_TRANSFORM_KEYS = (
    "location", "rotation", "scale",
    "offset", "rotation_offset", "local_offset", "local_rotation",
)


class RecordError(ValueError):
    """A stored record could not be read."""


def _vec(value, default):
    if value is None:
        return default
    return tuple(float(v) for v in value)


# --- to dict ---------------------------------------------------------------


def _rest_to_dict(r):
    return {
        "head": list(r.head),
        "tail": list(r.tail),
        "roll": r.roll,
        "parent": r.parent,
        "connect": r.connect,
        "deform": r.deform,
        "inherit_scale": r.inherit_scale,
        "use_local_location": r.use_local_location,
        "envelope_distance": r.envelope_distance,
        "envelope_weight": r.envelope_weight,
        "head_radius": r.head_radius,
        "tail_radius": r.tail_radius,
    }


def _display_to_dict(d):
    return {
        "shape": d.shape,
        "preset": d.preset,
        "scale": list(d.scale),
        "translation": list(d.translation),
        "rotation": list(d.rotation),
        "wire_width": d.wire_width,
        "use_bone_size": d.use_bone_size,
        "override": d.override,
        "show_wire": d.show_wire,
        "color": d.color,
    }


def _bone_to_dict(b):
    return {
        "rest": _rest_to_dict(b.rest),
        "membership": {
            "collections": list(b.membership.collections),
            "layers": list(b.membership.layers) if b.membership.layers is not None else None,
        },
        "display": _display_to_dict(b.display),
        "pose": {
            "rotation_mode": b.pose.rotation_mode,
            "locks": dict(b.pose.locks),
            "ik": dict(b.pose.ik),
        },
        "transform": {
            key: (list(getattr(b.transform, key)) if getattr(b.transform, key) else None)
            for key in _TRANSFORM_KEYS
        },
        "constraints": [
            {"type": c.type, "name": c.name, "props": dict(c.props)} for c in b.constraints
        ],
    }


def record_to_dict(record):
    return {
        "version": record.version,
        "source": {
            "object": record.source.object,
            "armature": record.source.armature,
            "captured_at": record.source.captured_at,
            "blender": record.source.blender,
        },
        "armature": {
            "display_type": record.armature.display_type,
            "show_in_front": record.armature.show_in_front,
            "pose_position": record.armature.pose_position,
            "collections": list(record.armature.collections),
        },
        "bones": {name: _bone_to_dict(b) for name, b in record.bones.items()},
    }


# --- from dict -------------------------------------------------------------


def _rest_from_dict(d):
    d = d or {}
    ref = RestDef()
    return RestDef(
        head=_vec(d.get("head"), ref.head),
        tail=_vec(d.get("tail"), ref.tail),
        roll=float(d.get("roll", ref.roll)),
        parent=d.get("parent", ref.parent),
        connect=bool(d.get("connect", ref.connect)),
        deform=bool(d.get("deform", ref.deform)),
        inherit_scale=d.get("inherit_scale", ref.inherit_scale),
        use_local_location=bool(d.get("use_local_location", ref.use_local_location)),
        envelope_distance=float(d.get("envelope_distance", ref.envelope_distance)),
        envelope_weight=float(d.get("envelope_weight", ref.envelope_weight)),
        head_radius=float(d.get("head_radius", ref.head_radius)),
        tail_radius=float(d.get("tail_radius", ref.tail_radius)),
    )


def _display_from_dict(d):
    d = d or {}
    ref = DisplayDef()
    return DisplayDef(
        shape=d.get("shape", ref.shape) or "",
        preset=d.get("preset", ref.preset) or "NONE",
        scale=_vec(d.get("scale"), ref.scale),
        translation=_vec(d.get("translation"), ref.translation),
        rotation=_vec(d.get("rotation"), ref.rotation),
        wire_width=float(d.get("wire_width", ref.wire_width)),
        use_bone_size=bool(d.get("use_bone_size", ref.use_bone_size)),
        override=d.get("override", ref.override) or "",
        show_wire=bool(d.get("show_wire", ref.show_wire)),
        color=d.get("color", ref.color),
    )


def _transform_from_dict(d):
    d = d or {}
    def opt(key):
        v = d.get(key)
        return tuple(float(x) for x in v) if v else None

    return TransformDef(**{key: opt(key) for key in _TRANSFORM_KEYS})


def _bone_from_dict(name, d):
    d = d or {}
    m = d.get("membership") or {}
    layers = m.get("layers")
    p = d.get("pose") or {}
    return BoneDef(
        name=name,
        rest=_rest_from_dict(d.get("rest")),
        membership=MembershipDef(
            collections=tuple(m.get("collections") or ()),
            layers=tuple(layers) if layers is not None else None,
        ),
        display=_display_from_dict(d.get("display")),
        pose=PoseDef(
            rotation_mode=p.get("rotation_mode", "QUATERNION"),
            locks=dict(p.get("locks") or {}),
            ik=dict(p.get("ik") or {}),
        ),
        transform=_transform_from_dict(d.get("transform")),
        constraints=tuple(
            ConstraintDef(
                type=c.get("type", ""),
                name=c.get("name", ""),
                props=dict(c.get("props") or {}),
            )
            for c in (d.get("constraints") or ())
        ),
    )


def record_from_dict(data):
    """Build a record from a plain dict. Raises :class:`RecordError`."""
    if not isinstance(data, dict):
        raise RecordError("record is not an object")
    version = data.get("version")
    if version != RECORD_VERSION:
        raise RecordError(
            f"record version {version!r}, expected {RECORD_VERSION} "
            "(run model.migrate first)"
        )
    src = data.get("source") or {}
    arm = data.get("armature") or {}
    bones = data.get("bones")
    if not isinstance(bones, dict):
        raise RecordError("record has no bones map")
    return RigRecord(
        version=RECORD_VERSION,
        source=SourceDef(
            object=src.get("object", ""),
            armature=src.get("armature", ""),
            captured_at=src.get("captured_at", ""),
            blender=src.get("blender", ""),
        ),
        armature=ArmatureDef(
            display_type=arm.get("display_type", "OCTAHEDRAL"),
            show_in_front=bool(arm.get("show_in_front", False)),
            pose_position=arm.get("pose_position", "POSE"),
            collections=tuple(arm.get("collections") or ()),
        ),
        bones={name: _bone_from_dict(name, b) for name, b in bones.items()},
    )


# --- JSON ------------------------------------------------------------------


def to_json(record):
    return json.dumps(record_to_dict(record), separators=(",", ":"), sort_keys=True)


def from_json(text):
    if not text:
        raise RecordError("empty record")
    try:
        data = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise RecordError(f"record is not valid JSON: {exc}") from exc
    return record_from_dict(data)


def peek_version(text):
    """The version of a stored record without fully parsing it.

    Used by ``migrate`` to decide what to do, and by the UI to report a record
    it cannot read yet. Returns ``None`` when the text is not a record at all.
    """
    if not text:
        return None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    version = data.get("version")
    return version if isinstance(version, int) else None


# --- widgets ---------------------------------------------------------------


def widgets_to_dict(lib):
    return {
        name: {
            "verts": [list(v) for v in geo.get("verts", ())],
            "edges": [list(e) for e in geo.get("edges", ())],
            "faces": [list(f) for f in geo.get("faces", ())],
        }
        for name, geo in lib.widgets.items()
    }


def widgets_from_dict(data):
    out = {}
    for name, geo in (data or {}).items():
        geo = geo or {}
        out[name] = {
            "verts": [tuple(float(x) for x in v) for v in geo.get("verts", ())],
            "edges": [tuple(int(x) for x in e) for e in geo.get("edges", ())],
            "faces": [tuple(int(x) for x in f) for f in geo.get("faces", ())],
        }
    return WidgetLib(widgets=out)


def validate(record):
    """Structural problems with a record, as a list of human-readable strings.

    Not raised: a record with a dangling parent is still worth loading and
    reporting, because refusing it would strand the user with no way back.
    """
    problems = []
    if record.version != RECORD_VERSION:
        problems.append(f"version {record.version}, expected {RECORD_VERSION}")
    for name, bone in record.bones.items():
        if bone.name != name:
            problems.append(f"{name}: keyed as {name} but named {bone.name}")
        parent = bone.rest.parent
        if parent and parent not in record.bones:
            problems.append(f"{name}: parent {parent!r} is not in the record")
        if bone.display.override and bone.display.override not in record.bones:
            problems.append(
                f"{name}: custom shape override {bone.display.override!r} is not in the record"
            )
    return problems
