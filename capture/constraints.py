"""Generic constraint serialisation -> ConstraintDef.

Every writable RNA property, by name. Generic on purpose: the v1
implementation serialised a hard-coded list of eight attributes, so every
other setting on every constraint was silently dropped and could never be
restored. A rig is not a fixed set of constraint types, and a capture that
only knows eight fields is wrong the first time someone uses a ninth.

ID pointers are stored as names. A record holding live pointers would keep
deleted datablocks alive and break the moment the file is reopened.
"""

from ..model.types import ConstraintDef

#: UI state and derived values. Recording these produces diffs that mean
#: nothing -- ``active`` flips when the user clicks a constraint panel, and
#: ``is_valid`` changes when an unrelated object is renamed.
SKIP = frozenset(
    {
        "active",
        "show_expanded",
        "is_valid",
        "is_override_data",
        "error_location",
        "error_rotation",
        "rna_type",
        "type",
        "name",
    }
)

__all__ = ["SKIP", "capture_constraint", "capture_constraints"]


def _value(constraint, prop):
    """One RNA property as plain JSON-able data."""
    raw = getattr(constraint, prop.identifier)
    if prop.type == "POINTER":
        return raw.name if raw is not None else None
    if prop.type == "COLLECTION":
        return None  # nothing a constraint exposes here belongs in a record
    if getattr(prop, "is_array", False):
        if prop.type == "BOOLEAN":
            return [bool(v) for v in raw]
        if prop.type == "INT":
            return [int(v) for v in raw]
        return [float(v) for v in raw]
    if prop.type == "BOOLEAN":
        return bool(raw)
    if prop.type == "FLOAT":
        return float(raw)
    if prop.type == "INT":
        return int(raw)
    return raw  # STRING / ENUM are already plain


def capture_constraint(constraint):
    props = {}
    for prop in constraint.bl_rna.properties:
        if prop.is_readonly or prop.identifier in SKIP:
            continue
        try:
            value = _value(constraint, prop)
        except (AttributeError, TypeError, ValueError):
            # A property this build exposes but cannot read is not worth
            # failing a whole capture over.
            continue
        if prop.type == "COLLECTION" and value is None:
            continue
        props[prop.identifier] = value
    return ConstraintDef(type=constraint.type, name=constraint.name, props=props)


def capture_constraints(pose_bone):
    if pose_bone is None:
        return ()
    return tuple(capture_constraint(c) for c in pose_bone.constraints)
