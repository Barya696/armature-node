"""Pure rig transforms — the operations the nodes are built from.

Every function here takes a :class:`RigRecord` and returns a new one. Nothing
mutates, nothing touches ``bpy``, so a node is a one-line wrapper around a
function that can be tested without Blender.

Selection is by name so that a node can say "hand.L" or "*_fk.?" without
knowing anything about the rig it will be run against.
"""

import fnmatch
from dataclasses import replace

from .types import DisplayDef, RigRecord, TransformDef

__all__ = [
    "select",
    "select_names",
    "set_shape",
    "clear_shape",
    "set_pose",
    "offset_pose",
    "clear_pose",
    "set_deform",
    "mirror_names",
    "map_bones",
]


# --- selection -------------------------------------------------------------


def select_names(record, pattern=""):
    """Bone names matching ``pattern``.

    Empty selects everything -- the Geometry Nodes convention, where no
    selection means the whole stream. Semicolons separate alternatives, and
    each alternative is an fnmatch glob, so ``hand.?;*_fk.L`` works.
    """
    if not pattern or not pattern.strip():
        return list(record.bones)
    parts = [p.strip() for p in pattern.split(";") if p.strip()]
    out = []
    for name in record.bones:
        if any(fnmatch.fnmatchcase(name, p) for p in parts):
            out.append(name)
    return out


def select(record, pattern=""):
    """The :class:`BoneDef` objects matching ``pattern``."""
    return [record.bones[n] for n in select_names(record, pattern)]


# --- generic rewrite -------------------------------------------------------


def map_bones(record, names, fn):
    """A new record with ``fn(bone)`` applied to each of ``names``.

    The single primitive every other operation here is built on: it keeps the
    "never mutate, always rebuild" rule in one place, and preserves bone order
    because capture stores them parent-first and apply relies on that.
    """
    wanted = set(names)
    if not wanted:
        return record
    bones = {}
    for name, bone in record.bones.items():
        bones[name] = fn(bone) if name in wanted else bone
    return replace(record, bones=bones)


# --- display ---------------------------------------------------------------


def set_shape(record, names, shape=None, preset=None, scale=None, translation=None,
               rotation=None, wire_width=None, use_bone_size=None, show_wire=None,
               override=None):
    """Assign a custom shape. Only the arguments given are changed.

    ``None`` means "leave it as the record has it" rather than "clear it", so
    a node that only sets a widget name does not silently reset that widget's
    scale to the dataclass default.
    """
    changes = {
        k: v
        for k, v in (
            ("shape", shape),
            ("preset", preset),
            ("scale", tuple(scale) if scale is not None else None),
            ("translation", tuple(translation) if translation is not None else None),
            ("rotation", tuple(rotation) if rotation is not None else None),
            ("wire_width", wire_width),
            ("use_bone_size", use_bone_size),
            ("show_wire", show_wire),
            ("override", override),
        )
        if v is not None
    }
    if not changes:
        return record
    return map_bones(record, names, lambda b: replace(b, display=replace(b.display, **changes)))


def clear_shape(record, names):
    """Remove the custom shape, leaving its placement values alone."""
    return map_bones(
        record, names, lambda b: replace(b, display=replace(b.display, shape="", preset="NONE"))
    )


# --- pose ------------------------------------------------------------------


def set_pose(record, names, location=None, rotation=None, scale=None):
    """Set the world-space pose the graph asks for.

    A component left at ``None`` is not set by this call, which is what lets a
    node move a bone without also pinning its rotation.
    """
    changes = {
        k: tuple(v)
        for k, v in (("location", location), ("rotation", rotation), ("scale", scale))
        if v is not None
    }
    if not changes:
        return record
    return map_bones(
        record, names, lambda b: replace(b, transform=replace(b.transform, **changes))
    )


def offset_pose(record, names, location=None, rotation=None, local=False):
    """Move bones relative to their rest pose.

    Offsets accumulate, so several Transform nodes in a row stack instead of
    overwriting each other. They are stored as offsets -- not folded into an
    absolute location -- because the rest pose they are relative to is only
    known at apply time, and resolving against the live pose would add the
    offset again on every build.
    """
    loc_key = "local_offset" if local else "offset"
    rot_key = "local_rotation" if local else "rotation_offset"

    def bump(b):
        t = b.transform
        changes = {}
        if location is not None:
            base = getattr(t, loc_key) or (0.0, 0.0, 0.0)
            changes[loc_key] = tuple(a + o for a, o in zip(base, location))
        if rotation is not None:
            base = getattr(t, rot_key) or (0.0, 0.0, 0.0)
            changes[rot_key] = tuple(a + o for a, o in zip(base, rotation))
        return replace(b, transform=replace(t, **changes))

    if location is None and rotation is None:
        return record
    return map_bones(record, names, bump)


def clear_pose(record, names):
    """Drop any graph-set pose, returning those bones to the recorded state."""
    return map_bones(record, names, lambda b: replace(b, transform=TransformDef()))


# --- rest ------------------------------------------------------------------


def set_deform(record, names, deform):
    """Turn Deform on or off — what "these are controls, not deformers" means."""
    return map_bones(record, names, lambda b: replace(b, rest=replace(b.rest, deform=deform)))


# --- naming ----------------------------------------------------------------

_SIDE_FLIP = (
    (".L", ".R"), (".l", ".r"), ("_L", "_R"), ("_l", "_r"),
    (".Left", ".Right"), ("_left", "_right"),
)


def mirror_names(name):
    """The opposite-side name, or ``None`` when the name has no side.

    Suffix only. Blender's own mirroring also understands prefixes and infixes,
    but a rig that uses those is rare enough that guessing wrong is worse than
    declining to guess.
    """
    for a, b in _SIDE_FLIP:
        if name.endswith(a):
            return name[: -len(a)] + b
        if name.endswith(b):
            return name[: -len(b)] + a
    return None
