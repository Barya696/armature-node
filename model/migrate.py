"""Upgrade stored records to the current version.

Migration is **never lossy**. Every field a v1 record carries is copied across
byte-for-byte; fields v2 added are filled with type defaults and nothing else.
In particular a migration never consults the live armature: by the time it
runs, the rig may already have been modified by a graph, so "read the missing
data off the object" would quietly record modifications as the original.

That is the same mistake the v1 implementation made with its implicit
re-capture, and it is why the v1 records being migrated here may already be
wrong. Migration cannot fix that -- it can only refuse to make it worse.
"""

from .schema import RECORD_VERSION, RecordError, record_from_dict
from .types import (
    BoneDef,
    ConstraintDef,
    DisplayDef,
    MembershipDef,
    PoseDef,
    RestDef,
    RigRecord,
    SourceDef,
    TransformDef,
)

#: Custom property the v1 implementation wrote.
V1_KEY = "an_baseline"

__all__ = ["V1_KEY", "needs_migration", "migrate_dict", "migrate_v1_bones"]


def needs_migration(version):
    """True when a record of this version can be upgraded.

    ``None`` (unreadable) and versions newer than this build are both refused:
    a newer record is not something an older reader should guess at.
    """
    return isinstance(version, int) and 0 < version < RECORD_VERSION


def _v1_bone(d):
    """One v1 bone dict -> :class:`BoneDef`.

    v1 stored a flat bone with an optional ``shape`` sub-dict and constraints
    whose ``params`` came from a hard-coded attribute list. Everything it had
    is carried over; everything it lacked (collections, colours, pose settings,
    rest radii, inherit scale, full constraint props) takes the type default.
    """
    shape = d.get("shape") or {}
    display = DisplayDef(
        shape=shape.get("widget", "") or "",
        # v1's generator preset has no other home in v2; dropping it would
        # lose the only record of how a missing widget should be rebuilt.
        preset=shape.get("preset", "NONE") or "NONE",
        scale=tuple(shape.get("scale", (1.0, 1.0, 1.0))),
        translation=tuple(shape.get("translation", (0.0, 0.0, 0.0))),
        rotation=tuple(shape.get("rotation", (0.0, 0.0, 0.0))),
        wire_width=float(shape.get("wire_width", 1.0)),
        use_bone_size=bool(shape.get("scale_to_bone_length", True)),
        show_wire=bool(shape.get("show_wire", False)),
    ) if shape else DisplayDef()

    return BoneDef(
        name=d["name"],
        rest=RestDef(
            head=tuple(float(x) for x in d.get("head", (0.0, 0.0, 0.0))),
            tail=tuple(float(x) for x in d.get("tail", (0.0, 0.0, 1.0))),
            roll=float(d.get("roll", 0.0)),
            parent=d.get("parent"),
            connect=bool(d.get("use_connect", False)),
            deform=bool(d.get("use_deform", True)),
            envelope_distance=float(d.get("envelope_distance", 0.25)),
            envelope_weight=float(d.get("envelope_weight", 1.0)),
        ),
        membership=MembershipDef(),
        display=display,
        pose=PoseDef(),
        transform=TransformDef(),
        constraints=tuple(
            ConstraintDef(
                type=c.get("type", ""),
                name=c.get("name", ""),
                props=dict(c.get("params") or {}),
            )
            for c in (d.get("constraints") or ())
        ),
    )


def migrate_v1_bones(bones, source=None):
    """A v1 ``bones`` list -> a v2 :class:`RigRecord`."""
    out = {}
    for d in bones or ():
        name = d.get("name")
        if not name:
            continue
        out[name] = _v1_bone(d)
    return RigRecord(
        version=RECORD_VERSION,
        source=source or SourceDef(),
        bones=out,
    )


def migrate_dict(data, source=None):
    """Upgrade a parsed record dict to the current version.

    Returns a :class:`RigRecord`. Raises :class:`RecordError` when the version
    is unknown, so a caller can tell "too new to read" from "upgraded fine".
    """
    if not isinstance(data, dict):
        raise RecordError("record is not an object")
    version = data.get("version")
    if version == RECORD_VERSION:
        return record_from_dict(data)
    if version == 1:
        return migrate_v1_bones(data.get("bones"), source=source)
    raise RecordError(
        f"record version {version!r} cannot be migrated to {RECORD_VERSION}"
    )
