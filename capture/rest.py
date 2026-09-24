"""Bone rest geometry -> RestDef. Reads ``Bone``, never ``EditBone``.

Capturing from ``Bone`` rather than ``EditBone`` means the armature does not
have to be dropped into Edit mode to be recorded, which matters because Bind
runs on whatever the user has selected. ``roll`` is the one value ``Bone``
does not expose, so it is recovered from the rest matrix.
"""

import bpy

from ..model.types import RestDef

__all__ = ["bone_roll", "capture_rest"]


def bone_roll(bone):
    """Roll of a (non-edit) Bone, recovered from its rest matrix."""
    try:
        return float(bpy.types.Bone.AxisRollFromMatrix(bone.matrix_local.to_3x3())[1])
    except Exception:  # noqa: BLE001 - RNA shape varies across versions
        return 0.0


def capture_rest(bone):
    return RestDef(
        head=tuple(bone.head_local),
        tail=tuple(bone.tail_local),
        roll=bone_roll(bone),
        parent=bone.parent.name if bone.parent else None,
        connect=bool(bone.use_connect),
        deform=bool(bone.use_deform),
        inherit_scale=getattr(bone, "inherit_scale", "FULL"),
        use_local_location=bool(getattr(bone, "use_local_location", True)),
        envelope_distance=float(bone.envelope_distance),
        envelope_weight=float(bone.envelope_weight),
        head_radius=float(bone.head_radius),
        tail_radius=float(bone.tail_radius),
    )
