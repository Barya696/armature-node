"""Read a live armature into a :class:`RigRecord`. Never writes.

This package is the *only* thing that builds a record from a live rig, and it
is called from exactly two places: ``ops/bind.py`` (binding a rig) and
``ops/record.py`` (an explicit, confirmed re-capture). Never from evaluation,
never lazily, never "if the record is missing".

That restriction is the whole point. The v1 implementation captured implicitly
during node evaluation, so a rig the graph had already modified was recorded
as the original and saved into the .blend. A capture is only correct when the
caller can promise the rig is unmodified, and only an operator the user
invoked can make that promise -- which is why ``store.lock`` is checked here
rather than trusted to callers.
"""

import datetime

import bpy

from .. import compat
from ..model.types import ArmatureDef, BoneDef, RigRecord, SourceDef
from ..store import lock
from .collections import capture_membership
from .constraints import capture_constraints
from .pose_display import capture_display, capture_pose
from .rest import capture_rest
from .widgets import capture_widgets

class CaptureError(RuntimeError):
    """A rig could not be recorded."""


__all__ = [
    "CaptureError",
    "capture_record",
    "capture_widgets",
    "capture_all",
    "ordered_bones",
]


def ordered_bones(armature):
    """Bones parent-first.

    ``apply`` depends on it: a child cannot be reparented before its parent
    exists, and a record whose order is arbitrary makes that the applier's
    problem on every build. Blender does not guarantee a topological order, so
    it is imposed once, here.
    """
    seen = set()
    out = []

    def visit(bone):
        if bone.name in seen:
            return
        if bone.parent is not None:
            visit(bone.parent)
        seen.add(bone.name)
        out.append(bone)

    for bone in armature.bones:
        visit(bone)
    return out


def _source(obj):
    return SourceDef(
        object=obj.name,
        armature=obj.data.name,
        captured_at=datetime.datetime.now().isoformat(timespec="seconds"),
        blender=bpy.app.version_string,
    )


def _armature(obj):
    return ArmatureDef(
        display_type=obj.data.display_type,
        # show_in_front is on the OBJECT. bpy.types.Armature has no such
        # property -- probed against 5.2, not assumed.
        show_in_front=bool(getattr(obj, "show_in_front", False)),
        pose_position=obj.data.pose_position,
        collections=compat.armature_collection_names(obj.data),
    )


def capture_record(obj):
    """Serialise ``obj``'s rig into a record."""
    lock.guard("capture")
    if obj is None or getattr(obj, "type", "") != "ARMATURE":
        raise CaptureError("capture needs an armature object")

    pose = obj.pose
    bones = {}
    for bone in ordered_bones(obj.data):
        pose_bone = pose.bones.get(bone.name) if pose else None
        bones[bone.name] = BoneDef(
            name=bone.name,
            rest=capture_rest(bone),
            membership=capture_membership(bone),
            display=capture_display(bone, pose_bone),
            pose=capture_pose(pose_bone),
            constraints=capture_constraints(pose_bone),
        )
    return RigRecord(
        version=2, source=_source(obj), armature=_armature(obj), bones=bones
    )


def capture_all(obj):
    """``(record, widget_library)`` for ``obj``."""
    lock.guard("capture")
    return capture_record(obj), capture_widgets(obj)
