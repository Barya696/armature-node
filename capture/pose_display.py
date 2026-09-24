"""Custom shape and its placement -> DisplayDef; pose settings -> PoseDef.

The shape is recorded by OBJECT NAME here. Its geometry goes to the widget
library (``capture/widgets.py``) separately, so a rig whose ``WGT-*`` objects
have been deleted can still be rebuilt exactly.

The pose *transform* is deliberately not captured. Locks, rotation mode and IK
limits describe how a control may be manipulated, so they are rig definition;
where the control currently sits is animation, and the graph owns it.
"""

from .. import compat
from ..model.types import DisplayDef, PoseDef

__all__ = ["capture_display", "capture_pose"]


def capture_display(bone, pose_bone):
    if pose_bone is None:
        return DisplayDef(
            show_wire=bool(bone.show_wire), color=compat.bone_color(bone)
        )
    shape = pose_bone.custom_shape
    override = getattr(pose_bone, "custom_shape_transform", None)
    return DisplayDef(
        shape=shape.name if shape is not None else "",
        preset="NONE",
        scale=tuple(pose_bone.custom_shape_scale_xyz),
        translation=tuple(pose_bone.custom_shape_translation),
        rotation=tuple(pose_bone.custom_shape_rotation_euler),
        wire_width=compat.wire_width(pose_bone),
        use_bone_size=bool(pose_bone.use_custom_shape_bone_size),
        # A pose-bone pointer, stored by bone name like every other reference.
        override=override.name if override is not None else "",
        show_wire=bool(bone.show_wire),
        color=compat.bone_color(bone),
    )


def capture_pose(pose_bone):
    if pose_bone is None:
        return PoseDef()
    locks = {
        "location": [bool(v) for v in pose_bone.lock_location],
        "rotation": [bool(v) for v in pose_bone.lock_rotation],
        "scale": [bool(v) for v in pose_bone.lock_scale],
        "rotation_w": bool(pose_bone.lock_rotation_w),
        "rotations_4d": bool(pose_bone.lock_rotations_4d),
    }
    ik = {"ik_stretch": float(pose_bone.ik_stretch)}
    for axis in "xyz":
        ik[f"lock_ik_{axis}"] = bool(getattr(pose_bone, f"lock_ik_{axis}"))
        ik[f"ik_stiffness_{axis}"] = float(getattr(pose_bone, f"ik_stiffness_{axis}"))
        ik[f"use_ik_limit_{axis}"] = bool(getattr(pose_bone, f"use_ik_limit_{axis}"))
        ik[f"ik_min_{axis}"] = float(getattr(pose_bone, f"ik_min_{axis}"))
        ik[f"ik_max_{axis}"] = float(getattr(pose_bone, f"ik_max_{axis}"))
    return PoseDef(rotation_mode=pose_bone.rotation_mode, locks=locks, ik=ik)
