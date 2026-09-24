"""Write custom shape, its placement, pose settings and the graph's pose."""

import bpy

from .. import compat
from .widgets import ensure_widget

__all__ = ["apply_display", "apply_pose", "pose_target", "pose_pass"]

_DISPLAY_TO_ATTR = {
    "scale": "custom_shape_scale_xyz",
    "translation": "custom_shape_translation",
    "rotation": "custom_shape_rotation_euler",
    "use_bone_size": "use_custom_shape_bone_size",
}


def apply_display(obj, bone, pbone, leaf, value, library, writer):
    """Write one ``display.*`` leaf."""
    if leaf == "shape":
        if not value:
            if pbone.custom_shape is not None:
                pbone.custom_shape = None
                writer.count()
            return
        widget = ensure_widget(value, library)
        if widget is None:
            # Cannot rebuild it: leave whatever the bone has. Clearing here is
            # exactly the v1 data loss.
            writer.note(f"widget {value!r} is missing and has no stored geometry")
            return
        if pbone.custom_shape is not widget:
            pbone.custom_shape = widget
            writer.count()
        return
    if leaf == "override":
        target = obj.pose.bones.get(value) if value else None
        if getattr(pbone, "custom_shape_transform", None) is not target:
            writer.set(pbone, "custom_shape_transform", target)
        return
    if leaf == "wire_width":
        writer.count(compat.set_wire_width(pbone, value))
        return
    if leaf == "show_wire":
        writer.set(bone, "show_wire", bool(value))
        return
    if leaf == "color":
        writer.count(compat.set_bone_color(bone, value))
        return
    if leaf == "preset":
        return  # generator hint, not a property of the rig
    attr = _DISPLAY_TO_ATTR.get(leaf)
    if attr is not None:
        writer.set(pbone, attr, value)


def apply_pose(pbone, leaf, value, writer):
    """Write one ``pose.*`` leaf: rotation mode, locks, IK settings."""
    if pbone is None:
        return
    if leaf == "rotation_mode":
        writer.set(pbone, "rotation_mode", value)
        return
    if leaf == "locks":
        for key, attr in (
            ("location", "lock_location"),
            ("rotation", "lock_rotation"),
            ("scale", "lock_scale"),
            ("rotation_w", "lock_rotation_w"),
            ("rotations_4d", "lock_rotations_4d"),
        ):
            if key in (value or {}):
                writer.set(pbone, attr, value[key])
        return
    if leaf == "ik":
        for key, sub in (value or {}).items():
            writer.set(pbone, key, sub)


def pose_target(obj, pbone, transform):
    """The armature-space matrix the graph asks for, or ``None`` for rest.

    Resolution, per component:

    * **Absolute** (``location`` / ``rotation`` / ``scale``) wins outright.
    * **Relative** (``offset``, ``rotation_offset``, ``local_*``) is applied to
      the bone's REST pose -- identity basis, following its parent's current
      pose. Never to the live pose: that already contains the last build's
      offset, so it would be added again on every build and the bone would
      drift away.
    * **Neither** leaves that component exactly as the bone has it, which is
      what lets a node move a bone without also pinning its rotation.

    ``local_*`` are the bone's own Location / Rotation channels (the N-panel
    values), so a local offset follows the bone's axes and its parent.
    """
    from mathutils import Euler, Matrix, Vector

    if transform.is_empty():
        return None

    # The pose the relative parts build on: rest, moved along the bone's own
    # axes when there is a local part.
    local_loc = Vector(transform.local_offset or (0.0, 0.0, 0.0))
    local_rot = Euler(tuple(transform.local_rotation or (0.0, 0.0, 0.0)), "XYZ")
    basis = Matrix.LocRotScale(local_loc, local_rot, Vector((1.0, 1.0, 1.0)))
    base = obj.convert_space(
        pose_bone=pbone, matrix=basis, from_space="LOCAL", to_space="POSE"
    )
    base_loc, base_rot, _base_scale = base.decompose()
    cur_loc, cur_rot, cur_scale = pbone.matrix.decompose()

    to_arm = obj.matrix_world.inverted_safe()
    arm_rot = to_arm.to_quaternion()

    relative_loc = transform.offset is not None or transform.local_offset is not None
    relative_rot = (
        transform.rotation_offset is not None or transform.local_rotation is not None
    )

    if transform.location is not None:
        loc = to_arm @ Vector(transform.location)
    else:
        loc = base_loc if relative_loc else cur_loc
    if transform.offset is not None:
        # A world-axis offset, expressed in armature space.
        loc = loc + arm_rot @ Vector(transform.offset)

    if transform.rotation is not None:
        rot = arm_rot @ Euler(tuple(transform.rotation), "XYZ").to_quaternion()
    else:
        rot = base_rot if relative_rot else cur_rot
    if transform.rotation_offset is not None:
        # Along world axes, about the bone's own head: rotate in place.
        world_offset = Euler(tuple(transform.rotation_offset), "XYZ").to_quaternion()
        rot = (arm_rot @ world_offset @ arm_rot.inverted()) @ rot

    scale = Vector(transform.scale) if transform.scale is not None else cur_scale
    return Matrix.LocRotScale(loc, rot, scale)


def pose_pass(obj, transforms, writer):
    """Pose every bone in ``transforms`` ({name: TransformDef}).

    Parent-first, with a depsgraph flush between depths. A child's matrix is
    resolved against its parent's EVALUATED pose, so writing a child before
    its parent has been evaluated places it relative to where the parent used
    to be -- which is how moving the whole rig left the children below the
    origin.

    Every write is verified once the depsgraph has settled. Blender accepts
    ``pbone.matrix = ...`` on a connected bone or a locked channel and then
    silently discards it; counting that as a write is how the addon looked
    broken with no error to show for it.
    """
    from mathutils import Matrix

    if not transforms or obj.pose is None:
        return
    view_layer = getattr(bpy.context, "view_layer", None)

    def depth(pbone):
        d, parent = 0, pbone.parent
        while parent is not None:
            d += 1
            parent = parent.parent
        return d

    items = []
    for name, transform in transforms.items():
        pbone = obj.pose.bones.get(name)
        if pbone is not None:
            items.append((depth(pbone), name, pbone, transform))
    items.sort(key=lambda item: (item[0], item[1]))

    written = []
    level = None
    for d, _name, pbone, transform in items:
        if level is not None and d != level and view_layer is not None:
            view_layer.update()
        level = d
        target = pose_target(obj, pbone, transform)
        if target is None:
            # Nothing is asked of this bone any more: back to rest.
            if not _same_matrix(pbone.matrix_basis, Matrix.Identity(4)):
                pbone.matrix_basis = Matrix.Identity(4)
                writer.count()
            continue
        if _same_matrix(pbone.matrix, target):
            continue  # already there: two identical builds write nothing
        try:
            pbone.matrix = target
        except (AttributeError, ValueError) as exc:
            writer.note(f"pose {pbone.name}: {exc}")
            continue
        writer.count()
        written.append((pbone, target))

    if view_layer is not None:
        view_layer.update()
    for pbone, target in written:
        try:
            stuck = _same_matrix(pbone.matrix, target, eps=1e-4)
        except ReferenceError:
            continue
        if not stuck:
            writer.writes -= 1
            writer.note(_why_blocked(pbone))


def _why_blocked(pbone):
    """A specific reason a pose write did not stick."""
    reasons = []
    if pbone.bone.use_connect:
        reasons.append("it is connected to its parent (head is pinned)")
    locked = [axis for axis, is_locked in zip("XYZ", pbone.lock_location) if is_locked]
    if locked:
        reasons.append("location " + "/".join(locked) + " locked")
    if all(pbone.lock_rotation) and pbone.lock_rotation_w:
        reasons.append("rotation locked")
    if all(pbone.lock_scale):
        reasons.append("scale locked")
    if not reasons:
        reasons.append("a constraint or driver is overriding it")
    return f"'{pbone.name}' did not move: " + ", ".join(reasons)


def _same_matrix(a, b, eps=1e-6):
    for row_a, row_b in zip(a, b):
        for x, y in zip(row_a, row_b):
            if abs(x - y) > eps:
                return False
    return True
