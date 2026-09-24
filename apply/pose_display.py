"""Write custom shape, its placement, pose settings and the graph's pose."""

from .. import compat
from .widgets import ensure_widget

__all__ = ["apply_display", "apply_pose", "apply_transform"]

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


def apply_transform(obj, pbone, values, writer):
    """Write the graph's world-space pose for one bone.

    Only the components the graph set are written; a component back at
    ``None`` is *cleared* to the rest pose, which is how removing a node undoes
    its posing. Scale falls back to 1 rather than the bone's current scale, so
    clearing is deterministic.
    """
    from mathutils import Euler, Matrix, Vector

    if pbone is None:
        return
    world = obj.matrix_world @ pbone.matrix
    cur_loc, cur_rot, cur_scale = world.decompose()

    loc = Vector(values["location"]) if values.get("location") else None
    rot = values.get("rotation")
    scale = values.get("scale")

    if "location" in values and loc is None:
        loc = None  # cleared: fall through to the rest position below
    target_loc = loc if loc is not None else cur_loc
    target_rot = (
        Euler(tuple(rot), "XYZ").to_matrix().to_4x4()
        if rot
        else cur_rot.to_matrix().to_4x4()
    )
    target_scale = Vector(scale) if scale else cur_scale

    # Anything the graph cleared goes back to the bone's rest state, which for
    # a pose bone means an identity basis.
    cleared = [k for k in ("location", "rotation", "scale") if k in values and not values[k]]
    if cleared and not any(values.get(k) for k in ("location", "rotation", "scale")):
        if not _same_matrix(pbone.matrix_basis, Matrix.Identity(4)):
            pbone.matrix_basis = Matrix.Identity(4)
            writer.count()
        return

    target = (
        Matrix.Translation(target_loc)
        @ target_rot
        @ Matrix.Diagonal(target_scale).to_4x4()
    )
    new_basis_world = obj.matrix_world.inverted_safe() @ target
    if _same_matrix(pbone.matrix, new_basis_world):
        return
    try:
        pbone.matrix = new_basis_world
    except (AttributeError, ValueError) as exc:
        writer.note(f"pose {pbone.name}: {exc}")
        return
    writer.count()
    # Blender accepts the assignment and then silently discards it when the
    # bone cannot move: a connected bone's head is pinned to its parent's
    # tail, and locked channels are not writable. Queue it for checking once
    # the depsgraph has re-evaluated -- reporting a write that did nothing is
    # worse than not writing, because the user sees no error and no motion.
    writer.pending_poses.append((pbone, new_basis_world))


def verify_poses(writer):
    """Check queued poses after the view layer has been updated."""
    for pbone, intended in writer.pending_poses:
        try:
            stuck = _same_matrix(pbone.matrix, intended)
        except ReferenceError:
            continue  # bone went away mid-build; nothing useful to report
        if not stuck:
            writer.writes -= 1
            writer.note(_why_blocked(pbone))
    writer.pending_poses = []


def _why_blocked(pbone):
    """A specific reason a pose write did not stick."""
    reasons = []
    if pbone.bone.use_connect:
        reasons.append("it is connected to its parent (head is pinned)")
    locked = [
        axis
        for axis, is_locked in zip("XYZ", pbone.lock_location)
        if is_locked
    ]
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
