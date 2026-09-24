"""restore -> apply -> record touched. The only writer in the addon.

Every build is::

    base     = store.record.read(obj)        # the rig as it was
    target   = graph.evaluate(tree) or base  # what the graph asks for
    changes  = model.diff(base, target)
    pipeline.apply(obj, base, target, store.touched.read(obj))

and the pipeline does three things in order:

1. **Restore** every field the *last* build wrote that this one is not
   writing, back to its value in the record.
2. **Apply** this build's changes.
3. **Record** what it wrote, for the next build to restore.

Step 1 is why the viewport converges on ``record + graph`` no matter what the
user does. Deleting a node, unplugging the Input, or emptying the tree all
leave paths in the touched set that the next build puts back --- so there is no
"do nothing when the stream is empty" special case, and cannot be one.

Nothing else in the addon may assign to a Bone, EditBone, PoseBone,
Constraint or Armature property. That is enforced by a boundary test, not by
convention.
"""

from dataclasses import dataclass, field

import bpy

from ..model import diff as model_diff
from ..store import lock
from ..store import record as record_store
from ..store import touched as touched_store
from ..store import widgets_lib
from . import collections as _collections
from . import constraints as _constraints
from . import rest as _rest
from .pose_display import apply_display, apply_pose, pose_pass
from .writer import Writer

__all__ = ["Result", "apply"]


@dataclass
class Result:
    """What one build did, for the Output node to report."""

    writes: int = 0
    restored: int = 0
    bones: tuple = ()
    errors: list = field(default_factory=list)
    skipped: tuple = ()

    def __bool__(self):
        return not self.errors


def _values_by_section(paths_and_values):
    """Group ``{path: value}`` into ``{section: {leaf: value}}``."""
    out = {}
    for path, value in paths_and_values.items():
        section, _, leaf = path.partition(".")
        out.setdefault(section or path, {})[leaf or path] = value
    return out


def _restore_values(base, previously, changes):
    """``{bone: {path: recorded_value}}`` for everything going back."""
    stale = model_diff.restore_paths(previously, changes)
    out = {}
    for name, paths in stale.items():
        bone = base.bones.get(name)
        if bone is None:
            continue
        out[name] = {p: model_diff.value_at(bone, p) for p in paths}
    return out


def _merge(restore, changes):
    """One write list per bone: restores first, then this build's changes."""
    merged = {}
    for name, values in restore.items():
        merged.setdefault(name, {}).update(values)
    for name, values in changes.bones.items():
        merged.setdefault(name, {}).update(values)
    return merged


def apply(obj, base, target, previously_touched=None, library=None):
    """Bring ``obj`` to ``base`` + the changes that make it ``target``."""
    if obj is None or obj.type != "ARMATURE":
        return Result(errors=["apply needs an armature object"])

    changes = model_diff.diff(base, target)
    restore = _restore_values(base, previously_touched or {}, changes)
    plan = _merge(restore, changes)
    writer = Writer()
    result = Result(
        restored=sum(len(v) for v in restore.values()),
        skipped=tuple(changes.added) + tuple(changes.removed),
    )
    if not plan:
        touched_store.write(obj, changes)
        return result

    if library is None:
        library = widgets_lib.read(obj)

    with lock.held("build"):
        posed = _apply_plan(obj, plan, library, writer)
        # Pose last: the edit pass can change rest geometry, and relative
        # offsets are resolved against rest. Each bone gets its WHOLE target
        # transform rather than only the leaves that changed -- clearing one
        # component must not reset the others.
        if posed:
            pose_pass(
                obj,
                {n: target.bones[n].transform for n in posed if n in target.bones},
                writer,
            )

    # The record is READ-ONLY here. A build that rewrote it would fold its own
    # output into the original, which is the whole class of bug this replaces.
    touched_store.write(obj, changes)
    result.writes = writer.writes
    result.bones = tuple(sorted(plan))
    result.errors = writer.errors
    return result


def _apply_plan(obj, plan, library, writer):
    """Everything except posing. Returns the bones whose pose must be set."""
    armature = obj.data
    edit_plan = {}
    posed = []

    # --- Object-mode pass ----------------------------------------------------
    for name, values in plan.items():
        bone = armature.bones.get(name)
        if bone is None:
            writer.note(f"rig has no bone {name!r}")
            continue
        pbone = obj.pose.bones.get(name)
        sections = _values_by_section(values)
        writer.touch(name)

        edit = {
            leaf: v
            for leaf, v in sections.get("rest", {}).items()
            if f"rest.{leaf}" in _rest.EDIT_PATHS
        }
        if edit:
            edit_plan[name] = edit
        flags = {
            leaf: v
            for leaf, v in sections.get("rest", {}).items()
            if f"rest.{leaf}" in _rest.OBJECT_PATHS
        }
        if flags:
            _rest.apply_flags(bone, flags, writer)

        for leaf, value in sections.get("membership", {}).items():
            _collections.apply_membership(armature, bone, leaf, value, writer)
        for leaf, value in sections.get("display", {}).items():
            if pbone is not None:
                apply_display(obj, bone, pbone, leaf, value, library, writer)
        for leaf, value in sections.get("pose", {}).items():
            apply_pose(pbone, leaf, value, writer)
        if "constraints" in values:
            _constraints.apply_constraints(pbone, values["constraints"], writer)
        if sections.get("transform"):
            posed.append(name)

    # --- Edit-mode pass ------------------------------------------------------
    # Entered only when rest geometry actually changed, so a build that just
    # swaps a widget never leaves Object mode.
    if edit_plan:
        _apply_edit(obj, edit_plan, writer)
    return posed


def _apply_edit(obj, edit_plan, writer):
    previous_mode = obj.mode
    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    view_layer.objects.active = obj
    try:
        bpy.ops.object.mode_set(mode="EDIT")
        edit_bones = obj.data.edit_bones
        for name, values in edit_plan.items():
            _rest.apply_edit(edit_bones, name, values, writer)
    except RuntimeError as exc:
        writer.note(f"edit-mode pass failed: {exc}")
    finally:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
            if previous_mode not in ("OBJECT", "EDIT"):
                bpy.ops.object.mode_set(mode=previous_mode)
        except RuntimeError:
            pass
        view_layer.objects.active = previous_active


def restore_original(obj):
    """Put the rig back to its record and forget what the graph wrote.

    The Restore Original operator. Applies the record to itself with every
    touched path marked stale, so all of it is rewritten from the record.
    """
    base = record_store.read(obj)
    if base is None:
        return Result(errors=["this rig is not bound"])
    previously = touched_store.read(obj)
    result = apply(obj, base, base, previously)
    touched_store.clear(obj)
    return result
