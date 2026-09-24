"""A build must not drag the user out of Pose mode.

The reported bug: entering Pose mode bounced straight back to Object. Every
build called ``_ensure_object_mode()`` unconditionally, and ``mode_set`` is an
operator -- so each flip emitted a depsgraph update, which scheduled another
build, which flipped again. Most builds write nothing at all, so the whole
dance was for a no-op.
"""

import bpy

import fixtures


def _bind_and_tree():
    from armature_nodes.ops.bind import bind

    obj = fixtures.make_rig()
    bind(obj)
    tree, src, out = fixtures.make_tree(obj)
    return obj, tree, src, out


def _enter_pose(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="POSE")
    assert obj.mode == "POSE", "fixture could not enter Pose mode"


def test_build_leaves_pose_mode_alone():
    from armature_nodes.build import build_armature_from_tree

    obj, tree, _src, _out = _bind_and_tree()
    _enter_pose(obj)
    build_armature_from_tree(tree)
    assert obj.mode == "POSE", "the build kicked the user out of Pose mode"


def test_repeated_builds_stay_in_pose_mode():
    """The loop: each flip scheduled another build, which flipped again."""
    from armature_nodes.build import build_armature_from_tree

    obj, tree, _src, _out = _bind_and_tree()
    _enter_pose(obj)
    for i in range(5):
        build_armature_from_tree(tree)
        assert obj.mode == "POSE", f"build {i} left Pose mode"


def test_build_in_pose_mode_still_applies():
    """Skipping the mode switch must not skip the work."""
    from armature_nodes.apply import pipeline
    from armature_nodes.model import ops
    from armature_nodes.store import record as record_store
    from armature_nodes.store import touched as touched_store

    obj, _tree, _src, _out = _bind_and_tree()
    _enter_pose(obj)

    base = record_store.read(obj)
    target = ops.set_deform(base, ["bone.001"], not base.bones["bone.001"].rest.deform)
    result = pipeline.apply(obj, base, target, touched_store.read(obj))

    assert obj.mode == "POSE", "applying changed the mode"
    assert result.writes == 1, f"expected one write, got {result.writes}"
    assert obj.data.bones["bone.001"].use_deform == target.bones["bone.001"].rest.deform


def test_build_skips_while_in_edit_mode():
    """armature.bones is stale in Edit mode, so the build must wait."""
    from armature_nodes.build import build_armature_from_tree

    obj, tree, _src, _out = _bind_and_tree()
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    assert obj.mode == "EDIT"
    build_armature_from_tree(tree)
    assert obj.mode == "EDIT", "the build yanked the user out of Edit mode"
    bpy.ops.object.mode_set(mode="OBJECT")


def test_leaving_edit_mode_reschedules_the_build():
    from armature_nodes import sync

    obj, tree, _src, _out = _bind_and_tree()
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode="EDIT")
    sync.watch_armature_mode()  # records EDIT
    bpy.ops.object.mode_set(mode="OBJECT")
    tree.is_dirty = False
    sync.watch_armature_mode()  # notices the change
    assert tree.is_dirty, "leaving Edit mode did not reschedule the build"


# --- Full Rig mode ----------------------------------------------------------
#
# The Modify fix did not cover this path. Full Rig rebuilds edit bones through
# apply/legacy_full.py, which enters EDIT and POSE unconditionally -- so every
# depsgraph tick ran a whole mode cycle, and each mode_set emitted another
# depsgraph update that scheduled another build. That loop is what snapped
# Pose mode straight back to Object.


def _full_tree():
    obj, tree, src, out = _bind_and_tree()
    out.mode = "FULL"
    out.armature_name = obj.name
    return obj, tree, out


def test_full_rig_skips_when_nothing_changed():
    from armature_nodes.build import build_armature_from_tree

    obj, tree, _out = _full_tree()
    build_armature_from_tree(tree)  # first build does the work
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="POSE")
    assert obj.mode == "POSE"

    for i in range(5):
        build_armature_from_tree(tree)
        assert obj.mode == "POSE", f"Full Rig build {i} left Pose mode"


def test_full_rig_defers_while_posing():
    """A rebuild that IS needed still must not interrupt the user."""
    from armature_nodes.build import build_armature_from_tree

    obj, tree, _out = _full_tree()
    build_armature_from_tree(tree)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="POSE")

    # Make the graph produce something different, so a rebuild is warranted.
    bone_node = tree.nodes.new("ArmatureNodesBoneNode")
    bone_node.bone = "bone.001"
    build_armature_from_tree(tree)
    assert obj.mode == "POSE", "a needed rebuild still yanked the user"


def test_leaving_pose_mode_reschedules_the_build():
    from armature_nodes import sync

    obj, tree, _out = _full_tree()
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="POSE")
    sync.watch_armature_mode()  # records POSE
    bpy.ops.object.mode_set(mode="OBJECT")
    tree.is_dirty = False
    sync.watch_armature_mode()
    assert tree.is_dirty, "leaving Pose mode did not reschedule the deferred build"
