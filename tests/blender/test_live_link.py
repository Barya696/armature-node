"""Nodes mirror the bone they pose, both ways, live.

"Grab" here means what the user does in Pose mode: the bone's pose changes
behind the graph's back. The node must pick it up -- both on the sync tick
that runs on every depsgraph update, and at the start of any build, so a
build never snaps a grabbed bone back.

The trap is the node's own write. Read that back and the node chases its own
output; on a constrained bone the evaluated pose differs from what was
written, so it would oscillate. The snapshot taken after every build is what
tells the two apart, and several tests here exist to prove it holds.
"""

import math

import bpy
from mathutils import Euler, Vector

import fixtures

BUILDS = 4


def _fresh(node_type, keep_constraints=False):
    from armature_nodes.ops.bind import bind

    obj = fixtures.make_rig()
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    for eb in obj.data.edit_bones:
        eb.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in obj.pose.bones:
        pb.lock_location = (False, False, False)
        pb.lock_rotation = (False, False, False)
        if not keep_constraints:
            while pb.constraints:
                pb.constraints.remove(pb.constraints[0])
    bpy.context.view_layer.update()
    bind(obj)

    tree, src, out = fixtures.make_tree(obj)
    for link in list(tree.links):
        tree.links.remove(link)
    node = tree.nodes.new(node_type)
    # The Bone node takes the rig on "Parent"; the others on "Rig".
    tree.links.new(src.outputs["Rig"], node.inputs[getattr(node, "stream_input", "Rig")])
    tree.links.new(node.outputs["Rig"], out.inputs["Rig"])
    # Let the sync tick find this tree, as it would for the active rig.
    obj.armature_nodes_tree = tree
    return obj, tree, src, out, node


def _build(tree, times=1):
    from armature_nodes.build import build_armature_from_tree

    for _ in range(times):
        build_armature_from_tree(tree)


def _tick():
    """One depsgraph-update sync, as Blender runs after every user edit."""
    from armature_nodes import sync

    bpy.context.view_layer.update()
    sync.sync_bone_nodes()


def _head(obj, name="bone.002"):
    return (obj.matrix_world @ obj.pose.bones[name].matrix).to_translation()


def _grab(obj, world_loc, name="bone.002"):
    """Move the bone the way G does: its pose changes, the graph is not told."""
    pb = obj.pose.bones[name]
    world = (obj.matrix_world @ pb.matrix).copy()
    world.translation = Vector(world_loc)
    pb.matrix = obj.matrix_world.inverted() @ world
    bpy.context.view_layer.update()


def _close(a, b, eps=1e-4):
    return (Vector(a) - Vector(b)).length < eps


def _assert_close(got, want, what):
    assert _close(got, want), f"{what}: got {tuple(round(v, 4) for v in got)}, want {tuple(want)}"


# --- Position -----------------------------------------------------------------


def test_position_set_follows_a_grab_on_the_tick():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    node.use_position = True
    _build(tree)

    _grab(obj, (1.0, 2.0, 3.0))
    _tick()
    _assert_close(node.inputs["Position"].default_value, (1.0, 2.0, 3.0), "Position after grab")


def test_a_build_after_a_grab_does_not_snap_back():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    node.use_position = True
    _build(tree)

    _grab(obj, (1.0, 2.0, 3.0))  # no tick in between: the build itself absorbs it
    _build(tree, BUILDS)
    _assert_close(_head(obj), (1.0, 2.0, 3.0), "bone after builds")
    _assert_close(node.inputs["Position"].default_value, (1.0, 2.0, 3.0), "Position")


def test_editing_position_still_posts_to_the_rig():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    node.use_position = True
    _build(tree)
    _tick()
    node.inputs["Position"].default_value = (4.0, 0.0, 1.0)
    _build(tree)
    _tick()
    _assert_close(_head(obj), (4.0, 0.0, 1.0), "bone after typing")
    _assert_close(node.inputs["Position"].default_value, (4.0, 0.0, 1.0), "field kept")


def test_typing_takes_the_bone_even_with_set_off():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    _build(tree)
    node.inputs["Position"].default_value = (2.0, 2.0, 2.0)
    assert node.use_position, "typing a position must take the bone over"
    _build(tree)
    _assert_close(_head(obj), (2.0, 2.0, 2.0), "bone")


def test_position_is_a_readout_when_not_driving():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    _build(tree)
    _grab(obj, (0.5, 0.5, 0.5))
    _tick()
    _assert_close(node.inputs["Position"].default_value, (0.5, 0.5, 0.5), "readout")
    assert not node.use_position, "a readout must not take the bone over"
    _build(tree)
    _assert_close(_head(obj), (0.5, 0.5, 0.5), "undriven bone left alone")


def test_offset_absorbs_a_grab_but_not_a_parent_move():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    node.inputs["Offset"].default_value = (1.0, 0.0, 0.0)
    _build(tree)
    _tick()

    start = _head(obj)
    _grab(obj, start + Vector((0.0, 1.0, 0.0)))
    _tick()
    _assert_close(node.inputs["Offset"].default_value, (1.0, 1.0, 0.0), "grab folded into Offset")

    # Moving the parent carries the child, but its offset from rest is unchanged.
    parent = obj.pose.bones["bone.001"]
    parent.location = (0.0, 0.0, 2.0)
    _tick()
    _assert_close(node.inputs["Offset"].default_value, (1.0, 1.0, 0.0), "Offset after parent move")


def test_grabbing_the_bone_moves_a_wired_marker():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    marker_node = tree.nodes.new("ArmatureNodesMarkerNode")
    marker = marker_node.markers[0]
    marker.set_position((0.0, 0.0, 1.0))
    tree.links.new(marker_node.outputs[0], node.inputs["Position"])
    _build(tree)
    _assert_close(_head(obj), (0.0, 0.0, 1.0), "bone on marker")

    _grab(obj, (2.0, 0.0, 1.0))
    _tick()
    _assert_close(marker.position, (2.0, 0.0, 1.0), "marker followed the bone")
    _build(tree)
    _assert_close(_head(obj), (2.0, 0.0, 1.0), "bone stays where it was grabbed")


def test_constrained_bone_does_not_oscillate():
    """The trap: the evaluated pose differs from what was written."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode", keep_constraints=True)
    node.bone = "bone.003"  # 50% Copy Transforms to a still empty
    node.use_position = True
    node.inputs["Position"].default_value = (2.0, 0.0, 2.0)
    _build(tree)
    _tick()
    settled = tuple(node.inputs["Position"].default_value)
    head = _head(obj, "bone.003")
    for _ in range(BUILDS):
        _build(tree)
        _tick()
    _assert_close(node.inputs["Position"].default_value, settled, "field drifted")
    _assert_close(_head(obj, "bone.003"), head, "bone drifted")


def test_undo_does_not_double_apply():
    """Snapshots are not in the undo stack, so undo must clear them."""
    from armature_nodes import livelink

    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    node.use_position = True
    _build(tree)
    livelink.reset()  # what the undo_post handler does
    _tick()  # first look after the reset: remembers, folds nothing in
    before = tuple(node.inputs["Position"].default_value)
    _tick()
    _assert_close(node.inputs["Position"].default_value, before, "a reset produced a move")


# --- Rotation -----------------------------------------------------------------


def test_rotation_set_follows_a_turn():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesRotationNode")
    node.bone = "bone.002"
    node.use_rotation = True
    _build(tree)

    pb = obj.pose.bones["bone.002"]
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (math.radians(30.0), 0.0, 0.0)
    _tick()
    want = (obj.matrix_world @ pb.matrix).to_euler("XYZ")
    _assert_close(node.inputs["Rotation"].default_value, (want.x, want.y, want.z), "Rotation")
    _build(tree, BUILDS)
    now = (obj.matrix_world @ pb.matrix).to_euler("XYZ")
    _assert_close((now.x, now.y, now.z), (want.x, want.y, want.z), "turn was snapped back")


# --- Transform ----------------------------------------------------------------


def test_transform_world_picks_up_a_grab():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    _build(tree)
    rest = _head(obj)

    _grab(obj, rest + Vector((1.0, 0.0, 0.0)))
    _tick()
    _assert_close(node.inputs["Translation"].default_value, (1.0, 0.0, 0.0), "Translation")
    _build(tree, BUILDS)
    _assert_close(_head(obj), rest + Vector((1.0, 0.0, 0.0)), "bone kept")

    _grab(obj, rest + Vector((1.0, 2.0, 0.0)))
    _tick()
    _assert_close(node.inputs["Translation"].default_value, (1.0, 2.0, 0.0), "second grab")


def test_transform_does_not_jump_a_hand_posed_bone():
    """A bone posed before the node existed must not snap back on first grab."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    obj.pose.bones["bone.002"].location = (0.0, 0.0, 0.7)  # hand-posed earlier
    bpy.context.view_layer.update()
    node.bone = "bone.002"
    _build(tree)
    _tick()
    posed = _head(obj)

    _grab(obj, posed + Vector((0.5, 0.0, 0.0)))
    _tick()
    _build(tree, BUILDS)
    _assert_close(_head(obj), posed + Vector((0.5, 0.0, 0.0)), "bone jumped")


def test_transform_local_mirrors_the_location_channel():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    node.space = "LOCAL"
    _build(tree)
    obj.pose.bones["bone.002"].location = (0.0, 0.3, 0.0)
    _tick()
    _assert_close(node.inputs["Translation"].default_value, (0.0, 0.3, 0.0), "Translation")


# --- Bone -------------------------------------------------------------------


def test_bone_node_ticked_location_follows_a_grab():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesBoneNode")
    node.bone = "bone.002"
    node.use_location = True
    _build(tree)

    _grab(obj, (1.5, 0.0, 1.5))
    _tick()
    _assert_close(node.inputs["Position"].default_value, (1.5, 0.0, 1.5), "Position")
    _build(tree, BUILDS)
    _assert_close(_head(obj), (1.5, 0.0, 1.5), "grab snapped back")
