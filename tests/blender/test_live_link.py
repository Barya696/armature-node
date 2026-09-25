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


def test_ticking_set_on_a_constrained_bone_does_not_move_it():
    """The seed is the pose before constraints: written back, it changes nothing."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode", keep_constraints=True)
    node.bone = "bone.003"
    _build(tree)
    _tick()
    before = _head(obj, "bone.003")
    node.use_position = True
    _build(tree, BUILDS)
    _assert_close(_head(obj, "bone.003"), before, "constrained bone moved")
    # The build above may write nothing at all: the target matches the pose.
    # A nudge forces a write, and a seed taken from the evaluated pose would
    # then drop the bone by half its height under the 50% constraint.
    node.inputs["Position"].default_value = Vector(node.inputs["Position"].default_value) + Vector(
        (0.2, 0.0, 0.0)
    )
    _build(tree)
    _assert_close(_head(obj, "bone.003"), before + Vector((0.1, 0.0, 0.0)), "nudged bone")


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


# --- Marker -------------------------------------------------------------------


def _wire_marker(tree, node, socket="Position", at=(5.0, 5.0, 5.0)):
    """A Marker node dropped away from the bone, then wired in."""
    marker_node = tree.nodes.new("ArmatureNodesMarkerNode")
    marker = marker_node.markers[0]
    marker.set_position(at)
    tree.links.new(marker_node.outputs[0], node.inputs[socket])
    return marker_node, marker


def _handle(marker_node, marker):
    """The marker's viewport empty, created the way the sync tick creates it."""
    from armature_nodes import sync
    from armature_nodes.primary_rig import find_marker_empties

    sync.sync_marker_handles()
    return find_marker_empties(marker_node)[marker.key]


def test_wiring_a_marker_puts_it_on_the_bone():
    """The marker takes the bone's location; the bone does not jump to it."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    before = _head(obj)
    marker_node, marker = _wire_marker(tree, node)
    _build(tree, BUILDS)
    _assert_close(_head(obj), before, "bone jumped to the marker")
    _assert_close(marker.position, before, "marker on the bone")
    _assert_close(_handle(marker_node, marker).location, before, "handle on the bone")


def test_a_wired_marker_follows_a_grab():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    marker_node, marker = _wire_marker(tree, node)
    _build(tree)
    handle = _handle(marker_node, marker)

    _grab(obj, (2.0, 0.0, 1.0))
    _tick()
    _assert_close(marker.position, (2.0, 0.0, 1.0), "marker followed the bone")
    _assert_close(handle.location, (2.0, 0.0, 1.0), "handle followed the bone")
    _build(tree, BUILDS)
    _assert_close(_head(obj), (2.0, 0.0, 1.0), "bone stays where it was grabbed")


def test_typing_the_marker_moves_the_bone():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    _marker_node, marker = _wire_marker(tree, node)
    _build(tree)
    _tick()
    marker.position = (1.0, 1.0, 1.0)  # through the update callback, as typing is
    _build(tree)
    _tick()
    _assert_close(_head(obj), (1.0, 1.0, 1.0), "bone after typing")
    _assert_close(marker.position, (1.0, 1.0, 1.0), "typed value kept")


def test_dragging_the_handle_moves_the_bone():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    marker_node, marker = _wire_marker(tree, node)
    _build(tree)
    handle = _handle(marker_node, marker)
    handle.location = (0.5, 0.0, 2.5)
    _handle(marker_node, marker)  # the tick reads the drag back
    _build(tree)
    _tick()
    _assert_close(_head(obj), (0.5, 0.0, 2.5), "bone after the drag")
    _assert_close(marker.position, (0.5, 0.0, 2.5), "marker after the drag")


def test_marker_is_a_readout_when_the_bone_node_does_not_drive():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesBoneNode")
    node.bone = "bone.002"
    node.use_location = False
    _marker_node, marker = _wire_marker(tree, node)
    _build(tree)
    _tick()
    _assert_close(marker.position, _head(obj), "marker on the bone")

    _grab(obj, (1.0, 0.0, 2.0))
    _tick()
    _assert_close(marker.position, (1.0, 0.0, 2.0), "marker follows the bone")
    _build(tree)
    _assert_close(_head(obj), (1.0, 0.0, 2.0), "undriven bone left alone")


def test_picking_another_bone_moves_the_marker_onto_it():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    _marker_node, marker = _wire_marker(tree, node)
    _build(tree)
    other = _head(obj, "bone.001")
    node.bone = "bone.001"
    _build(tree, BUILDS)
    _assert_close(_head(obj, "bone.001"), other, "new bone jumped to the marker")
    _assert_close(marker.position, other, "marker on the new bone")


def test_marker_rotation_comes_from_the_bone():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesRotationNode")
    pb = obj.pose.bones["bone.002"]
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (math.radians(20.0), 0.0, 0.0)
    bpy.context.view_layer.update()
    want = (obj.matrix_world @ pb.matrix).to_euler("XYZ")
    node.bone = "bone.002"
    _marker_node, marker = _wire_marker(tree, node, socket="Rotation")
    marker.set_rotation((1.0, 1.0, 1.0))  # somewhere else entirely
    _build(tree, BUILDS)
    got = (obj.matrix_world @ pb.matrix).to_euler("XYZ")
    _assert_close((got.x, got.y, got.z), (want.x, want.y, want.z), "bone turned to the marker")
    _assert_close(marker.rotation, (want.x, want.y, want.z), "marker rotation")


def test_marker_on_a_constrained_bone_does_not_move_it():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode", keep_constraints=True)
    node.bone = "bone.003"  # 50% Copy Transforms to a still empty
    _build(tree)
    before = _head(obj, "bone.003")
    _marker_node, marker = _wire_marker(tree, node)
    _build(tree)
    _tick()
    settled = tuple(marker.position)
    for _ in range(BUILDS):
        _build(tree)
        _tick()
    _assert_close(_head(obj, "bone.003"), before, "wiring moved the constrained bone")
    _assert_close(marker.position, settled, "marker drifted")
    # Force a write (see the Set test above): only the nudge may show.
    marker.position = Vector(marker.position) + Vector((0.2, 0.0, 0.0))
    _build(tree)
    _assert_close(_head(obj, "bone.003"), before + Vector((0.1, 0.0, 0.0)), "nudged bone")


def test_skeleton_landmark_is_not_pulled_onto_the_bone():
    """Landmarks are a layout to drag onto a character: the bone goes to them."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    skel = tree.nodes.new("ArmatureNodesSkeletonNode")
    marker = skel.markers[0]
    marker.set_position((0.5, 0.0, 1.5))
    sock = next(s for s in skel.outputs if s.marker_key == marker.key)
    tree.links.new(sock, node.inputs["Position"])
    _build(tree, BUILDS)
    _assert_close(marker.position, (0.5, 0.0, 1.5), "landmark moved")
    _assert_close(_head(obj), (0.5, 0.0, 1.5), "bone on the landmark")


def test_a_marker_on_several_bones_is_left_where_it_is():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.001;bone.002"
    marker_node, marker = _wire_marker(tree, node, at=(1.0, 0.0, 1.0))
    _build(tree)
    _tick()
    assert marker_node.live_bone() == (None, None), "several bones cannot be one link"
    _assert_close(marker.position, (1.0, 0.0, 1.0), "marker moved")


# --- Marker through Rotation and Transform --------------------------------------


def _full_tick():
    """Both halves of the sync, in the order Blender's handler runs them."""
    from armature_nodes import sync

    bpy.context.view_layer.update()
    sync.sync_marker_handles()
    sync.sync_bone_nodes()


def _world_rot(obj, name="bone.002"):
    return (obj.matrix_world @ obj.pose.bones[name].matrix).to_quaternion()


def _assert_turn(got, want, what):
    angle = 2.0 * math.acos(min(1.0, abs(got.dot(want))))
    assert angle < 1e-3, f"{what}: off by {math.degrees(angle):.3f} degrees"


def test_marker_in_rotation_turns_live_both_ways():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesRotationNode")
    pb = obj.pose.bones["bone.002"]
    pb.rotation_mode = "XYZ"
    node.bone = "bone.002"
    marker_node, marker = _wire_marker(tree, node, socket="Rotation")
    _build(tree)
    handle = _handle(marker_node, marker)

    # Wired into a rotation, the marker is a rotation: shown, and turnable.
    assert marker_node.marker_uses_rotation(marker.key), "rotation not shown"
    assert handle.empty_display_type == "ARROWS", "handle is not an axis gizmo"
    assert not any(handle.lock_rotation), "handle cannot be turned"
    # Its position drives nothing: it sits on the bone and cannot be dragged.
    _assert_close(handle.location, _head(obj), "handle on the bone")
    assert all(handle.lock_location), "a readout position must be locked"

    pb.rotation_euler = (math.radians(30.0), 0.0, 0.0)  # rig -> marker
    _full_tick()
    _assert_turn(Euler(marker.rotation, "XYZ").to_quaternion(), _world_rot(obj), "marker")
    _assert_turn(handle.rotation_euler.to_quaternion(), _world_rot(obj), "handle")

    handle.rotation_euler = (0.0, 0.0, math.radians(45.0))  # marker -> rig
    _full_tick()
    _build(tree)
    _assert_turn(_world_rot(obj), Euler((0.0, 0.0, math.radians(45.0))).to_quaternion(), "bone")


def test_marker_in_world_location_is_the_bones_location():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    live = _head(obj)
    marker_node, marker = _wire_marker(tree, node, socket="Location")
    _build(tree, BUILDS)
    _assert_close(marker.position, live, "marker did not take the bone's location")
    _assert_close(_head(obj), live, "bone jumped on wiring")
    handle = _handle(marker_node, marker)
    _assert_close(handle.location, live, "handle on the bone")

    _grab(obj, (1.0, 1.0, 2.0))  # rig -> marker
    _full_tick()
    _assert_close(marker.position, (1.0, 1.0, 2.0), "Location after grab")
    _assert_close(handle.location, (1.0, 1.0, 2.0), "handle after grab")

    handle.location = (0.0, 1.0, 1.5)  # marker -> rig
    _full_tick()
    _build(tree)
    _assert_close(marker.position, (0.0, 1.0, 1.5), "Location after drag")
    _assert_close(_head(obj), (0.0, 1.0, 1.5), "bone after drag")


def test_marker_in_local_location_moves_along_the_bone():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    node.space = "LOCAL"
    rest = _head(obj)
    marker_node, marker = _wire_marker(tree, node, socket="Location")
    _build(tree)
    _assert_close(marker.position, (0.0, 0.0, 0.0), "the channel, at rest")
    handle = _handle(marker_node, marker)
    _assert_close(handle.location, rest, "handle on the bone")

    # The bone points up world Z, which is its own Y axis.
    handle.location = rest + Vector((0.0, 0.0, 0.3))
    _full_tick()
    _build(tree)
    _assert_close(marker.position, (0.0, 0.3, 0.0), "local Location")
    _assert_close(_head(obj), rest + Vector((0.0, 0.0, 0.3)), "bone under the handle")


def test_marker_in_transform_rotation_turns_live_both_ways():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    pb = obj.pose.bones["bone.002"]
    pb.rotation_mode = "XYZ"
    node.bone = "bone.002"
    rest_rot = _world_rot(obj)
    marker_node, marker = _wire_marker(tree, node, socket="Rotation")
    _build(tree)
    handle = _handle(marker_node, marker)
    _assert_turn(Euler(marker.rotation, "XYZ").to_quaternion(), rest_rot, "marker took the bone's")
    _assert_turn(handle.rotation_euler.to_quaternion(), rest_rot, "handle shows the bone")

    spin = Euler((0.0, 0.0, math.radians(30.0))).to_quaternion()
    handle.rotation_euler = (spin @ rest_rot).to_euler("XYZ")  # marker -> rig
    _full_tick()
    _build(tree)
    _assert_turn(Euler(marker.rotation, "XYZ").to_quaternion(), spin @ rest_rot, "Rotation")
    _assert_turn(_world_rot(obj), spin @ rest_rot, "bone turned")

    pb.rotation_euler = (0.0, 0.0, 0.0)  # rig -> marker: back to rest
    _full_tick()
    _assert_turn(Euler(marker.rotation, "XYZ").to_quaternion(), rest_rot, "Rotation after reset")
    _assert_turn(handle.rotation_euler.to_quaternion(), rest_rot, "handle after reset")


def test_marker_in_transform_scale_scales_live_both_ways():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    pb = obj.pose.bones["bone.002"]
    pb.lock_scale = (False, False, False)
    node.bone = "bone.002"
    marker_node, marker = _wire_marker(tree, node, socket="Scale")
    _build(tree)
    handle = _handle(marker_node, marker)
    _assert_close(marker.scale, (1.0, 1.0, 1.0), "Scale took the bone's")
    assert not any(handle.lock_scale), "handle cannot be scaled"

    handle.scale = (2.0, 2.0, 2.0)  # marker -> rig
    _full_tick()
    _build(tree)
    _assert_close(pb.matrix.to_scale(), (2.0, 2.0, 2.0), "bone scale")

    pb.scale = (3.0, 3.0, 3.0)  # rig -> marker
    _full_tick()
    _assert_close(marker.scale, (3.0, 3.0, 3.0), "Scale after scaling the bone")
    _assert_close(handle.scale, (3.0, 3.0, 3.0), "handle after scaling the bone")


def test_moving_the_parent_carries_a_relative_handle_without_a_drag():
    """A Local value is measured from rest, which the parent carries."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    node.space = "LOCAL"
    marker_node, marker = _wire_marker(tree, node, socket="Location")
    _build(tree)
    marker.position = (0.5, 0.0, 0.0)
    _build(tree)
    handle = _handle(marker_node, marker)

    obj.pose.bones["bone.001"].location = (0.0, 0.0, 1.0)
    _full_tick()
    _assert_close(marker.position, (0.5, 0.0, 0.0), "parent move read as a drag")
    _assert_close(handle.location, _head(obj), "handle left behind by the parent")
    _build(tree, BUILDS)
    _assert_close(marker.position, (0.5, 0.0, 0.0), "Location drifted")


def test_unplugging_a_marker_hands_its_value_to_the_field():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    _marker_node, marker = _wire_marker(tree, node, socket="Location")
    _build(tree)
    marker.position = (0.0, 0.7, 2.0)
    _build(tree)
    placed = _head(obj)

    for link in list(node.inputs["Location"].links):
        tree.links.remove(link)
    _build(tree, BUILDS)
    _assert_close(node.inputs["Location"].default_value, (0.0, 0.7, 2.0), "field")
    assert node.use_location, "unplugged, the location must stay set"
    _assert_close(_head(obj), placed, "bone snapped back when unplugged")


def test_deleting_a_wired_marker_leaves_the_bone_in_place():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    marker_node, marker = _wire_marker(tree, node, socket="Location")
    _build(tree)
    marker.position = (0.0, 0.7, 2.0)
    _build(tree)
    placed = _head(obj)

    tree.nodes.remove(marker_node)
    _build(tree, BUILDS)
    _assert_close(_head(obj), placed, "bone snapped back when the marker was deleted")


def test_a_marker_from_before_tracking_keeps_driving():
    """No record of its wires: adopt them as they are rather than re-seed."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    marker_node, marker = _wire_marker(tree, node, socket="Location", at=(0.5, 0.0, 2.0))
    marker_node.live_links = ""  # as saved by an older version
    _build(tree, BUILDS)
    _assert_close(marker.position, (0.5, 0.0, 2.0), "marker was re-seeded")
    _assert_close(_head(obj), (0.5, 0.0, 2.0), "bone")


# --- The Transform input: all three on one wire --------------------------------


def _wire_transform(tree, node):
    """A Marker node wired into the Transform node's Transform input."""
    marker_node = tree.nodes.new("ArmatureNodesMarkerNode")
    marker = marker_node.markers[0]
    marker.set_position((5.0, 5.0, 5.0))  # dropped away from the bone
    tree.links.new(marker_node.outputs[0], node.inputs["Transform"])
    return marker_node, marker


def test_marker_outputs_a_transform():
    obj, tree, _s, _o, _node = _fresh("ArmatureNodesTransformNode")
    marker_node = tree.nodes.new("ArmatureNodesMarkerNode")
    assert marker_node.outputs[0].bl_idname == "ArmatureNodesTransformSocket"


def test_transform_input_takes_the_bones_live_transform_first():
    """Wired in, the marker takes where the bone is -- world values -- so
    the bone does not move, and the Marker node shows the bone's real
    Location / Rotation / Scale."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    pb = obj.pose.bones["bone.002"]
    pb.rotation_mode = "XYZ"
    pb.lock_scale = (False, False, False)
    node.bone = "bone.002"
    node.inputs["Location"].default_value = (0.5, 0.0, 2.0)
    _build(tree)
    live, live_rot = _head(obj), _world_rot(obj)

    marker_node, marker = _wire_transform(tree, node)
    _build(tree, BUILDS)
    _assert_close(marker.position, live, "marker did not take the live location")
    _assert_turn(Euler(marker.rotation, "XYZ").to_quaternion(), live_rot, "live rotation")
    _assert_close(marker.scale, (1.0, 1.0, 1.0), "live scale")
    _assert_close(_head(obj), live, "bone jumped on wiring")
    _assert_turn(_world_rot(obj), live_rot, "bone turned on wiring")
    for name in ("Location", "Rotation", "Scale"):
        assert node.inputs[name].hide, f"{name} field still shown while the wire drives"

    handle = _handle(marker_node, marker)
    _assert_close(handle.location, live, "handle on the bone")
    assert not any(handle.lock_location), "handle cannot move"
    assert not any(handle.lock_rotation), "handle cannot turn"
    assert not any(handle.lock_scale), "handle cannot scale"

    # Marker -> rig: move, turn and scale the one handle.
    spin = Euler((0.0, 0.0, math.radians(30.0))).to_quaternion()
    handle.location = (0.0, 1.0, 2.0)
    handle.rotation_euler = (spin @ live_rot).to_euler("XYZ")
    handle.scale = (2.0, 2.0, 2.0)
    _full_tick()
    _build(tree)
    _assert_close(_head(obj), (0.0, 1.0, 2.0), "bone location")
    _assert_turn(_world_rot(obj), spin @ live_rot, "bone rotation")
    _assert_close(pb.matrix.to_scale(), (2.0, 2.0, 2.0), "bone scale")
    _assert_close(marker.position, (0.0, 1.0, 2.0), "Location")
    _assert_turn(Euler(marker.rotation, "XYZ").to_quaternion(), spin @ live_rot, "Rotation")
    _assert_close(marker.scale, (2.0, 2.0, 2.0), "Scale")

    # Rig -> marker: grab and scale the bone.
    _grab(obj, (1.0, 1.0, 2.0))
    pb.scale = (3.0, 3.0, 3.0)
    _full_tick()
    _assert_close(marker.position, (1.0, 1.0, 2.0), "Location after grab")
    _assert_close(marker.scale, (3.0, 3.0, 3.0), "Scale after scaling the bone")
    _assert_close(handle.location, (1.0, 1.0, 2.0), "handle after grab")
    _build(tree, BUILDS)
    _assert_close(_head(obj), (1.0, 1.0, 2.0), "grab snapped back")


def test_transform_input_scale_is_the_bones_not_the_objects():
    """A scaled armature object must not have its scale applied twice."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    obj.scale = (2.0, 2.0, 2.0)
    bpy.context.view_layer.update()
    node.bone = "bone.002"
    _build(tree)
    _marker_node, marker = _wire_transform(tree, node)
    _build(tree, BUILDS)
    _assert_close(marker.scale, (1.0, 1.0, 1.0), "Scale")
    _assert_close(obj.pose.bones["bone.002"].matrix.to_scale(), (1.0, 1.0, 1.0), "bone scale")


def test_unplugging_the_transform_input_keeps_the_bone():
    """The fields take the wire's values, set, so the bone stays put."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    obj.pose.bones["bone.002"].lock_scale = (False, False, False)
    node.bone = "bone.002"
    rest, rest_rot = _head(obj), _world_rot(obj)
    _marker_node, marker = _wire_transform(tree, node)
    _build(tree)
    spin = Euler((0.0, 0.0, math.radians(20.0))).to_quaternion()
    marker.position = rest + Vector((0.0, 0.7, 0.0))
    marker.rotation = (spin @ rest_rot).to_euler("XYZ")
    marker.scale = (1.5, 1.5, 1.5)
    _build(tree)
    placed, turned = _head(obj), _world_rot(obj)

    for link in list(node.inputs["Transform"].links):
        tree.links.remove(link)
    _build(tree, BUILDS)
    rotation = Euler(node.inputs["Rotation"].default_value, "XYZ").to_quaternion()
    _assert_close(node.inputs["Location"].default_value, rest + Vector((0.0, 0.7, 0.0)), "Location")
    _assert_turn(rotation, spin @ rest_rot, "Rotation")
    _assert_close(node.inputs["Scale"].default_value, (1.5, 1.5, 1.5), "Scale")
    for name in ("Location", "Rotation", "Scale"):
        assert not node.inputs[name].hide, f"{name} field still hidden after unplugging"
    assert node.use_location and node.use_rotation and node.use_scale, "not set"
    _assert_close(_head(obj), placed, "bone moved when unplugged")
    _assert_turn(_world_rot(obj), turned, "bone turned when unplugged")


def test_transform_input_takes_a_skeleton_landmark():
    """Any node's marker output feeds it -- and a landmark with rotation off
    is a position, so it moves the bone without turning it."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    rest_rot = _world_rot(obj)
    skel = tree.nodes.new("ArmatureNodesSkeletonNode")
    marker = skel.markers[0]
    marker.set_position((0.0, 0.4, 1.5))
    sock = next(s for s in skel.outputs if s.marker_key == marker.key)
    tree.links.new(sock, node.inputs["Transform"])
    _build(tree, BUILDS)
    _assert_close(_head(obj), (0.0, 0.4, 1.5), "bone on the landmark")
    _assert_turn(_world_rot(obj), rest_rot, "a position-only landmark turned the bone")


def test_an_old_marker_output_becomes_a_transform_and_keeps_its_wire():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    marker_node = tree.nodes.new("ArmatureNodesMarkerNode")
    marker = marker_node.markers[0]
    # As an older version made it: a Vector output.
    old = marker_node.outputs[0]
    marker_node.outputs.remove(old)
    old = marker_node.outputs.new("ArmatureNodesVectorSocket", marker.name)
    old.marker_key = marker.key
    tree.links.new(old, node.inputs["Position"])
    _build(tree)
    out = marker_node.outputs[0]
    assert out.bl_idname == "ArmatureNodesTransformSocket", out.bl_idname
    assert out.marker_key == marker.key
    assert node.inputs["Position"].is_linked, "the wire was lost"
    assert node.inputs["Position"].links[0].from_socket == out


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


def test_transform_fields_follow_the_bone_until_set():
    """Unset, the fields are readouts: the node shows the bone, live."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    _build(tree)
    _grab(obj, (1.0, 0.0, 2.0))
    _tick()
    _assert_close(node.inputs["Location"].default_value, (1.0, 0.0, 2.0), "readout")
    assert not node.use_location, "a readout must not take the bone over"
    _build(tree, BUILDS)
    _assert_close(_head(obj), (1.0, 0.0, 2.0), "unset bone left alone")


def test_transform_set_location_follows_a_grab():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    node.use_location = True
    _build(tree)

    _grab(obj, (1.0, 0.0, 2.0))
    _tick()
    _assert_close(node.inputs["Location"].default_value, (1.0, 0.0, 2.0), "Location")
    _build(tree, BUILDS)
    _assert_close(_head(obj), (1.0, 0.0, 2.0), "bone kept")

    _grab(obj, (1.0, 2.0, 2.0))
    _tick()
    _assert_close(node.inputs["Location"].default_value, (1.0, 2.0, 2.0), "second grab")


def test_transform_starts_from_a_hand_posed_bone():
    """Posed before the node existed: the node reads that, and setting one
    part does not snap the bone back."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    obj.pose.bones["bone.002"].location = (0.0, 0.0, 0.7)  # hand-posed earlier
    bpy.context.view_layer.update()
    posed = _head(obj)
    node.bone = "bone.002"
    _assert_close(node.inputs["Location"].default_value, posed, "fields read the pose")
    node.use_location = True  # set it: starts from where the bone is
    _build(tree, BUILDS)
    _assert_close(_head(obj), posed, "bone jumped when its location was set")


def test_setting_a_constrained_bones_location_does_not_move_it():
    """The readout shows the bone after its constraint; setting the value has
    to start from before it, or the constraint is applied twice."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode", keep_constraints=True)
    node.bone = "bone.003"  # 50% Copy Transforms to a still empty
    _build(tree)
    _tick()
    before = _head(obj, "bone.003")
    node.use_location = True
    _build(tree, BUILDS)
    # A build can write nothing when the target matches; a nudge forces it.
    sock = node.inputs["Location"]
    sock.default_value = Vector(sock.default_value) + Vector((0.2, 0.0, 0.0))
    _build(tree)
    _assert_close(_head(obj, "bone.003"), before + Vector((0.1, 0.0, 0.0)), "nudged bone")


def test_transform_local_mirrors_the_location_channel():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    node.space = "LOCAL"
    _build(tree)
    obj.pose.bones["bone.002"].location = (0.0, 0.3, 0.0)
    _tick()
    _assert_close(node.inputs["Location"].default_value, (0.0, 0.3, 0.0), "Location")


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
