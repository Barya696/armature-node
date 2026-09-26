"""Marker handles you grab in any mode: the gizmo layer (``handles.py``).

A gizmo cannot be clicked without a window, so the parts are tested apart:
``pick`` decides what is under the mouse, ``Drag`` turns mouse motion into a
move, turn or scale of the handle, and from there the path is the ordinary
one -- the handle's empty moved, the marker synced, the bone rebuilt. The
view is a stand-in: a front orthographic view at 100 pixels per metre.
"""

import math

import bpy
from mathutils import Quaternion, Vector

import test_live_link as live


class FrontView:
    """Looking down +Y: screen X is world X, screen Y is world Z."""

    def to_screen(self, co):
        return Vector((500.0 + 100.0 * co[0], 300.0 + 100.0 * co[2]))

    def on_plane(self, xy, depth):
        return Vector(((xy[0] - 500.0) / 100.0, depth[1], (xy[1] - 300.0) / 100.0))

    def ray(self, xy):
        origin = Vector(((xy[0] - 500.0) / 100.0, -50.0, (xy[1] - 300.0) / 100.0))
        return origin, Vector((0.0, 1.0, 0.0))

    def facing(self):
        return Vector((0.0, -1.0, 0.0))  # towards the viewer

    def right_up(self):
        return Vector((1.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))

    def world_per_pixel(self, co):
        return 0.01


VIEW = FrontView()


def _drag(obj, mode, start, end, **options):
    from armature_nodes.handles import Drag

    drag = Drag(obj, mode, VIEW, start)
    drag.update(end, **options)
    return drag


def _marker_on(node_type, socket="Position"):
    """A rig, a node on bone.002 and a Marker wired into ``socket``."""
    obj, tree, _s, _o, node = live._fresh(node_type)
    pb = obj.pose.bones["bone.002"]
    pb.rotation_mode = "XYZ"
    pb.lock_scale = (False, False, False)
    node.bone = "bone.002"
    if socket == "Transform":
        marker_node, marker = live._wire_transform(tree, node)
    else:
        marker_node, marker = live._wire_marker(tree, node, socket=socket)
    live._build(tree)
    return obj, tree, node, marker_node, marker, live._handle(marker_node, marker)


# --- the parts -------------------------------------------------------------------


def test_handles_are_registered():
    from armature_nodes import handles

    for cls in handles.classes:
        # Registering gives a class its own RNA definition.
        assert "bl_rna" in cls.__dict__, f"{cls.__name__} is not registered"


def test_pick_finds_the_part_under_the_mouse():
    from armature_nodes.handles import MOVE, RING_RADIUS, ROTATE, SCALE, knob_offset, pick

    points = [
        ((100.0, 100.0), 1.0, MOVE, True, True),  # moves, turns and scales
        ((300.0, 100.0), 1.0, MOVE, False, False),  # only moves
    ]
    assert pick(points, (102.0, 101.0)) == (0, MOVE)
    assert pick(points, (100.0 + RING_RADIUS, 100.0)) == (0, ROTATE)
    knob = Vector((100.0, 100.0)) + knob_offset(1.0)
    assert pick(points, tuple(knob)) == (0, SCALE), "the square sits on the ring and wins"
    assert pick(points, (300.0 + RING_RADIUS, 100.0)) is None, "no ring where nothing turns"
    assert pick(points, (200.0, 250.0)) is None
    assert pick(points, (300.0, 100.0)) == (1, MOVE)


def test_bigger_handles_are_easier_to_hit():
    from armature_nodes.handles import CORE_RADIUS, MOVE, SLACK, pick

    edge = (100.0 + CORE_RADIUS * 1.5 + SLACK - 0.5, 100.0)
    assert pick([((100.0, 100.0), 1.0, MOVE, False, False)], edge) is None
    assert pick([((100.0, 100.0), 1.5, MOVE, False, False)], edge) == (0, MOVE)


# --- dragging --------------------------------------------------------------------


def test_dragging_a_handle_in_pose_mode_moves_the_bone():
    """The point of the gizmos: an empty cannot be clicked in Pose mode."""
    from armature_nodes.handles import MOVE

    obj, tree, _node, _mn, _marker, handle = _marker_on("ArmatureNodesPositionNode")
    before = live._head(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="POSE")

    start = VIEW.to_screen(handle.location)
    _drag(handle, MOVE, start, start + Vector((100.0, 0.0)))
    live._full_tick()
    live._build(tree)
    assert obj.mode == "POSE", "the drag must not leave Pose mode"
    live._assert_close(live._head(obj), before + Vector((1.0, 0.0, 0.0)), "bone")


def test_x_locks_the_move_and_shift_is_fine():
    from armature_nodes.handles import Drag, MOVE

    _obj, _tree, _node, _mn, _marker, handle = _marker_on("ArmatureNodesPositionNode")
    before = handle.location.copy()
    start = VIEW.to_screen(before)
    drag = Drag(handle, MOVE, VIEW, start)
    drag.set_axis("X")
    drag.update(start + Vector((100.0, 100.0)))
    live._assert_close(handle.location, before + Vector((1.0, 0.0, 0.0)), "locked to X")
    drag.set_axis("X")  # pressed again: free
    drag.update(start + Vector((100.0, 100.0)), precise=True)
    live._assert_close(handle.location, before + Vector((0.1, 0.0, 0.1)), "fine move")


def test_escape_puts_it_back():
    from armature_nodes.handles import MOVE

    _obj, _tree, _node, _mn, _marker, handle = _marker_on("ArmatureNodesPositionNode")
    before = handle.location.copy()
    start = VIEW.to_screen(before)
    drag = _drag(handle, MOVE, start, start + Vector((150.0, -40.0)))
    drag.cancel()
    live._assert_close(handle.location, before, "handle after cancel")


def test_ctrl_drops_the_marker_on_the_surface():
    from armature_nodes.handles import MOVE

    obj, tree, _node, _mn, _marker, handle = _marker_on("ArmatureNodesPositionNode")
    # A wall facing the viewer, one metre deep.
    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 1.0, 1.0), rotation=(math.radians(90.0), 0.0, 0.0))
    bpy.context.view_layer.objects.active = obj
    bpy.context.view_layer.update()
    start = VIEW.to_screen(handle.location)
    _drag(handle, MOVE, start, start + Vector((50.0, 0.0)), snap=True, context=bpy.context)
    want = Vector((VIEW.on_plane(start, handle.location).x + 0.5, 1.0, VIEW.on_plane(start, handle.location).z))
    live._assert_close(handle.location, want, "on the wall")
    live._full_tick()
    live._build(tree)
    live._assert_close(live._head(obj), want, "bone on the wall")


def test_turning_the_ring_turns_the_bone():
    from armature_nodes.handles import ROTATE

    obj, tree, _node, _mn, _marker, handle = _marker_on("ArmatureNodesTransformNode", "Transform")
    before = live._world_rot(obj)
    c = VIEW.to_screen(handle.location)
    # A quarter turn, counter-clockwise on screen.
    _drag(handle, ROTATE, c + Vector((50.0, 0.0)), c + Vector((0.0, 50.0)))
    live._full_tick()
    live._build(tree)
    turn = Quaternion(VIEW.facing(), math.radians(90.0))
    live._assert_turn(live._world_rot(obj), turn @ before, "bone")


def test_ctrl_turns_in_five_degree_steps():
    from armature_nodes.handles import ROTATE

    _obj, _tree, _node, _mn, _marker, handle = _marker_on("ArmatureNodesTransformNode", "Transform")
    before = handle.rotation_euler.to_quaternion()
    c = VIEW.to_screen(handle.location)
    tilt = Vector((math.cos(math.radians(12.0)), math.sin(math.radians(12.0)))) * 50.0
    _drag(handle, ROTATE, c + Vector((50.0, 0.0)), c + tilt, snap=True)
    turned = handle.rotation_euler.to_quaternion() @ before.inverted()
    assert abs(math.degrees(2.0 * math.acos(min(1.0, abs(turned.w)))) - 10.0) < 1e-3


def test_dragging_the_square_scales_the_bone():
    from armature_nodes.handles import SCALE

    obj, tree, _node, _mn, _marker, handle = _marker_on("ArmatureNodesTransformNode", "Transform")
    c = VIEW.to_screen(handle.location)
    _drag(handle, SCALE, c + Vector((40.0, 0.0)), c + Vector((80.0, 0.0)))
    live._assert_close(handle.scale, (2.0, 2.0, 2.0), "handle")
    live._full_tick()
    live._build(tree)
    live._assert_close(obj.pose.bones["bone.002"].matrix.to_scale(), (2.0, 2.0, 2.0), "bone")


# --- what is offered -------------------------------------------------------------


def test_a_marker_that_only_turns_turns_from_its_glow():
    """Wired into Rotation, the position is a readout: the glow turns it."""
    from armature_nodes.handles import MOVE, ROTATE, visible_handles

    _obj, _tree, _node, _mn, marker, handle = _marker_on("ArmatureNodesRotationNode", "Rotation")
    [h] = [h for h in visible_handles() if h.key == marker.key]
    assert not h.move and h.turn, (h.move, h.turn)
    assert h.core_mode() == ROTATE
    before = handle.location.copy()
    start = VIEW.to_screen(before)
    _drag(handle, MOVE, start, start + Vector((80.0, 40.0)))
    live._assert_close(handle.location, before, "a readout position moved")


def test_a_mirrored_landmark_is_not_grabbable():
    """Symmetric mode: right-side landmarks only follow their left partner."""
    from armature_nodes import sync
    from armature_nodes.handles import visible_handles

    obj, tree, _s, _o, node = live._fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    skel = tree.nodes.new("ArmatureNodesSkeletonNode")
    sock = next(s for s in skel.outputs if s.marker_key == "wrist_l")
    tree.links.new(sock, node.inputs["Position"])
    live._build(tree)
    sync.sync_marker_handles()
    keys = {h.key for h in visible_handles() if h.node == skel.name}
    assert "wrist_l" in keys, "the left wrist should be grabbable"
    assert "wrist_r" not in keys, "a mirrored landmark should not be"


def test_a_second_marker_node_gets_its_handle_too():
    """The marker collection used to be re-linked on every call after it
    existed -- an error -- so only the first marker node ever got a handle."""
    from armature_nodes import sync
    from armature_nodes.primary_rig import find_marker_empties

    obj, tree, _s, out, node = live._fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    first_node, first = live._wire_marker(tree, node)
    rotation = tree.nodes.new("ArmatureNodesRotationNode")
    rotation.bone = "bone.001"
    tree.links.new(node.outputs["Rig"], rotation.inputs["Rig"])
    for link in list(out.inputs["Rig"].links):
        tree.links.remove(link)
    tree.links.new(rotation.outputs["Rig"], out.inputs["Rig"])
    live._build(tree)
    sync.sync_marker_handles()  # the first handle, and the collection
    second_node, second = live._wire_marker(tree, rotation, socket="Rotation")
    live._build(tree)
    sync.sync_marker_handles()
    assert first.key in find_marker_empties(first_node), "first handle"
    assert second.key in find_marker_empties(second_node), "second handle never created"


def test_handle_size_and_colour_are_the_nodes():
    from armature_nodes.handles import visible_handles

    _obj, _tree, _node, marker_node, marker, _handle = _marker_on("ArmatureNodesPositionNode")
    marker_node.handle_size = 2.0
    marker.color = (0.1, 0.9, 0.3)
    [h] = [h for h in visible_handles() if h.key == marker.key]
    assert h.size == 2.0
    live._assert_close(h.color[:3], (0.1, 0.9, 0.3), "colour")
