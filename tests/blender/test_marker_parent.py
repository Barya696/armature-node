"""Markers with a parent, like bones.

Wire one marker's output into another Marker node's Parent input: the second
is the child. Its values are relative to the parent, it moves, turns and
scales with it, and the rest of the graph receives its world transform. A
line is drawn from parent to child.
"""

import math

import bpy
from mathutils import Vector

import test_live_link as live


def _world(node):
    return Vector(node.marker_value(node.markers[0], "position"))


def _parented(parent_at=(0.0, 0.0, 1.0)):
    """A rig with a child marker B driving bone.002, and a parent marker A."""
    obj, tree, _s, _o, pos = live._fresh("ArmatureNodesPositionNode")
    pos.bone = "bone.002"
    child_node, _child = live._wire_marker(tree, pos)
    live._build(tree)
    live._tick()
    parent_node = tree.nodes.new("ArmatureNodesMarkerNode")
    parent_node.markers[0].set_position(parent_at)
    tree.links.new(parent_node.outputs[0], child_node.inputs["Parent"])
    live._tick()
    return obj, tree, parent_node, child_node


def test_parenting_keeps_the_child_where_it_is():
    obj, tree, parent_node, child_node = _parented()
    head = live._head(obj)
    live._assert_close(_world(child_node), head, "child moved when parented")
    live._assert_close(child_node.markers[0].position, head - Vector((0.0, 0.0, 1.0)), "local")
    live._build(tree, live.BUILDS)
    live._assert_close(live._head(obj), head, "bone moved when its marker was parented")


def test_moving_the_parent_carries_the_child():
    obj, tree, parent_node, child_node = _parented()
    head = live._head(obj)
    parent_node.markers[0].position = (0.5, 0.0, 1.0)
    live._tick()
    live._assert_close(_world(child_node), head + Vector((0.5, 0.0, 0.0)), "child")
    live._build(tree)
    live._assert_close(live._head(obj), head + Vector((0.5, 0.0, 0.0)), "the bone followed")
    handle = live._handle(child_node, child_node.markers[0])
    live._full_tick()
    live._assert_close(handle.location, head + Vector((0.5, 0.0, 0.0)), "the child's handle")


def test_turning_the_parent_swings_the_child_around_it():
    obj, tree, parent_node, child_node = _parented()
    offset = _world(child_node) - Vector((0.0, 0.0, 1.0))  # straight up from the parent
    parent_node.markers[0].use_rotation = True
    parent_node.markers[0].rotation = (math.radians(90.0), 0.0, 0.0)
    live._tick()
    # +90 degrees about X turns "up" (+Z) into -Y.
    want = Vector((0.0, -offset.z, 1.0))
    live._assert_close(_world(child_node), want, "child after the parent turned")
    live._build(tree)
    live._assert_close(live._head(obj), want, "bone after the parent turned")


def test_scaling_the_parent_scales_the_child_offset():
    _obj, _tree, parent_node, child_node = _parented()
    offset = _world(child_node) - Vector((0.0, 0.0, 1.0))
    parent_node.markers[0].scale = (2.0, 2.0, 2.0)
    live._tick()
    live._assert_close(_world(child_node), Vector((0.0, 0.0, 1.0)) + offset * 2.0, "child")


def test_a_chain_keeps_everyone_in_place_whatever_order_it_was_made_in():
    """The tip exists before its parent is itself parented: it must not be
    measured against a parent that has not settled yet."""
    obj, tree, _s, _o, pos = live._fresh("ArmatureNodesPositionNode")
    pos.bone = "bone.002"
    tip, _t = live._wire_marker(tree, pos)  # made first
    live._build(tree)
    live._tick()
    root = tree.nodes.new("ArmatureNodesMarkerNode")
    middle = tree.nodes.new("ArmatureNodesMarkerNode")
    root.markers[0].set_position((0.0, 0.0, 0.8))
    middle.markers[0].set_position((0.35, 0.0, 1.3))
    tip_at = _world(tip)
    tree.links.new(root.outputs[0], middle.inputs["Parent"])
    tree.links.new(middle.outputs[0], tip.inputs["Parent"])
    live._tick()
    live._assert_close(_world(root), (0.0, 0.0, 0.8), "root")
    live._assert_close(_world(middle), (0.35, 0.0, 1.3), "middle")
    live._assert_close(_world(tip), tip_at, "tip")
    live._build(tree)
    live._assert_close(live._head(obj), tip_at, "the bone the tip drives")


def test_dragging_the_child_moves_only_the_child():
    _obj, _tree, parent_node, child_node = _parented()
    handle = live._handle(child_node, child_node.markers[0])
    handle.location = (0.3, 0.0, 1.5)
    live._full_tick()
    live._assert_close(_world(child_node), (0.3, 0.0, 1.5), "child world")
    live._assert_close(child_node.markers[0].position, (0.3, 0.0, 0.5), "child local")
    live._assert_close(_world(parent_node), (0.0, 0.0, 1.0), "the parent moved")


def test_unparenting_keeps_the_child_where_it_is():
    _obj, tree, parent_node, child_node = _parented()
    parent_node.markers[0].position = (0.5, 0.0, 1.0)
    live._tick()
    before = _world(child_node)
    for link in list(child_node.inputs["Parent"].links):
        tree.links.remove(link)
    live._tick()
    live._assert_close(_world(child_node), before, "child jumped when unparented")
    live._assert_close(child_node.markers[0].position, before, "its own values are world again")


def test_grabbing_the_parent_bone_does_not_move_the_child_twice():
    """The trap: the child bone is carried by the parent bone, and the child
    marker is carried by the parent marker. Taking both moves naively moves
    the child marker -- and so its bone -- twice."""
    obj, tree, _s, out, first = live._fresh("ArmatureNodesPositionNode")
    first.bone = "bone.001"
    second = tree.nodes.new("ArmatureNodesPositionNode")
    second.bone = "bone.002"
    for link in list(out.inputs["Rig"].links):
        tree.links.remove(link)
    tree.links.new(first.outputs["Rig"], second.inputs["Rig"])
    tree.links.new(second.outputs["Rig"], out.inputs["Rig"])
    parent_node, _pm = live._wire_marker(tree, first)
    child_node, _cm = live._wire_marker(tree, second)
    live._build(tree)
    live._tick()
    tree.links.new(parent_node.outputs[0], child_node.inputs["Parent"])
    live._tick()
    live._build(tree)

    p0, c0 = live._head(obj, "bone.001"), live._head(obj, "bone.002")
    step = Vector((0.4, 0.0, 0.0))
    live._grab(obj, p0 + step, name="bone.001")  # carries bone.002 with it
    live._assert_close(live._head(obj, "bone.002"), c0 + step, "Blender carried the child bone")
    live._tick()
    live._assert_close(_world(parent_node), p0 + step, "the parent marker")
    live._assert_close(_world(child_node), c0 + step, "the child marker moved twice")
    live._build(tree, live.BUILDS)
    live._assert_close(live._head(obj, "bone.001"), p0 + step, "parent bone")
    live._assert_close(live._head(obj, "bone.002"), c0 + step, "child bone moved twice")


def test_a_loop_of_parents_does_not_hang():
    """Blender marks the wire that closes the loop invalid, so it is ignored;
    ``parent_matrix`` also cuts a loop itself, for one Blender cannot see
    (through a group)."""
    _obj, tree, parent_node, child_node = _parented()
    tree.links.new(child_node.outputs[0], parent_node.inputs["Parent"])
    live._tick()
    for node in (parent_node, child_node):
        value = _world(node)
        assert all(math.isfinite(v) for v in value), value


def test_an_older_marker_node_gets_a_parent_input():
    _obj, tree, _s, _o, pos = live._fresh("ArmatureNodesPositionNode")
    marker_node, _marker = live._wire_marker(tree, pos)
    marker_node.inputs.remove(marker_node.inputs["Parent"])  # as saved before parenting
    live._tick()
    assert marker_node.inputs.get("Parent") is not None


def test_a_skeleton_landmark_can_be_a_parent():
    _obj, tree, _s, _o, pos = live._fresh("ArmatureNodesPositionNode")
    pos.bone = "bone.002"
    child_node, _child = live._wire_marker(tree, pos)
    live._build(tree)
    live._tick()
    skel = tree.nodes.new("ArmatureNodesSkeletonNode")
    sock = next(s for s in skel.outputs if s.marker_key == "nose")
    tree.links.new(sock, child_node.inputs["Parent"])
    live._tick()
    before = _world(child_node)
    nose = skel.marker_by_key("nose")
    # Typed, as in the node: that moves the landmark's handle too. (The
    # quiet setter leaves the handle behind, and the next sync would read the
    # stale handle as a drag back.)
    nose.position = Vector(nose.position) + Vector((0.2, 0.0, 0.0))
    live._tick()
    live._assert_close(_world(child_node), before + Vector((0.2, 0.0, 0.0)), "child of a landmark")


def test_the_line_runs_from_parent_to_child():
    from armature_nodes.primary_rig import marker_color, marker_lines

    _obj, _tree, parent_node, child_node = _parented()
    lines = dict(marker_lines([parent_node, child_node]))
    color = marker_color(child_node, child_node.markers[0])
    assert color in lines, "no line in the child's colour"
    start, end = lines[color][0], lines[color][1]
    live._assert_close(start, _world(parent_node), "line start")
    live._assert_close(end, _world(child_node), "line end")
    assert len(lines) == 1, "a marker without a parent draws no line"
