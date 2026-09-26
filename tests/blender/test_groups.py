"""Node groups: the Blender principle, in this tree type (``groups.py``).

A group is its own tree with Group Input and Group Output nodes; a group node
runs it like a function, with what is wired into it arriving at Group Input.
Making a group must not change the rig; nor must ungrouping it again.
"""

import math
import types

import bpy
from mathutils import Vector

import fixtures


def _rig():
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
        while pb.constraints:
            pb.constraints.remove(pb.constraints[0])
    bpy.context.view_layer.update()
    bind(obj)
    return obj


def _chain(obj, *node_types):
    """Input -> the given nodes, in order -> Output. Returns (tree, nodes)."""
    tree, src, out = fixtures.make_tree(obj)
    for link in list(tree.links):
        tree.links.remove(link)
    nodes, previous = [], src
    for i, kind in enumerate(node_types):
        node = tree.nodes.new(kind)
        node.location = (200.0 * (i + 1), 0.0)
        tree.links.new(previous.outputs["Rig"], node.inputs[getattr(node, "stream_input", "Rig")])
        nodes.append(node)
        previous = node
    tree.links.new(previous.outputs["Rig"], out.inputs["Rig"])
    obj.armature_nodes_tree = tree
    return tree, nodes


def _build(tree, times=2):
    from armature_nodes.build import build_armature_from_tree

    for _ in range(times):
        build_armature_from_tree(tree)


def _head(obj, name="bone.002"):
    return (obj.matrix_world @ obj.pose.bones[name].matrix).to_translation()


def _rot(obj, name="bone.002"):
    return (obj.matrix_world @ obj.pose.bones[name].matrix).to_quaternion()


def _close(a, b, what, eps=1e-4):
    assert (Vector(a) - Vector(b)).length < eps, f"{what}: got {tuple(round(v, 4) for v in a)}, want {tuple(b)}"


def _turn_close(a, b, what):
    angle = 2.0 * math.acos(min(1.0, abs(a.dot(b))))
    assert angle < 1e-3, f"{what}: off by {math.degrees(angle):.3f} degrees"


def _kinds(tree):
    return sorted(n.bl_idname for n in tree.nodes)


def _offset_rotation_chain():
    """A rig posed by a Position (offset) and a Rotation (offset) node."""
    obj = _rig()
    tree, (pos, rot) = _chain(obj, "ArmatureNodesPositionNode", "ArmatureNodesRotationNode")
    pos.bone = rot.bone = "bone.002"
    pos.inputs["Offset"].default_value = (1.0, 0.0, 0.0)
    rot.inputs["Offset"].default_value = (0.0, 0.0, math.radians(30.0))
    _build(tree)
    return obj, tree, pos, rot


# --- making and unmaking a group -----------------------------------------------


def test_making_a_group_keeps_the_rig_as_it_was():
    from armature_nodes.groups import GROUP_NODE, make_group

    obj, tree, pos, rot = _offset_rotation_chain()
    head, turn = _head(obj), _rot(obj)
    group_node = make_group(tree, [pos, rot])
    group = group_node.node_tree

    assert _kinds(tree) == sorted(["ArmatureNodesInputNode", "ArmatureNodesOutputNode", GROUP_NODE])
    assert _kinds(group) == sorted([
        "NodeGroupInput", "NodeGroupOutput",
        "ArmatureNodesPositionNode", "ArmatureNodesRotationNode",
    ])
    items = [(i.in_out, i.name, i.socket_type) for i in group.interface.items_tree if i.item_type == "SOCKET"]
    assert ("INPUT", "Rig", "ArmatureNodesBoneSocket") in items, items
    assert ("OUTPUT", "Rig", "ArmatureNodesBoneSocket") in items, items
    assert group_node.inputs[0].is_linked and group_node.outputs[0].is_linked, "not wired in"

    _build(tree)
    _close(_head(obj), head, "bone moved when grouped")
    _turn_close(_rot(obj), turn, "bone turned when grouped")


def test_ungrouping_puts_it_all_back():
    from armature_nodes.groups import GROUP_NODE, make_group, ungroup

    obj, tree, pos, rot = _offset_rotation_chain()
    head, turn = _head(obj), _rot(obj)
    group_node = make_group(tree, [pos, rot])
    _build(tree)
    ungroup(tree, group_node)

    assert GROUP_NODE not in _kinds(tree), "the group node is still there"
    kinds = _kinds(tree)
    assert "ArmatureNodesPositionNode" in kinds and "ArmatureNodesRotationNode" in kinds
    new_pos = next(n for n in tree.nodes if n.bl_idname == "ArmatureNodesPositionNode")
    _close(new_pos.inputs["Offset"].default_value, (1.0, 0.0, 0.0), "Offset kept")
    assert new_pos.bone == "bone.002", "the node's settings were not carried back"
    _build(tree)
    _close(_head(obj), head, "bone moved when ungrouped")
    _turn_close(_rot(obj), turn, "bone turned when ungrouped")


def test_a_value_wired_in_from_outside_arrives_inside():
    """A marker left outside the group drives a Position node inside it."""
    from armature_nodes.groups import make_group

    obj = _rig()
    tree, (pos,) = _chain(obj, "ArmatureNodesPositionNode")
    pos.bone = "bone.002"
    marker_node = tree.nodes.new("ArmatureNodesMarkerNode")
    tree.links.new(marker_node.outputs[0], pos.inputs["Position"])
    _build(tree)
    group_node = make_group(tree, [pos])  # the marker not selected

    inputs = [s.name for s in group_node.inputs]
    assert "Position" in inputs, inputs
    assert group_node.inputs["Position"].links[0].from_node == tree.nodes[marker_node.name]

    marker_node.markers[0].position = (0.5, 0.0, 1.5)
    _build(tree)
    _close(_head(obj), (0.5, 0.0, 1.5), "bone did not follow the marker through the group")


# --- markers inside a group ------------------------------------------------------


def _marker_in(group):
    return next(n for n in group.nodes if n.bl_idname == "ArmatureNodesMarkerNode")


def _grouped_marker():
    """Input -> Position (bone.002, from a live marker) -> Output, then the
    marker and the Position node made into a group."""
    import test_live_link as live
    from armature_nodes.groups import make_group

    obj = _rig()
    tree, (pos,) = _chain(obj, "ArmatureNodesPositionNode")
    pos.bone = "bone.002"
    marker_node = tree.nodes.new("ArmatureNodesMarkerNode")
    tree.links.new(marker_node.outputs[0], pos.inputs["Position"])
    _build(tree)
    live._tick()
    head = _head(obj)
    group_node = make_group(tree, [marker_node, pos])
    return obj, tree, group_node, head


def test_a_marker_goes_into_the_group_and_the_bone_stays():
    from armature_nodes import sync
    from armature_nodes.primary_rig import find_marker_empties

    obj, tree, group_node, head = _grouped_marker()
    group = group_node.node_tree
    assert "ArmatureNodesMarkerNode" not in _kinds(tree), "the marker should be in the group"
    inner = _marker_in(group)
    _build(tree)
    _close(_head(obj), head, "bone moved when its marker was grouped")

    # Its handle shows, through the group node, and dragging it moves the bone.
    sync.sync_marker_handles()
    handle = find_marker_empties(inner).get(inner.markers[0].key)
    assert handle is not None, "no handle for a marker inside a group"
    _close(handle.location, inner.markers[0].position, "handle on the marker")
    handle.location = (0.5, 0.0, 1.5)
    sync.sync_marker_handles()
    _build(tree)
    _close(_head(obj), (0.5, 0.0, 1.5), "dragging the handle inside a group")


def test_a_marker_inside_a_group_used_once_is_live():
    """Wiring it in takes the bone's place; grabbing the bone moves it."""
    import test_live_link as live

    obj, tree, group_node, _head0 = _grouped_marker()
    group = group_node.node_tree
    inner_pos = next(n for n in group.nodes if n.bl_idname == "ArmatureNodesPositionNode")
    assert inner_pos.resolve_armature() == obj, "one user: the group knows its rig"

    fresh = group.nodes.new("ArmatureNodesMarkerNode")
    fresh.markers[0].set_position((5.0, 5.0, 5.0))  # dropped away from the bone
    for link in list(inner_pos.inputs["Position"].links):
        group.links.remove(link)
    group.links.new(fresh.outputs[0], inner_pos.inputs["Position"])
    before = _head(obj)
    _build(tree)
    _close(_head(obj), before, "wiring a marker inside a group moved the bone")
    _close(fresh.markers[0].position, before, "the marker should take the bone's place")

    live._grab(obj, (1.0, 0.0, 1.8))
    live._tick()
    _close(fresh.markers[0].position, (1.0, 0.0, 1.8), "the marker did not follow the grab")
    _build(tree)
    _close(_head(obj), (1.0, 0.0, 1.8), "the grab snapped back")


def test_a_marker_in_a_group_used_twice_drives_both_but_follows_neither():
    """Shared, like any value in a group; which bone to follow is not known."""
    from armature_nodes.groups import GROUP_NODE

    obj, tree, group_node, _head0 = _grouped_marker()
    group = group_node.node_tree
    second = tree.nodes.new(GROUP_NODE)
    second.node_tree = group
    inner_pos = next(n for n in group.nodes if n.bl_idname == "ArmatureNodesPositionNode")
    assert inner_pos.resolve_armature() is None, "two users: no one rig"
    assert inner_pos.rig_for_ui() == obj, "the bone list still shows a rig"
    _marker_in(group).markers[0].position = (0.0, 0.3, 1.8)
    _build(tree)
    _close(_head(obj), (0.0, 0.3, 1.8), "the shared marker still drives")


def test_a_marker_inside_is_hidden_when_the_group_is_unplugged():
    from armature_nodes import sync
    from armature_nodes.primary_rig import find_marker_empties, marker_node_visible

    _obj, tree, group_node, _head0 = _grouped_marker()
    inner = _marker_in(group_node.node_tree)
    sync.sync_marker_handles()
    assert marker_node_visible(inner) and find_marker_empties(inner), "shown while plugged in"
    for link in list(group_node.outputs[0].links):
        tree.links.remove(link)
    sync.sync_marker_handles()
    assert not marker_node_visible(inner), "still shown with the group unplugged"
    assert not find_marker_empties(inner), "the handle should go with it"


def test_a_value_typed_on_the_group_node_arrives_inside():
    from armature_nodes.groups import make_group

    obj = _rig()
    tree, (pos,) = _chain(obj, "ArmatureNodesPositionNode")
    pos.bone = "bone.002"
    marker_node = tree.nodes.new("ArmatureNodesMarkerNode")
    tree.links.new(marker_node.outputs[0], pos.inputs["Position"])
    group_node = make_group(tree, [pos])
    for link in list(group_node.inputs["Position"].links):
        tree.links.remove(link)
    group_node.inputs["Position"].default_value = (0.25, 0.0, 1.0)
    _build(tree)
    _close(_head(obj), (0.25, 0.0, 1.0), "the group node's own field")


def test_ungrouping_hands_a_typed_value_to_the_node_inside():
    from armature_nodes.groups import make_group, ungroup

    obj = _rig()
    tree, (pos,) = _chain(obj, "ArmatureNodesPositionNode")
    pos.bone = "bone.002"
    marker_node = tree.nodes.new("ArmatureNodesMarkerNode")
    tree.links.new(marker_node.outputs[0], pos.inputs["Position"])
    group_node = make_group(tree, [pos])
    for link in list(group_node.inputs["Position"].links):
        tree.links.remove(link)
    group_node.inputs["Position"].default_value = (0.25, 0.0, 1.0)
    _build(tree)
    ungroup(tree, group_node)
    back = next(n for n in tree.nodes if n.bl_idname == "ArmatureNodesPositionNode")
    _close(back.inputs["Position"].default_value, (0.25, 0.0, 1.0), "field")
    _build(tree)
    _close(_head(obj), (0.25, 0.0, 1.0), "bone")


# --- one group, used more than once ---------------------------------------------


def test_one_group_used_twice_changes_in_both():
    from armature_nodes.groups import GROUP_NODE, make_group

    obj = _rig()
    rest = _head(obj)
    tree, (pos,) = _chain(obj, "ArmatureNodesPositionNode")
    pos.bone = "bone.002"
    pos.inputs["Offset"].default_value = (0.0, 0.0, 0.5)
    first = make_group(tree, [pos])
    group = first.node_tree

    # A second group node running the same group, after the first.
    out = next(n for n in tree.nodes if n.bl_idname == "ArmatureNodesOutputNode")
    second = tree.nodes.new(GROUP_NODE)
    second.node_tree = group
    for link in list(out.inputs["Rig"].links):
        tree.links.remove(link)
    tree.links.new(first.outputs[0], second.inputs[0])
    tree.links.new(second.outputs[0], out.inputs["Rig"])
    _build(tree)
    _close(_head(obj), rest + Vector((0.0, 0.0, 1.0)), "two offsets")

    inner = next(n for n in group.nodes if n.bl_idname == "ArmatureNodesPositionNode")
    inner.inputs["Offset"].default_value = (0.0, 0.0, 0.25)
    _build(tree)
    _close(_head(obj), rest + Vector((0.0, 0.0, 0.5)), "one edit, both group nodes")


def test_editing_inside_a_group_rebuilds_the_rig_that_uses_it():
    from armature_nodes import tree as tree_module
    from armature_nodes.groups import make_group

    obj = _rig()
    tree, (pos,) = _chain(obj, "ArmatureNodesPositionNode")
    pos.bone = "bone.002"
    group = make_group(tree, [pos]).node_tree
    _build(tree)
    tree.is_dirty = group.is_dirty = False
    tree_module._pending.clear()

    inner = next(n for n in group.nodes if n.bl_idname == "ArmatureNodesPositionNode")
    inner.inputs["Offset"].default_value = (0.0, 0.3, 0.0)
    assert tree.is_dirty and tree.name in tree_module._pending, "the rig's tree was not rebuilt"
    assert not group.is_dirty and group.name not in tree_module._pending, (
        "a group builds nothing by itself"
    )


def test_groups_inside_groups():
    from armature_nodes.groups import GROUP_NODE, new_group_tree

    obj = _rig()
    rest = _head(obj)
    inner = new_group_tree("Inner")
    gin = next(n for n in inner.nodes if n.bl_idname == "NodeGroupInput")
    gout = next(n for n in inner.nodes if n.bl_idname == "NodeGroupOutput")
    for link in list(inner.links):
        inner.links.remove(link)
    pos = inner.nodes.new("ArmatureNodesPositionNode")
    pos.bone = "bone.002"
    pos.inputs["Offset"].default_value = (1.0, 0.0, 0.0)
    inner.links.new(gin.outputs[0], pos.inputs["Rig"])
    inner.links.new(pos.outputs["Rig"], gout.inputs[0])

    outer = new_group_tree("Outer")
    ogin = next(n for n in outer.nodes if n.bl_idname == "NodeGroupInput")
    ogout = next(n for n in outer.nodes if n.bl_idname == "NodeGroupOutput")
    for link in list(outer.links):
        outer.links.remove(link)
    use_inner = outer.nodes.new(GROUP_NODE)
    use_inner.node_tree = inner
    outer.links.new(ogin.outputs[0], use_inner.inputs[0])
    outer.links.new(use_inner.outputs[0], ogout.inputs[0])

    tree, _nodes = _chain(obj)
    out = next(n for n in tree.nodes if n.bl_idname == "ArmatureNodesOutputNode")
    src = next(n for n in tree.nodes if n.bl_idname == "ArmatureNodesInputNode")
    for link in list(tree.links):
        tree.links.remove(link)
    use_outer = tree.nodes.new(GROUP_NODE)
    use_outer.node_tree = outer
    tree.links.new(src.outputs["Rig"], use_outer.inputs[0])
    tree.links.new(use_outer.outputs[0], out.inputs["Rig"])
    _build(tree)
    _close(_head(obj), rest + Vector((1.0, 0.0, 0.0)), "offset from two groups down")


def test_a_group_cannot_run_itself():
    fixtures.ensure_registered()  # first test to run: nothing has registered yet
    from armature_nodes.build import evaluate_tree
    from armature_nodes.groups import GROUP_NODE, add_menu_items, new_group_tree

    a, b = new_group_tree("A"), new_group_tree("B")
    use_a = b.nodes.new(GROUP_NODE)
    use_a.node_tree = a
    # In A's editor, B is not offered: it would put A inside itself.
    context = types.SimpleNamespace(space_data=types.SimpleNamespace(edit_tree=a))
    offered = [getattr(item, "label", "") for item in add_menu_items(context)]
    assert "B" not in offered, offered
    assert "A" not in offered, "a group is not offered inside itself"

    use_b = a.nodes.new(GROUP_NODE)
    use_b.node_tree = b  # Blender accepts this
    if use_b.node_tree is None:
        return  # refused: nothing more to check
    # Marking one dirty must not chase the loop for ever (it once recursed
    # until Python gave up -- inside an update callback, where Blender only
    # prints the error).
    a.mark_dirty()
    b.mark_dirty()
    for group, node in ((a, use_b), (b, use_a)):
        gin = next(n for n in group.nodes if n.bl_idname == "NodeGroupInput")
        gout = next(n for n in group.nodes if n.bl_idname == "NodeGroupOutput")
        for link in list(group.links):
            group.links.remove(link)
        group.links.new(gin.outputs[0], node.inputs[0])
        group.links.new(node.outputs[0], gout.inputs[0])
    obj = _rig()
    tree, _nodes = _chain(obj)
    out = next(n for n in tree.nodes if n.bl_idname == "ArmatureNodesOutputNode")
    run_a = tree.nodes.new(GROUP_NODE)
    run_a.node_tree = a
    tree.links.new(run_a.outputs[0], out.inputs["Rig"])
    try:
        evaluate_tree(tree, strict=False)
    except RuntimeError as exc:
        assert "contains itself" in str(exc), exc
    else:
        raise AssertionError("a group running itself should be refused")


# --- the interface -------------------------------------------------------------


def test_group_nodes_follow_the_interface():
    from armature_nodes.groups import make_group, sync_all_group_nodes

    obj = _rig()
    tree, (pos,) = _chain(obj, "ArmatureNodesPositionNode")
    group_node = make_group(tree, [pos])
    group = group_node.node_tree

    item = group.interface.new_socket("Extra", in_out="INPUT", socket_type="ArmatureNodesVectorSocket")
    sync_all_group_nodes()
    assert "Extra" in [s.name for s in group_node.inputs], "new input missing"
    item.name = "Renamed"
    sync_all_group_nodes()
    assert "Renamed" in [s.name for s in group_node.inputs], "rename not followed"
    rig_in = group_node.inputs[0]
    assert rig_in.is_linked, "renaming another socket dropped the rig's wire"
    group.interface.remove(item)
    sync_all_group_nodes()
    assert "Renamed" not in [s.name for s in group_node.inputs], "removed input still there"


def test_only_this_trees_socket_types_can_be_group_sockets():
    from armature_nodes.groups import new_group_tree

    group = new_group_tree("Types")
    try:
        group.interface.new_socket("F", in_out="INPUT", socket_type="NodeSocketFloat")
    except TypeError:
        pass
    else:
        raise AssertionError("a Float socket should not be allowed")


# --- around the edges ----------------------------------------------------------


def test_the_add_menu_offers_groups_and_their_io():
    from armature_nodes.groups import GROUP_INPUT, GROUP_OUTPUT, add_menu_items, make_group

    obj = _rig()
    tree, (pos,) = _chain(obj, "ArmatureNodesPositionNode")
    group = make_group(tree, [pos]).node_tree

    in_rig = add_menu_items(types.SimpleNamespace(space_data=types.SimpleNamespace(edit_tree=tree)))
    assert group.name in [getattr(i, "label", "") for i in in_rig]
    assert not any(getattr(i, "nodetype", "") == GROUP_INPUT for i in in_rig), "no Group Input in a rig's tree"
    in_group = add_menu_items(types.SimpleNamespace(space_data=types.SimpleNamespace(edit_tree=group)))
    kinds = [getattr(i, "nodetype", "") for i in in_group]
    assert GROUP_INPUT in kinds and GROUP_OUTPUT in kinds, kinds


def test_bone_names_are_listed_inside_a_group():
    from armature_nodes.groups import make_group

    obj = _rig()
    tree, (pos,) = _chain(obj, "ArmatureNodesPositionNode")
    group = make_group(tree, [pos]).node_tree
    inner = next(n for n in group.nodes if n.bl_idname == "ArmatureNodesPositionNode")
    assert not any(n.bl_idname == "ArmatureNodesInputNode" for n in group.nodes)
    assert inner.rig_for_ui() == obj, "the dropdown should list the rig that uses the group"


def test_a_reroute_passes_the_rig_on():
    obj = _rig()
    rest = _head(obj)
    tree, (pos,) = _chain(obj, "ArmatureNodesPositionNode")
    pos.bone = "bone.002"
    pos.inputs["Offset"].default_value = (0.0, 1.0, 0.0)
    src = next(n for n in tree.nodes if n.bl_idname == "ArmatureNodesInputNode")
    for link in list(pos.inputs["Rig"].links):
        tree.links.remove(link)
    reroute = tree.nodes.new("NodeReroute")
    tree.links.new(src.outputs["Rig"], reroute.inputs[0])
    tree.links.new(reroute.outputs[0], pos.inputs["Rig"])
    _build(tree)
    _close(_head(obj), rest + Vector((0.0, 1.0, 0.0)), "through a reroute")
