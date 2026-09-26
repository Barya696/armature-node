"""Several branches wired into the Armature Output, side by side.

Each branch carries a copy of the whole rig, changed by its own nodes. The
Output used to put the copies side by side and keep one copy of each bone --
so one branch's changes won and every other branch's were dropped: two
Transform nodes wired into the Output, one of them did nothing.

Now the branches are chained, in the order the wires come in: each one runs
on top of the ones before it, from where it split off from them. Side by
side is the same as one plugged into the other, and a node the branches
share runs once.
"""

import bpy
from mathutils import Vector

import fixtures
import test_transform_nodes as t

# Five bones: the fixture pins the last one (bone.004) with a constraint.
N_BONES = 5


def _rig():
    """A bound rig and an empty stack: (obj, tree, input, output)."""
    obj, tree, src, out, node = t._fresh("ArmatureNodesTransformNode", n_bones=N_BONES)
    tree.nodes.remove(node)
    return obj, tree, src, out


def _node(tree, kind, bone, **inputs):
    node = tree.nodes.new(kind)
    node.bone = bone
    for name, value in inputs.items():
        node.inputs[name].default_value = value
    return node


def _wire(tree, *nodes):
    """Wire ``nodes`` one after another."""
    for a, b in zip(nodes, nodes[1:]):
        socket = b.inputs.get(getattr(b, "stream_input", "Rig")) or b.inputs["Rig"]
        tree.links.new(a.outputs["Rig"], socket)


def _defs(tree):
    from armature_nodes.build import evaluate_tree

    _name, defs = evaluate_tree(tree)
    return {b.name: b for b in defs}


def _rest_heads(obj):
    return {pb.name: (obj.matrix_world @ pb.bone.matrix_local).to_translation() for pb in obj.pose.bones}


def _pose(b):
    """Everything a node can ask of a bone, for comparing two evaluations."""
    return (
        b.pose_location,
        b.pose_rotation,
        b.pose_scale,
        tuple(round(v, 6) for v in b.pose_offset),
        tuple(round(v, 6) for v in b.pose_rotation_offset),
        tuple(round(v, 6) for v in b.pose_local_offset),
        tuple(round(v, 6) for v in b.pose_local_rotation),
        tuple(b.pose_local_set),
        repr(b.shape),
        b.use_deform,
        len(b.constraints),
    )


def test_two_transform_nodes_side_by_side_both_apply():
    """What was reported: two Transform nodes into the Output, one did nothing."""
    obj, tree, src, out = _rig()
    first = _node(tree, "ArmatureNodesTransformNode", "bone.002", Location=(1.0, 0.0, 2.0))
    second = _node(tree, "ArmatureNodesTransformNode", "bone.003", Location=(-1.0, 0.0, 3.0))
    _wire(tree, src, first, out)
    _wire(tree, src, second, out)
    t._build(tree, t.BUILDS)
    t._assert_close(t._head(obj, "bone.002"), (1.0, 0.0, 2.0), "the first branch")
    t._assert_close(t._head(obj, "bone.003"), (-1.0, 0.0, 3.0), "the second branch")


def test_the_rig_arrives_once_whatever_the_number_of_branches():
    obj, tree, src, out = _rig()
    for bone in ("bone.001", "bone.002", "bone.003"):
        _wire(tree, src, _node(tree, "ArmatureNodesPositionNode", bone, Offset=(0.1, 0.0, 0.0)), out)
    from armature_nodes.build import evaluate_tree

    _name, defs = evaluate_tree(tree)
    names = [b.name for b in defs]
    assert sorted(names) == sorted(pb.name for pb in obj.pose.bones), names
    for bone in ("bone.001", "bone.002", "bone.003"):
        got = next(b for b in defs if b.name == bone).pose_offset
        t._assert_close(got, (0.1, 0.0, 0.0), f"{bone}'s offset")


def test_one_bone_moved_in_one_branch_and_turned_in_another():
    import math

    obj, tree, src, out = _rig()
    move = _node(tree, "ArmatureNodesPositionNode", "bone.002", Position=(0.5, 0.0, 2.0))
    turn = _node(tree, "ArmatureNodesRotationNode", "bone.002", Rotation=(0.0, math.radians(30.0), 0.0))
    _wire(tree, src, move, out)
    _wire(tree, src, turn, out)
    b = _defs(tree)["bone.002"]
    t._assert_close(b.pose_location, (0.5, 0.0, 2.0), "moved")
    t._assert_close(b.pose_rotation, (0.0, math.radians(30.0), 0.0), "turned")
    t._build(tree, t.BUILDS)
    t._assert_close(t._head(obj, "bone.002"), (0.5, 0.0, 2.0), "the built bone")


def test_offsets_side_by_side_add_up():
    obj, tree, src, out = _rig()
    rest = _rest_heads(obj)["bone.002"]
    _wire(tree, src, _node(tree, "ArmatureNodesPositionNode", "bone.002", Offset=(0.5, 0.0, 0.0)), out)
    _wire(tree, src, _node(tree, "ArmatureNodesPositionNode", "bone.002", Offset=(0.0, 0.0, 0.25)), out)
    t._build(tree, t.BUILDS)
    t._assert_close(t._head(obj, "bone.002"), rest + Vector((0.5, 0.0, 0.25)), "both offsets")


def test_the_wire_connected_last_wins_a_value_both_set():
    obj, tree, src, out = _rig()
    a = _node(tree, "ArmatureNodesTransformNode", "bone.002", Location=(1.0, 0.0, 2.0))
    b = _node(tree, "ArmatureNodesTransformNode", "bone.002", Location=(0.0, 1.0, 2.0))
    _wire(tree, src, a, out)
    _wire(tree, src, b, out)
    t._build(tree, t.BUILDS)
    t._assert_close(t._head(obj, "bone.002"), (0.0, 1.0, 2.0), "the later wire")
    # Plug the first one in again: now it is the later one.
    tree.links.remove(next(l for l in out.inputs["Rig"].links if l.from_node == a))
    _wire(tree, a, out)
    t._build(tree, t.BUILDS)
    t._assert_close(t._head(obj, "bone.002"), (1.0, 0.0, 2.0), "re-plugged, it wins")


def test_a_node_the_branches_share_runs_once():
    """Input -> Offset -> two branches: the offset is not applied twice."""
    obj, tree, src, out = _rig()
    shared = _node(tree, "ArmatureNodesPositionNode", "bone.001", Offset=(0.5, 0.0, 0.0))
    a = _node(tree, "ArmatureNodesTransformNode", "bone.002", Location=(1.0, 0.0, 2.0))
    b = _node(tree, "ArmatureNodesTransformNode", "bone.003", Location=(-1.0, 0.0, 3.0))
    _wire(tree, src, shared)
    _wire(tree, shared, a, out)
    _wire(tree, shared, b, out)
    defs = _defs(tree)
    t._assert_close(defs["bone.001"].pose_offset, (0.5, 0.0, 0.0), "the shared offset, once")
    t._assert_close(defs["bone.002"].pose_location, (1.0, 0.0, 2.0), "branch a")
    t._assert_close(defs["bone.003"].pose_location, (-1.0, 0.0, 3.0), "branch b")


def test_a_split_inside_a_split_runs_each_node_once():
    """Branch 1: P1 -> A. Branch 2: P1 -> P2 -> B. Branch 3: P2 -> C.
    Chained, that is P1, A, P2, B, C: P2 once, though two branches share it."""
    obj, tree, src, out = _rig()
    p1 = _node(tree, "ArmatureNodesPositionNode", "bone.001", Offset=(0.5, 0.0, 0.0))
    a = _node(tree, "ArmatureNodesTransformNode", "bone.002", Location=(1.0, 0.0, 2.0))
    p2 = _node(tree, "ArmatureNodesPositionNode", "bone.003", Offset=(0.0, 0.0, 0.3))
    b = _node(tree, "ArmatureNodesPositionNode", "bone.001", Offset=(0.0, 0.4, 0.0))
    c = _node(tree, "ArmatureNodesPositionNode", "bone.003", Offset=(0.2, 0.0, 0.0))
    _wire(tree, src, p1, a, out)
    _wire(tree, p1, p2, b, out)
    _wire(tree, p2, c, out)
    defs = _defs(tree)
    t._assert_close(defs["bone.001"].pose_offset, (0.5, 0.4, 0.0), "P1 and B")
    t._assert_close(defs["bone.002"].pose_location, (1.0, 0.0, 2.0), "A")
    t._assert_close(defs["bone.003"].pose_offset, (0.2, 0.0, 0.3), "P2 once, and C")


def test_side_by_side_is_the_same_as_one_after_another():
    import math

    obj, tree, src, out = _rig()
    nodes = [
        _node(tree, "ArmatureNodesTransformNode", "bone.002", Location=(1.0, 0.0, 2.0)),
        _node(tree, "ArmatureNodesRotationNode", "bone.002", Offset=(0.0, 0.0, math.radians(20.0))),
        _node(tree, "ArmatureNodesPositionNode", "bone.001", Offset=(0.0, 0.3, 0.0)),
        _node(tree, "ArmatureNodesCustomShapeNode", "bone.003"),
    ]
    for node in nodes:
        _wire(tree, src, node, out)
    side_by_side = {name: _pose(b) for name, b in _defs(tree).items()}
    for link in list(tree.links):
        tree.links.remove(link)
    _wire(tree, src, *nodes, out)
    chained = {name: _pose(b) for name, b in _defs(tree).items()}
    assert side_by_side == chained, {
        n: (side_by_side.get(n), chained.get(n)) for n in chained if side_by_side.get(n) != chained[n]
    }


def test_a_shape_in_one_branch_and_a_pose_in_another():
    obj, tree, src, out = _rig()
    shape = _node(tree, "ArmatureNodesCustomShapeNode", "bone.001")
    shape.widget_scale = (3.0, 3.0, 3.0)  # the rig's own shapes are not this size
    _wire(tree, src, shape, out)
    _wire(tree, src, _node(tree, "ArmatureNodesTransformNode", "bone.002", Location=(1.0, 0.0, 2.0)), out)
    defs = _defs(tree)
    assert tuple(defs["bone.001"].shape.scale) == (3.0, 3.0, 3.0), "the shape was dropped"
    t._assert_close(defs["bone.002"].pose_location, (1.0, 0.0, 2.0), "the pose")


def test_a_group_in_one_branch_and_a_node_in_another():
    from armature_nodes.groups import make_group

    obj, tree, src, out = _rig()
    inside = _node(tree, "ArmatureNodesTransformNode", "bone.002", Location=(1.0, 0.0, 2.0))
    beside = _node(tree, "ArmatureNodesTransformNode", "bone.003", Location=(-1.0, 0.0, 3.0))
    _wire(tree, src, inside, out)
    _wire(tree, src, beside, out)
    make_group(tree, [inside])
    t._build(tree, t.BUILDS)
    t._assert_close(t._head(obj, "bone.002"), (1.0, 0.0, 2.0), "the grouped branch")
    t._assert_close(t._head(obj, "bone.003"), (-1.0, 0.0, 3.0), "the other branch")


def test_a_muted_branch_is_left_out():
    obj, tree, src, out = _rig()
    _wire(tree, src, _node(tree, "ArmatureNodesPositionNode", "bone.002", Offset=(0.5, 0.0, 0.0)), out)
    muted = _node(tree, "ArmatureNodesPositionNode", "bone.003", Offset=(0.5, 0.0, 0.0))
    _wire(tree, src, muted, out)
    next(l for l in out.inputs["Rig"].links if l.from_node == muted).is_muted = True
    defs = _defs(tree)
    t._assert_close(defs["bone.002"].pose_offset, (0.5, 0.0, 0.0), "the live branch")
    t._assert_close(defs["bone.003"].pose_offset, (0.0, 0.0, 0.0), "the muted branch")


def test_unplugging_a_branch_puts_its_bone_back():
    obj, tree, src, out = _rig()
    keep = _node(tree, "ArmatureNodesTransformNode", "bone.002", Location=(1.0, 0.0, 2.0))
    drop = _node(tree, "ArmatureNodesTransformNode", "bone.003", Location=(-1.0, 0.0, 3.0))
    _wire(tree, src, keep, out)
    _wire(tree, src, drop, out)
    t._build(tree, t.BUILDS)
    tree.links.remove(next(l for l in out.inputs["Rig"].links if l.from_node == drop))
    t._build(tree, t.BUILDS)
    t._assert_close(t._head(obj, "bone.002"), (1.0, 0.0, 2.0), "the branch still wired")
    # bone.003 hangs off bone.002, which moved: its rest follows its parent.
    parent = obj.pose.bones["bone.002"]
    want = (obj.matrix_world @ parent.matrix @ parent.bone.matrix_local.inverted()
            @ obj.pose.bones["bone.003"].bone.matrix_local).to_translation()
    t._assert_close(t._head(obj, "bone.003"), want, "the unplugged branch's bone")
    assert not t._close(t._head(obj, "bone.003"), (-1.0, 0.0, 3.0)), "still where the branch put it"


def test_rigs_with_nothing_in_common_still_sit_side_by_side():
    """Two Chain nodes building bones from scratch: all of both arrive."""
    fixtures.ensure_registered()
    tree = bpy.data.node_groups.new("Chains", "ArmatureNodeTreeType")
    out = tree.nodes.new("ArmatureNodesOutputNode")
    out.mode = "FULL"
    for prefix in ("arm", "leg"):
        chain = tree.nodes.new("ArmatureNodesChainNode")
        chain.prefix, chain.count = prefix, 2
        tree.links.new(chain.outputs["Rig"], out.inputs["Rig"])
    assert sorted(_defs(tree)) == ["arm.001", "arm.002", "leg.001", "leg.002"]
