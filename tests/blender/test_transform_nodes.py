"""The Transform category: Position, Rotation and Transform nodes.

What was broken, each pinned by a test below:

* A **Transform** node did nothing. Its translation went into an offset, and
  the bridge only folded an offset into an absolute location -- with none set,
  the offset was dropped without an error.
* A fresh **Position** node wrecked the rig: empty selection means every bone,
  and the unlinked socket defaulted to (0, 0, 0), so the whole rig went to the
  origin. Children ended up below it, because poses were written in arbitrary
  order with no depsgraph flush and each child was placed against a stale
  parent.
* **Rotation** inputs were Vector sockets: shown in metres, read as radians.

Offsets are resolved against the REST pose, so every test also rebuilds
several times: resolving against the live pose would add the move again each
build and the bone would drift.
"""

import math

import bpy
from mathutils import Vector

import fixtures

BUILDS = 4


def _fresh(node_type, n_bones=4):
    """A bound rig with free bones and ``node_type`` on the wire."""
    from armature_nodes.ops.bind import bind

    obj = fixtures.make_rig(n_bones=n_bones)
    # The fixture connects bones and locks channels; Blender silently refuses
    # a pose write on either, which has its own test elsewhere.
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    for eb in obj.data.edit_bones:
        eb.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in obj.pose.bones:
        pb.lock_location = (False, False, False)
        pb.lock_rotation = (False, False, False)
        pb.lock_scale = (False, False, False)
    bind(obj)

    tree, src, out = fixtures.make_tree(obj)
    for link in list(tree.links):
        tree.links.remove(link)
    node = tree.nodes.new(node_type)
    tree.links.new(src.outputs["Rig"], node.inputs["Rig"])
    tree.links.new(node.outputs["Rig"], out.inputs["Rig"])
    return obj, tree, src, out, node


def _build(tree, times=1):
    from armature_nodes.build import build_armature_from_tree

    for _ in range(times):
        build_armature_from_tree(tree)


def _head(obj, name):
    return (obj.matrix_world @ obj.pose.bones[name].matrix).to_translation()


def _heads(obj):
    return {pb.name: _head(obj, pb.name) for pb in obj.pose.bones}


def _close(a, b, eps=1e-4):
    return (Vector(a) - Vector(b)).length < eps


def _assert_close(got, want, what):
    assert _close(got, want), f"{what}: got {tuple(round(v, 4) for v in got)}, want {tuple(want)}"


# --- a freshly added node must not touch the rig -----------------------------


def _assert_no_op(node_type):
    obj, tree, *_ = _fresh(node_type)
    before = _heads(obj)
    _build(tree, BUILDS)
    for name, head in _heads(obj).items():
        _assert_close(head, before[name], f"{node_type} moved {name}")


def test_fresh_position_node_is_a_no_op():
    """It used to send every bone to the origin."""
    _assert_no_op("ArmatureNodesPositionNode")


def test_fresh_rotation_node_is_a_no_op():
    _assert_no_op("ArmatureNodesRotationNode")


def test_fresh_transform_node_is_a_no_op():
    _assert_no_op("ArmatureNodesTransformNode")


# --- Position ----------------------------------------------------------------


def test_set_position_moves_the_bone():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    node.use_position = True
    node.inputs["Position"].default_value = (2.0, 0.0, 3.0)
    _build(tree, BUILDS)
    _assert_close(_head(obj, "bone.002"), (2.0, 0.0, 3.0), "set position")


def test_ticking_set_position_does_not_jump():
    """The box seeds Position from the bone, so turning it on changes nothing."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    before = _head(obj, "bone.002")
    node.use_position = True
    _assert_close(node.inputs["Position"].default_value, before, "seeded value")
    _build(tree)
    _assert_close(_head(obj, "bone.002"), before, "bone after ticking Set Position")


def test_position_offset_is_relative_and_does_not_drift():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    rest = _head(obj, "bone.002")
    node.inputs["Offset"].default_value = (1.0, 0.0, 0.0)
    _build(tree, BUILDS)
    _assert_close(_head(obj, "bone.002"), rest + Vector((1.0, 0.0, 0.0)), "offset")


def test_position_wired_marker_counts_as_set():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    marker_node = tree.nodes.new("ArmatureNodesMarkerNode")
    tree.links.new(marker_node.outputs[0], node.inputs["Position"])
    assert not node.use_position, "the wire alone should be enough"
    _build(tree)  # wiring puts the marker on the bone first
    marker_node.markers[0].position = (0.5, 1.0, 2.0)
    _build(tree)
    _assert_close(_head(obj, "bone.002"), (0.5, 1.0, 2.0), "marker-driven position")


# --- Transform -----------------------------------------------------------------


def test_transform_world_translation_moves_the_bone():
    """The headline bug: this did nothing at all."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    rest = _head(obj, "bone.002")
    node.inputs["Translation"].default_value = (1.0, 0.0, 0.0)
    _build(tree, BUILDS)
    _assert_close(_head(obj, "bone.002"), rest + Vector((1.0, 0.0, 0.0)), "world +1 X")


def test_transform_local_writes_the_location_channel():
    """Local space IS a Location transform: the N-panel values."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    node.space = "LOCAL"
    node.inputs["Translation"].default_value = (0.0, 1.0, 0.0)
    _build(tree, BUILDS)
    pb = obj.pose.bones["bone.002"]
    _assert_close(pb.location, (0.0, 1.0, 0.0), "Location channel")


def test_transform_local_follows_the_bone_axes():
    """Bone Y runs along the bone, which points up the world Z axis here."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    node.space = "LOCAL"
    rest = _head(obj, "bone.002")
    node.inputs["Translation"].default_value = (0.0, 1.0, 0.0)
    _build(tree, BUILDS)
    _assert_close(_head(obj, "bone.002"), rest + Vector((0.0, 0.0, 1.0)), "local +Y")


def test_transform_rotation_is_degrees_and_turns_in_place():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    sock = node.inputs["Rotation"]
    assert type(sock).bl_rna.properties["default_value"].subtype == "EULER", (
        "rotation must be shown in degrees, not metres"
    )
    rest = _head(obj, "bone.002")
    sock.default_value = (math.radians(90.0), 0.0, 0.0)
    _build(tree, BUILDS)
    world = obj.matrix_world @ obj.pose.bones["bone.002"].matrix
    _assert_close(world.to_translation(), rest, "head must stay put")
    # The bone pointed up world +Z; 90 degrees about world X turns +Z to -Y.
    _assert_close(world.to_3x3().col[1].normalized(), (0.0, -1.0, 0.0), "bone axis")


def test_transform_scale():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    node.inputs["Scale"].default_value = (2.0, 2.0, 2.0)
    _build(tree, BUILDS)
    scale = (obj.matrix_world @ obj.pose.bones["bone.002"].matrix).to_scale()
    _assert_close(scale, (2.0, 2.0, 2.0), "scale")


def test_transform_moves_every_bone_exactly_once():
    """Parent and child selected together: the child is carried, not moved twice.

    Also the ordering bug: children must be placed after their parents.
    """
    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    # The fixture's last bone has a 50% Copy Transforms to an empty that does
    # not move, so it rightly lands halfway -- grabbing the root by hand in
    # Blender gives the same. That is the constraint's behaviour, not the
    # node's, so it is taken out of this test.
    for pb in obj.pose.bones:
        while pb.constraints:
            pb.constraints.remove(pb.constraints[0])
    # pbone.matrix is computed: until the depsgraph re-evaluates it still
    # shows the constrained position.
    bpy.context.view_layer.update()
    rest = _heads(obj)
    node.inputs["Translation"].default_value = (1.0, 0.0, 0.0)  # every bone
    _build(tree, BUILDS)
    for name, head in _heads(obj).items():
        _assert_close(head, rest[name] + Vector((1.0, 0.0, 0.0)), f"{name} moved")


def test_removing_the_node_restores_the_bone():
    obj, tree, src, out, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    rest = _head(obj, "bone.002")
    node.inputs["Translation"].default_value = (1.0, 0.0, 0.0)
    _build(tree)
    tree.nodes.remove(node)
    tree.links.new(src.outputs["Rig"], out.inputs["Rig"])
    _build(tree, 2)
    _assert_close(_head(obj, "bone.002"), rest, "after deleting the Transform node")


def test_stacked_transforms_add_up():
    obj, tree, src, out, first = _fresh("ArmatureNodesTransformNode")
    first.bone = "bone.002"
    first.inputs["Translation"].default_value = (1.0, 0.0, 0.0)
    second = tree.nodes.new("ArmatureNodesTransformNode")
    second.bone = "bone.002"
    second.inputs["Translation"].default_value = (0.0, 2.0, 0.0)
    for link in list(tree.links):
        if link.to_node == out:
            tree.links.remove(link)
    tree.links.new(first.outputs["Rig"], second.inputs["Rig"])
    tree.links.new(second.outputs["Rig"], out.inputs["Rig"])
    rest = _head(obj, "bone.002")
    _build(tree, BUILDS)
    _assert_close(_head(obj, "bone.002"), rest + Vector((1.0, 2.0, 0.0)), "stacked")


def test_identical_builds_write_nothing():
    from armature_nodes import bridge
    from armature_nodes.apply import pipeline
    from armature_nodes.build import evaluate_tree
    from armature_nodes.store import record as record_store
    from armature_nodes.store import touched as touched_store

    obj, tree, _s, _o, node = _fresh("ArmatureNodesTransformNode")
    node.bone = "bone.002"
    node.inputs["Translation"].default_value = (1.0, 0.0, 0.0)
    base = record_store.read(obj)
    _name, defs = evaluate_tree(tree)
    target = bridge.overlay(base, defs)
    pipeline.apply(obj, base, target, touched_store.read(obj))
    second = pipeline.apply(obj, base, target, touched_store.read(obj))
    assert second.writes == 0, f"second identical build wrote {second.writes}"


# --- Rotation ------------------------------------------------------------------


def test_set_rotation_orients_the_bone():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesRotationNode")
    node.bone = "bone.002"
    node.use_rotation = True
    node.inputs["Rotation"].default_value = (math.radians(90.0), 0.0, 0.0)
    rest = _head(obj, "bone.002")
    _build(tree, BUILDS)
    world = obj.matrix_world @ obj.pose.bones["bone.002"].matrix
    _assert_close(world.to_translation(), rest, "head must stay put")
    e = world.to_euler("XYZ")
    _assert_close((e.x, e.y, e.z), (math.radians(90.0), 0.0, 0.0), "world rotation")


def test_ticking_set_rotation_does_not_jump():
    obj, tree, _s, _o, node = _fresh("ArmatureNodesRotationNode")
    node.bone = "bone.002"
    before = (obj.matrix_world @ obj.pose.bones["bone.002"].matrix).copy()
    node.use_rotation = True
    _build(tree)
    after = obj.matrix_world @ obj.pose.bones["bone.002"].matrix
    for ra, rb in zip(before, after):
        _assert_close(ra, rb, "matrix after ticking Set Rotation")


# --- nodes saved by the previous version -----------------------------------


def test_old_position_node_keeps_its_value():
    """Older Position nodes always applied their value; keep that intent."""
    obj, tree, _s, _o, node = _fresh("ArmatureNodesPositionNode")
    node.bone = "bone.002"
    node.inputs.remove(node.inputs["Offset"])  # as saved before Offset existed
    node.inputs["Position"].default_value = (2.0, 0.0, 3.0)
    node.ensure_inputs()
    assert node.use_position, "a set Position must stay in effect"
    _assert_close(node.inputs["Position"].default_value, (2.0, 0.0, 3.0), "value kept")
    _build(tree)
    _assert_close(_head(obj, "bone.002"), (2.0, 0.0, 3.0), "old node still moves the bone")


def test_old_rotation_socket_is_upgraded_to_degrees():
    """The old Rotation input was a metres-labelled Vector holding radians."""
    from armature_nodes.sockets import VectorSocket

    obj, tree, _s, _o, node = _fresh("ArmatureNodesRotationNode")
    node.inputs.remove(node.inputs["Offset"])
    node.inputs.remove(node.inputs["Rotation"])
    old = node.inputs.new(VectorSocket.bl_idname, "Rotation")
    old.default_value = (0.5, 0.0, 0.0)

    node.ensure_inputs()
    sock = node.inputs["Rotation"]
    assert sock.bl_idname == "ArmatureNodesRotationSocket"
    _assert_close(sock.default_value, (0.5, 0.0, 0.0), "radians carried over")
    assert node.use_rotation
    assert "Offset" in node.inputs
