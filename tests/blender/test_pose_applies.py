"""A Bone node's pose must actually reach the rig -- in both Output modes.

The reported failure: mode toggling stopped, but adjusting a control did
nothing. Two causes, both covered here.

* **Full Rig never applied pose at all.** ``pose_location`` was produced by
  every Transform-category node and consumed only by ``bridge.overlay``, which
  only Modify mode calls. In Full Rig the legacy pass wrote constraints and
  custom shapes and silently dropped the pose.
* **The defer guard blocked pose-only edits.** Rebuilding bones has to wait
  until the user leaves Pose mode, but writing ``pbone.matrix`` needs no
  operator and no mode change -- so a pose-only edit must never defer, or
  adjusting a control from the node while posing does nothing.
"""

import bpy

import fixtures


def _world_head(obj, name):
    return (obj.matrix_world @ obj.pose.bones[name].matrix).to_translation()


def _setup(mode):
    from armature_nodes.ops.bind import bind

    obj = fixtures.make_rig()
    bind(obj)
    tree, src, out = fixtures.make_tree(obj)
    out.mode = mode
    if mode == "FULL":
        out.armature_name = obj.name

    # Insert the Bone node ON the wire. An unconnected node is never reached
    # by evaluate_tree, which walks back from the Output.
    for link in list(tree.links):
        tree.links.remove(link)
    # The fixture connects bones and locks channels, and Blender silently
    # refuses a pose write on either. Free the test bone so this exercises the
    # apply path rather than Blender's lock semantics -- the blocked case has
    # its own test below.
    import bpy

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    obj.data.edit_bones["bone.002"].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.pose.bones["bone.002"].lock_location = (False, False, False)

    bone = tree.nodes.new("ArmatureNodesBoneNode")
    bone.bone = "bone.002"
    bone.use_location = True
    tree.links.new(src.outputs["Rig"], bone.inputs["Parent"])
    tree.links.new(bone.outputs["Rig"], out.inputs["Rig"])
    return obj, tree, bone


def _assert_moved(obj, tree, bone, mode):
    from armature_nodes.build import build_armature_from_tree

    build_armature_from_tree(tree)
    target = (3.0, 0.0, 5.0)
    bone.inputs["Position"].default_value = target
    build_armature_from_tree(tree)

    got = _world_head(obj, "bone.002")
    for axis, (a, b) in enumerate(zip(got, target)):
        assert abs(a - b) < 1e-4, f"{mode}: axis {axis} is {a}, expected {b}"


def test_pose_applies_in_modify_mode():
    obj, tree, bone = _setup("MODIFY")
    _assert_moved(obj, tree, bone, "MODIFY")


def test_pose_applies_in_full_rig_mode():
    """The path that discarded every pose."""
    obj, tree, bone = _setup("FULL")
    _assert_moved(obj, tree, bone, "FULL")


def test_pose_applies_while_in_pose_mode():
    """A pose-only edit must not be deferred: that is when it matters most."""
    from armature_nodes.build import build_armature_from_tree

    obj, tree, bone = _setup("FULL")
    build_armature_from_tree(tree)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="POSE")
    assert obj.mode == "POSE"

    bone.inputs["Position"].default_value = (0.0, 2.0, 4.0)
    build_armature_from_tree(tree)

    assert obj.mode == "POSE", "applying a pose changed the mode"
    got = _world_head(obj, "bone.002")
    assert abs(got.y - 2.0) < 1e-4 and abs(got.z - 4.0) < 1e-4, f"bone did not move: {got[:]}"


def test_unchecked_components_are_left_alone():
    from armature_nodes.build import build_armature_from_tree

    obj, tree, bone = _setup("MODIFY")
    bone.use_rotation = False
    bone.use_scale = False
    before = obj.pose.bones["bone.002"].matrix_basis.to_scale()

    bone.inputs["Position"].default_value = (1.0, 1.0, 1.0)
    build_armature_from_tree(tree)

    after = obj.pose.bones["bone.002"].matrix_basis.to_scale()
    for a, b in zip(before, after):
        assert abs(a - b) < 1e-4, "scale changed although its checkbox was off"


def test_convert_rig_produces_modify_mode():
    """A generated rig must not be handed to the bone-rebuilding path."""
    from armature_nodes.build import find_output_node
    from armature_nodes.sync import resync_tree_from_armature

    obj = fixtures.make_rig()
    fixtures.ensure_registered()
    tree = resync_tree_from_armature(obj, shapes_only=False, full=True)
    out = find_output_node(tree)
    # resync_tree_from_armature(full=True) is the low-level call; the operator
    # now passes full=False. Assert the operator's own default instead.
    tree2 = resync_tree_from_armature(obj, shapes_only=False, full=False)
    assert find_output_node(tree2).mode == "MODIFY"
    assert out is not None


def test_a_blocked_pose_is_reported_not_silently_dropped():
    """Connected / locked bones refuse the write; saying so is the fix.

    Blender accepts ``pbone.matrix = ...`` and then discards it. Counting that
    as a successful write is why the addon looked broken: no error, no motion.
    """
    from armature_nodes.apply import pipeline
    from armature_nodes.model import ops
    from armature_nodes.ops.bind import bind
    from armature_nodes.store import record as record_store

    obj = fixtures.make_rig()
    bind(obj)
    pb = obj.pose.bones["bone.002"]
    assert pb.bone.use_connect and any(pb.lock_location), "fixture should be blocked"

    base = record_store.read(obj)
    target = ops.set_pose(base, ["bone.002"], location=(3.0, 0.0, 5.0))
    result = pipeline.apply(obj, base, target, {})

    assert result.writes == 0, "a write that did nothing must not be counted"
    assert result.errors, "a blocked pose must be reported"
    message = result.errors[0]
    assert "bone.002" in message
    assert "connected" in message or "locked" in message, message


def test_typing_into_an_unlinked_position_socket_works():
    """The reported bug: the socket accepted typing and discarded it."""
    from armature_nodes.build import build_armature_from_tree

    obj, tree, bone = _setup("MODIFY")
    sock = bone.inputs["Position"]
    assert not sock.is_linked, "this test is about the UNLINKED socket"

    sock.default_value = (2.0, 0.0, 4.0)
    build_armature_from_tree(tree)

    got = _world_head(obj, "bone.002")
    assert abs(got.x - 2.0) < 1e-4 and abs(got.z - 4.0) < 1e-4, f"got {got[:]}"


def test_a_linked_marker_overrides_the_typed_value():
    from armature_nodes.build import build_armature_from_tree

    obj, tree, bone = _setup("MODIFY")
    bone.inputs["Position"].default_value = (9.0, 9.0, 9.0)

    skel = tree.nodes.new("ArmatureNodesSkeletonNode")
    marker = skel.markers[0]
    marker.set_position((1.0, 0.0, 3.0))
    sock = next(s for s in skel.outputs if s.marker_key == marker.key)
    tree.links.new(sock, bone.inputs["Position"])

    build_armature_from_tree(tree)
    got = _world_head(obj, "bone.002")
    assert abs(got.x - 1.0) < 1e-4 and abs(got.z - 3.0) < 1e-4, (
        f"marker should win over the typed value, got {got[:]}"
    )


def test_unticked_location_is_a_live_readout():
    """Off, the node does not drive the bone: it shows where the bone is."""
    import bpy
    from armature_nodes.build import build_armature_from_tree

    obj, tree, bone = _setup("MODIFY")
    bone.use_location = False
    build_armature_from_tree(tree)

    pb = obj.pose.bones["bone.002"]
    pb.location = (0.0, 0.5, 0.0)  # the user grabs the bone
    bpy.context.view_layer.update()
    moved = _world_head(obj, "bone.002")

    build_armature_from_tree(tree)
    got = _world_head(obj, "bone.002")
    assert (got - moved).length < 1e-4, "an undriven bone was snapped back"
    shown = bone.inputs["Position"].default_value
    assert (moved - type(moved)(shown)).length < 1e-4, "Position does not show the bone"


def test_typing_into_position_takes_the_bone():
    """Typing a position is asking for it, whatever the checkbox said."""
    from armature_nodes.build import build_armature_from_tree

    obj, tree, bone = _setup("MODIFY")
    bone.use_location = False
    bone.inputs["Position"].default_value = (7.0, 0.0, 7.0)
    assert bone.use_location, "typing did not take the bone over"
    build_armature_from_tree(tree)
    got = _world_head(obj, "bone.002")
    assert abs(got.x - 7.0) < 1e-4 and abs(got.z - 7.0) < 1e-4, f"got {got[:]}"
