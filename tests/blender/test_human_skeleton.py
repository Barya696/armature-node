"""The Human Skeleton group, and fitting it onto a character procedurally:
a small rig named like Rigify's controls, and a metaball mannequin whose
joints are known."""

import math

import bpy
from mathutils import Vector

import fixtures
import test_live_link as live

H = 1.6  # the mannequin, and the rig, are this tall


def _rig():
    """An armature with Rigify's control names at a 1.6 m human's places --
    not the group's 1.8 m defaults, so a marker that takes its bone moves --
    and the limbs' ``*_parent`` bones holding Rigify's pole switch, off."""
    from armature_nodes.human_skeleton import spec
    from armature_nodes.ops.bind import bind

    fixtures.ensure_registered()
    arm = bpy.data.armatures.new("rig")
    obj = bpy.data.objects.new("rig", arm)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    for _name, control, _parent, where, _how, side in spec():
        bone = arm.edit_bones.new(control)
        bone.head = (side * where[0] * H, where[1] * H, where[2] * H)
        bone.tail = bone.head + Vector((0.0, 0.0, 0.1))
    for name in ("upper_arm_parent.L", "upper_arm_parent.R", "thigh_parent.L", "thigh_parent.R"):
        bone = arm.edit_bones.new(name)
        bone.head, bone.tail = (0.0, 0.3, 1.0), (0.0, 0.3, 1.1)
    bpy.ops.object.mode_set(mode="OBJECT")
    for name in ("upper_arm_parent.L", "upper_arm_parent.R", "thigh_parent.L", "thigh_parent.R"):
        obj.pose.bones[name]["pole_vector"] = False
    bind(obj)
    tree, _src, _out = fixtures.make_tree(obj)
    obj.armature_nodes_tree = tree
    return obj, tree


def _mannequin(arms_down=45.0):
    """A character standing in an A-pose, metaball capsules turned into one
    mesh, and {joint: where it is}. Its feet are on z = 0 and the top of its
    head at H, as the proportions have them."""
    s = math.radians(arms_down)
    joints = {
        "pelvis": Vector((0.0, 0.0, 0.544 * H)), "chest": Vector((0.0, 0.0, 0.65 * H)),
        "neck": Vector((0.0, 0.0, 0.83 * H)), "head": Vector((0.0, 0.0, 0.899 * H)),
        "crown": Vector((0.0, 0.0, 0.94 * H)),  # a head 0.06 H round: its top at H
    }
    limbs = [("pelvis", "chest", 0.075), ("chest", "neck", 0.07), ("neck", "head", 0.03), ("head", "crown", 0.06)]
    for side, x in (("L", 1.0), ("R", -1.0)):
        shoulder = Vector((x * 0.099 * H, 0.0, 0.799 * H))
        down = Vector((x * math.cos(s), 0.0, -math.sin(s)))
        joints.update({
            f"shoulder.{side}": shoulder,
            f"elbow.{side}": shoulder + down * 0.15 * H,
            f"wrist.{side}": shoulder + down * 0.276 * H,
            f"tip.{side}": shoulder + down * 0.376 * H,
            f"hip.{side}": Vector((x * 0.05 * H, 0.0, 0.53 * H)),
            f"knee.{side}": Vector((x * 0.05 * H, 0.0, 0.28 * H)),
            f"ankle.{side}": Vector((x * 0.05 * H, 0.0, 0.035 * H)),
            f"toe.{side}": Vector((x * 0.05 * H, -0.09 * H, 0.022 * H)),  # its sole on the floor
        })
        limbs += [
            ("chest", f"shoulder.{side}", 0.04), (f"shoulder.{side}", f"elbow.{side}", 0.028),
            (f"elbow.{side}", f"wrist.{side}", 0.024), (f"wrist.{side}", f"tip.{side}", 0.02),
            ("pelvis", f"hip.{side}", 0.06), (f"hip.{side}", f"knee.{side}", 0.045),
            (f"knee.{side}", f"ankle.{side}", 0.033), (f"ankle.{side}", f"toe.{side}", 0.022),
        ]
    balls = bpy.data.metaballs.new("Mannequin")
    balls.resolution, balls.threshold = 0.012, 0.6
    for a, b, radius in limbs:
        head, tail = joints[a], joints[b]
        el = balls.elements.new(type="CAPSULE")
        el.co, el.radius, el.size_x = (head + tail) / 2.0, radius * H / 0.475, (tail - head).length / 2.0
        el.rotation = Vector((1.0, 0.0, 0.0)).rotation_difference(tail - head)
    obj = bpy.data.objects.new("Mannequin", balls)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.update()
    mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    bpy.data.objects.remove(obj)
    body = bpy.data.objects.new("Character", mesh)
    bpy.context.scene.collection.objects.link(body)
    bpy.context.view_layer.update()
    return body, joints


def _group_parts(group):
    markers = {n.markers[0].name: n for n in group.nodes if n.bl_idname == "ArmatureNodesMarkerNode"}
    wrap = next(n for n in group.nodes if n.bl_idname == "ArmatureNodesWrapNode")
    return markers, wrap


def _at(node):
    return Vector(node.marker_value(node.markers[0], "position"))


def test_the_group_holds_the_skeleton_and_is_made_once():
    from armature_nodes.human_skeleton import human_skeleton_group

    fixtures.ensure_registered()
    group = human_skeleton_group()
    assert human_skeleton_group() == group
    markers, _wrap = _group_parts(group)
    moves = {n.bone for n in group.nodes if n.bl_idname == "ArmatureNodesTransformNode"}
    assert len(markers) == 13 and len(moves) == 13
    assert {"torso", "chest", "head", "hand_ik.L", "foot_ik.R", "upper_arm_ik_target.L"} <= moves
    assert markers["Hand.L"]._parent_source().node == markers["Elbow.L"]
    assert markers["Knee.R"]._parent_source().node == markers["Foot.R"]
    assert len(group.marker_links) == 2, "the thighs: pelvis to each knee"
    assert markers["Elbow.L"].markers[0].wrap_role == "FREE", "a pole target is carried, not wrapped"
    switch = next(n for n in group.nodes if n.bl_idname == "ArmatureNodesRigifySwitchNode")
    pole = switch.switches["pole_vector"]
    assert pole.use and pole.flag


def test_on_a_rig_every_marker_takes_its_controls_place_and_nothing_moves():
    from armature_nodes.human_skeleton import add_to_tree, spec

    rig, tree = _rig()
    heads = {pb.name: (rig.matrix_world @ pb.matrix).to_translation() for pb in rig.pose.bones}
    node = add_to_tree(tree)
    live._build(tree)
    live._tick()
    markers, _wrap = _group_parts(node.node_tree)
    for name, control, _parent, _where, _how, _side in spec():
        live._assert_close(_at(markers[name]), heads[control], f"{name} on {control}")
    live._build(tree, live.BUILDS)
    for name, head in heads.items():
        live._assert_close((rig.matrix_world @ rig.pose.bones[name].matrix).to_translation(), head, name)


def test_the_poles_come_on_though_the_rig_is_read_before_the_first_build():
    """A switch takes a value set in Rigify's panel into itself -- but not
    while a build is due. A group is never dirty itself; asking only it,
    the switch took the rig's Off before the first build could turn the
    poles on, and they stayed off."""
    from armature_nodes import sync
    from armature_nodes.human_skeleton import add_to_tree

    rig, tree = _rig()
    add_to_tree(tree)
    sync.sync_bone_nodes()  # the rig read back before any build
    live._build(tree)
    assert all(rig.pose.bones[f"{n}.{s}"]["pole_vector"] for n in ("upper_arm_parent", "thigh_parent") for s in "LR")


def test_auto_pairs_finds_the_joints_of_a_standing_character():
    from armature_nodes.human_skeleton import human_skeleton_group
    from armature_nodes.nodes.wrap import MeshTarget, auto_pairs

    fixtures.ensure_registered()
    body, joints = _mannequin()
    markers, wrap = _group_parts(human_skeleton_group())
    wrap.target = body
    assert auto_pairs(wrap, MeshTarget(body, bpy.context.evaluated_depsgraph_get())) == 9
    misses = []
    for name, joint, tolerance in (
        ("Pelvis", "pelvis", 0.02), ("Chest", "chest", 0.02), ("Head", "head", 0.02),
        ("Foot.L", "ankle.L", 0.03), ("Foot.R", "ankle.R", 0.03),
        ("Hand.L", "wrist.L", 0.04), ("Hand.R", "wrist.R", 0.04),
    ):
        got = Vector(markers[name].markers[0].wrap_target)
        off = (got - joints[joint]).length
        if off >= tolerance * H:
            misses.append(f"{name} {off * 100:.1f} cm from the {joint}, at {tuple(round(v, 3) for v in got)}")
    assert not misses, "; ".join(misses)
    assert not markers["Elbow.L"].markers[0].wrap_pair, "a pole target is not paired"


def test_fit_to_mesh_pairs_and_wraps_in_one_go():
    from armature_nodes.human_skeleton import human_skeleton_group

    fixtures.ensure_registered()
    body, joints = _mannequin(arms_down=30.0)
    group = human_skeleton_group()
    markers, wrap = _group_parts(group)
    wrap.target = body
    result = bpy.ops.armature_nodes.wrap_run(step="FIT", tree=group.name, node=wrap.name)
    assert result == {"FINISHED"}, result
    assert wrap.wrap_stage == 3
    assert (_at(markers["Hand.L"]) - joints["wrist.L"]).length < 0.04 * H
    assert (_at(markers["Pelvis"]) - joints["pelvis"]).length < 0.02 * H
    elbow = _at(markers["Elbow.L"])
    assert elbow.x > joints["shoulder.L"].x and elbow.z < joints["shoulder.L"].z, "the pole target was not carried down the arm"


def test_fit_keeps_your_picks_and_redoes_its_own_pairs():
    """With two buttons there is nothing to clear: Fit keeps what you picked
    and finds the rest afresh every time -- a character moved since follows."""
    from armature_nodes.human_skeleton import human_skeleton_group
    from armature_nodes.nodes.wrap import MeshTarget, pair

    fixtures.ensure_registered()
    body, joints = _mannequin()
    group = human_skeleton_group()
    markers, wrap = _group_parts(group)
    wrap.target = body
    elbow = joints["elbow.L"]
    mesh = MeshTarget(body, bpy.context.evaluated_depsgraph_get())
    assert pair(wrap, (markers["Hand.L"].name, "marker"), mesh, Vector((elbow.x, -5.0, elbow.z)), Vector((0, 1, 0)))
    picked = Vector(markers["Hand.L"].markers[0].wrap_target)
    assert bpy.ops.armature_nodes.wrap_run(step="FIT", tree=group.name, node=wrap.name) == {"FINISHED"}
    live._assert_close(_at(markers["Hand.L"]), picked, "the picked hand")
    live._assert_close(_at(markers["Hand.R"]), picked * Vector((-1.0, 1.0, 1.0)), "its mirror, picked with it")
    pelvis = _at(markers["Pelvis"])
    body.location.x += 0.5
    bpy.context.view_layer.update()
    assert bpy.ops.armature_nodes.wrap_run(step="FIT", tree=group.name, node=wrap.name) == {"FINISHED"}
    live._assert_close(_at(markers["Pelvis"]), pelvis + Vector((0.5, 0.0, 0.0)), "the pelvis, after the character moved")
    live._assert_close(Vector(markers["Hand.L"].markers[0].wrap_target), picked, "your pick stays yours")


def test_selecting_another_character_fits_to_it_and_forgets_the_last_ones_picks():
    from armature_nodes.human_skeleton import human_skeleton_group
    from armature_nodes.nodes.wrap import MeshTarget, pair

    fixtures.ensure_registered()
    first, joints = _mannequin()
    second, _joints = _mannequin()
    second.location.x = 2.0
    bpy.context.view_layer.update()
    group = human_skeleton_group()
    markers, wrap = _group_parts(group)
    wrap.target = first
    wrist = joints["wrist.L"]
    mesh = MeshTarget(first, bpy.context.evaluated_depsgraph_get())
    assert pair(wrap, (markers["Hand.L"].name, "marker"), mesh, Vector((wrist.x, -5.0, wrist.z)), Vector((0, 1, 0)))
    with bpy.context.temp_override(active_object=second, selected_objects=[second]):
        assert bpy.ops.armature_nodes.wrap_run(step="FIT", tree=group.name, node=wrap.name) == {"FINISHED"}
    assert wrap.target == second
    assert not markers["Hand.L"].markers[0].wrap_picked, "a pick on the other character was kept"
    assert (_at(markers["Hand.L"]) - (wrist + Vector((2.0, 0.0, 0.0)))).length < 0.04 * H


def test_the_viewport_panel_finds_the_wrap_node_inside_the_group():
    from armature_nodes.human_skeleton import add_to_tree
    from armature_nodes.nodes.wrap import wrap_node_for

    rig, tree = _rig()
    node = add_to_tree(tree)
    with bpy.context.temp_override(active_object=rig):
        wrap = wrap_node_for(bpy.context)
    assert wrap is not None and wrap.id_data == node.node_tree
