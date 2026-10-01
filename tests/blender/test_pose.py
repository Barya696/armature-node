"""Pose: the rig posed by a mesh. A Rigify rig with a body skinned to it, the
Human Skeleton on the rig, and 'Generated' -- the same body in another pose,
its own mesh -- with the markers put on its joints, as the user does: Pose
finds the turns the markers cannot say."""

import math

import bpy
from mathutils import Euler, Vector

import fixtures
import test_live_link as live


def _rigify_scene():
    """(rig, body, tree, Human Skeleton group): a generated Rigify human, a
    body of capsules on its deform bones -- a nose and thumbs to show the
    turns -- skinned with automatic weights, and the Human Skeleton on it."""
    import addon_utils

    from armature_nodes.human_skeleton import add_to_tree
    from armature_nodes.ops.bind import bind

    addon_utils.enable("rigify", default_set=True)
    fixtures.ensure_registered()
    bpy.ops.object.armature_human_metarig_add()
    bpy.ops.pose.rigify_generate()
    rig = bpy.data.objects["rig"]
    bpy.data.objects["metarig"].hide_set(True)
    mw = rig.matrix_world
    balls = bpy.data.metaballs.new("Body")
    balls.resolution, balls.threshold = 0.025, 0.6

    def capsule(a, b, radius):
        el = balls.elements.new(type="CAPSULE")
        el.co, el.radius, el.size_x = (a + b) / 2.0, radius / 0.475, (b - a).length / 2.0
        el.rotation = Vector((1.0, 0.0, 0.0)).rotation_difference(b - a)

    radii = {"spine": 0.13, "spine.001": 0.12, "spine.002": 0.13, "spine.003": 0.14, "spine.004": 0.055,
             "spine.005": 0.05, "shoulder": 0.05, "upper_arm": 0.05, "forearm": 0.042, "hand": 0.035,
             "thigh": 0.075, "shin": 0.055, "foot": 0.045, "toe": 0.035}
    for bone in rig.data.bones:
        key = bone.name[4:] if bone.name.startswith("DEF-") else None
        radius = key and radii.get(key.rsplit(".", 1)[0] if key.endswith((".L", ".R")) else key)
        if radius:
            capsule(mw @ bone.head_local, mw @ bone.tail_local, radius)
    head = rig.data.bones["DEF-spine.006"]
    middle = mw @ head.head_local.lerp(head.tail_local, 0.45)
    el = balls.elements.new(type="BALL")
    el.co, el.radius = middle, 0.105 / 0.475
    capsule(middle + Vector((0.0, -0.08, 0.0)), middle + Vector((0.0, -0.13, -0.02)), 0.022)
    for side in "LR":
        hand = rig.data.bones[f"DEF-hand.{side}"]
        base = (mw @ hand.head_local).lerp(mw @ hand.tail_local, 0.3) + Vector((0.0, -0.03, 0.0))
        capsule(base, base + Vector((0.0, -0.06, -0.02)), 0.016)
    temp = bpy.data.objects.new("Balls", balls)
    bpy.context.scene.collection.objects.link(temp)
    bpy.context.view_layer.update()
    body = bpy.data.objects.new("Body", bpy.data.meshes.new_from_object(temp.evaluated_get(bpy.context.evaluated_depsgraph_get())))
    bpy.data.objects.remove(temp)
    bpy.context.scene.collection.objects.link(body)
    for obj in bpy.context.view_layer.objects:
        obj.select_set(obj in (body, rig))
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    bind(rig)
    tree, _src, _out = fixtures.make_tree(rig)
    rig.armature_nodes_tree = tree
    group = add_to_tree(tree).node_tree
    _settle(tree)
    return rig, body, tree, group


def _settle(tree, times=3):
    for _ in range(times):
        live._build(tree)
        live._tick()


def _markers(group):
    return {n.markers[0].name: n for n in group.nodes if n.bl_idname == "ArmatureNodesMarkerNode"}


def _turn(group, name, degrees):
    """Turn a marker as a rigid body turns: its children swing with it, and
    the offsets of the Position nodes they drive."""
    from armature_nodes.nodes.marker_base import deferred_marker_writes

    markers = _markers(group)
    R = Euler([math.radians(v) for v in degrees], "XYZ").to_matrix()
    node = markers[name]
    with deferred_marker_writes():
        was = Euler(node.marker_value(node.markers[0], "rotation"), "XYZ").to_matrix()
        node.write_marker(node.markers[0], "rotation", tuple((R @ was).to_euler("XYZ")))
    below, todo = [], [node]
    while todo:
        n = todo.pop()
        below.append(n)
        todo += [m for m in markers.values() if getattr(m._parent_source(), "node", None) == n]
    for n in below:
        for link in n.outputs[0].links:
            if link.to_node.bl_idname == "ArmatureNodesPositionNode":
                link.to_node.write_socket("Offset", tuple(R @ Vector(link.to_node.socket_value("Offset"))))
    group.mark_dirty()


def _joints(rig):
    """Where each Human Skeleton marker's joint is on the rig as it stands."""
    from armature_nodes.human_skeleton import spec

    out = {}
    for name, control, seed, _parent, _where, _side in spec():
        bone, along = seed or (control, 0.0) if control else (f"DEF-thigh{name[-2:]}", 0.0)
        pb = rig.pose.bones[bone]
        out[name] = rig.matrix_world @ pb.head.lerp(pb.tail, along)
    return out


def _gap(body, target):
    """How far the rigged body is from the target, on average."""
    from mathutils.bvhtree import BVHTree

    mesh = target.data
    bvh = BVHTree.FromPolygons([target.matrix_world @ v.co for v in mesh.vertices], [tuple(p.vertices) for p in mesh.polygons])
    evaluated = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    posed = evaluated.to_mesh()
    total = sum(bvh.find_nearest(body.matrix_world @ v.co)[3] for v in posed.vertices) / len(posed.vertices)
    evaluated.to_mesh_clear()
    return total


def _off(node, want):
    got = Euler(node.marker_value(node.markers[0], "rotation"), "XYZ").to_quaternion()
    angle = math.degrees(got.rotation_difference(want).angle)
    return min(angle, 360.0 - angle)


def test_pose_finds_the_turns_the_markers_cannot_say():
    """The chest leaning 12 degrees and the right foot turned 20: with every
    marker on the generated mesh's joints but the turns left as they were,
    Pose leans the chest and turns the foot, and the body fits closer."""
    from armature_nodes.nodes.marker_base import deferred_marker_writes

    rig, body, tree, group = _rigify_scene()
    markers = _markers(group)
    base = {k: (Vector(n.marker_value(n.markers[0], "position")), Euler(n.marker_value(n.markers[0], "rotation"), "XYZ"))
            for k, n in markers.items()}
    offsets = {n.name: tuple(n.socket_value("Offset")) for n in group.nodes if n.bl_idname == "ArmatureNodesPositionNode"}
    _turn(group, "Chest", (12.0, 0.0, 0.0))
    _turn(group, "Foot.R", (-20.0, 0.0, 0.0))
    _settle(tree)
    truth = {k: Euler(n.marker_value(n.markers[0], "rotation"), "XYZ").to_quaternion() for k, n in markers.items()}
    joints = _joints(rig)
    target = bpy.data.objects.new("Generated", bpy.data.meshes.new_from_object(body.evaluated_get(bpy.context.evaluated_depsgraph_get())))
    bpy.context.scene.collection.objects.link(target)
    # Back to the rig's own pose, then the markers put on Generated's joints.
    with deferred_marker_writes():
        for k, (_pos, rot) in base.items():
            markers[k].write_marker(markers[k].markers[0], "rotation", tuple(rot))
    for name, offset in offsets.items():
        group.nodes[name].write_socket("Offset", offset)
    with deferred_marker_writes():
        for k, at in joints.items():
            markers[k].write_marker(markers[k].markers[0], "position", tuple(at))
    group.mark_dirty()
    _settle(tree)
    wrap = next(n for n in group.nodes if n.bl_idname == "ArmatureNodesWrapNode")
    wrap.target = target
    before = _gap(body, target)
    chest, foot = _off(markers["Chest"], truth["Chest"]), _off(markers["Foot.R"], truth["Foot.R"])
    assert chest > 10.0 and foot > 15.0, (chest, foot)
    assert bpy.ops.armature_nodes.wrap_pose(tree=group.name, node=wrap.name) == {"FINISHED"}
    _settle(tree)
    after = _gap(body, target)
    assert _off(markers["Chest"], truth["Chest"]) < 3.0, _off(markers["Chest"], truth["Chest"])
    assert _off(markers["Foot.R"], truth["Foot.R"]) < foot / 2.0, _off(markers["Foot.R"], truth["Foot.R"])
    assert after < 0.7 * before, (before, after)
    for k, at in joints.items():  # the markers stay about where they were put: a strong landmark
        got = Vector(markers[k].marker_value(markers[k].markers[0], "position"))
        assert (got - at).length < 0.01, f"{k} moved {(got - at).length * 1000:.1f} mm"
    # Original and Wrapped: before Pose, and after it -- the turns and all.
    assert wrap.preview == "WRAPPED"
    wrap.preview = "ORIGINAL"
    assert _off(markers["Chest"], truth["Chest"]) > 10.0, "Original shows the chest as it was"
    for name, offset in offsets.items():  # and the poles, the head, the shoulders as they were
        assert (Vector(group.nodes[name].socket_value("Offset")) - Vector(offset)).length < 1e-5, name
    wrap.preview = "WRAPPED"
    assert _off(markers["Chest"], truth["Chest"]) < 3.0, "and Wrapped as Pose left it"


def test_only_the_marker_moved_drives_the_rig():
    """A marker moved by hand drives its control alone. The rest of the rig
    moves as Rigify moves it -- the chest rides down with the torso instead
    of staying pinned -- and every other marker is read off its bone: the
    hands on their IK controls, the elbow on the elbow."""
    rig, _body, tree, group = _rigify_scene()
    markers = _markers(group)
    pose, mw = rig.pose.bones, rig.matrix_world

    def at(name):
        return Vector(markers[name].marker_value(markers[name].markers[0], "position"))

    torso, chest = mw @ pose["torso"].head, mw @ pose["chest"].head
    handle = live._handle(markers["Pelvis"], markers["Pelvis"].markers[0])
    drop = Vector((0.0, 0.0, -0.1))
    handle.location = Vector(handle.location) + drop
    _settle(tree)
    assert (mw @ pose["torso"].head - (torso + drop)).length < 1e-4, "the moved marker drives its control"
    assert (mw @ pose["chest"].head - (chest + drop)).length < 1e-3, "the chest rides on the torso"
    for side in "LR":
        assert (at(f"Hand.{side}") - mw @ pose[f"hand_ik.{side}"].head).length < 1e-4, "a hand marker on its control"
        assert (at(f"Elbow.{side}") - mw @ pose[f"ORG-forearm.{side}"].head).length < 1e-3, "an elbow marker on the elbow"
    assert (at("Chest") - mw @ pose["chest"].head).length < 1e-4


def test_the_hips_sit_on_the_rigs_hip_joints():
    """A hip moves no control, so nothing seeds it: added to a rig, the
    skeleton puts its hips on the rig's own hip joints, whatever its size."""
    rig, _body, _tree, group = _rigify_scene()
    markers = _markers(group)
    for side in "LR":
        hip = Vector(markers[f"Hip.{side}"].marker_value(markers[f"Hip.{side}"].markers[0], "position"))
        joint = rig.matrix_world @ rig.pose.bones[f"DEF-thigh.{side}"].head
        assert (hip - joint).length < 1e-3, f"Hip.{side} is {(hip - joint).length * 1000:.0f} mm off its joint"


def test_a_chest_turned_far_does_not_swing_the_arms():
    """A shoulder's turn is a readout of its bone, re-read as a build starts
    -- before that build is evaluated, so it lags the chest by one. The
    elbows and the hands, parented to the shoulders but each driving a bone
    of its own, stay where they were put."""
    from armature_nodes.nodes.marker_base import deferred_marker_writes

    _rig, _body, tree, group = _rigify_scene()
    markers = _markers(group)
    places = {k: Vector(n.marker_value(n.markers[0], "position")) for k, n in markers.items()}
    R = Euler((math.radians(-15.0), 0.0, math.radians(40.0)), "XYZ").to_matrix()
    with deferred_marker_writes():
        chest = markers["Chest"]
        was = Euler(chest.marker_value(chest.markers[0], "rotation"), "XYZ")
        chest.write_marker(chest.markers[0], "rotation", tuple((R @ was.to_matrix()).to_euler("XYZ", was)))
        for k, n in markers.items():
            n.write_marker(n.markers[0], "position", tuple(places[k]))
    group.mark_dirty()
    _settle(tree)
    for k in ("Elbow.L", "Hand.L", "Elbow.R", "Hand.R"):
        got = Vector(markers[k].marker_value(markers[k].markers[0], "position"))
        assert (got - places[k]).length < 1e-3, f"{k} swung {(got - places[k]).length * 1000:.1f} mm"


def test_pose_needs_a_mesh_rigged_to_the_rig():
    import test_human_skeleton as hs

    from armature_nodes.human_skeleton import add_to_tree

    _rig, tree = hs._rig()
    group = add_to_tree(tree).node_tree
    wrap = next(n for n in group.nodes if n.bl_idname == "ArmatureNodesWrapNode")
    wrap.target, _joints_ = hs._mannequin()
    assert bpy.ops.armature_nodes.wrap_pose(tree=group.name, node=wrap.name) == {"CANCELLED"}
