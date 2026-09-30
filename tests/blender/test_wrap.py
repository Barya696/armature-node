"""The Wrap Markers node: pairs, Snap, Attract, Stick, symmetry and Preview,
on meshes whose middle is known -- upright cylinders, a limb or two."""

import math

import bpy
from mathutils import Matrix, Vector

import fixtures
import test_live_link as live


def _limbs(xs=(0.0,), at=(0.0, 0.0, 0.0), radius=0.1, height=2.0, rings=41, segments=24):
    """One object of upright capped cylinders, one per local x in ``xs``,
    rings 5 cm apart: the middle of each is the line through (x, 0)."""
    verts, faces = [], []
    for x in xs:
        first = len(verts)
        for r in range(rings):
            z = height * r / (rings - 1)
            for s in range(segments):
                a = 2.0 * math.pi * s / segments
                verts.append((x + radius * math.cos(a), radius * math.sin(a), z))
        for r in range(rings - 1):
            for s in range(segments):
                a = first + r * segments + s
                b = first + r * segments + (s + 1) % segments
                faces.append((a, b, b + segments, a + segments))
        faces.append(tuple(reversed(range(first, first + segments))))
        faces.append(tuple(range(first + (rings - 1) * segments, first + rings * segments)))
    mesh = bpy.data.meshes.new("Limbs")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("Limbs", mesh)
    obj.location = at
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.update()
    return obj


def _tree():
    fixtures.ensure_registered()
    return bpy.data.node_groups.new("Wrap", "ArmatureNodeTreeType")


def _marker(tree, name, at, parent=None):
    node = tree.nodes.new("ArmatureNodesMarkerNode")
    node.name = node.label = name
    node.markers[0].name = name
    node.markers[0].set_position(at)
    if parent is not None:
        tree.links.new(parent.outputs[0], node.inputs["Parent"])
        node._track_parent()  # as the sync does: values relative, world kept
    return node


def _wrap(tree, target):
    wrap = tree.nodes.new("ArmatureNodesWrapNode")
    wrap.target = target
    return wrap


def _at(node):
    return Vector(node.marker_value(node.markers[0], "position"))


def _pair(wrap, node, z, x=0.0):
    """Pair a Marker node through a ray from the front, at (x, z)."""
    from armature_nodes.nodes.wrap import MeshTarget, pair

    mesh = MeshTarget(wrap.target, bpy.context.evaluated_depsgraph_get())
    return pair(wrap, (node.name, node.markers[0].key), mesh, Vector((x, -5.0, z)), Vector((0.0, 1.0, 0.0)))


def _run(wrap, step):
    result = bpy.ops.armature_nodes.wrap_run(step=step, tree=wrap.id_data.name, node=wrap.name)
    assert result == {"FINISHED"}, f"{step} did not run: {result}"


def _chain(tree):
    hip = _marker(tree, "Hip", (0.5, 0.2, 1.8))
    knee = _marker(tree, "Knee", (0.5, 0.2, 1.0), parent=hip)
    ankle = _marker(tree, "Ankle", (0.5, 0.2, 0.2), parent=knee)
    return hip, knee, ankle


def test_a_joint_is_paired_halfway_through_the_mesh():
    tree = _tree()
    wrap = _wrap(tree, _limbs())
    hip = _marker(tree, "Hip", (0.5, 0.2, 1.8))
    assert _pair(wrap, hip, 1.9) == 1
    live._assert_close(hip.markers[0].wrap_target, (0.0, 0.0, 1.9), "pair target")
    hip.markers[0].wrap_role = "SURFACE"
    _pair(wrap, hip, 1.9)
    live._assert_close(hip.markers[0].wrap_target, (0.0, -0.1, 1.9), "a landmark goes on the skin")


def test_a_ray_from_an_orthographic_view_still_finds_the_middle():
    """A front or side view's ray starts a clip distance away (~1000 m),
    where 32-bit floats lose the hit's place on the skin: the ray going on
    met the same skin again, and about half the joints went onto it."""
    from armature_nodes.nodes.wrap import MeshTarget

    mesh = MeshTarget(_limbs(), bpy.context.evaluated_depsgraph_get())
    for k in range(20):
        x, z = 0.004 * (k - 10), 0.3 + 0.07 * k
        spot = mesh.pick(Vector((x, -1000.0, z)), Vector((0.0, 1.0, 0.0)), "INSIDE")
        assert abs(spot.y) < 1e-3, f"ray {k} put the joint at depth {spot.y:.4f}, on the skin"


def test_snap_puts_the_pairs_on_their_places_and_the_skeleton_follows():
    tree = _tree()
    wrap = _wrap(tree, _limbs())
    hip, knee, ankle = _chain(tree)
    _pair(wrap, hip, 1.9)
    _pair(wrap, ankle, 0.1)
    _run(wrap, "SNAP")
    live._assert_close(_at(hip), (0.0, 0.0, 1.9), "hip")
    live._assert_close(_at(ankle), (0.0, 0.0, 0.1), "ankle, a grandchild")
    live._assert_close(_at(knee), (0.0, 0.0, 1.0), "the knee, carried between them")
    assert wrap.wrap_stage == 1


def test_attract_and_stick_put_a_joint_in_the_middle_of_its_limb():
    tree = _tree()
    wrap = _wrap(tree, _limbs())
    hip = _marker(tree, "Hip", (0.0, 0.0, 1.9))
    knee = _marker(tree, "Knee", (0.04, 0.03, 1.0), parent=hip)
    ankle = _marker(tree, "Ankle", (0.0, 0.0, 0.1), parent=knee)
    _pair(wrap, hip, 1.9)
    _pair(wrap, ankle, 0.1)
    _run(wrap, "ATTRACT")
    off = Vector((_at(knee).x, _at(knee).y)).length
    assert off < 0.025, f"attract left the knee {off:.4f} off the middle"
    _run(wrap, "STICK")
    live._assert_close(_at(knee), (0.0, 0.0, 1.0), "stuck knee")
    live._assert_close(_at(hip), (0.0, 0.0, 1.9), "the pairs held")


def test_a_surface_marker_sticks_to_the_skin():
    tree = _tree()
    limb = _limbs()
    wrap = _wrap(tree, limb)
    # Sided, with no partner here: a marker without a side near the middle
    # would be a middle-line one, and Symmetric would keep it on the plane.
    marker = _marker(tree, "Knee.L", (0.05, 0.0, 1.0))
    marker.markers[0].wrap_role = "SURFACE"
    _run(wrap, "STICK")
    local = limb.matrix_world.inverted() @ _at(marker)
    found, co, _normal, _index = limb.closest_point_on_mesh(local)
    assert found and (co - local).length < 1e-4, f"not on the skin: {tuple(_at(marker))}"


def test_a_fixed_marker_stays_and_a_free_one_is_carried():
    tree = _tree()
    wrap = _wrap(tree, _limbs())
    hip, knee, ankle = _chain(tree)
    knee.markers[0].wrap_role = "FREE"
    ankle.markers[0].wrap_role = "FIXED"
    _pair(wrap, hip, 1.9)
    before = _at(knee)
    _run(wrap, "SNAP")
    live._assert_close(_at(ankle), (0.5, 0.2, 0.2), "the fixed ankle")
    live._assert_close(_at(hip), (0.0, 0.0, 1.9), "hip")
    assert (_at(knee) - before).length > 0.1, "the free knee was not carried"


def test_a_fixed_marker_takes_no_pair():
    tree = _tree()
    wrap = _wrap(tree, _limbs())
    hip = _marker(tree, "Hip", (0.5, 0.2, 1.8))
    hip.markers[0].wrap_role = "FIXED"
    assert _pair(wrap, hip, 1.9) == 0
    assert not hip.markers[0].wrap_pair


def test_symmetric_pairs_mirror_across_the_middle_of_the_mesh():
    """The mesh stands at x = 2, legs at 1.7 and 2.3: Knee.R is paired with
    the mirror image across x = 2, not across the world's x = 0."""
    tree = _tree()
    wrap = _wrap(tree, _limbs(xs=(-0.3, 0.3), at=(2.0, 0.0, 0.0)))
    left = _marker(tree, "Knee.L", (2.35, 0.0, 1.0))
    right = _marker(tree, "Knee.R", (1.6, 0.05, 1.0))
    assert _pair(wrap, left, 1.0, x=2.3) == 2
    live._assert_close(right.markers[0].wrap_target, (1.7, 0.0, 1.0), "the mirrored pair")
    _run(wrap, "SNAP")
    live._assert_close(_at(left), (2.3, 0.0, 1.0), "Knee.L")
    live._assert_close(_at(right), (1.7, 0.0, 1.0), "Knee.R")
    from armature_nodes.nodes.wrap import unpair

    unpair(wrap, (left.name, left.markers[0].key))
    assert not left.markers[0].wrap_pair and not right.markers[0].wrap_pair, "forget one, forget both"
    wrap.symmetric = False  # the mirror toggle off: the marker alone
    assert _pair(wrap, left, 1.0, x=2.3) == 1 and not right.markers[0].wrap_pair


def test_a_middle_marker_stays_on_the_middle_of_a_mesh_imported_at_a_hundredth():
    """A character imported from centimetres: scale 0.01, turned up. The
    middle is measured in the world, not in the mesh's own units -- in
    those, a pelvis 5 mm off the middle is half a unit off, far past 5% of
    this 20 cm skeleton, and it was not counted as a middle marker."""
    tree = _tree()
    legs = _limbs(xs=(-0.1, 0.1))
    imported = Matrix.Rotation(math.radians(90.0), 4, "X") @ Matrix.Scale(0.01, 4)
    legs.data.transform(imported.inverted())
    legs.matrix_world = imported
    bpy.context.view_layer.update()
    wrap = _wrap(tree, legs)
    pelvis = _marker(tree, "Pelvis", (0.005, 0.0, 1.0))
    pelvis.markers[0].wrap_role = "FREE"
    # Both hips move 4 cm in -x, so the bones alone leave the pelvis 3.5 cm
    # past the middle: only the symmetry puts it back.
    left = _marker(tree, "Hip.L", (0.14, 0.0, 0.9), parent=pelvis)
    right = _marker(tree, "Hip.R", (-0.06, 0.0, 0.9), parent=pelvis)
    assert _pair(wrap, left, 0.9, x=0.1) == 2
    _run(wrap, "SNAP")
    live._assert_close(_at(left), (0.1, 0.0, 0.9), "Hip.L")
    live._assert_close(_at(right), (-0.1, 0.0, 0.9), "Hip.R, mirrored")
    assert abs(_at(pelvis).x) < 1e-4, f"the pelvis stayed {_at(pelvis).x:.3f} off the middle"


def test_a_click_picks_the_marker_under_it_not_the_one_lit_last():
    """The glow lit under the mouse is only as fresh as the last mouse move
    over the view: leave the view over its header and it still names the
    marker passed last. A click is tested where it lands."""
    from armature_nodes import handles
    from armature_nodes.nodes.wrap import _marker_under

    _obj, tree, _s, _o, pos = live._fresh("ArmatureNodesPositionNode")
    first, _m = live._wire_marker(tree, pos, at=(0.0, 0.0, 1.0))
    second = tree.nodes.new("ArmatureNodesMarkerNode")
    second.markers[0].set_position((0.5, 0.0, 1.0))
    tree.links.new(second.outputs[0], pos.inputs["Offset"])
    live._build(tree)
    live._full_tick()

    class FrontView:
        """Front orthographic, 1 mm a pixel, the origin at pixel (500, 0)."""

        def to_screen(self, co):
            return Vector((500.0 + co[0] * 1000.0, co[2] * 1000.0))

        def world_per_pixel(self, co):
            return 0.001

    lit = (tree.name, second.name, "marker")
    handles._hovered = lit  # the stale glow: the other marker
    try:
        at = FrontView().to_screen(_at(first))
        assert _marker_under(FrontView(), at, tree.name) == (first.name, "marker")
        assert _marker_under(FrontView(), at + Vector((200.0, 0.0)), tree.name) is None
    finally:
        handles._hovered = None


def test_symmetrize_mirrors_the_skeleton_and_its_pairs():
    tree = _tree()
    wrap = _wrap(tree, None)
    left = _marker(tree, "Hand.L", (0.5, 0.1, 1.4))
    right = _marker(tree, "Hand.R", (-0.3, 0.0, 1.5))
    chest = _marker(tree, "Chest", (0.12, 0.0, 1.3))
    left.markers[0].wrap_pair = True
    left.markers[0].wrap_target = (0.6, 0.0, 1.4)
    result = bpy.ops.armature_nodes.wrap_symmetrize(tree=tree.name, node=wrap.name)
    assert result == {"FINISHED"}
    # The skeleton's own middle is x = 0.1, where the hands meet.
    live._assert_close(_at(left), (0.5, 0.05, 1.45), "Hand.L")
    live._assert_close(_at(right), (-0.3, 0.05, 1.45), "Hand.R")
    live._assert_close(_at(chest), (0.1, 0.0, 1.3), "Chest, onto the middle")
    assert right.markers[0].wrap_pair
    live._assert_close(right.markers[0].wrap_target, (-0.4, 0.0, 1.4), "the mirrored pair")


def test_original_and_wrapped_show_the_skeleton_before_and_after_the_fit():
    """The switch at the bottom of the node. Each pose keeps what was done
    to it while it showed."""
    tree = _tree()
    wrap = _wrap(tree, _limbs())
    hip, knee, ankle = _chain(tree)
    chain = (hip, knee, ankle)
    wrap.preview = "WRAPPED"
    assert wrap.preview == "ORIGINAL", "nothing fitted yet, nothing to show"
    _pair(wrap, hip, 1.9)
    _run(wrap, "SNAP")
    _run(wrap, "STICK")
    assert wrap.preview == "WRAPPED"
    del wrap["wrap_preview"]  # a node saved before the switch: its kept original means a fit shows
    assert wrap.preview == "WRAPPED"
    knee.markers[0].set_position(knee.markers[0].position + Vector((0.0, 0.0, 0.1)))  # touched up
    fitted = [_at(n) for n in chain]
    wrap.preview = "ORIGINAL"
    for node, want in zip(chain, ((0.5, 0.2, 1.8), (0.5, 0.2, 1.0), (0.5, 0.2, 0.2))):
        live._assert_close(_at(node), want, f"{node.name}, original")
    wrap.preview = "ORIGINAL"  # again: nothing to keep from the other
    wrap.preview = "WRAPPED"
    for node, want in zip(chain, fitted):
        live._assert_close(_at(node), want, f"{node.name}, wrapped as touched up")


def test_a_fit_from_the_original_fits_the_skeleton_as_it_is_now():
    tree = _tree()
    wrap = _wrap(tree, _limbs())
    hip, knee, ankle = _chain(tree)
    _pair(wrap, hip, 1.9)
    _run(wrap, "SNAP")
    wrap.preview = "ORIGINAL"
    ankle.markers[0].set_position(ankle.markers[0].position + Vector((0.0, 0.0, -0.1)))  # a longer shin
    shin = (_at(knee) - _at(ankle)).length
    assert abs(shin - 0.9) < 1e-4, shin
    _run(wrap, "SNAP")
    assert abs((_at(knee) - _at(ankle)).length - shin) < 1e-3, "the fit kept the shin it was shown"


def test_undo_in_pick_pairs_takes_back_the_last_pair():
    """Ctrl Z while picking, or the button beside Pick Pairs: a pair made
    or forgotten goes back to what it was, its marker picked again to put
    it right."""
    from armature_nodes.nodes import wrap as module

    tree = _tree()
    wrap = _wrap(tree, _limbs())
    hip, knee, _ankle = _chain(tree)
    mesh = module.MeshTarget(wrap.target, bpy.context.evaluated_depsgraph_get())
    ident = (hip.name, hip.markers[0].key)
    front = Vector((0.0, 1.0, 0.0))
    module._picking = {"tree": tree.name, "chosen": None, "spots": [], "undo": [], "cursor": None}
    try:
        for z in (1.9, 1.7):  # paired, then again elsewhere
            module._picking["chosen"] = ident
            assert module.set_pair(wrap, mesh, (Vector((0.0, -5.0, z)), front))
        knee.markers[0].wrap_pair = True  # paired since, not by a pick: a Fit's own pairs
        module._picking["chosen"] = ident
        assert not module.set_pair(wrap, mesh, (Vector((3.0, -5.0, 1.0)), front)), "a miss"
        assert len(module._picking["undo"]) == 2 and module._picking["chosen"] == ident
        module.set_pair(wrap)  # X: forgotten
        assert not hip.markers[0].wrap_pair and module._picking["chosen"] is None
        assert module.undo_pick(wrap)
        assert hip.markers[0].wrap_pair and module._picking["chosen"] == ident, "back, and picked again"
        live._assert_close(hip.markers[0].wrap_target, (0.0, 0.0, 1.7), "the last pair")
        module.undo_pick(wrap)
        live._assert_close(hip.markers[0].wrap_target, (0.0, 0.0, 1.9), "the first pair")
        module.undo_pick(wrap)
        assert not hip.markers[0].wrap_pair and not hip.markers[0].wrap_picked
        assert not module.undo_pick(wrap), "nothing left to undo"
        assert knee.markers[0].wrap_pair, "a pair no pick made is left alone"
    finally:
        module._picking = None


def test_snap_starts_from_the_original_every_time():
    tree = _tree()
    wrap = _wrap(tree, _limbs())
    hip, knee, ankle = _chain(tree)
    _pair(wrap, hip, 1.9)
    _pair(wrap, ankle, 0.1)
    _run(wrap, "SNAP")
    knee.markers[0].set_position(knee.markers[0].position + Vector((0.3, 0.0, 0.0)))  # dragged since
    _run(wrap, "SNAP")
    live._assert_close(_at(knee), (0.0, 0.0, 1.0), "the knee after a second Snap")


def test_stopping_a_step_puts_the_markers_back():
    """What Esc does to a step running in the background."""
    from armature_nodes.nodes.wrap import WrapRun

    tree = _tree()
    wrap = _wrap(tree, _limbs())
    hip, knee, ankle = _chain(tree)
    _pair(wrap, hip, 1.9)
    run = WrapRun(wrap, "SNAP", bpy.context.evaluated_depsgraph_get())
    for _ in range(3):
        assert run.advance()
    assert (_at(hip) - Vector((0.5, 0.2, 1.8))).length > 1e-3, "the step had not started"
    run.cancel()
    live._assert_close(_at(hip), (0.5, 0.2, 1.8), "hip")
    live._assert_close(_at(ankle), (0.5, 0.2, 0.2), "ankle")


def test_a_step_needs_a_mesh_and_snap_needs_a_pair():
    tree = _tree()
    wrap = _wrap(tree, None)
    _marker(tree, "Hip", (0.0, 0.0, 1.0))
    assert bpy.ops.armature_nodes.wrap_run(step="SNAP", tree=tree.name, node=wrap.name) == {"CANCELLED"}
    wrap.target = _limbs()
    assert bpy.ops.armature_nodes.wrap_run(step="SNAP", tree=tree.name, node=wrap.name) == {"CANCELLED"}
    assert bpy.ops.armature_nodes.wrap_run(step="STICK", tree=tree.name, node=wrap.name) == {"FINISHED"}


def test_a_skeleton_landmark_carries_its_face():
    tree = _tree()
    wrap = _wrap(tree, _limbs())
    skeleton = tree.nodes.new("ArmatureNodesSkeletonNode")
    nose, eye = skeleton.marker_by_key("nose"), skeleton.marker_by_key("eye_l")
    nose0, eye0 = Vector(nose.position), Vector(eye.position)
    from armature_nodes.nodes.wrap import MeshTarget, pair

    mesh = MeshTarget(wrap.target, bpy.context.evaluated_depsgraph_get())
    pair(wrap, (skeleton.name, "nose"), mesh, Vector((0.0, -5.0, 1.5)), Vector((0.0, 1.0, 0.0)))
    _run(wrap, "SNAP")
    live._assert_close(nose.position, (0.0, 0.0, 1.5), "nose")
    live._assert_close(Vector(eye.position) - eye0, Vector(nose.position) - nose0, "the eye rode with it")


def test_the_rig_follows_the_wrapped_marker():
    obj, tree, _s, _o, pos = live._fresh("ArmatureNodesPositionNode")
    pos.bone = "bone.002"
    marker_node, _m = live._wire_marker(tree, pos)
    live._build(tree)
    live._tick()
    wrap = _wrap(tree, _limbs(at=(1.0, 0.0, 0.0)))
    from armature_nodes.nodes.wrap import MeshTarget, pair

    mesh = MeshTarget(wrap.target, bpy.context.evaluated_depsgraph_get())
    assert pair(wrap, (marker_node.name, "marker"), mesh, Vector((1.0, -5.0, 1.2)), Vector((0.0, 1.0, 0.0)))
    _run(wrap, "SNAP")
    live._assert_close(_at(marker_node), (1.0, 0.0, 1.2), "the marker")
    live._build(tree, live.BUILDS)
    live._assert_close(live._head(obj), (1.0, 0.0, 1.2), "the bone it drives")
