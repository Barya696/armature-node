"""Marker handles (``handles.py``): a marker is its empty, moved with G / R /
S like any object; the gizmo only outlines the glows and lights the one under
the mouse. ``pick`` decides what is under it -- Pick Pairs uses it too. The
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


def test_g_moves_a_marker_as_it_moves_any_object():
    """A marker is its empty: moved as G moves an object, the marker takes
    the move on the next tick, and the bone follows."""
    obj, tree, _node, _mn, _marker, handle = _marker_on("ArmatureNodesPositionNode")
    handle.location = Vector(handle.location) + Vector((0.3, 0.0, 0.2))
    goal = Vector(handle.location)
    live._tick()
    live._build(tree)
    live._tick()
    assert (live._head(obj) - goal).length < 1e-3, (live._head(obj), goal)


def test_dragging_a_glow_moves_the_marker_freely():
    """Press on a glow and drag: the marker follows the mouse in the view's
    plane -- X locks it to that axis -- and the bone follows the marker."""
    from armature_nodes.handles import Drag

    obj, tree, _node, _mn, _marker, handle = _marker_on("ArmatureNodesPositionNode")
    was = Vector(handle.location)
    start = VIEW.to_screen(was)
    drag = Drag(handle, VIEW, start)
    drag.set_axis("X")
    drag.update(start + Vector((40.0, 20.0)))
    assert (Vector(handle.location) - was - Vector((0.4, 0.0, 0.0))).length < 1e-6, "X only"
    drag.set_axis("X")  # free again
    drag.update(start + Vector((40.0, 20.0)))
    goal = Vector(handle.location)
    assert (goal - was - Vector((0.4, 0.0, 0.2))).length < 1e-6
    live._tick()
    live._build(tree)
    live._tick()
    assert (live._head(obj) - goal).length < 1e-3, (live._head(obj), goal)


def test_a_click_selects_the_marker_for_g_r_s():
    from armature_nodes.handles import select_marker

    obj, _tree, _node, _mn, _marker, handle = _marker_on("ArmatureNodesPositionNode")
    obj.select_set(True)
    select_marker(bpy.context, handle)
    assert handle.select_get() and bpy.context.view_layer.objects.active == handle and not obj.select_get()
    select_marker(bpy.context, handle, extend=True)  # Shift on the active one takes it off
    assert not handle.select_get()


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


# --- sized in the scene, like the character ------------------------------------------


class ZoomView(FrontView):
    """The front view at ``ppm`` pixels per metre: zoomed in, or out."""

    def __init__(self, ppm):
        self.ppm = ppm

    def to_screen(self, co):
        return Vector((500.0 + self.ppm * co[0], 300.0 + self.ppm * co[2]))

    def on_plane(self, xy, depth):
        return Vector(((xy[0] - 500.0) / self.ppm, depth[1], (xy[1] - 300.0) / self.ppm))

    def world_per_pixel(self, co):
        return 1.0 / self.ppm


def test_a_handle_grows_as_you_zoom_in():
    from armature_nodes.handles import screen_scale

    near = screen_scale(ZoomView(4000.0), (0.0, 0.0, 1.0), 1.0, 1.8)
    far = screen_scale(ZoomView(1000.0), (0.0, 0.0, 1.0), 1.0, 1.8)
    assert abs(near / far - 4.0) < 1e-6, (near, far)


def test_a_full_body_view_shows_a_handle_at_its_sizes():
    """Framed so the figure is FIGURE_PIXELS tall, one size pixel is one pixel,
    and the node's Size still multiplies it."""
    from armature_nodes.handles import FIGURE_PIXELS, screen_scale

    view = ZoomView(FIGURE_PIXELS / 1.8)
    assert abs(screen_scale(view, (0.0, 0.0, 1.0), 1.0, 1.8) - 1.0) < 1e-6
    assert abs(screen_scale(view, (0.0, 0.0, 1.0), 2.0, 1.8) - 2.0) < 1e-6


def test_a_handle_is_a_share_of_its_figure():
    from armature_nodes.handles import screen_scale

    view = ZoomView(2000.0)
    small = screen_scale(view, (0.0, 0.0, 1.0), 1.0, 0.9)
    tall = screen_scale(view, (0.0, 0.0, 1.0), 1.0, 1.8)
    assert abs(tall / small - 2.0) < 1e-6, (small, tall)


def test_zoomed_far_out_a_handle_can_still_be_grabbed():
    from armature_nodes.handles import SMALLEST, screen_scale, ui_scale

    tiny = screen_scale(ZoomView(1.0), (0.0, 0.0, 1.0), 1.0, 1.8)
    assert abs(tiny - SMALLEST * ui_scale()) < 1e-6, tiny


def test_what_lights_up_follows_the_zoom():
    """The mouse 30 pixels from the handle: on it zoomed in, off it zoomed
    out -- the hover test uses the size the glow is drawn at. And that size
    is the figure's: this rig is 3.8 m tall, so 20 pixels out is still on it
    where a 1.8 m figure's handle would have ended. What lights up is what
    a press grabs."""
    import types

    from armature_nodes import handles

    _obj, _tree, _node, _mn, _marker, handle = _marker_on("ArmatureNodesPositionNode")
    gizmo = handles.ARMATURE_NODES_GT_marker_handles
    context = types.SimpleNamespace(region=object(), region_data=object())
    real = handles.View
    try:
        found = {}
        for ppm, gap in ((4000.0, 30.0), (400.0, 30.0), (400.0, 20.0)):
            handles.View = lambda region, rv3d, ppm=ppm: ZoomView(ppm)
            press = ZoomView(ppm).to_screen(handle.location) + Vector((gap, 0.0))
            found[ppm, gap] = gizmo.test_select(types.SimpleNamespace(hit=None), context, press)
            assert (found[ppm, gap] == handles.MOVE) == (handles.hovered_ident() is not None)
    finally:
        handles.View = real
        handles._set_hovered(None)
    assert found[4000.0, 30.0] == handles.MOVE, "zoomed in, the handle is under the mouse"
    assert found[400.0, 30.0] == -1, "zoomed out, the handle is smaller than 30 pixels"
    assert found[400.0, 20.0] == handles.MOVE, "sized to its 3.8 m rig, it reaches 20 pixels"


def test_a_marker_is_sized_to_its_rig_at_rest():
    from armature_nodes import primary_rig

    obj, _tree, _node, marker_node, _marker, _handle = _marker_on("ArmatureNodesPositionNode")
    primary_rig._rig_sizes.clear()
    # The fixture's four bones stand one above the other, from z = 0 to 3.8.
    assert abs(primary_rig.rig_size(obj) - 3.8) < 1e-4, primary_rig.rig_size(obj)
    assert abs(marker_node.effective_height() - 3.8) < 1e-4
    # Posing moves no handle size: it is the rig at rest.
    obj.pose.bones["bone.003"].location = (0.0, 2.0, 0.0)
    bpy.context.view_layer.update()
    primary_rig._rig_sizes.clear()
    assert abs(primary_rig.rig_size(obj) - 3.8) < 1e-4
