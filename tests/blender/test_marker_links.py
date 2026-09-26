"""Lines between Marker nodes (``marker_links``).

A marker joins any number of other markers. A join is a line and nothing
more: nothing follows anything. In the node editor its two ends are ports on
the nodes' borders, each facing the other node, sliding round the node as
either one moves -- Fusion's way, not Blender's fixed sockets.

The editor itself cannot run here (no window, and nodes are never drawn, so
they have no size), so the node rectangles are given, and the port gizmo's
callbacks are called the way Blender calls them.
"""

import contextlib
import types

import bpy
from mathutils import Vector

import fixtures

MARKER = "ArmatureNodesMarkerNode"
W, H = 180.0, 200.0  # a node's size in the editor, at 100% UI scale


def _tree():
    fixtures.ensure_registered()
    return bpy.data.node_groups.new("Lines", "ArmatureNodeTreeType")


def _markers(tree, *spots):
    nodes = []
    for x, y in spots:
        node = tree.nodes.new(MARKER)
        node.location = (x, y)
        nodes.append(node)
    return nodes


def _pairs(tree):
    from armature_nodes.marker_links import links_of

    return {frozenset((a.name, b.name)) for a, b in links_of(tree)}


def _pair(a, b):
    return frozenset((a.name, b.name))


@contextlib.contextmanager
def _drawn():
    """Every node has the size it would be drawn at."""
    from armature_nodes import marker_links as ml

    def view_rect(node):
        scale = ml.ui_scale()
        loc = node.location_absolute
        left, top = loc.x * scale, loc.y * scale
        return (left, top, left + W * scale, top - H * scale)

    real = ml.view_rect
    ml.view_rect = view_rect
    try:
        yield ml
    finally:
        ml.view_rect = real


class _View2D:
    """The editor's view: view units to region pixels, with zoom and pan."""

    def __init__(self, zoom=1.0, pan=(300.0, 400.0)):
        self.zoom, self.pan = zoom, pan

    def view_to_region(self, x, y, clip=True):
        return (round(x * self.zoom + self.pan[0]), round(y * self.zoom + self.pan[1]))

    def region_to_view(self, x, y):
        return ((x - self.pan[0]) / self.zoom, (y - self.pan[1]) / self.zoom)


def _context(tree, zoom=1.0):
    region = types.SimpleNamespace(view2d=_View2D(zoom), width=1600, height=1000)
    space = types.SimpleNamespace(type="NODE_EDITOR", tree_type="ArmatureNodeTreeType", edit_tree=tree)
    return types.SimpleNamespace(region=region, space_data=space)


def _mouse(xy):
    return types.SimpleNamespace(mouse_region_x=xy[0], mouse_region_y=xy[1])


# --- The joins ------------------------------------------------------------------


def test_a_marker_joins_as_many_markers_as_you_like():
    from armature_nodes.marker_links import join, partners

    tree = _tree()
    a, b, c, d = _markers(tree, (0, 0), (400, 0), (0, 400), (400, 400))
    for other in (b, c, d):
        assert join(a, other), f"could not join {other.name}"
    assert join(b, c)
    assert {n.name for n in partners(a)} == {b.name, c.name, d.name}
    assert {n.name for n in partners(c)} == {a.name, b.name}
    assert len(tree.marker_links) == 4


def test_a_pair_is_joined_once_either_way_round_and_never_to_itself():
    from armature_nodes.marker_links import join, joined

    tree = _tree()
    a, b = _markers(tree, (0, 0), (400, 0))
    assert join(a, b)
    assert not join(b, a), "the same pair, the other way round"
    assert not join(a, a), "a marker joined to itself"
    assert joined(b, a) and len(tree.marker_links) == 1


def test_a_line_is_only_a_line():
    """Joining moves nothing, and moving one marker leaves the other."""
    from armature_nodes.marker_links import join

    tree = _tree()
    a, b = _markers(tree, (0, 0), (400, 0))
    a.markers[0].position = (1.0, 0.0, 0.0)
    b.markers[0].position = (0.0, 0.0, 2.0)
    join(a, b)
    a.markers[0].position = (3.0, 3.0, 3.0)
    assert Vector(b.markers[0].position) == Vector((0.0, 0.0, 2.0)), "the other marker moved"


def test_unjoining_leaves_the_other_lines():
    from armature_nodes.marker_links import join, unjoin

    tree = _tree()
    a, b, c = _markers(tree, (0, 0), (400, 0), (0, 400))
    join(a, b)
    join(a, c)
    assert unjoin(b, a)
    assert _pairs(tree) == {_pair(a, c)}
    assert not unjoin(a, b), "removing a line that is not there"


def test_deleting_a_marker_takes_its_lines_with_it():
    from armature_nodes.marker_links import join

    tree = _tree()
    a, b, c = _markers(tree, (0, 0), (400, 0), (0, 400))
    join(a, b)
    join(a, c)
    join(b, c)
    tree.nodes.remove(a)
    assert _pairs(tree) == {_pair(b, c)}
    assert len(tree.marker_links) == 1, "a line to a deleted marker was kept"


def test_renaming_a_marker_keeps_its_lines():
    from armature_nodes.marker_links import join

    tree = _tree()
    a, b = _markers(tree, (0, 0), (400, 0))
    join(a, b)
    a.name = "Hip"
    assert _pairs(tree) == {frozenset(("Hip", b.name))}


def test_a_marker_from_before_lines_gets_an_id():
    tree = _tree()
    (a,) = _markers(tree, (0, 0))
    a.uid = ""
    a.follow_live()
    assert a.uid, "an old marker was given no ID for its lines"


def test_duplicating_joined_markers_joins_the_copies():
    """Shift+D keeps the links among the nodes it copies; so it is with lines.
    A line to a marker left behind is not copied, as Blender drops that link."""
    from armature_nodes import marker_links

    tree = _tree()
    a, b, c = _markers(tree, (0, 0), (400, 0), (0, 400))
    marker_links.join(a, b)
    marker_links.join(a, c)
    copies = []
    for original in (a, b):  # Blender copies the node, then calls copy()
        copy = tree.nodes.new(MARKER)
        copy.uid = original.uid
        copy.copy(original)
        copies.append(copy)
    a2, b2 = copies
    assert a2.uid != a.uid and b2.uid != b.uid, "a copy kept its original's ID"
    marker_links._copy_noted()  # the timer, once Shift+D has finished
    assert _pairs(tree) == {_pair(a, b), _pair(a, c), _pair(a2, b2)}


def test_a_copied_tree_keeps_its_lines():
    from armature_nodes.marker_links import join

    tree = _tree()
    a, b, c = _markers(tree, (0, 0), (400, 0), (0, 400))
    join(a, b)
    join(b, c)
    copy = tree.copy()
    assert _pairs(copy) == _pairs(tree)


# --- Ports that face each other ----------------------------------------------------


def _ends(ml, tree, node, other):
    for a, b, pa, pb in ml.wire_ends(tree):
        if (a, b) == (node, other):
            return pa
        if (a, b) == (other, node):
            return pb
    raise AssertionError(f"no line between {node.name} and {other.name}")


def _side(ml, node, point, eps=1e-3):
    left, top, right, bottom = ml.view_rect(node)
    x, y = point
    for name, hit in (
        ("left", abs(x - left) < eps),
        ("right", abs(x - right) < eps),
        ("top", abs(y - top) < eps),
        ("bottom", abs(y - bottom) < eps),
    ):
        if hit:
            return name
    raise AssertionError(f"{point} is not on the border of {node.name}")


def test_the_port_faces_the_node_it_joins_and_moves_when_it_moves():
    tree = _tree()
    a, b = _markers(tree, (0, 0), (500, 0))
    with _drawn() as ml:
        ml.join(a, b)
        assert _side(ml, a, _ends(ml, tree, a, b)) == "right"
        assert _side(ml, b, _ends(ml, tree, b, a)) == "left"
        b.location = (0, 700)  # above
        assert _side(ml, a, _ends(ml, tree, a, b)) == "top"
        assert _side(ml, b, _ends(ml, tree, b, a)) == "bottom"
        b.location = (-600, -40)  # to the left
        assert _side(ml, a, _ends(ml, tree, a, b)) == "left"
        assert _side(ml, b, _ends(ml, tree, b, a)) == "right"
        b.location = (0, -900)  # below
        assert _side(ml, a, _ends(ml, tree, a, b)) == "bottom"


def test_each_line_has_its_own_port():
    tree = _tree()
    a, b, c, d = _markers(tree, (0, 0), (600, 0), (0, 800), (-700, 0))
    with _drawn() as ml:
        for other in (b, c, d):
            ml.join(a, other)
        sides = {_side(ml, a, _ends(ml, tree, a, other)) for other in (b, c, d)}
        assert sides == {"right", "top", "left"}


def test_a_port_on_the_side_stays_below_the_sockets():
    """Blender's sockets sit on the sides under the header; a port there
    would cover the socket you meant to grab."""
    tree = _tree()
    a, b = _markers(tree, (0, 0), (600, 160))  # up and to the right
    with _drawn() as ml:
        ml.join(a, b)
        rect = ml.view_rect(a)
        natural = ml.border_point(rect, ml.center(ml.view_rect(b)))
        port = _ends(ml, tree, a, b)
        band = ml.socket_band(a)
        assert natural[1] > rect[1] - band, "the test needs a port among the sockets"
        assert _side(ml, a, port) == "right"
        assert abs(port[1] - (rect[1] - band)) < 1e-3, "the port covers a socket"


def test_a_collapsed_node_keeps_its_ports_off_its_sides():
    tree = _tree()
    a, b = _markers(tree, (0, 0), (600, -20))
    a.hide = True
    with _drawn() as ml:
        ml.join(a, b)
        port = _ends(ml, tree, a, b)
        # Its bottom corner, facing b, under the socket at its end.
        assert abs(port[1] - ml.view_rect(a)[3]) < 1e-3, f"{port} is not on the bottom edge"


def test_the_free_port_steps_aside_for_a_line_below():
    tree = _tree()
    a, b = _markers(tree, (0, 0), (0, -900))
    with _drawn() as ml:
        ml.join(a, b)
        context = _context(tree)
        ports = ml.region_ports(tree, context.region)
        end = next(p for p in ports if p[0] == "end" and p[1] == a)
        free = next(p for p in ports if p[0] == "free" and p[1] == a)
        gap = Vector(end[3]) - Vector(free[3])
        assert gap.length >= end[4] + free[4], "the free port is under a line's port"


def test_ports_grow_and_shrink_with_the_zoom():
    tree = _tree()
    (a,) = _markers(tree, (0, 0))
    with _drawn() as ml:
        near = ml.region_ports(tree, _context(tree, zoom=2.0).region)[0][4]
        far = ml.region_ports(tree, _context(tree, zoom=0.5).region)[0][4]
        assert abs(near / far - 4.0) < 1e-6


# --- Dragging a wire -------------------------------------------------------------


def _gizmo():
    from armature_nodes.marker_links import ARMATURE_NODES_GT_marker_ports

    gz = types.SimpleNamespace(hit=None, is_highlight=False, is_modal=False)
    return ARMATURE_NODES_GT_marker_ports, gz


def _drag(ml, context, start, stop):
    """Press on ``start``, move to ``stop`` and let go -- as Blender runs the
    port gizmo. Returns what test_select made of the press."""
    cls, gz = _gizmo()
    hit = cls.test_select(gz, context, start)
    if hit != 0:
        return hit
    assert cls.invoke(gz, context, _mouse(start)) == {"RUNNING_MODAL"}
    middle = ((start[0] + stop[0]) // 2, (start[1] + stop[1]) // 2)
    for xy in (middle, stop):
        cls.modal(gz, context, _mouse(xy), None)
    cls.exit(gz, context, False)
    return hit


def _node_middle(ml, context, node):
    x, y = ml.center(ml.view_rect(node))
    return context.region.view2d.view_to_region(x, y)


def _port_of(ml, tree, context, kind, node, other=None):
    for port in ml.region_ports(tree, context.region):
        if port[0] == kind and port[1] == node and (other is None or port[2] == other):
            return port[3]
    raise AssertionError(f"no {kind} port on {node.name}")


def test_drag_from_the_free_port_onto_a_marker_joins_them():
    tree = _tree()
    a, b = _markers(tree, (0, 0), (500, 100))
    with _drawn() as ml:
        context = _context(tree)
        start = _port_of(ml, tree, context, "free", a)
        assert _drag(ml, context, start, _node_middle(ml, context, b)) == 0
        assert _pairs(tree) == {_pair(a, b)}


def test_drag_from_the_free_port_onto_nothing_joins_nothing():
    tree = _tree()
    a, b = _markers(tree, (0, 0), (500, 100))
    with _drawn() as ml:
        context = _context(tree)
        start = _port_of(ml, tree, context, "free", a)
        _drag(ml, context, start, (start[0] - 900, start[1] - 900))
        assert not _pairs(tree)


def test_drag_a_lines_end_onto_another_marker_moves_it():
    tree = _tree()
    a, b, c = _markers(tree, (0, 0), (500, 0), (0, 700))
    with _drawn() as ml:
        ml.join(a, b)
        context = _context(tree)
        start = _port_of(ml, tree, context, "end", b, a)  # the end at b
        _drag(ml, context, start, _node_middle(ml, context, c))
        assert _pairs(tree) == {_pair(a, c)}


def test_drag_a_lines_end_onto_nothing_removes_it():
    tree = _tree()
    a, b, c = _markers(tree, (0, 0), (500, 0), (0, 700))
    with _drawn() as ml:
        ml.join(a, b)
        ml.join(a, c)
        context = _context(tree)
        start = _port_of(ml, tree, context, "end", b, a)
        _drag(ml, context, start, (start[0] + 2000, start[1] - 2000))
        assert _pairs(tree) == {_pair(a, c)}


def test_drag_a_lines_end_back_onto_its_node_changes_nothing():
    tree = _tree()
    a, b = _markers(tree, (0, 0), (500, 0))
    with _drawn() as ml:
        ml.join(a, b)
        context = _context(tree)
        start = _port_of(ml, tree, context, "end", b, a)
        _drag(ml, context, start, _node_middle(ml, context, b))
        assert _pairs(tree) == {_pair(a, b)}


def test_a_drag_onto_a_marker_already_joined_leaves_both_lines():
    tree = _tree()
    a, b, c = _markers(tree, (0, 0), (500, 0), (0, 700))
    with _drawn() as ml:
        ml.join(a, b)
        ml.join(a, c)
        context = _context(tree)
        start = _port_of(ml, tree, context, "end", b, a)
        _drag(ml, context, start, _node_middle(ml, context, c))
        assert _pairs(tree) == {_pair(a, b), _pair(a, c)}


def test_a_click_on_a_lines_end_leaves_it():
    """A press and release on the port, the mouse still -- even when the port
    is a pixel outside the node, where letting go would mean "remove"."""
    tree = _tree()
    a, b = _markers(tree, (0, 0), (500, 0))
    with _drawn() as ml:
        ml.join(a, b)
        context = _context(tree)
        x, y = _port_of(ml, tree, context, "end", b, a)
        cls, gz = _gizmo()
        assert cls.test_select(gz, context, (x - 1, y)) == 0
        cls.invoke(gz, context, _mouse((x - 1, y)))
        cls.modal(gz, context, _mouse((x - 2, y)), None)
        cls.exit(gz, context, False)
        assert _pairs(tree) == {_pair(a, b)}, "a click removed the line"


def test_a_press_away_from_the_ports_is_left_to_blender():
    tree = _tree()
    a, b = _markers(tree, (0, 0), (500, 0))
    with _drawn() as ml:
        ml.join(a, b)
        context = _context(tree)
        cls, gz = _gizmo()
        # The middle of a node: selecting and moving it is Blender's.
        assert cls.test_select(gz, context, _node_middle(ml, context, a)) == -1


def test_the_dragged_wire_snaps_to_the_node_it_would_join():
    tree = _tree()
    a, b = _markers(tree, (0, 0), (600, 0))
    with _drawn() as ml:
        context = _context(tree)
        cls, gz = _gizmo()
        start = _port_of(ml, tree, context, "free", a)
        assert cls.test_select(gz, context, start) == 0
        cls.invoke(gz, context, _mouse(start))
        try:
            cls.modal(gz, context, _mouse(_node_middle(ml, context, b)), None)
            begin, end, anchor, target = ml.drag_wire(context.region)
            assert anchor == a and target == b
            # Back from whole pixels to the view: within a pixel of the border.
            to_view = context.region.view2d.region_to_view
            assert _side(ml, b, to_view(*end), eps=1.0) == "left"
            assert _side(ml, a, to_view(*begin), eps=1.0) == "right"
        finally:
            cls.exit(gz, context, True)
        assert not _pairs(tree), "a cancelled drag joined them"


# --- Menu operators ------------------------------------------------------------------


def _run(op_cls, context, **props):
    reports = []
    op = types.SimpleNamespace(report=lambda kind, text: reports.append(text), a="", b="")
    for key, value in props.items():
        setattr(op, key, value)
    return op_cls.execute(op, context), reports


def test_join_markers_joins_the_selected_to_the_active():
    from armature_nodes.marker_links import ARMATURE_NODES_OT_marker_join

    tree = _tree()
    a, b, c, d = _markers(tree, (0, 0), (500, 0), (0, 700), (900, 900))
    for node in tree.nodes:
        node.select = node in (a, b, c)
    tree.nodes.active = a
    context = _context(tree)
    assert ARMATURE_NODES_OT_marker_join.poll(context)
    _run(ARMATURE_NODES_OT_marker_join, context)
    assert _pairs(tree) == {_pair(a, b), _pair(a, c)}


def test_remove_lines_removes_those_among_the_selected():
    from armature_nodes.marker_links import ARMATURE_NODES_OT_marker_unjoin, join

    tree = _tree()
    a, b, c = _markers(tree, (0, 0), (500, 0), (0, 700))
    join(a, b)
    join(a, c)
    join(b, c)
    for node in tree.nodes:
        node.select = node in (a, b)
    _run(ARMATURE_NODES_OT_marker_unjoin, _context(tree))
    assert _pairs(tree) == {_pair(a, c), _pair(b, c)}
    # From a node's list of lines: that one line.
    _run(ARMATURE_NODES_OT_marker_unjoin, _context(tree), a=c.uid, b=b.uid)
    assert _pairs(tree) == {_pair(a, c)}


# --- Groups ------------------------------------------------------------------------


def test_lines_go_into_a_group_with_their_markers():
    from armature_nodes.groups import make_group, ungroup
    from armature_nodes.marker_links import join

    tree = _tree()
    a, b, c = _markers(tree, (0, 0), (500, 0), (0, 700))
    join(a, b)
    join(a, c)
    group_node = make_group(tree, [a, b])
    group = group_node.node_tree
    assert len(_pairs(group)) == 1, "the line did not go into the group"
    assert not _pairs(tree), "a line to a marker that went into the group stayed outside"

    back = ungroup(tree, group_node)
    markers = [n for n in back if n.bl_idname == MARKER]
    assert len(markers) == 2
    assert _pairs(tree) == {_pair(*markers)}, "ungrouping lost the line"
    assert len(_pairs(group)) == 1, "the group itself keeps its line"


# --- In the viewport -----------------------------------------------------------------


def _two_live_markers():
    import test_live_link as live

    obj = fixtures.make_rig()
    bpy.context.view_layer.objects.active = obj
    tree, src, out = fixtures.make_tree(obj)
    for link in list(tree.links):
        tree.links.remove(link)
    first = tree.nodes.new("ArmatureNodesPositionNode")
    second = tree.nodes.new("ArmatureNodesPositionNode")
    tree.links.new(src.outputs["Rig"], first.inputs["Rig"])
    tree.links.new(first.outputs["Rig"], second.inputs["Rig"])
    tree.links.new(second.outputs["Rig"], out.inputs["Rig"])
    first.bone, second.bone = "bone.002", "bone.003"
    a, _ma = live._wire_marker(tree, first, at=(1.0, 0.0, 1.0))
    b, _mb = live._wire_marker(tree, second, at=(-1.0, 0.0, 2.0))
    return tree, a, b


def test_joined_markers_are_joined_in_the_viewport_in_their_colours():
    from armature_nodes.marker_links import join
    from armature_nodes.primary_rig import displayed_marker_nodes, join_segments, marker_color

    tree, a, b = _two_live_markers()
    a.markers[0].color = (1.0, 0.0, 0.0)
    b.markers[0].color = (0.0, 0.0, 1.0)
    shown = list(displayed_marker_nodes())
    assert not join_segments(shown), "a line with nothing joined"
    join(a, b)
    ((p0, p1, c0, c1),) = join_segments(list(displayed_marker_nodes()))
    want = {
        tuple(round(v, 4) for v in a.marker_value(a.markers[0], "position")),
        tuple(round(v, 4) for v in b.marker_value(b.markers[0], "position")),
    }
    assert {tuple(round(v, 4) for v in p) for p in (p0, p1)} == want
    colors = {tuple(c0), tuple(c1)}
    assert colors == {marker_color(a, a.markers[0]), marker_color(b, b.markers[0])}


def test_a_hidden_marker_draws_no_line():
    from armature_nodes.marker_links import join
    from armature_nodes.primary_rig import displayed_marker_nodes, join_segments

    tree, a, b = _two_live_markers()
    join(a, b)
    b.show_handles = False
    assert not join_segments(list(displayed_marker_nodes()))
