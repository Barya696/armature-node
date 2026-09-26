"""Lines between Marker nodes -- as many as you like, drawn, nothing more.

A marker can be joined to any number of other markers. A join is only a line:
nothing inherits anything. It is drawn twice:

* **In the 3D viewport**, between the two markers' handles, shading from one
  marker's colour to the other's.
* **In the node editor**, as a wire between the two Marker nodes, the way
  DaVinci Resolve's Fusion page draws its connections: each end sits on the
  node's border where the wire leaves it, facing the other node, so the port
  slides round the node as you move either one.

Blender's own sockets are fixed -- inputs on the left, outputs on the right --
and cannot be moved, so these wires are not Blender links. The joins live on
the tree (``ArmatureNodeTree.marker_links``, a pair of node IDs each), and
this module draws them: the wires under the nodes, the ports on top. A port
never sits on one of Blender's sockets: it stays below them on the sides.

**Making one**: every Marker node has a free port, a small ring at the middle
of its bottom edge. Drag from it and drop on another Marker node. Drag a
wire's end onto a different node to move it there, or onto empty space to
remove the join. Each drag is one undo step. With Marker nodes selected,
*Join Markers* in the Node menu joins them all to the active one.

Nothing here keeps a node between redraws, only its name: an undo replaces
every node, and a node kept from before it is no longer safe to touch.
"""

import math
import uuid

import bpy
from bpy.props import StringProperty
from bpy.types import Operator

from .core import TREE_IDNAME

try:
    import gpu
    from gpu_extras.batch import batch_for_shader
except ImportError:  # outside Blender (unit tests)
    gpu = batch_for_shader = None

MARKER_NODE = "ArmatureNodesMarkerNode"
# Sizes in view units at 100% UI scale: like Blender's sockets, the ports
# and wires grow and shrink with the zoom.
PORT_RADIUS = 5.5
FREE_PORT_RADIUS = 6.5
WIRE_WIDTH = 3.0
# Blender's sockets sit on a node's sides, one row each below the header.
SOCKET_ROW = 24.0
SOCKET_MARGIN = 10.0
SLACK = 5.0  # pixels outside a port that still count as on it


def new_uid():
    return uuid.uuid4().hex[:12]


def ui_scale():
    try:
        scale = bpy.context.preferences.system.ui_scale
    except AttributeError:
        scale = 1.0
    return scale or 1.0


def _redraw():
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type in ("NODE_EDITOR", "VIEW_3D"):
                    area.tag_redraw()
    except (AttributeError, ReferenceError):
        pass


# ---------------------------------------------------------------------------
# The joins
# ---------------------------------------------------------------------------


def marker_nodes(tree):
    return [n for n in tree.nodes if n.bl_idname == MARKER_NODE]


def links_of(tree):
    """[(node a, node b)] for every join in ``tree`` whose two nodes exist."""
    if tree is None or not hasattr(tree, "marker_links"):
        return []
    by_uid = {n.uid: n for n in marker_nodes(tree) if n.uid}
    out = []
    for link in tree.marker_links:
        a, b = by_uid.get(link.a), by_uid.get(link.b)
        if a is not None and b is not None and a != b:
            out.append((a, b))
    return out


def _index(tree, a, b):
    pair = {a.uid, b.uid}
    for index, link in enumerate(tree.marker_links):
        if {link.a, link.b} == pair:
            return index
    return -1


def joined(a, b):
    return a.id_data == b.id_data and _index(a.id_data, a, b) >= 0


def can_join(a, b):
    return (
        a is not None
        and b is not None
        and a != b
        and a.bl_idname == MARKER_NODE
        and b.bl_idname == MARKER_NODE
        and a.id_data == b.id_data
        and bool(a.uid)
        and bool(b.uid)
        and not joined(a, b)
    )


def join(a, b):
    """Join Marker nodes ``a`` and ``b``. False when they already are, or can't be."""
    if not can_join(a, b):
        return False
    link = a.id_data.marker_links.add()
    link.a, link.b = a.uid, b.uid
    _redraw()
    return True


def unjoin(a, b):
    tree = a.id_data
    index = _index(tree, a, b)
    if index < 0:
        return False
    tree.marker_links.remove(index)
    _redraw()
    return True


def partners(node):
    """The Marker nodes joined to ``node``."""
    out = []
    for a, b in links_of(node.id_data):
        if a == node:
            out.append(b)
        elif b == node:
            out.append(a)
    return out


def forget(node):
    """A Marker node is being deleted: its joins go with it."""
    try:
        tree, uid = node.id_data, node.uid
        links = getattr(tree, "marker_links", None)
    except (AttributeError, ReferenceError):
        return
    if not uid or links is None:
        return
    for index in reversed(range(len(links))):
        if uid in (links[index].a, links[index].b):
            links.remove(index)
    _redraw()


def copy_joins(src_tree, dst_tree, uid_map):
    """Joins among nodes copied from ``src_tree`` into ``dst_tree``.

    ``uid_map`` is {old uid: uid of the copy}. A join with only one end
    copied cannot cross from one tree to another, and is left behind.
    """
    pairs = [(link.a, link.b) for link in getattr(src_tree, "marker_links", ())]
    for old_a, old_b in pairs:
        a, b = uid_map.get(old_a), uid_map.get(old_b)
        if a and b and a != b:
            if not any({l.a, l.b} == {a, b} for l in dst_tree.marker_links):
                new = dst_tree.marker_links.add()
                new.a, new.b = a, b


def prune(tree):
    """Drop joins naming a node the tree no longer has."""
    uids = {n.uid for n in marker_nodes(tree) if n.uid}
    links = tree.marker_links
    for index in reversed(range(len(links))):
        if links[index].a not in uids or links[index].b not in uids:
            links.remove(index)


# Shift+D copies nodes one at a time; the joins among them are copied once
# it has finished. {(source tree name, tree name): {uid: uid of the copy}}
_copied = {}


def note_copy(src, dst):
    """``dst`` is a new copy of Marker node ``src`` (Shift+D, paste).

    Blender keeps the links among the nodes it duplicates; the joins among
    them are kept the same way -- after the duplicate, when every copy exists.
    """
    try:
        key = (src.id_data.name, dst.id_data.name)
        old, new = src.uid, dst.uid
    except (AttributeError, ReferenceError):
        return
    if not old or not new:
        return
    _copied.setdefault(key, {})[old] = new
    if not bpy.app.timers.is_registered(_copy_noted):
        bpy.app.timers.register(_copy_noted, first_interval=0.0)


def _copy_noted():
    batches = dict(_copied)
    _copied.clear()
    for (src_name, dst_name), uid_map in batches.items():
        src = bpy.data.node_groups.get(src_name)
        dst = bpy.data.node_groups.get(dst_name)
        if src is not None and dst is not None and hasattr(dst, "marker_links"):
            copy_joins(src, dst, uid_map)
    _redraw()
    return None


# ---------------------------------------------------------------------------
# Where the ports are
#
# Worked out in the editor's view space, where a node's rectangle is: its
# location times the UI scale, and its dimensions, which already include it.
# ---------------------------------------------------------------------------


def view_rect(node):
    """(left, top, right, bottom) of ``node`` in view space, or None before
    the node has been drawn once."""
    width, height = node.dimensions
    if width <= 0.0 or height <= 0.0:
        return None
    scale = ui_scale()
    loc = node.location_absolute if hasattr(node, "location_absolute") else node.location
    left, top = loc.x * scale, loc.y * scale
    return (left, top, left + width, top - height)


def center(rect):
    left, top, right, bottom = rect
    return ((left + right) / 2.0, (top + bottom) / 2.0)


def socket_band(node):
    """How far down its sides a node's own sockets reach, in view units.

    Ports keep below it, so a port never covers a socket you want to grab.
    A collapsed node has its sockets at its ends: its ports keep off its
    sides altogether.
    """
    if node.hide:
        return math.inf
    rows = 1  # the header
    rows += sum(1 for s in node.outputs if s.enabled and not s.hide)
    rows += sum(1 for s in node.inputs if s.enabled and not s.hide)
    return (SOCKET_ROW * rows + SOCKET_MARGIN) * ui_scale()


def border_point(rect, toward, band=0.0):
    """Where a line from the middle of ``rect`` toward ``toward`` leaves it --
    the port of a join, facing the other node.

    On a side, it is kept ``band`` below the top, clear of the sockets; where
    that leaves no room on the side, it goes to the bottom corner.
    """
    left, top, right, bottom = rect
    cx, cy = center(rect)
    dx, dy = toward[0] - cx, toward[1] - cy
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        dx = 1.0
    half_w, half_h = (right - left) / 2.0, (top - bottom) / 2.0
    tx = half_w / abs(dx) if abs(dx) > 1e-9 else math.inf
    ty = half_h / abs(dy) if abs(dy) > 1e-9 else math.inf
    x, y = cx + dx * min(tx, ty), cy + dy * min(tx, ty)
    if tx <= ty and y > top - band:  # on a side, among the sockets
        y = max(top - band, bottom)
    return (x, y)


def free_port(rect, taken=(), clear=0.0):
    """Where a new join is dragged from: the middle of the bottom edge --
    or beside it, when a join's port is already there."""
    left, _top, right, bottom = rect
    cx = (left + right) / 2.0
    for k in range(9):
        x = cx + ((k + 1) // 2) * clear * (1 if k % 2 else -1)
        if k and not (left + clear / 2.0 <= x <= right - clear / 2.0):
            continue
        if all(math.hypot(x - tx, bottom - ty) >= clear for tx, ty in taken):
            return (x, bottom)
    return (cx, bottom)


def wire_ends(tree, skip=None):
    """[(a, b, port on a, port on b)] in view space, for every join --
    except ``skip``, a pair of node names."""
    out = []
    for a, b in links_of(tree):
        if skip is not None and {a.name, b.name} == set(skip):
            continue
        ra, rb = view_rect(a), view_rect(b)
        if ra is None or rb is None:
            continue
        pa = border_point(ra, center(rb), socket_band(a))
        pb = border_point(rb, center(ra), socket_band(b))
        out.append((a, b, pa, pb))
    return out


def zoom(v2d):
    """Region pixels per view unit."""
    x0, _y = v2d.view_to_region(0.0, 0.0, clip=False)
    x1, _y = v2d.view_to_region(1000.0, 0.0, clip=False)
    return max((x1 - x0) / 1000.0, 1e-6)


def region_ports(tree, region, skip=None):
    """Every port in region pixels, as (kind, node, other, (x, y), radius):
    both ends of every join ("end"), and each Marker node's free port ("free")."""
    scale = ui_scale()
    v2d = region.view2d
    px = zoom(v2d)
    ports, taken = [], {}
    for a, b, pa, pb in wire_ends(tree, skip):
        ports.append(("end", a, b, v2d.view_to_region(*pa, clip=False), PORT_RADIUS * scale * px))
        ports.append(("end", b, a, v2d.view_to_region(*pb, clip=False), PORT_RADIUS * scale * px))
        taken.setdefault(a.name, []).append(pa)
        taken.setdefault(b.name, []).append(pb)
    clear = (PORT_RADIUS + FREE_PORT_RADIUS + 3.0) * scale
    for node in marker_nodes(tree):
        rect = view_rect(node)
        if rect is not None:
            spot = free_port(rect, taken.get(node.name, ()), clear)
            xy = v2d.view_to_region(*spot, clip=False)
            ports.append(("free", node, None, xy, FREE_PORT_RADIUS * scale * px))
    return ports


def pick(ports, mouse):
    """The port under ``mouse`` (region pixels): the closest within reach, or None."""
    best = None
    for port in ports:
        xy, radius = port[3], port[4]
        d = math.hypot(mouse[0] - xy[0], mouse[1] - xy[1])
        if d <= radius + SLACK * ui_scale() and (best is None or d < best[0]):
            best = (d, port)
    return None if best is None else best[1]


def node_under(tree, region, mouse):
    """The Marker node under ``mouse`` (region pixels), the top one first."""
    for node in reversed(list(tree.nodes)):
        if node.bl_idname != MARKER_NODE:
            continue
        rect = view_rect(node)
        if rect is None:
            continue
        left, top, right, bottom = rect
        x0, y1 = region.view2d.view_to_region(left, top, clip=False)
        x1, y0 = region.view2d.view_to_region(right, bottom, clip=False)
        if x0 <= mouse[0] <= x1 and y0 <= mouse[1] <= y1:
            return node
    return None


def port_ident(port):
    """What names a port between redraws: (kind, node name, other's name)."""
    kind, node, other = port[0], port[1], port[2]
    return (kind, node.name, other.name if other is not None else "")


# ---------------------------------------------------------------------------
# Dragging
# ---------------------------------------------------------------------------


class Drag:
    """A wire being dragged: a new one from a free port, or one end of a join.

    Grabbing a join's end at node A lifts it off A; the wire stays on the
    other node and follows the mouse. Nodes are held by name.
    """

    def __init__(self, tree, kind, node, other, mouse):
        self.tree_name = tree.name
        self.kind = kind
        self.node = node.name
        self.other = other.name if other is not None else ""
        self.press = self.mouse = tuple(mouse)
        self.target = ""  # the Marker node under the mouse

    def moved(self):
        """Past Blender's drag threshold. A click on a port is not a drop:
        without this, a click just off a node's edge removed its line."""
        try:
            threshold = bpy.context.preferences.inputs.drag_threshold_mouse
        except AttributeError:
            threshold = 3
        dx, dy = self.mouse[0] - self.press[0], self.mouse[1] - self.press[1]
        return math.hypot(dx, dy) > threshold * ui_scale()

    def tree(self):
        return bpy.data.node_groups.get(self.tree_name)

    def _get(self, name):
        tree = self.tree()
        return tree.nodes.get(name) if tree is not None and name else None

    @property
    def anchor(self):
        """The node the dragged wire stays on."""
        return self._get(self.node if self.kind == "free" else self.other)

    @property
    def lifted(self):
        """The join being moved, as a pair of node names, or None."""
        return (self.node, self.other) if self.kind == "end" else None

    def valid(self, target):
        """Whether dropping on ``target`` would make a join."""
        if target is None:
            return False
        if self.kind == "end" and target.name == self.node:
            return False  # where it already is
        return can_join(self.anchor, target)

    def finish(self, target):
        """Drop on ``target`` (a Marker node, or None for empty space).
        Returns what happened: "joined", "moved", "removed" or "".
        """
        anchor = self.anchor
        if anchor is None or not self.moved():
            return ""
        if self.kind == "free":
            return "joined" if self.valid(target) and join(anchor, target) else ""
        node = self._get(self.node)
        if target is not None and target.name == self.node:
            return ""  # dropped back where it was
        if node is None or not joined(anchor, node):
            return ""
        if target is None:
            unjoin(anchor, node)
            return "removed"
        if not self.valid(target):
            return ""  # onto a node it is already joined to: left as it was
        unjoin(anchor, node)
        join(anchor, target)
        return "moved"


_drag = None
_hover = None  # port_ident() of the port under the mouse


def _editor_tree(context):
    space = context.space_data
    if space is None or space.type != "NODE_EDITOR" or space.tree_type != TREE_IDNAME:
        return None
    return space.edit_tree


class ARMATURE_NODES_GT_marker_ports(bpy.types.Gizmo):
    """Every port in the node editor, hit-tested on the CPU. The drawing is
    done by the editor's draw handlers, under and over the nodes."""

    bl_idname = "ARMATURE_NODES_GT_marker_ports"

    __slots__ = ("hit",)

    def setup(self):
        self.hit = None

    def test_select(self, context, location):
        global _hover
        tree, region = _editor_tree(context), context.region
        found = pick(region_ports(tree, region), location) if tree and region else None
        self.hit = port_ident(found) if found is not None else None
        if self.hit != _hover:
            _hover = self.hit
            _redraw()
        return -1 if found is None else 0

    def invoke(self, context, event):
        global _drag
        tree = _editor_tree(context)
        if self.hit is None or tree is None:
            return {"CANCELLED"}
        kind, node_name, other_name = self.hit
        node, other = tree.nodes.get(node_name), tree.nodes.get(other_name)
        if node is None or (kind == "end" and other is None):
            return {"CANCELLED"}
        _drag = Drag(tree, kind, node, other, (event.mouse_region_x, event.mouse_region_y))
        return {"RUNNING_MODAL"}

    def modal(self, context, event, tweak):
        if _drag is None:
            return {"CANCELLED"}
        _drag.mouse = (event.mouse_region_x, event.mouse_region_y)
        tree, region = _drag.tree(), context.region
        target = node_under(tree, region, _drag.mouse) if tree and region else None
        _drag.target = target.name if target is not None else ""
        _redraw()
        return {"RUNNING_MODAL"}

    def exit(self, context, cancel):
        global _drag
        drag, _drag = _drag, None
        if drag is not None and not cancel:
            tree, region = drag.tree(), context.region
            target = node_under(tree, region, drag.mouse) if tree and region else None
            drag.finish(target)
        _redraw()

    def draw(self, context):
        global _hover
        if not (self.is_highlight or self.is_modal) and _hover is not None:
            _hover = None


class ARMATURE_NODES_GGT_marker_ports(bpy.types.GizmoGroup):
    bl_idname = "ARMATURE_NODES_GGT_marker_ports"
    bl_label = "Marker Ports"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "WINDOW"
    bl_options = {"PERSISTENT"}

    @classmethod
    def poll(cls, context):
        tree = _editor_tree(context)
        return tree is not None and any(n.bl_idname == MARKER_NODE for n in tree.nodes)

    def setup(self, context):
        gz = self.gizmos.new(ARMATURE_NODES_GT_marker_ports.bl_idname)
        gz.use_draw_modal = True
        gz.use_undo = True  # one undo step per drag


# ---------------------------------------------------------------------------
# Drawing in the node editor
# ---------------------------------------------------------------------------


def marker_color(node):
    from .primary_rig import COLOR_CUSTOM, marker_color as color_of

    marker = node.markers[0] if len(node.markers) else None
    return tuple(color_of(node, marker)) if marker is not None else COLOR_CUSTOM


def _faded(color, alpha):
    return (color[0], color[1], color[2], alpha)


def drag_wire(region):
    """The wire being dragged, as (start, end, anchor, target) in region
    pixels, or None. It leaves the anchor facing the mouse -- or, over a node
    it can join, snaps to that node's facing port."""
    if _drag is None:
        return None
    anchor = _drag.anchor
    rect = view_rect(anchor) if anchor is not None else None
    if rect is None:
        return None
    v2d = region.view2d
    tree = _drag.tree()
    target = tree.nodes.get(_drag.target) if tree is not None and _drag.target else None
    other = view_rect(target) if _drag.valid(target) else None
    band = socket_band(anchor)
    if other is not None:
        start = border_point(rect, center(other), band)
        end = v2d.view_to_region(*border_point(other, center(rect), socket_band(target)), clip=False)
    else:
        target = None
        start = border_point(rect, v2d.region_to_view(*_drag.mouse), band)
        end = _drag.mouse
    return v2d.view_to_region(*start, clip=False), end, anchor, target


def _draw_wires():
    """Under the nodes: every join as a wire, and the one being dragged."""
    context = bpy.context
    tree = _editor_tree(context)
    if tree is None or gpu is None:
        return
    region = context.region
    v2d = region.view2d
    coords, colors = [], []
    for a, b, pa, pb in wire_ends(tree, _drag.lifted if _drag is not None else None):
        coords += [v2d.view_to_region(*pa, clip=False), v2d.view_to_region(*pb, clip=False)]
        colors += [marker_color(a), marker_color(b)]
    wire = drag_wire(region)
    if wire is not None:
        start, end, anchor, target = wire
        color = marker_color(anchor)
        coords += [start, end]
        colors += [color, marker_color(target) if target is not None else _faded(color, 0.45)]
    if not coords:
        return
    from .handles import draw_smooth_lines

    width = max(1.5, WIRE_WIDTH * ui_scale() * zoom(v2d))
    gpu.state.blend_set("ALPHA")
    try:
        draw_smooth_lines(region, [(x, y, 0.0) for x, y in coords], colors, width)
    finally:
        gpu.state.blend_set("NONE")
        gpu.state.line_width_set(1.0)


def _disc(center, radius, color, segments=24):
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    pts = []
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        pts += [
            center,
            (center[0] + radius * math.cos(a0), center[1] + radius * math.sin(a0)),
            (center[0] + radius * math.cos(a1), center[1] + radius * math.sin(a1)),
        ]
    shader.uniform_float("color", color)
    batch_for_shader(shader, "TRIS", {"pos": pts}).draw(shader)


def _outline(region, rect, color, width):
    """A rectangle round a node, in region pixels."""
    from .handles import draw_lines, line_shader

    v2d = region.view2d
    left, top, right, bottom = rect
    x0, y1 = v2d.view_to_region(left, top, clip=False)
    x1, y0 = v2d.view_to_region(right, bottom, clip=False)
    pad = 3.0
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    corners = [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]
    coords = []
    for i in range(4):
        coords += [corners[i], corners[(i + 1) % 4]]
    draw_lines(line_shader(region), coords, color, width)


def _port(xy, radius, color, free=False, hot=False):
    r = radius * (1.35 if hot else 1.0)
    edge = max(1.0, radius * 0.3)
    _disc(xy, r + edge, (0.05, 0.05, 0.05, 0.9))  # an outline, for any background
    _disc(xy, r, _faded(color, 1.0 if (hot or not free) else 0.85))
    if free:
        _disc(xy, r * 0.45, (0.05, 0.05, 0.05, 0.9))  # a ring: "drag from here"


def _draw_ports():
    """On top of the nodes: a dot at each end of each join, a ring for each
    node's free port -- the one under the mouse larger -- and, while
    dragging, the node the wire would join."""
    context = bpy.context
    tree = _editor_tree(context)
    if tree is None or gpu is None:
        return
    region = context.region
    scale, px = ui_scale(), zoom(region.view2d)
    gpu.state.blend_set("ALPHA")
    try:
        lifted = _drag.lifted if _drag is not None else None
        for port in region_ports(tree, region, lifted):
            kind, node, _other, xy, radius = port
            hot = _drag is None and _hover == port_ident(port)
            _port(xy, radius, marker_color(node), free=(kind == "free"), hot=hot)
        wire = drag_wire(region)
        if wire is not None:
            start, end, anchor, target = wire
            color = marker_color(anchor)
            _port(start, PORT_RADIUS * scale * px, color, hot=True)
            if target is not None:
                _outline(region, view_rect(target), _faded(color, 0.9), 2.0 * scale)
                _port(end, PORT_RADIUS * scale * px, marker_color(target), hot=True)
    finally:
        gpu.state.blend_set("NONE")
        gpu.state.line_width_set(1.0)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def _selected_markers(context):
    tree = _editor_tree(context)
    if tree is None:
        return None, []
    return tree, [n for n in tree.nodes if n.select and n.bl_idname == MARKER_NODE]


class ARMATURE_NODES_OT_marker_join(Operator):
    """Draw a line from the active Marker node to each other selected one"""

    bl_idname = "armature_nodes.marker_join"
    bl_label = "Join Markers"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        tree, nodes = _selected_markers(context)
        active = tree.nodes.active if tree is not None else None
        return active is not None and active.bl_idname == MARKER_NODE and len(nodes) > 1

    def execute(self, context):
        tree, nodes = _selected_markers(context)
        active = tree.nodes.active
        made = sum(join(active, n) for n in nodes if n != active)
        self.report({"INFO"}, f"{made} line(s) added")
        return {"FINISHED"}


class ARMATURE_NODES_OT_marker_unjoin(Operator):
    """Remove the lines between the selected Marker nodes"""

    bl_idname = "armature_nodes.marker_unjoin"
    bl_label = "Remove Marker Lines"
    bl_options = {"REGISTER", "UNDO"}

    # Named, from a node's list of lines: that one line.
    a: StringProperty(options={"HIDDEN", "SKIP_SAVE"})
    b: StringProperty(options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _editor_tree(context) is not None

    def execute(self, context):
        tree, nodes = _selected_markers(context)
        pairs = links_of(tree)
        if self.a and self.b:
            pairs = [(a, b) for a, b in pairs if {a.uid, b.uid} == {self.a, self.b}]
        else:
            pairs = [(a, b) for a, b in pairs if a in nodes and b in nodes]
        removed = sum(unjoin(a, b) for a, b in pairs)
        self.report({"INFO"}, f"{removed} line(s) removed")
        return {"FINISHED"} if removed else {"CANCELLED"}


def _draw_menu(self, context):
    if _editor_tree(context) is None:
        return
    layout = self.layout
    layout.separator()
    layout.operator(ARMATURE_NODES_OT_marker_join.bl_idname, icon="IPO_LINEAR")
    layout.operator(ARMATURE_NODES_OT_marker_unjoin.bl_idname, icon="X")


classes = (
    ARMATURE_NODES_GT_marker_ports,
    ARMATURE_NODES_GGT_marker_ports,
    ARMATURE_NODES_OT_marker_join,
    ARMATURE_NODES_OT_marker_unjoin,
)
_handles = []


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.NODE_MT_node.append(_draw_menu)
    bpy.types.NODE_MT_context_menu.append(_draw_menu)
    if gpu is not None:
        space = bpy.types.SpaceNodeEditor
        _handles.append(space.draw_handler_add(_draw_wires, (), "WINDOW", "BACKDROP"))
        _handles.append(space.draw_handler_add(_draw_ports, (), "WINDOW", "POST_PIXEL"))


def unregister():
    global _drag, _hover
    for handle in _handles:
        bpy.types.SpaceNodeEditor.draw_handler_remove(handle, "WINDOW")
    _handles.clear()
    bpy.types.NODE_MT_context_menu.remove(_draw_menu)
    bpy.types.NODE_MT_node.remove(_draw_menu)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    if bpy.app.timers.is_registered(_copy_noted):
        bpy.app.timers.unregister(_copy_noted)
    _copied.clear()
    _drag = _hover = None
