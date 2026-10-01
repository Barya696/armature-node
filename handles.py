"""Markers move the way everything in Blender moves: click one, then G, R, S.

A marker is its empty -- in the markers' collection, drawn in front of
everything -- so clicking its glow selects it like any object, and Blender's
own transform moves, turns and scales it: X / Y / Z, typed values, snapping
(Face snapping drops a marker onto a surface), several markers at once, Esc
to put it back, one undo step. What a marker cannot do is locked on its
empty: a position that is only a readout of the bone does not move, a marker
that supplies no rotation does not turn. Markers are picked in Object mode;
in Pose mode G / R / S move the bones.

The glow is sized in the scene, like the character it sits on: zoom in and
it grows, zoom out and it shrinks (see ``screen_scale``). This gizmo outlines
every glow and lights the one under the mouse, its name beside it. Press on
a glow and drag, and the marker moves freely with the mouse: X / Y / Z lock
the move to that axis, Shift moves it finely, Ctrl drops it onto the surface
under the cursor, Esc or right-click puts it back. Press and let go without
moving -- a click -- and the marker's empty is selected, as a click on any
object selects it, ready for G / R / S.
"""

import math

import bpy
from mathutils import Vector

try:
    import blf
    import gpu
    from bpy_extras import view3d_utils
    from gpu_extras.batch import batch_for_shader
except ImportError:  # outside Blender (unit tests)
    blf = gpu = view3d_utils = batch_for_shader = None

MOVE, ROTATE, SCALE = 0, 1, 2

# Sizes in pixels, before the node's Size -- the pixels of a full-body view,
# where the figure is FIGURE_PIXELS tall on screen. Zoomed in or out, the
# handle scales with the figure (``screen_scale``).
CORE_RADIUS = 10.0  # the glow you grab
RING_RADIUS = 22.0  # the rotation ring
KNOB_HALF = 5.0  # half the scale square
SLACK = 4.0  # how far outside a shape still counts as on it
FIGURE_PIXELS = 700.0
# Zoomed far out, a handle stops shrinking at this share of those sizes on
# screen, so it can still be seen and grabbed.
SMALLEST = 0.4

_AXES = {"X": Vector((1.0, 0.0, 0.0)), "Y": Vector((0.0, 1.0, 0.0)), "Z": Vector((0.0, 0.0, 1.0))}


def ui_scale():
    """The interface scale, so handles grow with the rest of the UI."""
    try:
        scale = bpy.context.preferences.system.ui_scale
    except AttributeError:
        scale = 1.0
    return scale or 1.0  # 0 without a window (background mode)


def screen_scale(view, co, size, height):
    """Screen pixels per pixel of the sizes above, for a handle at ``co``.

    ``size`` is the node's Size and ``height`` the figure's: a handle is a
    size in the scene, a fixed share of its figure, so it grows as you zoom
    in and shrinks as you zoom out -- down to ``SMALLEST``. None when ``co``
    is behind the view.
    """
    wpp = view.world_per_pixel(co)
    if not wpp:
        return None
    return max(size * height / (FIGURE_PIXELS * wpp), SMALLEST * size * ui_scale())


# ---------------------------------------------------------------------------
# What there is to grab
# ---------------------------------------------------------------------------


class Handle:
    """One grabbable marker, as the viewport sees it."""

    __slots__ = (
        "tree", "node", "key", "label", "obj", "color", "size", "height", "move", "turn", "grow",
    )

    def __init__(self, tree, node, key, label, obj, color, size, move, turn, grow, height=1.8):
        self.tree, self.node, self.key, self.label = tree, node, key, label
        self.obj, self.color, self.size, self.height = obj, color, size, height
        self.move, self.turn, self.grow = move, turn, grow

    def ident(self):
        return (self.tree, self.node, self.key)

    def scale(self, view):
        """Screen pixels per size pixel, here and now; None when behind the view."""
        return screen_scale(view, self.obj.location, self.size, self.height)

    def core_mode(self):
        """What dragging the glow does: move, or failing that turn or scale."""
        if self.move:
            return MOVE
        return ROTATE if self.turn else SCALE


def visible_handles():
    """Every marker handle displayed now, in drawing order.

    A handle whose empty is locked in every channel -- a right-side landmark
    in Symmetric mode, which only mirrors its partner -- has nothing to grab
    and is left out; its glow is still drawn.
    """
    from .primary_rig import (
        displayed_marker_nodes,
        find_marker_empties,
        marker_color,
        marker_empties_index,
    )

    out = []
    index = marker_empties_index()
    for node in displayed_marker_nodes():
        empties = find_marker_empties(node, index)
        size = float(getattr(node, "handle_size", 1.0))
        height = node.effective_height()
        for marker in node.markers:
            obj = empties.get(marker.key)
            if obj is None:
                continue
            move = not all(obj.lock_location)
            turn = not all(obj.lock_rotation)
            grow = not all(obj.lock_scale)
            if not (move or turn or grow):
                continue
            out.append(
                Handle(
                    node.id_data.name,
                    node.name,
                    marker.key,
                    marker.name or marker.key,
                    obj,
                    marker_color(node, marker),
                    size,
                    move,
                    turn,
                    grow,
                    height,
                )
            )
    return out


def knob_offset(scale):
    """Where the scale square sits: on the ring, up and to the right."""
    return Vector((1.0, 1.0)) * (RING_RADIUS * scale * math.sqrt(0.5))


def pick(points, mouse):
    """(index, part) under ``mouse``, or None.

    ``points`` holds, per handle, ``(screen position or None, scale, core
    mode, turn, grow)``. The closest shape wins, so a small scale square on
    the ring is not swallowed by the ring around it.
    """
    mouse = Vector(mouse)
    best = None
    for index, (xy, scale, core_mode, turn, grow) in enumerate(points):
        if xy is None:
            continue
        xy = Vector(xy)
        d = (mouse - xy).length
        found = []
        if d <= CORE_RADIUS * scale + SLACK:
            found.append((d, core_mode))
        if turn and abs(d - RING_RADIUS * scale) <= SLACK + 2.0:
            found.append((abs(d - RING_RADIUS * scale), ROTATE))
        if grow:
            k = mouse - (xy + knob_offset(scale))
            if max(abs(k.x), abs(k.y)) <= KNOB_HALF * scale + SLACK:
                found.append((0.0, SCALE))  # on the square: always the square
        for dist, part in found:
            if best is None or dist < best[0]:
                best = (dist, index, part)
    return None if best is None else (best[1], best[2])


# ---------------------------------------------------------------------------
# Screen <-> world
# ---------------------------------------------------------------------------


class View:
    """Screen to world and back, for one 3D viewport region."""

    def __init__(self, region, rv3d):
        self.region, self.rv3d = region, rv3d

    def to_screen(self, co):
        xy = view3d_utils.location_3d_to_region_2d(self.region, self.rv3d, co)
        return None if xy is None else Vector(xy)

    def on_plane(self, xy, depth):
        """The point under ``xy`` on the view plane through ``depth``."""
        return view3d_utils.region_2d_to_location_3d(self.region, self.rv3d, xy, depth)

    def ray(self, xy):
        origin = view3d_utils.region_2d_to_origin_3d(self.region, self.rv3d, xy)
        return origin, view3d_utils.region_2d_to_vector_3d(self.region, self.rv3d, xy)

    def facing(self):
        """The view axis, pointing at the viewer."""
        return self.rv3d.view_rotation @ Vector((0.0, 0.0, 1.0))

    def right_up(self):
        rot = self.rv3d.view_rotation
        return rot @ Vector((1.0, 0.0, 0.0)), rot @ Vector((0.0, 1.0, 0.0))

    def world_per_pixel(self, co):
        xy = self.to_screen(co)
        if xy is None:
            return None
        return (self.on_plane(xy + Vector((1.0, 0.0)), co) - Vector(co)).length


# ---------------------------------------------------------------------------
# One drag
# ---------------------------------------------------------------------------


class Drag:
    """One free drag of a marker's glow: its empty follows the mouse in the
    view's plane. X / Y / Z lock it to that axis, Shift moves it finely, Ctrl
    drops it onto the surface under the cursor. Its locked channels stay
    put, as they do under G."""

    def __init__(self, obj, view, mouse):
        self.obj, self.view = obj, view
        self.loc0 = obj.location.copy()
        self.mouse0 = Vector(mouse)
        self.axis = None

    def set_axis(self, name):
        """X / Y / Z pressed: lock to that axis, or free it again."""
        self.axis = None if self.axis == name else name

    def update(self, mouse, precise=False, snap=False, context=None):
        mouse = Vector(mouse)
        if precise:
            mouse = self.mouse0 + (mouse - self.mouse0) * 0.1
        target = self._surface(mouse, context) if snap else None
        if target is None:
            delta = self.view.on_plane(mouse, self.loc0) - self.view.on_plane(self.mouse0, self.loc0)
            if self.axis:
                axis = _AXES[self.axis]
                delta = axis * delta.dot(axis)
            target = self.loc0 + delta
        new = Vector(target)
        for i, locked in enumerate(self.obj.lock_location):
            if locked:
                new[i] = self.loc0[i]
        if (Vector(self.obj.location) - new).length > 1e-9:
            self.obj.location = new

    def cancel(self):
        self.obj.location = self.loc0

    def _surface(self, mouse, context):
        """The first surface under the cursor, or None."""
        if context is None:
            return None
        origin, direction = self.view.ray(mouse)
        hit, location, *_rest = context.scene.ray_cast(context.evaluated_depsgraph_get(), origin, direction)
        return Vector(location) if hit else None


def select_marker(context, obj, extend=False):
    """A click on a marker: its empty selected and made active, as a click
    on any object does -- Shift adds it, or takes it off when it is already
    the active one. In Pose mode Blender picks no other object, and neither
    does this."""
    layer = context.view_layer
    if context.mode != "OBJECT" or layer.objects.get(obj.name) is None:
        return
    if extend and obj.select_get() and layer.objects.active == obj:
        obj.select_set(False)
        return
    if not extend:
        for other in context.selected_objects:
            other.select_set(False)
    obj.select_set(True)
    layer.objects.active = obj


_hovered = None


def hovered_ident():
    """The marker to draw brighter: the one under the mouse."""
    return _hovered


def _set_hovered(ident):
    global _hovered
    if ident != _hovered:
        _hovered = ident
        _redraw()


def _redraw():
    from .primary_rig import tag_viewports_redraw

    tag_viewports_redraw()


# ---------------------------------------------------------------------------
# The gizmo
# ---------------------------------------------------------------------------


def _circle(center, right, up, radius, segments=40):
    points = []
    for i in range(segments):
        for j in (i, i + 1):
            a = 2.0 * math.pi * j / segments
            points.append(center + (right * math.cos(a) + up * math.sin(a)) * radius)
    return points


class ARMATURE_NODES_GT_marker_handles(bpy.types.Gizmo):
    """Every marker's glow in one gizmo, hit-tested on the CPU: lit under the
    mouse; pressed and dragged, it moves the marker freely; clicked, it
    selects the marker's empty for G / R / S."""

    bl_idname = "ARMATURE_NODES_GT_marker_handles"

    __slots__ = ("hit", "drag", "press", "moved", "extend")

    def setup(self):
        self.hit = self.drag = self.press = None
        self.moved = self.extend = False

    def test_select(self, context, location):
        handles = visible_handles()
        region, rv3d = context.region, context.region_data
        self.hit = None
        if region is not None and rv3d is not None and handles:
            view = View(region, rv3d)
            points = []
            for h in handles:
                xy, s = view.to_screen(h.obj.location), h.scale(view)
                points.append((xy if s else None, s or 0.0, MOVE, False, False))
            found = pick(points, location)
            if found is not None:
                self.hit = handles[found[0]]
        _set_hovered(self.hit.ident() if self.hit is not None else None)
        return MOVE if self.hit is not None else -1

    def invoke(self, context, event):
        if self.hit is None:
            return {"CANCELLED"}
        self.press = Vector((event.mouse_region_x, event.mouse_region_y))
        self.drag = Drag(self.hit.obj, View(context.region, context.region_data), self.press)
        self.moved, self.extend = False, event.shift
        return {"RUNNING_MODAL"}

    def modal(self, context, event, tweak):
        if self.drag is None:
            return {"CANCELLED"}
        if event.value == "PRESS" and event.type in _AXES:
            self.drag.set_axis(event.type)
        mouse = Vector((event.mouse_region_x, event.mouse_region_y))
        if not self.moved and (mouse - self.press).length > context.preferences.inputs.drag_threshold_mouse:
            self.moved = True  # past Blender's drag threshold: a drag, not a click
        if self.moved:
            self.drag.update(mouse, precise="PRECISE" in tweak, snap="SNAP" in tweak, context=context)
            _redraw()
        return {"RUNNING_MODAL"}

    def exit(self, context, cancel):
        if self.drag is not None:
            if cancel:
                self.drag.cancel()
            elif not self.moved:
                select_marker(context, self.drag.obj, self.extend)
        self.drag = None
        _redraw()

    def draw(self, context):
        _draw_outlines(context, self.hit)


def _draw_outlines(context, hot):
    """A white outline round every glow, so a marker reads against any
    background -- brighter round the one under the mouse. The glow itself is
    drawn by the marker overlay (``primary_rig``)."""
    if gpu is None:
        return
    region, rv3d = context.region, context.region_data
    if region is None or rv3d is None:
        return
    view = View(region, rv3d)
    right, up = view.right_up()
    line = line_shader(region)
    scale = ui_scale()
    hot_ident = hot.ident() if hot is not None else None
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")
    try:
        for handle in visible_handles():
            co = Vector(handle.obj.location)
            wpp, s = view.world_per_pixel(co), handle.scale(view)
            if not wpp or not s:
                continue
            lit = handle.ident() == hot_ident
            draw_lines(line, _circle(co, right, up, CORE_RADIUS * s * (1.25 if lit else 1.0) * wpp),
                       (1.0, 1.0, 1.0, 1.0 if lit else 0.55), (2.5 if lit else 1.5) * scale)
    finally:
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set("NONE")
        gpu.state.depth_test_set("LESS_EQUAL")


def line_shader(region):
    """(shader, draws real widths) for lines.

    The plain UNIFORM_COLOR shader ignores the line width on Blender 5's
    backends, so every ring would be one pixel thin. POLYLINE draws them as
    thin quads instead, at whatever width it is told.
    """
    try:
        shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    except (ValueError, SystemError):
        return gpu.shader.from_builtin("UNIFORM_COLOR"), False
    shader.bind()
    shader.uniform_float("viewportSize", (float(region.width), float(region.height)))
    return shader, True


def draw_lines(line, coords, color, width):
    """LINES through ``coords``, in ``color``, ``width`` pixels wide."""
    shader, real_width = line
    shader.bind()
    shader.uniform_float("color", color)
    if real_width:
        shader.uniform_float("lineWidth", width)
    else:
        gpu.state.line_width_set(width)
    batch_for_shader(shader, "LINES", {"pos": coords}).draw(shader)


def draw_smooth_lines(region, coords, colors, width):
    """LINES through ``coords`` (3D), each end in its own colour, so a line
    shades from one to the other -- ``width`` pixels wide."""
    try:
        shader = gpu.shader.from_builtin("POLYLINE_SMOOTH_COLOR")
    except (ValueError, SystemError):
        shader = gpu.shader.from_builtin("SMOOTH_COLOR")
        gpu.state.line_width_set(width)
    else:
        shader.bind()
        shader.uniform_float("viewportSize", (float(region.width), float(region.height)))
        shader.uniform_float("lineWidth", width)
    batch_for_shader(shader, "LINES", {"pos": coords, "color": colors}).draw(shader)


class ARMATURE_NODES_GGT_marker_handles(bpy.types.GizmoGroup):
    bl_idname = "ARMATURE_NODES_GGT_marker_handles"
    bl_label = "Marker Handles"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {"3D", "PERSISTENT", "SHOW_MODAL_ALL"}

    @classmethod
    def poll(cls, context):
        from .primary_rig import displayed_marker_nodes

        return bool(displayed_marker_nodes())

    def setup(self, context):
        gz = self.gizmos.new(ARMATURE_NODES_GT_marker_handles.bl_idname)
        gz.use_draw_modal = True
        gz.use_undo = True  # one undo step per drag or click


# ---------------------------------------------------------------------------
# The name beside the handle
# ---------------------------------------------------------------------------

_label_handle = None


def _draw_label():
    """Name the marker under the mouse."""
    if blf is None:
        return
    ident = hovered_ident()
    if ident is None:
        return
    context = bpy.context
    region, rv3d = context.region, context.region_data
    if region is None or rv3d is None:
        return
    handle = next((h for h in visible_handles() if h.ident() == ident), None)
    if handle is None:
        return
    view = View(region, rv3d)
    xy, s = view.to_screen(handle.obj.location), handle.scale(view)
    if xy is None or not s:
        return
    scale = ui_scale()
    text = handle.label
    offset = RING_RADIUS * s + 8.0 * scale  # beside the ring, however big it is now
    font = 0
    blf.size(font, 13.0 * scale)
    blf.position(font, xy.x + offset, xy.y + offset * 0.4, 0.0)
    blf.enable(font, blf.SHADOW)
    blf.shadow(font, 3, 0.0, 0.0, 0.0, 0.8)
    blf.color(font, 1.0, 1.0, 1.0, 0.95)
    blf.draw(font, text)
    blf.disable(font, blf.SHADOW)


classes = (ARMATURE_NODES_GT_marker_handles, ARMATURE_NODES_GGT_marker_handles)


def register():
    global _label_handle
    for cls in classes:
        bpy.utils.register_class(cls)
    if blf is not None and _label_handle is None:
        _label_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_label, (), "WINDOW", "POST_PIXEL"
        )


def unregister():
    global _label_handle, _hovered
    if _label_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_label_handle, "WINDOW")
        _label_handle = None
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    _hovered = None
