"""Marker handles you can grab in any mode -- Pose mode included.

A marker's handle used to be only its empty. An empty is an object, and
Blender does not let you click another object while the armature is in Pose
mode (Lock Object Modes) -- which is exactly when you want the marker:
posing a bone and placing its marker are the same job.

So the glow you see is now the handle: a gizmo, like the ones on a light or
a camera. A gizmo belongs to the viewport, not the scene, so it works in
Object, Pose and Edit mode alike, stays the same size on screen however far
you zoom, and lights up under the mouse with the marker's name beside it.

* **Drag the glow** to move the marker. **Ctrl** drops it onto the surface
  under the cursor, **Shift** moves it finely, **X / Y / Z** lock the move to
  that axis. **Esc** or right-click puts it back.
* **Drag the ring** -- there when the marker supplies a rotation -- to turn
  it about the view, or with X / Y / Z about that axis. Ctrl: 5 degree steps.
* **Drag the square** on the ring -- there when it supplies a scale -- to
  scale it. X / Y / Z: that axis only. Ctrl: steps of 0.1.
* A marker that cannot move (its position is only a readout of the bone)
  turns or scales when you drag its glow instead.

Every drag is one undo step.

The gizmo does not write the marker itself. It moves the marker's empty,
exactly as grabbing the empty would, so everything downstream of a drag --
locks, mirroring, rigid groups, the live link to the bone -- is the path that
already exists.
"""

import math

import bpy
from mathutils import Euler, Quaternion, Vector

try:
    import blf
    import gpu
    from bpy_extras import view3d_utils
    from gpu_extras.batch import batch_for_shader
except ImportError:  # outside Blender (unit tests)
    blf = gpu = view3d_utils = batch_for_shader = None

MOVE, ROTATE, SCALE = 0, 1, 2
_MODE_NAMES = ("Move", "Rotate", "Scale")

# Sizes on screen, in pixels at 100% UI scale, before the node's Size.
CORE_RADIUS = 10.0  # the glow you grab
RING_RADIUS = 22.0  # the rotation ring
KNOB_HALF = 5.0  # half the scale square
SLACK = 4.0  # how far outside a shape still counts as on it

_AXES = {"X": Vector((1.0, 0.0, 0.0)), "Y": Vector((0.0, 1.0, 0.0)), "Z": Vector((0.0, 0.0, 1.0))}


def ui_scale():
    """The interface scale, so handles grow with the rest of the UI."""
    try:
        scale = bpy.context.preferences.system.ui_scale
    except AttributeError:
        scale = 1.0
    return scale or 1.0  # 0 without a window (background mode)


# ---------------------------------------------------------------------------
# What there is to grab
# ---------------------------------------------------------------------------


class Handle:
    """One grabbable marker, as the viewport sees it."""

    __slots__ = ("tree", "node", "key", "label", "obj", "color", "size", "move", "turn", "grow")

    def __init__(self, tree, node, key, label, obj, color, size, move, turn, grow):
        self.tree, self.node, self.key, self.label = tree, node, key, label
        self.obj, self.color, self.size = obj, color, size
        self.move, self.turn, self.grow = move, turn, grow

    def ident(self):
        return (self.tree, self.node, self.key)

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
    from .primary_rig import displayed_marker_nodes, find_marker_empties, marker_color

    out = []
    for node in displayed_marker_nodes():
        empties = find_marker_empties(node)
        size = float(getattr(node, "handle_size", 1.0))
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


def _signed_angle(a, b):
    """Angle from 2D vector ``a`` to ``b``, counter-clockwise positive."""
    angle = math.atan2(b.y, b.x) - math.atan2(a.y, a.x)
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class Drag:
    """One drag of one handle: move, turn or scale it, the way G / R / S do.

    Works on the handle's empty and nothing else. Locked channels stay put,
    so a Skeleton's Lock Depth, a mirrored landmark or a marker whose
    position is only a readout are respected exactly as they are when the
    empty is grabbed directly.
    """

    def __init__(self, obj, mode, view, mouse):
        self.obj, self.mode, self.view = obj, mode, view
        self.loc0 = obj.location.copy()
        self.rot0 = obj.rotation_euler.copy()
        self.scale0 = obj.scale.copy()
        # Turns and scales are measured from where the drag started, so a
        # press on the ring or the square changes nothing by itself.
        self.mouse0 = Vector(mouse)
        self.center = view.to_screen(self.loc0) or Vector(self.mouse0)
        self.axis = None
        self.last = None

    def set_axis(self, name):
        """X / Y / Z pressed: lock to that axis, or free it again."""
        self.axis = None if self.axis == name else name

    def update(self, mouse, precise=False, snap=False, context=None):
        mouse = Vector(mouse)
        if precise:
            mouse = self.mouse0 + (mouse - self.mouse0) * 0.1
        (self._move, self._rotate, self._scale)[self.mode](mouse, snap, context)
        self.last = mouse

    def cancel(self):
        self.obj.location = self.loc0
        self.obj.rotation_euler = self.rot0
        self.obj.scale = self.scale0

    def describe(self):
        text = _MODE_NAMES[self.mode]
        return f"{text} {self.axis}" if self.axis else text

    # -- The three motions -------------------------------------------------------

    def _move(self, mouse, snap, context):
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

    def _surface(self, mouse, context):
        """The first surface under the cursor, or None."""
        if context is None:
            return None
        origin, direction = self.view.ray(mouse)
        hit, location, *_rest = context.scene.ray_cast(
            context.evaluated_depsgraph_get(), origin, direction
        )
        return Vector(location) if hit else None

    def _rotate(self, mouse, snap, context):
        angle = _signed_angle(self.mouse0 - self.center, mouse - self.center)
        facing = self.view.facing()
        if self.axis:
            axis = _AXES[self.axis]
            # Keep the turn going the way the mouse goes, whichever way the
            # axis happens to point relative to the viewer.
            if axis.dot(facing) < 0.0:
                axis = -axis
        else:
            axis = facing
        if snap:
            step = math.radians(5.0)
            angle = round(angle / step) * step
        q = Quaternion(axis, angle) @ self.rot0.to_quaternion()
        new = q.to_euler("XYZ", self.rot0)
        for i, locked in enumerate(self.obj.lock_rotation):
            if locked:
                new[i] = self.rot0[i]
        if (Vector(self.obj.rotation_euler) - Vector(new)).length > 1e-9:
            self.obj.rotation_euler = new

    def _scale(self, mouse, snap, context):
        start = max((self.mouse0 - self.center).length, 1.0)
        factor = (mouse - self.center).length / start
        if snap:
            factor = round(factor * 10.0) / 10.0
        factor = max(factor, 1e-3)
        new = self.scale0.copy()
        for i, name in enumerate("XYZ"):
            if self.obj.lock_scale[i] or (self.axis and self.axis != name):
                continue
            new[i] = self.scale0[i] * factor
        if (Vector(self.obj.scale) - new).length > 1e-9:
            self.obj.scale = new


# ---------------------------------------------------------------------------
# Hover and drag state, shared with the drawing code
# ---------------------------------------------------------------------------

# (tree, node, key) of the handle under the mouse, and of the one being
# dragged with how. The glow overlay brightens them; the label names them.
_hovered = None
_active = None


def hovered_ident():
    """The handle to draw brighter: the one being dragged, else under the mouse."""
    return _active[0] if _active else _hovered


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


def _square(center, right, up, half):
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    points = []
    for i in range(4):
        for cx, cy in (corners[i], corners[(i + 1) % 4]):
            points.append(center + (right * cx + up * cy) * half)
    return points


class ARMATURE_NODES_GT_marker_handles(bpy.types.Gizmo):
    """Every marker handle in one gizmo, hit-tested on the CPU.

    One gizmo rather than one per marker: the set of markers changes while
    you work -- added, deleted, wired, unwired -- and a gizmo group cannot
    safely rebuild its gizmos in the middle of drawing. ``test_select``
    reads the handles fresh on every mouse move instead.
    """

    bl_idname = "ARMATURE_NODES_GT_marker_handles"

    __slots__ = ("hit", "drag")

    def setup(self):
        self.hit = None
        self.drag = None

    def test_select(self, context, location):
        handles = visible_handles()
        region, rv3d = context.region, context.region_data
        if region is None or rv3d is None or not handles:
            self.hit = None
            _set_hovered(None)
            return -1
        view = View(region, rv3d)
        scale = ui_scale()
        points = [
            (view.to_screen(h.obj.location), h.size * scale, h.core_mode(), h.turn, h.grow)
            for h in handles
        ]
        found = pick(points, location)
        if found is None:
            self.hit = None
            _set_hovered(None)
            return -1
        index, part = found
        self.hit = (handles[index], part)
        _set_hovered(handles[index].ident())
        return part

    def invoke(self, context, event):
        global _active
        if self.hit is None:
            return {"CANCELLED"}
        handle, part = self.hit
        view = View(context.region, context.region_data)
        self.drag = Drag(handle.obj, part, view, (event.mouse_region_x, event.mouse_region_y))
        _active = (handle.ident(), handle.label, self.drag)
        return {"RUNNING_MODAL"}

    def modal(self, context, event, tweak):
        if self.drag is None:
            return {"CANCELLED"}
        if event.value == "PRESS" and event.type in _AXES:
            self.drag.set_axis(event.type)
        self.drag.update(
            (event.mouse_region_x, event.mouse_region_y),
            precise="PRECISE" in tweak,
            snap="SNAP" in tweak,
            context=context,
        )
        _redraw()
        return {"RUNNING_MODAL"}

    def exit(self, context, cancel):
        global _active
        if cancel and self.drag is not None:
            self.drag.cancel()
        self.drag = None
        _active = None
        _redraw()

    def draw(self, context):
        if not (self.is_highlight or self.is_modal):
            _set_hovered(None)
        _draw_rings(context, self.hit if (self.is_highlight or self.is_modal) else None)


def _draw_rings(context, hit):
    """Outline, rotation ring and scale square for every handle.

    The glow itself is drawn by the marker overlay (``primary_rig``); this
    adds the parts you grab. Outlined in white so a handle reads against any
    background, and brighter where the mouse is.
    """
    if gpu is None:
        return
    region, rv3d = context.region, context.region_data
    if region is None or rv3d is None:
        return
    view = View(region, rv3d)
    right, up = view.right_up()
    line = line_shader(region)
    scale = ui_scale()
    hot_ident = hit[0].ident() if hit else None
    hot_part = hit[1] if hit else None
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")
    try:
        for handle in visible_handles():
            co = Vector(handle.obj.location)
            wpp = view.world_per_pixel(co)
            if wpp is None:
                continue
            s = handle.size * scale
            hot = handle.ident() == hot_ident
            r, g, b, _a = handle.color
            shapes = []
            core_hot = hot and hot_part == handle.core_mode()
            shapes.append(
                (
                    _circle(co, right, up, CORE_RADIUS * s * (1.25 if core_hot else 1.0) * wpp),
                    (1.0, 1.0, 1.0, 1.0 if core_hot else 0.55),
                    2.5 if core_hot else 1.5,
                )
            )
            if handle.turn:
                ring_hot = hot and hot_part == ROTATE
                shapes.append(
                    (
                        _circle(co, right, up, RING_RADIUS * s * wpp, segments=56),
                        (r, g, b, 1.0 if ring_hot else 0.55),
                        3.0 if ring_hot else 1.5,
                    )
                )
            if handle.grow:
                knob_hot = hot and hot_part == SCALE
                offset = knob_offset(s)
                center = co + (right * offset.x + up * offset.y) * wpp
                shapes.append(
                    (
                        _square(center, right, up, KNOB_HALF * s * wpp),
                        (1.0, 1.0, 1.0, 1.0 if knob_hot else 0.7),
                        2.5 if knob_hot else 1.5,
                    )
                )
            for coords, color, width in shapes:
                draw_lines(line, coords, color, width * scale)
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


class ARMATURE_NODES_GGT_marker_handles(bpy.types.GizmoGroup):
    bl_idname = "ARMATURE_NODES_GGT_marker_handles"
    bl_label = "Marker Handles"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {"3D", "PERSISTENT", "SHOW_MODAL_ALL"}

    @classmethod
    def poll(cls, context):
        from .primary_rig import displayed_marker_nodes

        return next(iter(displayed_marker_nodes()), None) is not None

    def setup(self, context):
        gz = self.gizmos.new(ARMATURE_NODES_GT_marker_handles.bl_idname)
        gz.use_draw_modal = True
        gz.use_undo = True  # one undo step per drag
        # Not use_event_handle_all: a click on a handle already goes to the
        # gizmo first, and swallowing every event while hovering would also
        # eat G / R / S meant for the selected bone.


# ---------------------------------------------------------------------------
# The name beside the handle
# ---------------------------------------------------------------------------

_label_handle = None


def _draw_label():
    """Name the handle under the mouse; while dragging, say what the drag does."""
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
    xy = View(region, rv3d).to_screen(handle.obj.location)
    if xy is None:
        return
    scale = ui_scale()
    text = handle.label
    if _active and _active[0] == ident:
        text = f"{text}  ·  {_active[2].describe()}"
    offset = (RING_RADIUS * handle.size + 8.0) * scale
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
    global _label_handle, _hovered, _active
    if _label_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_label_handle, "WINDOW")
        _label_handle = None
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    _hovered = _active = None
