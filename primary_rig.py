"""Skeleton node support: marker handles, locks and the viewport overlay.

Markers are user-defined: a world position, optionally oriented, with a stable
key. Each is drawn as a draggable empty in the ``MRKS_rig`` collection and,
for the MediaPipe ones, joined by the standard ``POSE_CONNECTIONS`` skeleton
in MediaPipe's colours (left = orange, right = cyan); markers you add
yourself are drawn in the marker colour.

The 33-point MediaPipe Pose topology below (nose, eyes, ears, mouth,
shoulders, elbows, wrists, pinky / index / thumb, hips, knees, ankles, heels,
foot index) is a **preset**, loaded into a new Skeleton node so it starts with
a usable body. Nothing here depends on it being complete.

Scope: **markers only**. There is no mesh analysis, no auto placement and no
snapping -- markers are placed by hand (dragged in the viewport or typed on
the node). Turning a marker into rig is the Custom Shape node's job: wire a
marker into one and it poses that control.
"""

import bpy
from bpy.props import EnumProperty
from bpy.types import Operator
from mathutils import Vector

try:
    import gpu
    from gpu_extras.batch import batch_for_shader
except ImportError:  # outside Blender (unit tests)
    gpu = None
    batch_for_shader = None


MARKER_COLLECTION = "MRKS_rig"
# Node types that own draggable marker handles.
MARKER_NODE_IDNAMES = ("ArmatureNodesSkeletonNode", "ArmatureNodesMarkerNode")
DEFAULT_HEIGHT = 1.8

# ---------------------------------------------------------------------------
# MediaPipe Pose topology
# ---------------------------------------------------------------------------

# Index order is MediaPipe's. Blender: +X is the character's left, the
# character faces -Y. Defaults are a 1.8 m T-pose figure on the origin.
LANDMARKS = (
    # idx, key,               label,             side, default (x, y, z)
    (0, "nose", "Nose", "C", (0.0, -0.10, 1.66)),
    (1, "eye_inner_l", "Eye Inner.L", "L", (0.016, -0.085, 1.69)),
    (2, "eye_l", "Eye.L", "L", (0.032, -0.08, 1.69)),
    (3, "eye_outer_l", "Eye Outer.L", "L", (0.048, -0.07, 1.69)),
    (4, "eye_inner_r", "Eye Inner.R", "R", (-0.016, -0.085, 1.69)),
    (5, "eye_r", "Eye.R", "R", (-0.032, -0.08, 1.69)),
    (6, "eye_outer_r", "Eye Outer.R", "R", (-0.048, -0.07, 1.69)),
    (7, "ear_l", "Ear.L", "L", (0.08, 0.0, 1.67)),
    (8, "ear_r", "Ear.R", "R", (-0.08, 0.0, 1.67)),
    (9, "mouth_l", "Mouth.L", "L", (0.025, -0.085, 1.61)),
    (10, "mouth_r", "Mouth.R", "R", (-0.025, -0.085, 1.61)),
    (11, "shoulder_l", "Shoulder.L", "L", (0.19, 0.0, 1.48)),
    (12, "shoulder_r", "Shoulder.R", "R", (-0.19, 0.0, 1.48)),
    (13, "elbow_l", "Elbow.L", "L", (0.50, 0.0, 1.46)),
    (14, "elbow_r", "Elbow.R", "R", (-0.50, 0.0, 1.46)),
    (15, "wrist_l", "Wrist.L", "L", (0.78, 0.0, 1.44)),
    (16, "wrist_r", "Wrist.R", "R", (-0.78, 0.0, 1.44)),
    (17, "pinky_l", "Pinky.L", "L", (0.90, 0.02, 1.43)),
    (18, "pinky_r", "Pinky.R", "R", (-0.90, 0.02, 1.43)),
    (19, "index_l", "Index.L", "L", (0.91, -0.02, 1.44)),
    (20, "index_r", "Index.R", "R", (-0.91, -0.02, 1.44)),
    (21, "thumb_l", "Thumb.L", "L", (0.84, -0.05, 1.43)),
    (22, "thumb_r", "Thumb.R", "R", (-0.84, -0.05, 1.43)),
    (23, "hip_l", "Hip.L", "L", (0.10, 0.0, 0.93)),
    (24, "hip_r", "Hip.R", "R", (-0.10, 0.0, 0.93)),
    (25, "knee_l", "Knee.L", "L", (0.105, 0.0, 0.50)),
    (26, "knee_r", "Knee.R", "R", (-0.105, 0.0, 0.50)),
    (27, "ankle_l", "Ankle.L", "L", (0.11, 0.0, 0.08)),
    (28, "ankle_r", "Ankle.R", "R", (-0.11, 0.0, 0.08)),
    (29, "heel_l", "Heel.L", "L", (0.11, 0.05, 0.02)),
    (30, "heel_r", "Heel.R", "R", (-0.11, 0.05, 0.02)),
    (31, "foot_index_l", "Foot Index.L", "L", (0.115, -0.18, 0.02)),
    (32, "foot_index_r", "Foot Index.R", "R", (-0.115, -0.18, 0.02)),
)
LM_KEYS = tuple(lm[1] for lm in LANDMARKS)
LM_BY_INDEX = {lm[0]: lm[1] for lm in LANDMARKS}
LM_LABELS = {lm[1]: lm[2] for lm in LANDMARKS}
LM_SIDE = {lm[1]: lm[3] for lm in LANDMARKS}
LM_DEFAULTS = {lm[1]: lm[4] for lm in LANDMARKS}
LM_MIRROR = {}
for _k in LM_KEYS:
    if _k.endswith("_l"):
        LM_MIRROR[_k] = _k[:-2] + "_r"
    elif _k.endswith("_r"):
        LM_MIRROR[_k] = _k[:-2] + "_l"
    else:
        LM_MIRROR[_k] = None

# mp.solutions.pose.POSE_CONNECTIONS
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)

# Landmarks that move rigidly with an anchor: position the nose and the whole
# face follows, position the wrist and the finger landmarks follow, etc.
RIGID_GROUPS = {
    "nose": ("eye_inner_l", "eye_l", "eye_outer_l", "eye_inner_r", "eye_r",
             "eye_outer_r", "ear_l", "ear_r", "mouth_l", "mouth_r"),
    "wrist_l": ("pinky_l", "index_l", "thumb_l"),
    "wrist_r": ("pinky_r", "index_r", "thumb_r"),
    "ankle_l": ("heel_l", "foot_index_l"),
    "ankle_r": ("heel_r", "foot_index_r"),
}
GROUP_ANCHOR = {m: a for a, members in RIGID_GROUPS.items() for m in members}
# The joints you actually place; everything else is a rigid-group member.
PRIMARY_KEYS = tuple(k for k in LM_KEYS if k not in GROUP_ANCHOR)
COLOR_LEFT = (1.0, 0.54, 0.0, 1.0)
COLOR_RIGHT = (0.0, 0.85, 0.9, 1.0)
COLOR_CENTER = (0.92, 0.92, 0.92, 1.0)
COLOR_LINK = (0.75, 0.75, 0.75, 0.9)


# Custom markers the user added themselves (no MediaPipe key).
COLOR_CUSTOM = (0.95, 0.45, 0.75, 1.0)


def side_color(key):
    """Marker colour. A key outside the MediaPipe set is a custom marker."""
    if key not in LM_SIDE:
        return COLOR_CUSTOM
    return {"L": COLOR_LEFT, "R": COLOR_RIGHT}.get(LM_SIDE[key], COLOR_CENTER)


def is_landmark(key):
    """True when ``key`` is one of the 33 MediaPipe landmarks."""
    return key in LM_SIDE


def mirror_point(point, mid_x):
    p = Vector(point)
    return Vector((2.0 * mid_x - p.x, p.y, p.z))


def mirror_rotation(euler):
    """Mirror an Euler rotation across the YZ plane (X axis flip)."""
    x, y, z = euler
    return (x, -y, -z)

# ---------------------------------------------------------------------------
# Landmark handles (draggable empties)
# ---------------------------------------------------------------------------


def marker_collection(create=True):
    coll = bpy.data.collections.get(MARKER_COLLECTION)
    scene = bpy.context.scene
    if coll is None:
        if not create:
            return None
        coll = bpy.data.collections.new(MARKER_COLLECTION)
        scene.collection.children.link(coll)
    elif scene is not None and coll.name not in scene.collection.children_recursive:
        scene.collection.children.link(coll)
    return coll


def find_marker_empties(node):
    """key -> empty object for every landmark handle owned by ``node``."""
    coll = bpy.data.collections.get(MARKER_COLLECTION)
    if coll is None:
        return {}
    tree_name, node_name = node.id_data.name, node.name
    keys = set(node.marker_keys())
    found = {}
    for obj in coll.objects:
        if obj.get("an_tree") != tree_name or obj.get("an_node") != node_name:
            continue
        key = obj.get("an_marker")
        # Markers are user-defined now, so the node's own list is the only
        # authority on which keys exist. A handle whose marker was deleted is
        # not returned here; ensure_marker_empties() removes it.
        if key in keys:
            found[key] = obj
    return found


def prune_marker_empties(node):
    """Delete handles whose marker no longer exists on the node."""
    coll = bpy.data.collections.get(MARKER_COLLECTION)
    if coll is None:
        return
    tree_name, node_name = node.id_data.name, node.name
    keys = set(node.marker_keys())
    for obj in list(coll.objects):
        if obj.get("an_tree") != tree_name or obj.get("an_node") != node_name:
            continue
        if obj.get("an_marker") not in keys:
            bpy.data.objects.remove(obj, do_unlink=True)


def ensure_marker_empties(node):
    """Create (or refresh) one empty per marker on ``node``.

    MediaPipe landmarks keep their side colour and their large/small handle
    sizing (rigid-group members such as face, fingers and toes are small);
    custom markers get the marker-socket pink at the primary size. Markers
    with rotation enabled are drawn as axes so the orientation is visible and
    grabbable. Handles for deleted markers are removed.
    """
    coll = marker_collection(create=True)
    prune_marker_empties(node)
    existing = find_marker_empties(node)
    h = node.effective_height()
    for marker in node.markers:
        key = marker.key
        if not key:
            continue
        obj = existing.get(key)
        if obj is None:
            obj = bpy.data.objects.new(f"LM-{marker.name or key}", None)
            obj.show_in_front = True
            obj.hide_render = True
            obj.lock_scale = (True, True, True)
            obj.rotation_mode = "XYZ"
            obj.color = side_color(key)
            obj["an_tree"] = node.id_data.name
            obj["an_node"] = node.name
            obj["an_marker"] = key
            coll.objects.link(obj)
            existing[key] = obj
        obj.empty_display_size = (0.008 if key in GROUP_ANCHOR else 0.016) * h
        obj.location = tuple(marker.position)
        if marker.use_rotation:
            obj.rotation_euler = tuple(marker.rotation)
        apply_marker_locks(node, obj, key)
    # The node has the last word on placement: a Marker node wired into a
    # relative input draws its handle on the bone, not at the raw value.
    node.push_markers_to_empties()
    return existing


def apply_marker_locks(node, obj, key):
    """Lock Y (depth) for front-view adjustment, lock rotation unless this
    marker has rotation enabled, and, in Symmetric mode, lock right-side
    MediaPipe handles entirely so they only follow the mirrored left side.

    Symmetric mirroring is a MediaPipe-landmark feature: a custom marker has
    no mirror partner, so it is never locked by it.

    A node that knows better -- the Marker node, whose locks follow what its
    wire drives -- says so through ``handle_locks``.
    """
    use_rot = node.marker_uses_rotation(key)
    locks = getattr(node, "handle_locks", None)
    if locks is not None:
        lock_loc, lock_rot, lock_scale = locks(key)
        set_handle_locks(obj, lock_loc, lock_rot, lock_scale, use_rot)
        return
    mirrored = bool(getattr(node, "symmetric", False)) and LM_SIDE.get(key) == "R"
    if mirrored:
        lock_loc = lock_rot = (True, True, True)
    else:
        lock_loc = (False, bool(getattr(node, "lock_depth", False)), False)
        lock_rot = (not use_rot,) * 3
    set_handle_locks(obj, lock_loc, lock_rot, (True, True, True), use_rot, mirrored)


def set_handle_locks(obj, lock_loc, lock_rot, lock_scale, arrows, hide_select=False):
    """Write a handle's locks and look, touching only what differs.

    The Marker node re-applies these on every sync tick. A write of the same
    value still tags the object for a depsgraph update, which runs the sync
    again -- so an unconditional write here would never let it settle.
    """
    for attr, value in (
        ("lock_location", tuple(lock_loc)),
        ("lock_rotation", tuple(lock_rot)),
        ("lock_scale", tuple(lock_scale)),
    ):
        if tuple(getattr(obj, attr)) != value:
            setattr(obj, attr, value)
    display = "ARROWS" if arrows else "SPHERE"
    if obj.empty_display_type != display:
        obj.empty_display_type = display
    if obj.hide_select != hide_select:
        obj.hide_select = hide_select


def remove_marker_empties(node):
    for obj in find_marker_empties(node).values():
        bpy.data.objects.remove(obj, do_unlink=True)
    coll = bpy.data.collections.get(MARKER_COLLECTION)
    if coll is not None and not coll.objects and not coll.children:
        bpy.data.collections.remove(coll)


# ---------------------------------------------------------------------------
# Viewport skeleton overlay (MediaPipe style)
# ---------------------------------------------------------------------------

_draw_handle = None


def _shader():
    try:
        return gpu.shader.from_builtin("UNIFORM_COLOR")
    except (ValueError, SystemError):
        return gpu.shader.from_builtin("3D_UNIFORM_COLOR")


def _reaches_output(node, seen=None):
    """True when following this node's output links arrives at an Armature
    Output whose marker display is on."""
    if seen is None:
        seen = set()
    if node.name in seen:
        return False  # graphs can rejoin; never walk a node twice
    seen.add(node.name)
    for sock in node.outputs:
        for link in sock.links:
            if not link.is_valid:
                continue
            nxt = link.to_node
            if nxt.bl_idname == "ArmatureNodesOutputNode":
                if getattr(nxt, "show_markers", True):
                    return True
                continue  # this output hides markers; another may not
            if _reaches_output(nxt, seen):
                return True
    return False


def marker_node_visible(node):
    """Whether a marker node's handles belong in the viewport.

    A marker is displayed *through* the Armature Output it feeds, the way a
    value in Geometry Nodes only matters once it reaches the output. So an
    unconnected Marker node draws nothing, and turning Show Markers off on the
    Output hides every marker in that graph at once. The node's own Handles
    toggle still wins, for hiding one marker without unwiring it.
    """
    if not getattr(node, "show_handles", True):
        return False
    return _reaches_output(node)


def _overlay_nodes():
    """Every marker-holding node currently displayed.

    Both the Skeleton node and the single Marker node draw here -- a Marker
    node that drew nothing was invisible in the viewport, which defeats the
    point of a marker.
    """
    from .core import TREE_IDNAME

    for tree in bpy.data.node_groups:
        if tree.bl_idname != TREE_IDNAME:
            continue
        for node in tree.nodes:
            if node.bl_idname not in MARKER_NODE_IDNAMES:
                continue
            if marker_node_visible(node):
                yield node


# Round, soft-edged points. The builtin UNIFORM_COLOR shader draws points as
# hard squares, which read as debug gizmos rather than markers; this discards
# outside the radius and falls the alpha off towards the edge, so a stack of
# additively blended sizes bloom into a glowing sphere.
_GLOW_VERT = """
uniform mat4 ModelViewProjectionMatrix;
uniform float pointSize;
in vec3 pos;
void main()
{
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
    gl_PointSize = pointSize;
}
"""

_GLOW_FRAG = """
uniform vec4 color;
out vec4 fragColor;
void main()
{
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = length(d) * 2.0;
    if (r > 1.0) {
        discard;
    }
    float falloff = pow(1.0 - r, 1.5);
    fragColor = vec4(color.rgb, color.a * falloff);
}
"""

_glow_shader = None
_glow_failed = False


def _glow():
    """The round-point shader, or None when this build will not compile it.

    Custom GLSL is not guaranteed across Blender's GPU backends, so a failure
    here is not fatal: the caller falls back to square builtin points rather
    than losing the whole overlay.
    """
    global _glow_shader, _glow_failed

    if _glow_shader is not None or _glow_failed:
        return _glow_shader
    try:
        _glow_shader = gpu.types.GPUShader(_GLOW_VERT, _GLOW_FRAG)
    except Exception as exc:  # noqa: BLE001
        _glow_failed = True
        print(f"[Armature Nodes] Glow shader unavailable, using flat points: {exc}")
    return _glow_shader


# Halo rings drawn under the core, biggest and faintest first.
_GLOW_LAYERS = ((26.0, 0.10), (18.0, 0.16), (12.0, 0.30))
_CORE_SIZE = 7.0


def _draw_glow_points(coords, color, scale=1.0):
    """One glowing sphere per position."""
    if not coords:
        return
    shader = _glow()
    if shader is None:  # fallback: flat square points, still colour-coded
        flat = _shader()
        gpu.state.point_size_set(_CORE_SIZE * scale * 1.6)
        flat.uniform_float("color", color)
        batch_for_shader(flat, "POINTS", {"pos": coords}).draw(flat)
        return
    batch = batch_for_shader(shader, "POINTS", {"pos": coords})
    shader.bind()
    gpu.state.blend_set("ADDITIVE")
    for size, alpha in _GLOW_LAYERS:
        shader.uniform_float("pointSize", size * scale)
        shader.uniform_float("color", (color[0], color[1], color[2], alpha))
        batch.draw(shader)
    # Opaque core on top, so the marker still reads as a solid point against
    # a bright background where additive blending washes out.
    gpu.state.blend_set("ALPHA")
    shader.uniform_float("pointSize", _CORE_SIZE * scale)
    shader.uniform_float("color", (color[0], color[1], color[2], 1.0))
    batch.draw(shader)


def _draw_skeleton_overlay():
    if gpu is None or batch_for_shader is None:
        return
    nodes = list(_overlay_nodes())
    if not nodes:
        return
    flat = _shader()
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")
    try:
        for node in nodes:
            pos = {m.key: tuple(m.position) for m in node.markers if m.key}
            if not pos:
                continue
            # Skeleton bones, coloured by side (mixed = grey). Only drawn
            # between two landmarks that are both present: the MediaPipe set
            # is a preset, so a graph may hold part of it, or none of it.
            by_color = {}
            for a, b in POSE_CONNECTIONS:
                ka, kb = LM_BY_INDEX[a], LM_BY_INDEX[b]
                if ka not in pos or kb not in pos:
                    continue
                color = side_color(ka) if LM_SIDE[ka] == LM_SIDE[kb] else COLOR_LINK
                by_color.setdefault(color, []).extend((pos[ka], pos[kb]))
            if by_color:
                gpu.state.line_width_set(3.0)
                flat.bind()
                for color, coords in by_color.items():
                    flat.uniform_float("color", color)
                    batch_for_shader(flat, "LINES", {"pos": coords}).draw(flat)

            # Glowing spheres. Rigid-group members (face, fingers, toes) are
            # drawn smaller so the joints you actually place stand out.
            side_colors = {"L": COLOR_LEFT, "R": COLOR_RIGHT, "C": COLOR_CENTER}
            for scale, keys in ((1.0, PRIMARY_KEYS), (0.55, tuple(GROUP_ANCHOR))):
                for side, color in side_colors.items():
                    coords = [k for k in keys if k in pos and LM_SIDE[k] == side]
                    _draw_glow_points([pos[k] for k in coords], color, scale)
            custom = [pos[k] for k in pos if not is_landmark(k)]
            _draw_glow_points(custom, COLOR_CUSTOM)
    finally:
        gpu.state.line_width_set(1.0)
        gpu.state.point_size_set(1.0)
        gpu.state.blend_set("NONE")
        gpu.state.depth_test_set("LESS_EQUAL")


def tag_viewports_redraw():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


# Both marker-holding nodes answer to these operators.
_MARKER_NODES = set(MARKER_NODE_IDNAMES)


def _node(context, needs=None):
    """The marker-holding node this operator was invoked from.

    ``needs`` names a method the node must have, which is how the
    Skeleton-only operators (mirror, preset) keep themselves off the single
    Marker node without a separate poll each.
    """
    node = getattr(context, "node", None)
    if node is None or node.bl_idname not in _MARKER_NODES:
        return None
    if needs is not None and not hasattr(node, needs):
        return None
    return node


class ARMATURE_OT_skeleton_toggle_markers(Operator):
    """Show the landmarks as draggable handles in the viewport (or hide them)"""

    bl_idname = "armature_nodes.skeleton_toggle_markers"
    bl_label = "Toggle Handles"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _node(context, getattr(cls, "_needs", None)) is not None

    def execute(self, context):
        node = _node(context, getattr(self, "_needs", None))
        if not len(node.markers):
            self.report({"WARNING"}, "This node has no markers yet")
            return {"CANCELLED"}
        # Flip the node's own flag rather than just adding or removing the
        # empties: the flag is what the overlay and the periodic re-create
        # both read, so toggling anything else would be undone a tick later.
        node.show_handles = not node.show_handles
        if not node.show_handles:
            remove_marker_empties(node)
        elif marker_node_visible(node):
            ensure_marker_empties(node)
        else:
            # Creating them here would only have them removed again on the
            # next sync tick, which reads as a flicker rather than an answer.
            self.report(
                {"WARNING"},
                "Wire this node into an Armature Output to see its markers",
            )
        tag_viewports_redraw()
        return {"FINISHED"}


class ARMATURE_OT_skeleton_mirror(Operator):
    """Copy the left-side landmarks to the right (or the other way round)"""

    bl_idname = "armature_nodes.skeleton_mirror"
    bl_label = "Mirror Landmarks"
    bl_options = {"REGISTER", "UNDO"}
    _needs = "mirror_markers"

    direction: EnumProperty(
        name="Direction",
        items=(
            ("L_TO_R", "Left to Right", "Overwrite right landmarks with mirrored left ones"),
            ("R_TO_L", "Right to Left", "Overwrite left landmarks with mirrored right ones"),
        ),
        default="L_TO_R",
    )

    @classmethod
    def poll(cls, context):
        return _node(context, getattr(cls, "_needs", None)) is not None

    def execute(self, context):
        node = _node(context, getattr(self, "_needs", None))
        node.mirror_markers(self.direction)
        node.id_data.mark_dirty()
        tag_viewports_redraw()
        return {"FINISHED"}


class ARMATURE_OT_skeleton_front_view(Operator):
    """Switch a 3D viewport to the front orthographic view for 2D landmark
    adjustment"""

    bl_idname = "armature_nodes.skeleton_front_view"
    bl_label = "Front View (2D)"

    def execute(self, context):
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type != "VIEW_3D":
                    continue
                region = next((r for r in area.regions if r.type == "WINDOW"), None)
                if region is None:
                    continue
                with context.temp_override(window=window, area=area, region=region):
                    bpy.ops.view3d.view_axis(type="FRONT")
                    area.spaces.active.region_3d.view_perspective = "ORTHO"
                return {"FINISHED"}
        self.report({"WARNING"}, "Open a 3D Viewport first")
        return {"CANCELLED"}


class ARMATURE_OT_skeleton_toggle_rotation(Operator):
    """Enable or disable rotation adjustment on this landmark"""

    bl_idname = "armature_nodes.skeleton_toggle_rotation"
    bl_label = "Toggle Landmark Rotation"
    bl_options = {"REGISTER", "UNDO"}

    marker: bpy.props.StringProperty(name="Marker", default="")

    @classmethod
    def poll(cls, context):
        return _node(context, getattr(cls, "_needs", None)) is not None

    def execute(self, context):
        node = _node(context, getattr(self, "_needs", None))
        marker = node.marker_by_key(self.marker)
        if marker is None:
            self.report({"WARNING"}, "Unknown marker")
            return {"CANCELLED"}
        marker.use_rotation = not marker.use_rotation
        return {"FINISHED"}


class ARMATURE_OT_skeleton_add_marker(Operator):
    """Add one marker to this Skeleton node, at the 3D cursor"""

    bl_idname = "armature_nodes.skeleton_add_marker"
    bl_label = "Add Marker"
    bl_options = {"REGISTER", "UNDO"}

    name: bpy.props.StringProperty(name="Name", default="")

    @classmethod
    def poll(cls, context):
        return _node(context, getattr(cls, "_needs", None)) is not None

    def execute(self, context):
        node = _node(context, getattr(self, "_needs", None))
        # A Marker node is one handle by definition; a second would create an
        # empty in the scene that the node then ignores.
        if node.bl_idname == "ArmatureNodesMarkerNode" and len(node.markers):
            self.report({"WARNING"}, "A Marker node holds a single marker")
            return {"CANCELLED"}
        scene = context.scene
        # The 3D cursor, so a new marker lands somewhere the user chose and
        # is visible immediately rather than piling up on the origin.
        location = tuple(scene.cursor.location) if scene is not None else (0.0, 0.0, 0.0)
        name = self.name or f"Marker {len(node.markers) + 1}"
        marker = node.add_marker(name=name, position=location)
        node.id_data.mark_dirty()
        tag_viewports_redraw()
        self.report({"INFO"}, f"Added marker '{marker.name}'")
        return {"FINISHED"}


class ARMATURE_OT_skeleton_remove_marker(Operator):
    """Remove this marker, its output socket and its viewport handle"""

    bl_idname = "armature_nodes.skeleton_remove_marker"
    bl_label = "Remove Marker"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(name="Index", default=-1)

    @classmethod
    def poll(cls, context):
        return _node(context, getattr(cls, "_needs", None)) is not None

    def execute(self, context):
        node = _node(context, getattr(self, "_needs", None))
        if not node.remove_marker(self.index):
            self.report({"WARNING"}, "No such marker")
            return {"CANCELLED"}
        node.id_data.mark_dirty()
        tag_viewports_redraw()
        return {"FINISHED"}


class ARMATURE_OT_skeleton_load_preset(Operator):
    """Fill this Skeleton node with MediaPipe's 33 pose landmarks.

    A 1.8 m T-pose body, ready to drag onto the character. Adds only what is
    missing unless Replace Markers is ticked."""

    bl_idname = "armature_nodes.skeleton_load_preset"
    bl_label = "Load MediaPipe Preset"
    bl_options = {"REGISTER", "UNDO"}
    _needs = "load_mediapipe_preset"

    replace: bpy.props.BoolProperty(
        name="Replace Markers",
        description="Delete existing markers first; off keeps them and adds only what is missing",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        return _node(context, getattr(cls, "_needs", None)) is not None

    def execute(self, context):
        node = _node(context, getattr(self, "_needs", None))
        added = node.load_mediapipe_preset(replace=self.replace)
        node.id_data.mark_dirty()
        tag_viewports_redraw()
        self.report({"INFO"}, f"Added {added} landmark marker(s)")
        return {"FINISHED"}


class ARMATURE_OT_skeleton_clear_markers(Operator):
    """Delete every marker on this Skeleton node"""

    bl_idname = "armature_nodes.skeleton_clear_markers"
    bl_label = "Clear Markers"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        node = _node(context, getattr(self, "_needs", None))
        return node is not None and len(node.markers) > 0

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        node = _node(context, getattr(self, "_needs", None))
        count = len(node.markers)
        node.clear_markers()
        node.id_data.mark_dirty()
        tag_viewports_redraw()
        self.report({"INFO"}, f"Removed {count} marker(s)")
        return {"FINISHED"}


classes = (
    ARMATURE_OT_skeleton_toggle_markers,
    ARMATURE_OT_skeleton_mirror,
    ARMATURE_OT_skeleton_front_view,
    ARMATURE_OT_skeleton_toggle_rotation,
    ARMATURE_OT_skeleton_add_marker,
    ARMATURE_OT_skeleton_remove_marker,
    ARMATURE_OT_skeleton_load_preset,
    ARMATURE_OT_skeleton_clear_markers,
)


def register():
    global _draw_handle
    for cls in classes:
        bpy.utils.register_class(cls)
    if gpu is not None and _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_skeleton_overlay, (), "WINDOW", "POST_VIEW"
        )


def unregister():
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
        _draw_handle = None
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
