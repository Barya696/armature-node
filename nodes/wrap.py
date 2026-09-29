"""Wrap: fit the tree's markers onto a mesh -- the Wrap Markers node.

The wrap add-on's workflow, with the markers as the template and a mesh as
the target (``wrap_solver`` has the maths):

1. **Pick Pairs** -- click a marker, then where it goes on the mesh. A joint
   (Inside) goes into the mesh, halfway through it under the cursor; a
   landmark (Surface) onto the skin. With Symmetric on, its partner --
   Hand.R for Hand.L -- is paired with the mirror image across the mesh's
   own middle.
2. **Symmetrize** -- makes the skeleton symmetric, and gives every pair its
   partner.
3. **Snap**, **Attract**, **Stick** -- the wrap's three steps.

A step runs in the background: the runner feeds the solver's passes to the
markers one at a time, so the skeleton glides onto the mesh and the rig
follows it live. Esc stops it and puts the markers back. Each step is one
undo step, and **Original** puts the markers back where they were before the
first one.

The node has no sockets. It works on the markers of its own tree -- the
skeleton the viewport draws -- and moves them the way a drag does: a child
marker against its parent, a face or finger landmark with its anchor.
"""

import json

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Node, Operator
from mathutils import Matrix, Vector

from ..wrap_solver import Skeleton
from .base import ArmatureNodeBase, redraw_viewports

WRAP_NODE = "ArmatureNodesWrapNode"
_SKELETON_NODE = "ArmatureNodesSkeletonNode"

# (identifier, label, tooltip, icon), in the order they are run.
_STEPS = (
    ("SNAP", "Snap", "Pull the paired markers onto their places; the skeleton follows along its bones", "SNAP_ON"),
    ("ATTRACT", "Attract", "Draw each marker toward its place on the mesh: the middle of its limb, or the skin", "FORCE_MAGNETIC"),
    ("STICK", "Stick", "Put every marker within the distance exactly on its place on the mesh", "PINNED"),
)
_FIT = (
    "FIT",
    "Fit to Mesh",
    "Snap, Attract and Stick in one go -- pairing the markers Auto Pairs knows first, if none are",
    "PLAY",
)
_PAIR_COLOR = (1.0, 0.82, 0.25, 0.9)
# At most this many points are sampled down the middle of a dense mesh.
_MIDDLE_SAMPLES = 20000
# What a running step lets through: looking around, not editing.
_NAVIGATION = {
    "MOUSEMOVE", "INBETWEEN_MOUSEMOVE", "MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE",
    "TRACKPADPAN", "TRACKPADZOOM", "NDOF_MOTION",
}


# ---------------------------------------------------------------------------
# The mesh
# ---------------------------------------------------------------------------


class MeshTarget:
    """The mesh to wrap onto, in world space: rays through it, and where a
    marker belongs on it."""

    def __init__(self, obj, depsgraph):
        from mathutils.bvhtree import BVHTree

        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            count = len(mesh.vertices)
            co = np.empty(count * 3, dtype=np.float32)
            normals = np.empty(count * 3, dtype=np.float32)
            tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
            mesh.vertices.foreach_get("co", co)
            mesh.vertex_normals.foreach_get("vector", normals)
            mesh.loop_triangles.foreach_get("vertices", tris)
        finally:
            evaluated.to_mesh_clear()
        if not len(tris):
            raise ValueError(f"'{obj.name}' has no faces to wrap onto")
        mw = np.array(obj.matrix_world)
        self.points = co.reshape(-1, 3).astype(float) @ mw[:3, :3].T + mw[:3, 3]
        normals = normals.reshape(-1, 3).astype(float) @ np.linalg.pinv(mw[:3, :3])
        self.normals = normals / np.maximum(np.linalg.norm(normals, axis=1), 1e-12)[:, None]
        self.bvh = BVHTree.FromPolygons(
            self.points.tolist(), tris.reshape(-1, 3).tolist(), all_triangles=True
        )
        self.frame = obj.matrix_world.copy()  # its YZ plane is the mirror
        self.size = max(float(np.ptp(self.points, axis=0).max()), 1e-6)
        self.center = Vector(self.points.mean(axis=0))
        self._middles = None

    def pick(self, origin, direction, role):
        """Where a click along the ray puts a marker: where the ray meets the
        mesh -- or, for a joint (Inside), halfway to where it comes out
        again. None when it misses."""
        d = Vector(direction).normalized()
        origin = Vector(origin)
        # Start near the mesh. An orthographic view's ray starts a clip
        # distance away, and at 1000 m mathutils' 32-bit floats put the hit
        # off the skin by more than the nudge below: the ray going on would
        # meet the same skin again, and the middle would be the skin.
        ahead = (self.center - origin).dot(d) - self.size
        if ahead > 0.0:
            origin += d * ahead
        entry = self.bvh.ray_cast(origin, d)[0]
        if entry is None or role != "INSIDE":
            return entry
        exit_ = self.bvh.ray_cast(entry + d * (1e-5 * self.size), d)[0]
        return entry if exit_ is None else (entry + exit_) * 0.5

    def middles(self):
        """A KD-tree of points down the middle of the mesh, or None.

        From each vertex -- a sample of them, on a dense mesh -- halfway to
        the far side, straight in. On a limb they gather on its axis, in the
        body on its mid-depth sheet: where joints sit.
        """
        if self._middles is None:
            from mathutils.kdtree import KDTree

            step = max(1, len(self.points) // _MIDDLE_SAMPLES)
            samples = list(zip(self.points[::step], self.normals[::step]))
            nudge = 1e-5 * self.size
            # In against the normals -- or, for a mesh turned inside out, along them.
            for sign in (-1.0, 1.0):
                found = []
                for p, n in samples:
                    p, inward = Vector(p), Vector(n) * sign
                    hit = self.bvh.ray_cast(p + inward * nudge, inward)[0]
                    if hit is not None:
                        found.append((p + hit) * 0.5)
                if 2 * len(found) >= len(samples):
                    break
            self._middles = False
            if found:
                self._middles = KDTree(len(found))
                for i, co in enumerate(found):
                    self._middles.insert(co, i)
                self._middles.balance()
        return self._middles if self._middles is not False else None

    def place(self, points, roles):
        """Where each marker belongs on the mesh: the middle for a joint
        (Inside), the skin for a landmark (Surface), NaN for the rest."""
        middles = self.middles() if "INSIDE" in roles else None
        out = np.full((len(points), 3), np.nan)
        for i, (p, role) in enumerate(zip(points, roles)):
            if role == "INSIDE" and middles is not None:
                co = middles.find(Vector(p))[0]
            elif role in ("INSIDE", "SURFACE"):
                co = self.bvh.find_nearest(Vector(p))[0]
            else:
                continue
            if co is not None:
                out[i] = co
        return out


def _mirrored(frame, point):
    """``point`` reflected across the YZ plane of ``frame``."""
    local = frame.inverted_safe() @ Vector(point)
    local.x = -local.x
    return frame @ local


# ---------------------------------------------------------------------------
# The markers, as a skeleton
# ---------------------------------------------------------------------------


def tree_markers(tree):
    """[(node, marker)] for every marker in ``tree`` a wrap moves by itself:
    all of them but a Skeleton node's face, finger and toe landmarks, which
    ride with their anchor as they do when dragged."""
    from ..primary_rig import GROUP_ANCHOR, MARKER_NODE_IDNAMES

    return [
        (node, m)
        for node in tree.nodes
        if node.bl_idname in MARKER_NODE_IDNAMES
        for m in node.markers
        if m.key and not (node.bl_idname == _SKELETON_NODE and m.key in GROUP_ANCHOR)
    ]


def gather(tree):
    """The markers a wrap places, and the bones between them.

    The markers are ``tree_markers`` less any whose position is no place of
    its own -- a readout of its bone, or an offset from rest. The bones, as
    index pairs, are the lines the viewport draws: a marker and its parent,
    joined markers, MediaPipe's bones between a Skeleton node's landmarks.
    """
    from ..marker_links import links_of
    from ..primary_rig import LM_BY_INDEX, POSE_CONNECTIONS
    from ..sockets import source_marker

    modes = {}

    def placeable(node):
        if not hasattr(node, "link_state"):
            return True
        if node.name not in modes:
            modes[node.name] = node.link_state().modes["position"]
        return modes[node.name] in (None, "absolute")

    entries = [(n, m) for n, m in tree_markers(tree) if placeable(n)]
    index = {(n.name, m.key): i for i, (n, m) in enumerate(entries)}
    bones = []

    def bone(a, b):
        i, j = index.get(a), index.get(b)
        if i is not None and j is not None and i != j:
            bones.append((i, j))

    for node, marker in entries:
        if hasattr(node, "_parent_source"):
            parent, pm = source_marker(node._parent_source())
            if pm is not None and parent.id_data == tree:
                bone((node.name, marker.key), (parent.name, pm.key))
    for a, b in links_of(tree):
        if a.marker is not None and b.marker is not None:
            bone((a.name, a.marker.key), (b.name, b.marker.key))
    for node in tree.nodes:
        if node.bl_idname == _SKELETON_NODE:
            for a, b in POSE_CONNECTIONS:
                bone((node.name, LM_BY_INDEX[a]), (node.name, LM_BY_INDEX[b]))
    return entries, bones


def partners(entries, points=None, frame=None, size=1.0):
    """Each marker's mirror partner, by name: Hand.R for Hand.L, a landmark
    by its key; -1 when the partner is missing. A marker with no side in its
    name is its own partner, on the middle line -- unless ``points`` has it
    well away from the middle of ``frame``, when it has none either."""
    from ..primary_rig import LM_MIRROR

    by_key = {(n.name, m.key): i for i, (n, m) in enumerate(entries)}
    by_name = {}
    for i, (_n, m) in enumerate(entries):
        by_name.setdefault(m.name, i)
    out = []
    for i, (node, marker) in enumerate(entries):
        if node.bl_idname == _SKELETON_NODE and marker.key in LM_MIRROR:
            other = LM_MIRROR[marker.key]
            j = by_key.get((node.name, other), -1) if other else i
        else:
            flipped = bpy.utils.flip_name(marker.name)
            j = by_name.get(flipped, -1) if flipped != marker.name else i
        # Measured in the world, as twice the way to the middle: the mesh's
        # own units can be anything (a character imported at 0.01 scale).
        if j == i and frame is not None and (_mirrored(frame, points[i]) - Vector(points[i])).length > 0.1 * size:
            j = -1
        out.append(j)
    return out


def world_positions(entries):
    """(n, 3) where the markers are, in the world."""
    points = [node.marker_value(m, "position") for node, m in entries]
    return np.array(points, dtype=float).reshape(-1, 3)


def place_markers(entries, points):
    """Move each marker to its world position in ``points`` -- parents first,
    each child against its parent's new place, as a drag does -- with a
    Skeleton node's face, finger and toe landmarks riding along. The rig
    follows on the next tick."""
    from ..tree import apply_soon
    from .marker_base import deferred_marker_writes

    if not entries:
        return
    with deferred_marker_writes():
        for (node, marker), point in zip(entries, points):
            for key, value in node._with_group_followers({marker.key: Vector(point)}).items():
                moved = node.marker_by_key(key)
                if moved is not None:
                    node.write_marker(moved, "position", tuple(value))
    entries[0][0].id_data.mark_dirty()
    apply_soon()


def _height(points):
    """How big the skeleton is: the distances are shares of it."""
    from ..primary_rig import DEFAULT_HEIGHT

    extent = float(np.ptp(points, axis=0).max()) if len(points) else 0.0
    return extent if extent > 1e-3 else DEFAULT_HEIGHT


def _redraw():
    """Viewports for the pairs, node editors for the node's buttons."""
    wm = bpy.context.window_manager
    for window in wm.windows if wm is not None else ():
        for area in window.screen.areas:
            if area.type in {"VIEW_3D", "NODE_EDITOR"}:
                area.tag_redraw()


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------

# While Pick Pairs runs: {"tree", "chosen": (node name, key) or None, "spot"}.
_picking = None


def pair(wrap, ident, mesh, origin, direction):
    """Pair marker ``ident`` -- (node name, key) -- with the place the ray
    finds on ``mesh`` and, with Symmetric on, its partner with the mirror
    image. Returns how many markers were paired: 0 when the ray misses."""
    entries = tree_markers(wrap.id_data)
    keys = [(n.name, m.key) for n, m in entries]
    if tuple(ident) not in keys:
        return 0
    i = keys.index(tuple(ident))
    rays = [(i, Vector(origin), Vector(direction))]
    j = partners(entries)[i] if wrap.symmetric else -1
    if j not in (-1, i):
        near = _mirrored(mesh.frame, origin)
        rays.append((j, near, _mirrored(mesh.frame, Vector(origin) + Vector(direction)) - near))
    done = 0
    for k, o, d in rays:
        marker = entries[k][1]
        spot = mesh.pick(o, d, marker.wrap_role) if marker.wrap_role != "FIXED" else None
        if spot is None:
            if k == i:
                return 0
            continue
        marker.wrap_target = spot
        marker.wrap_pair = True
        done += 1
    _redraw()
    return done


def auto_pairs(wrap, mesh):
    """Pair every unpaired marker the Human Skeleton knows by name (Pelvis,
    Hand.L...) with its place on ``mesh``, found from a character's
    proportions (``human_skeleton.landmarks``) -- in the middle of the body
    there for a joint, on the skin for a Surface marker. A marker found by
    its height keeps that height: the middle of a round head is its centre,
    well above the neck. With Symmetric on, the right side is the left one's
    mirror image. Returns how many."""
    from ..human_skeleton import landmarks, spec

    found = landmarks(mesh)
    level = {name for name, _c, _p, _w, how, _s in spec() if how == "height"}
    todo = [
        m for _n, m in tree_markers(wrap.id_data)
        if m.name in found and not m.wrap_pair and m.wrap_role in ("INSIDE", "SURFACE")
    ]
    if not todo:
        return 0
    if wrap.symmetric:
        for m in todo:
            other = bpy.utils.flip_name(m.name)
            if other != m.name and other in found and found[m.name].x < found[other].x:
                found[m.name] = _mirrored(mesh.frame, found[other])  # the right, from the left
    places = mesh.place(np.array([found[m.name] for m in todo]), [m.wrap_role for m in todo])
    for m, place in zip(todo, places):
        spot = Vector(place) if np.isfinite(place).all() else found[m.name]
        if m.name in level and m.wrap_role == "INSIDE":
            spot.z = found[m.name].z
        m.wrap_target = spot
        m.wrap_pair = True
    _redraw()
    return len(todo)


def unpair(wrap, ident=None):
    """Forget marker ``ident``'s pair and, with Symmetric on, its partner's --
    or every pair in the tree when ``ident`` is None."""
    entries = tree_markers(wrap.id_data)
    keys = [(n.name, m.key) for n, m in entries]
    if ident is None:
        chosen = range(len(entries))
    elif tuple(ident) in keys:
        i = keys.index(tuple(ident))
        j = partners(entries)[i] if wrap.symmetric else -1
        chosen = [i] if j in (-1, i) else [i, j]
    else:
        chosen = ()
    for k in chosen:
        entries[k][1].wrap_pair = False
    _redraw()


def draw_pairs(nodes, height):
    """For the viewport overlay: each pair as a line from its marker to its
    place, the place lit -- in trees whose Wrap Markers node shows pairs --
    and, while Pick Pairs runs, the marker picked and where it would go."""
    from ..handles import draw_lines, line_shader, ui_scale
    from ..primary_rig import _draw_glow_points

    showing, coords, spots = {}, [], []
    for node in nodes:
        tree = node.id_data
        if tree.name not in showing:
            showing[tree.name] = any(n.bl_idname == WRAP_NODE and n.show_pairs for n in tree.nodes)
        if showing[tree.name]:
            for marker in node.markers:
                if marker.wrap_pair:
                    spot = tuple(marker.wrap_target)
                    coords += [tuple(node.marker_value(marker, "position")), spot]
                    spots.append(spot)
    lit = None
    if _picking is not None and _picking["chosen"] is not None:
        tree = bpy.data.node_groups.get(_picking["tree"])
        node = tree.nodes.get(_picking["chosen"][0]) if tree is not None else None
        marker = node.marker_by_key(_picking["chosen"][1]) if node is not None else None
        if marker is not None:
            lit = tuple(node.marker_value(marker, "position"))
            if _picking["spot"] is not None:
                coords += [lit, tuple(_picking["spot"])]
                spots.append(tuple(_picking["spot"]))
    if coords:
        draw_lines(line_shader(bpy.context.region), coords, _PAIR_COLOR, 2.0 * ui_scale())
    _draw_glow_points(spots, _PAIR_COLOR, 0.45, height)
    if lit is not None:
        _draw_glow_points([lit], _PAIR_COLOR, 1.4, height, bright=True)


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


class Refused(Exception):
    """A wrap step that cannot run, with the reason to report."""


class WrapRun:
    """One wrap step in flight: the backend it runs on.

    The solver's passes, fed to the markers one at a time -- by the modal
    operator, one per timer tick, so the skeleton glides onto the mesh and
    the viewport keeps drawing; or all at once from a script (``finish``).
    """

    def __init__(self, wrap, step, depsgraph):
        if wrap.target is None or wrap.target.type != "MESH":
            raise Refused("Choose the mesh to wrap onto")
        self.tree_name, self.node_name, self.step = wrap.id_data.name, wrap.name, step
        self.entries, bones = gather(wrap.id_data)
        if not self.entries:
            raise Refused("There are no markers in this tree to wrap")
        mesh = MeshTarget(wrap.target, depsgraph)
        if step == "FIT" and not any(m.wrap_pair for _n, m in self.entries):
            auto_pairs(wrap, mesh)
        pairs = {i: tuple(m.wrap_target) for i, (_n, m) in enumerate(self.entries) if m.wrap_pair}
        if step in ("SNAP", "FIT") and not pairs:
            raise Refused("Pick at least one pair first")
        self.start = world_positions(self.entries)
        wrap.remember(self.entries)
        rest = wrap.originals(self.entries)
        size = _height(rest)
        mirror = partners(self.entries, self.start, mesh.frame, size) if wrap.symmetric else None
        roles = [m.wrap_role for _n, m in self.entries]
        skeleton = Skeleton(rest, bones, roles, pairs, mirror, np.array(mesh.frame))
        reach = {"ATTRACT": wrap.attract_distance * 0.01 * size, "STICK": wrap.stick_distance * 0.01 * size}
        steps = ("SNAP", "ATTRACT", "STICK") if step == "FIT" else (step,)
        self.passes = self._steps(skeleton, steps, mesh.place, reach)

    def _steps(self, skeleton, steps, place, reach):
        """Each step's passes, one after the other, each from where the last
        left the markers."""
        points = self.start
        for step in steps:
            if step == "SNAP":
                passes = skeleton.snap(points)
            else:
                passes = getattr(skeleton, step.lower())(points, place, reach[step])
            for points in passes:
                yield points

    def advance(self):
        """Feed the next pass to the markers; False once there are none left."""
        points = next(self.passes, None)
        if points is None:
            return False
        place_markers(self.entries, points)
        return True

    def finish(self):
        while self.advance():
            pass

    def cancel(self):
        """Put the markers back where the step found them."""
        place_markers(self.entries, self.start)

    def node(self):
        tree = bpy.data.node_groups.get(self.tree_name)
        return tree.nodes.get(self.node_name) if tree is not None else None


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class _WrapOperator:
    """What the wrap operators share: the node they act for -- the one whose
    button was pressed, or the one a script names."""

    tree: StringProperty(name="Tree", options={"HIDDEN", "SKIP_SAVE"})
    node: StringProperty(name="Node", options={"HIDDEN", "SKIP_SAVE"})

    def wrap_node(self, context):
        tree = bpy.data.node_groups.get(self.tree) if self.tree else None
        node = tree.nodes.get(self.node) if tree is not None else getattr(context, "node", None)
        return node if getattr(node, "bl_idname", "") == WRAP_NODE else None

    def with_mesh(self, context):
        """The node, with a mesh to wrap onto -- the selected one if it has
        none yet -- or None, reported."""
        wrap = self.wrap_node(context)
        if wrap is not None and wrap.target is None:
            for obj in (getattr(context, "active_object", None), *getattr(context, "selected_objects", ())):
                if obj is not None and obj.type == "MESH":
                    wrap.target = obj
                    break
        if wrap is None or wrap.target is None:
            self.report({"WARNING"}, "Choose the mesh to wrap onto first")
            return None
        return wrap


class ARMATURE_NODES_OT_wrap_run(_WrapOperator, Operator):
    """Run a wrap step on the tree's markers"""

    bl_idname = "armature_nodes.wrap_run"
    bl_label = "Wrap"
    bl_options = {"UNDO"}  # one step, one undo; no redo panel re-running it

    step: EnumProperty(name="Step", items=[(s, label, tip) for s, label, tip, _icon in (*_STEPS, _FIT)])

    @classmethod
    def description(cls, context, properties):
        return next(tip for s, _label, tip, _icon in (*_STEPS, _FIT) if s == properties.step)

    def _begin(self, context):
        wrap = self.with_mesh(context)
        if wrap is None:
            return None
        try:
            return WrapRun(wrap, self.step, context.evaluated_depsgraph_get())
        except (Refused, ValueError) as exc:
            self.report({"WARNING"}, str(exc))
            return None

    def execute(self, context):
        run = self._begin(context)
        if run is None:
            return {"CANCELLED"}
        run.finish()
        return self._done(run)

    def invoke(self, context, event):
        self._run = self._begin(context)
        if self._run is None:
            return {"CANCELLED"}
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        context.workspace.status_text_set(f"Wrap Markers: {self.step.title()}...   Esc: stop")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC" and event.value == "PRESS":
            self._stop(context)
            self._run.cancel()
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"} if event.type in _NAVIGATION else {"RUNNING_MODAL"}
        try:
            more = self._run.advance()
        except Exception as exc:  # noqa: BLE001 -- a marker deleted under the step, say
            self._stop(context)
            self.report({"WARNING"}, f"Wrap stopped: {exc}")
            return {"CANCELLED"}
        if more:
            return {"RUNNING_MODAL"}
        self._stop(context)
        return self._done(self._run)

    def _stop(self, context):
        context.window_manager.event_timer_remove(self._timer)
        context.workspace.status_text_set(None)

    def _done(self, run):
        wrap = run.node()
        if wrap is not None:
            wrap.wrap_stage = len(_STEPS) if self.step == "FIT" else [s[0] for s in _STEPS].index(self.step) + 1
        _redraw()
        self.report({"INFO"}, f"{self.step.title()}: {len(run.entries)} markers")
        return {"FINISHED"}


class ARMATURE_NODES_OT_wrap_pick(_WrapOperator, Operator):
    """Pair markers with their places on the mesh: click a marker, then where
    it goes -- then the next. Right-click drops the picked marker, X forgets
    its pair; Esc or right-click with none picked ends"""

    bl_idname = "armature_nodes.wrap_pick"
    bl_label = "Pick Pairs"
    bl_options = {"UNDO"}  # the whole picking session is one undo step

    def invoke(self, context, event):
        global _picking
        wrap = self.with_mesh(context)
        if wrap is None:
            return {"CANCELLED"}
        try:
            self._mesh = MeshTarget(wrap.target, context.evaluated_depsgraph_get())
        except ValueError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        self.tree, self.node = wrap.id_data.name, wrap.name
        _picking = {"tree": self.tree, "chosen": None, "spot": None}
        context.window_manager.modal_handler_add(self)
        self._say(context, wrap)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        wrap = self.wrap_node(context)
        if wrap is None:
            return self._end(context)
        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            if _picking["chosen"] is None:
                return self._end(context)
            _picking["chosen"] = _picking["spot"] = None  # drop it, keep picking
            self._say(context, wrap)
            _redraw()
            return {"RUNNING_MODAL"}
        at = _view_under(context, event)
        if event.type == "MOUSEMOVE":
            _picking["spot"] = self._spot(wrap, at)
            _redraw()
            return {"PASS_THROUGH"}  # so the handle under the mouse lights up
        if event.value != "PRESS" or at is None:
            return {"PASS_THROUGH"}
        if event.type == "LEFTMOUSE":
            self._click(wrap, at)
        elif event.type in {"X", "DEL"} and _picking["chosen"] is not None:
            unpair(wrap, _picking["chosen"])
            _picking["chosen"] = _picking["spot"] = None
        else:
            return {"PASS_THROUGH"}
        self._say(context, wrap)
        return {"RUNNING_MODAL"}

    def _chosen_marker(self, wrap, ident):
        node = wrap.id_data.nodes.get(ident[0]) if ident is not None else None
        return node.marker_by_key(ident[1]) if hasattr(node, "marker_by_key") else None

    def _spot(self, wrap, at):
        marker = self._chosen_marker(wrap, _picking["chosen"])
        if marker is None or at is None:
            return None
        view, xy = at
        return self._mesh.pick(*view.ray(xy), marker.wrap_role)

    def _click(self, wrap, at):
        """With a marker picked, the mesh under the cursor pairs it -- even
        through a glow, since its place is often right behind one. Off the
        mesh, a marker's glow picks that marker."""
        view, xy = at
        if _picking["chosen"] is not None and pair(wrap, _picking["chosen"], self._mesh, *view.ray(xy)):
            _picking["chosen"] = _picking["spot"] = None
            return
        under = _marker_under(view, xy, self.tree)
        marker = self._chosen_marker(wrap, under)
        if marker is None:
            return
        if marker.wrap_role == "FIXED":
            self.report({"WARNING"}, f"'{marker.name}' is Fixed: it takes no pair")
        else:
            _picking["chosen"] = under

    def _say(self, context, wrap):
        marker = self._chosen_marker(wrap, _picking["chosen"])
        if marker is None:
            text = "Pick Pairs: click a marker   Esc / right-click: done"
        else:
            text = (
                f"Pick Pairs: click where {marker.name} goes on {wrap.target.name}"
                "   Right-click: another marker   X: forget its pair"
            )
        context.workspace.status_text_set(text)

    def _end(self, context):
        global _picking
        _picking = None
        context.workspace.status_text_set(None)
        _redraw()
        return {"FINISHED"}


def _view_under(context, event):
    """(View, mouse position in its region) for the 3D viewport under the
    mouse, or None -- also over its header, toolbar or sidebar, which lie on
    top of the view and keep their clicks."""
    from ..handles import View

    def under(region):
        return (
            region.x <= event.mouse_x < region.x + region.width
            and region.y <= event.mouse_y < region.y + region.height
        )

    for area in context.window.screen.areas if context.window is not None else ():
        if area.type != "VIEW_3D":
            continue
        regions = [r for r in area.regions if r.width > 1 and r.height > 1 and under(r)]
        view = next((r for r in regions if r.type == "WINDOW"), None)
        if view is None or any(r.type != "WINDOW" for r in regions):
            return None
        return View(view, view.data), Vector((event.mouse_x - view.x, event.mouse_y - view.y))
    return None


def _marker_under(view, xy, tree_name):
    """(node name, key) of the marker glow under ``xy`` in ``tree_name``.

    Tested at the click, as the handles test a drag: the glow lit last is
    only as fresh as the last mouse move over the view, so trusting it would
    pair whatever marker the mouse passed before leaving the view.
    """
    from ..handles import pick, visible_handles

    handles = [h for h in visible_handles() if h.tree == tree_name]
    points = []
    for h in handles:
        at, scale = view.to_screen(h.obj.location), h.scale(view)
        points.append((at if scale else None, scale or 0.0, h.core_mode(), False, False))
    found = pick(points, xy)
    return None if found is None else (handles[found[0]].node, handles[found[0]].key)


class ARMATURE_NODES_OT_wrap_symmetrize(_WrapOperator, Operator):
    """Make the skeleton symmetric -- each marker averaged with its partner's
    mirror image, one without a side put on the middle -- and give every pair
    its partner on the other side"""

    bl_idname = "armature_nodes.wrap_symmetrize"
    bl_label = "Symmetrize"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        wrap = self.wrap_node(context)
        if wrap is None:
            return {"CANCELLED"}
        entries, _bones = gather(wrap.id_data)
        points = world_positions(entries)
        sides = [(i, j) for i, j in enumerate(partners(entries)) if j not in (-1, i)]
        if not sides:
            self.report({"WARNING"}, "No left and right markers to mirror: name them .L and .R")
            return {"CANCELLED"}
        # The skeleton's own middle: where each left and right meet.
        frame = Matrix.Translation((float(np.mean([points[i][0] + points[j][0] for i, j in sides])) / 2, 0, 0))
        mirror = partners(entries, points, frame, _height(points))
        skeleton = Skeleton(points, [], mirror=mirror, frame=np.array(frame))
        place_markers(entries, skeleton.symmetrize(points))
        if wrap.wrap_original:  # what Original goes back to is symmetric too
            wrap.remember(entries, skeleton.symmetrize(wrap.originals(entries)), replace=True)
        across = wrap.target.matrix_world if wrap.target is not None else frame
        added = 0
        for i, j in sides:
            a, b = entries[i][1], entries[j][1]
            if a.wrap_pair and not b.wrap_pair and b.wrap_role != "FIXED":
                b.wrap_target = _mirrored(across, a.wrap_target)
                b.wrap_pair = True
                added += 1
        _redraw()
        self.report({"INFO"}, f"Symmetric: {len(sides) // 2} left and right, {added} pair(s) mirrored")
        return {"FINISHED"}


class ARMATURE_NODES_OT_wrap_original(_WrapOperator, Operator):
    """Put the markers back where they were before the first wrap step"""

    bl_idname = "armature_nodes.wrap_original"
    bl_label = "Original"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        wrap = self.wrap_node(context)
        if wrap is None or not wrap.wrap_original:
            self.report({"INFO"}, "The markers are where they started")
            return {"CANCELLED"}
        entries = tree_markers(wrap.id_data)
        place_markers(entries, wrap.originals(entries))
        wrap.wrap_original, wrap.wrap_stage = "", 0
        _redraw()
        return {"FINISHED"}


class ARMATURE_NODES_OT_wrap_unpair(_WrapOperator, Operator):
    """Forget this marker's pair -- or, on the node, every pair"""

    bl_idname = "armature_nodes.wrap_unpair"
    bl_label = "Forget Pairs"
    bl_options = {"REGISTER", "UNDO"}

    marker: StringProperty(
        name="Marker",
        description="Node name and marker key, tab separated; empty for every pair",
        options={"HIDDEN", "SKIP_SAVE"},
    )

    def execute(self, context):
        wrap = self.wrap_node(context)
        if wrap is None:
            return {"CANCELLED"}
        unpair(wrap, tuple(self.marker.split("\t")) if self.marker else None)
        return {"FINISHED"}


class ARMATURE_NODES_OT_wrap_auto_pairs(_WrapOperator, Operator):
    """Pair the markers the Human Skeleton knows by name -- Pelvis, Head,
    Hand.L... -- with their places on the mesh, found from a character's
    proportions: no clicks. Markers already paired keep their pairs"""

    bl_idname = "armature_nodes.wrap_auto_pairs"
    bl_label = "Auto Pairs"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        wrap = self.with_mesh(context)
        if wrap is None:
            return {"CANCELLED"}
        try:
            count = auto_pairs(wrap, MeshTarget(wrap.target, context.evaluated_depsgraph_get()))
        except ValueError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        if not count:
            self.report({"WARNING"}, "No unpaired marker here has a name Auto Pairs knows (Pelvis, Hand.L...)")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Paired {count} marker(s)")
        return {"FINISHED"}


class ARMATURE_NODES_OT_wrap_add(Operator):
    """Add a Wrap Markers node to the selected rig's tree, to fit the
    markers it holds onto a character. Markers named like the Human
    Skeleton's pole targets (Elbow, Knee) are made Free, as there"""

    bl_idname = "armature_nodes.wrap_add"
    bl_label = "Wrap These Markers"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from ..human_skeleton import spec

        obj = context.active_object
        tree = getattr(obj, "armature_nodes_tree", None) if obj is not None and obj.type == "ARMATURE" else None
        if tree is None:
            self.report({"WARNING"}, "Select a rig with an Armature Nodes tree first")
            return {"CANCELLED"}
        carried = {name for name, _c, _p, _w, how, _s in spec() if how is None}
        for _node, marker in tree_markers(tree):
            if marker.name in carried and marker.wrap_role == "INSIDE":
                marker.wrap_role = "FREE"
        wrap = tree.nodes.new(WRAP_NODE)
        spots = [n.location for n in tree.nodes if n != wrap]
        wrap.location = (min(p.x for p in spots) - 300.0, max(p.y for p in spots)) if spots else (0.0, 0.0)
        _redraw()
        return {"FINISHED"}


def wrap_node_for(context):
    """The Wrap Markers node the viewport panel works with: one in the
    selected rig's tree or a group it runs, else one wrapping onto the
    selected mesh. None when there is none."""
    from ..groups import armature_trees, trees_in_use

    obj = context.active_object
    tree = getattr(obj, "armature_nodes_tree", None) if obj is not None and obj.type == "ARMATURE" else None
    trees = trees_in_use(tree) if tree is not None else []
    for tree in trees:
        for node in tree.nodes:
            if node.bl_idname == WRAP_NODE:
                return node
    if obj is not None and obj.type == "MESH":
        for tree in armature_trees():
            for node in tree.nodes:
                if node.bl_idname == WRAP_NODE and node.target == obj:
                    return node
    return None


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------


def _redraw_only(self, context):
    redraw_viewports()


class WrapMarkersNode(ArmatureNodeBase, Node):
    """Wrap the tree's markers onto a mesh, the way the wrap add-on wraps a
    template onto a scan: pairs, then Snap, Attract, Stick.

    No sockets: it works on the markers of its tree, the skeleton the
    viewport draws, and each marker's own Wrap role says what happens to it
    (Inside, Surface, Free, Fixed -- in the sidebar).
    """

    bl_idname = WRAP_NODE
    bl_label = "Wrap Markers"
    bl_icon = "MOD_SHRINKWRAP"

    target: PointerProperty(
        name="Mesh",
        description="The mesh to wrap the markers onto",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "MESH",
        update=_redraw_only,
    )
    symmetric: BoolProperty(
        name="Symmetric",
        description=(
            "Pair a marker's partner (Hand.R for Hand.L) with the mirror image "
            "across the mesh's middle, and keep the skeleton symmetric as it wraps"
        ),
        default=True,
    )
    attract_distance: FloatProperty(
        name="Attract Distance",
        description="How far Attract reaches on its last pass, as a share of the skeleton's height",
        default=5.0,
        min=0.1,
        max=50.0,
        precision=1,
        subtype="PERCENTAGE",
        update=_redraw_only,
    )
    stick_distance: FloatProperty(
        name="Stick Distance",
        description=(
            "A marker this close to its place, as a share of the skeleton's "
            "height, lands exactly on it"
        ),
        default=3.0,
        min=0.1,
        max=50.0,
        precision=1,
        subtype="PERCENTAGE",
        update=_redraw_only,
    )
    show_pairs: BoolProperty(
        name="Show Pairs",
        description="Draw each pair in the viewport: a line from the marker to its place",
        default=True,
        update=_redraw_only,
    )
    wrap_stage: IntProperty(name="Steps Run", default=0, options={"HIDDEN"})
    wrap_original: StringProperty(name="Original", default="", options={"HIDDEN"})

    def init(self, context):
        self.width = 220

    def remember(self, entries, points=None, replace=False):
        """Keep where the markers are -- what Original goes back to, and the
        shape Snap starts from -- unless that is already kept."""
        if self.wrap_original and not replace:
            return
        points = world_positions(entries) if points is None else points
        saved = {f"{n.name}\t{m.key}": [float(v) for v in p] for (n, m), p in zip(entries, points)}
        self.wrap_original = json.dumps(saved)

    def originals(self, entries):
        """(n, 3) where the markers were before the first step -- where they
        are now, for one added since."""
        saved = json.loads(self.wrap_original or "{}")
        points = [saved.get(f"{n.name}\t{m.key}") or n.marker_value(m, "position") for n, m in entries]
        return np.array(points, dtype=float).reshape(-1, 3)

    def draw_buttons(self, context, layout):
        layout.context_pointer_set("node", self)
        layout.prop(self, "target", text="", icon="MESH_DATA")
        count = sum(m.wrap_pair for _n, m in tree_markers(self.id_data))
        row = layout.row(align=True)
        row.operator("armature_nodes.wrap_auto_pairs", text="Auto", icon="AUTO")
        row.operator("armature_nodes.wrap_pick", text=f"Pick Pairs ({count})", icon="EYEDROPPER")
        row.operator("armature_nodes.wrap_unpair", text="", icon="X")
        row.prop(self, "show_pairs", text="", icon="HIDE_OFF" if self.show_pairs else "HIDE_ON")
        row = layout.row(align=True)
        row.prop(self, "symmetric", toggle=True, icon="MOD_MIRROR")
        row.operator("armature_nodes.wrap_symmetrize")
        row = layout.row()
        row.scale_y = 1.5
        row.operator("armature_nodes.wrap_run", text=_FIT[1], icon=_FIT[3]).step = _FIT[0]
        col = layout.column(align=True)
        col.scale_y = 1.2
        for number, (step, label, _tip, icon) in enumerate(_STEPS, 1):
            row = col.row(align=True)
            op = row.operator("armature_nodes.wrap_run", text=label, icon=icon, depress=self.wrap_stage >= number)
            op.step = step
            if step != "SNAP":
                row.prop(self, "attract_distance" if step == "ATTRACT" else "stick_distance", text="")
        row = layout.row()
        row.enabled = bool(self.wrap_original)
        row.operator("armature_nodes.wrap_original", icon="LOOP_BACK")

    def draw_buttons_ext(self, context, layout):
        """The sidebar: the node, then each marker's role and pair."""
        self.draw_buttons(context, layout)
        box = layout.box()
        box.label(text="Markers", icon="EMPTY_AXIS")
        for node, marker in tree_markers(self.id_data):
            row = box.row(align=True)
            row.label(text=marker.name or marker.key)
            row.prop(marker, "wrap_role", text="")
            if marker.wrap_pair:
                op = row.operator("armature_nodes.wrap_unpair", text="", icon="X")
                op.marker = f"{node.name}\t{marker.key}"
            else:
                row.label(text="", icon="BLANK1")


classes = (
    WrapMarkersNode,
    ARMATURE_NODES_OT_wrap_run,
    ARMATURE_NODES_OT_wrap_pick,
    ARMATURE_NODES_OT_wrap_auto_pairs,
    ARMATURE_NODES_OT_wrap_symmetrize,
    ARMATURE_NODES_OT_wrap_original,
    ARMATURE_NODES_OT_wrap_unpair,
    ARMATURE_NODES_OT_wrap_add,
)
