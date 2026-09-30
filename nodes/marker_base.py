"""What the Marker and Skeleton nodes share.

The marker itself (``SkeletonMarker``); the list of them a node holds, with
its output sockets and viewport handles (``MarkerHolderMixin``); and the hold
on marker writes for the length of a live pass (``deferred_marker_writes``).
"""

import contextlib
import logging

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, StringProperty
from mathutils import Vector

from ..sockets import VectorSocket
from .base import redraw_viewports, same_vec

log = logging.getLogger(__name__)

_EPS = 1e-5


# True while markers are being written in bulk, so the per-marker callback
# does not rebuild once per component.
_syncing_markers = False


def handle_size_prop():
    """Size of a marker node's handles on screen. Display only: no rebuild."""
    return FloatProperty(
        name="Size",
        description="How big this node's marker handles are drawn in the viewport",
        default=1.0,
        min=0.5,
        max=3.0,
        update=lambda self, context: redraw_viewports(),
    )


def _slug(text):
    """A key-safe token from a display name."""
    out = "".join(c if c.isalnum() else "_" for c in (text or "").strip().lower())
    return out.strip("_")


def marker_owner(marker):
    """The node a marker belongs to.

    A PropertyGroup only knows its owning ID (the node tree), not the node, so
    the node is found by identity. A tree holds a handful of marker nodes at
    most and this only runs on an edit, so the scan is cheaper than keeping a
    back-reference correct across copy / paste / rename.
    """
    tree = marker.id_data
    if tree is None:
        return None
    for node in tree.nodes:
        if not hasattr(node, "markers"):
            continue
        for m in node.markers:
            if m == marker:
                return node
    return None


def _on_marker_changed(self, context):
    """A marker position/rotation was typed: move its handle and rebuild."""
    if _syncing_markers:
        return
    from ..primary_rig import tag_viewports_redraw

    node = marker_owner(self)
    if node is not None:
        node.push_markers_to_empties()
    tag_viewports_redraw()
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


def _on_marker_name_changed(self, context):
    """Renaming a marker renames its output socket; links survive because they
    are attached to the socket, not to its name."""
    node = marker_owner(self)
    if node is not None and hasattr(node, "sync_marker_sockets"):
        node.sync_marker_sockets()


def _on_marker_use_rotation_changed(self, context):
    """Rotation enabled/disabled on one marker: re-lock and redraw its handle,
    then rebuild -- an oriented marker also turns what it drives."""
    from ..primary_rig import (
        apply_marker_locks,
        find_marker_empties,
        tag_viewports_redraw,
    )

    node = marker_owner(self)
    if node is None:
        return
    for key, obj in find_marker_empties(node).items():
        apply_marker_locks(node, obj, key)
    # Through the node, not straight onto the handle: a Marker node wired
    # into a relative input draws its handle on the bone, not at the value.
    node.push_markers_to_empties()
    tag_viewports_redraw()
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


class SkeletonMarker(bpy.types.PropertyGroup):
    """One marker: a world position, optionally oriented.

    ``key`` is the stable identity -- the output socket and the viewport
    handle are both bound to it, so it is assigned once and never changes.
    ``name`` is only the label and is free to be edited.
    """

    key: StringProperty(name="Key", default="", options={"HIDDEN"})
    name: StringProperty(
        name="Name",
        description="Label for this marker; also names its output socket",
        default="Marker",
        update=_on_marker_name_changed,
    )
    position: FloatVectorProperty(
        name="Position",
        description="World position of the marker handle",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="TRANSLATION",
        update=_on_marker_changed,
    )
    rotation: FloatVectorProperty(
        name="Rotation",
        description="Orientation of the marker, when rotation is enabled on it",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="EULER",
        update=_on_marker_changed,
    )
    use_rotation: BoolProperty(
        name="Rotation",
        description=(
            "Adjust this marker's rotation as well as its position. Off by "
            "default: the handle is position-only until this is enabled"
        ),
        default=False,
        update=_on_marker_use_rotation_changed,
    )
    scale: FloatVectorProperty(
        name="Scale",
        description="Scale the marker supplies to a Scale input",
        size=3,
        default=(1.0, 1.0, 1.0),
        subtype="XYZ",
        update=_on_marker_changed,
    )
    color: FloatVectorProperty(
        name="Color",
        description=(
            "Glow colour of this marker's handle. MediaPipe landmarks keep "
            "their side colours"
        ),
        size=3,
        min=0.0,
        max=1.0,
        default=(0.95, 0.45, 0.75),
        subtype="COLOR",
        update=lambda self, context: redraw_viewports(),
    )
    # What a Wrap Markers node does with this marker (nodes/wrap.py).
    wrap_role: EnumProperty(
        name="Wrap",
        description="What a Wrap Markers node does with this marker",
        items=(
            ("INSIDE", "Inside", "A joint: drawn into the middle of the limb it sits in"),
            ("SURFACE", "Surface", "A landmark: drawn onto the skin"),
            ("FREE", "Free", "Carried along by the skeleton, never drawn to the mesh"),
            ("FIXED", "Fixed", "Never moved by a wrap"),
        ),
        default="INSIDE",
    )
    wrap_pair: BoolProperty(name="Paired", default=False, options={"HIDDEN"})
    # Picked by hand: kept by Fit, which redoes the pairs it made itself.
    wrap_picked: BoolProperty(name="Picked", default=False, options={"HIDDEN"})
    wrap_target: FloatVectorProperty(
        name="Pair",
        description="Where the pair puts this marker, on or in the mesh",
        size=3,
        subtype="TRANSLATION",
        options={"HIDDEN"},
    )

    def set_position(self, value):
        """Write without firing the per-marker rebuild callback.

        Callers that set many markers at once (a preset, a mirror, a handle
        drag) push to the viewport and mark the tree dirty themselves; letting
        each component fire would rebuild the rig dozens of times per edit.
        """
        global _syncing_markers
        was = _syncing_markers
        _syncing_markers = True
        try:
            self.position = tuple(value)
        finally:
            _syncing_markers = was

    def set_rotation(self, value):
        self._set_quietly("rotation", value)

    def set_scale(self, value):  # called as "set_" + attr
        self._set_quietly("scale", value)

    def _set_quietly(self, attr, value):
        global _syncing_markers
        was = _syncing_markers
        _syncing_markers = True
        try:
            setattr(self, attr, tuple(value))
        finally:
            _syncing_markers = was


# While a live pass runs (``deferred_marker_writes``): the world values nodes
# write into markers, waiting to be applied parent-first. None otherwise.
_deferred = None


@contextlib.contextmanager
def deferred_marker_writes():
    """Hold marker writes until a live pass is over, then apply them
    parent-first.

    Why: grab a parent bone and Blender carries its child bone with it. The
    parent's marker takes the parent bone's move and the child's marker takes
    the child bone's -- which already contains the carry. If the child's
    marker is parented to the parent's, applying the two one after the other
    moves it twice: once through its parent, once itself. So every node
    reads the markers as they were when the pass began, and the writes land
    together at the end, each child re-expressed against its parent's new
    place.
    """
    global _deferred
    if _deferred is not None:
        yield  # already inside a pass: the outer one applies
        return
    _deferred = {}
    try:
        yield
    finally:
        pending, _deferred = _deferred, None
        _apply_deferred(pending)


def _apply_deferred(pending):
    moved = []
    for node, marker, parts in sorted(pending.values(), key=lambda e: e[0].parent_depth()):
        try:
            node.apply_world(marker, parts)
            moved.append(node)
        except ReferenceError:
            continue  # the node went away during the pass
    for node in moved:
        node.push_markers_to_empties()


class MarkerHolderMixin:
    """Shared marker list, viewport handles and sockets.

    Both the single Marker node and the Skeleton node hold markers and both
    draw handles through ``primary_rig``, which asks a node for ``markers``
    and ``marker_keys()``. Keeping that in one place is what lets the handle,
    lock, mirror and overlay code stay ignorant of which node it is serving.
    """

    #: Socket type of each marker's output.
    output_socket = VectorSocket

    def sync_marker_sockets(self):
        """One output per marker, bound by ``marker_key``.

        Sockets are matched and renamed rather than rebuilt, because a link is
        attached to the socket itself: dropping and re-adding one would break
        every wire. New markers append, so inserting in the middle of the list
        leaves socket order behind list order -- harmless, and the alternative
        costs links. The one exception is a socket of the wrong type, left by
        an older version: that is replaced, and its wires moved across.
        """
        wanted = [(m.key, m.name or m.key) for m in self.markers if m.key]
        wanted_keys = {k for k, _n in wanted}
        idname = self.output_socket.bl_idname
        for sock in list(self.outputs):
            if sock.marker_key not in wanted_keys:
                self.outputs.remove(sock)
        existing = {s.marker_key: s for s in self.outputs}
        for key, label in wanted:
            sock = existing.get(key)
            if sock is not None and sock.bl_idname != idname:
                self._retype_output(sock, idname)
            elif sock is None:
                sock = self.outputs.new(idname, label)
                sock.marker_key = key
            elif sock.name != label:
                sock.name = label

    def _retype_output(self, sock, idname):
        """Swap ``sock`` for one of type ``idname``, keeping its wires and place."""
        tree = self.id_data
        index = list(self.outputs).index(sock)
        targets = [(l.to_node.name, l.to_socket.identifier) for l in sock.links]
        name, key = sock.name, sock.marker_key
        self.outputs.remove(sock)
        new = self.outputs.new(idname, name)
        new.marker_key = key
        try:
            self.outputs.move(len(self.outputs) - 1, index)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        for node_name, identifier in targets:
            node = tree.nodes.get(node_name)
            target = next((s for s in node.inputs if s.identifier == identifier), None) if node else None
            if target is not None:
                tree.links.new(new, target)

    def marker_keys(self):
        return [m.key for m in self.markers if m.key]

    def marker_by_key(self, key):
        for m in self.markers:
            if m.key == key:
                return m
        return None

    def unique_marker_key(self, base):
        """A key not already used on this node. Keys are stable identifiers:
        sockets and handles are bound to them, so they never change once
        assigned -- renaming a marker only changes its label."""
        base = base or "marker"
        taken = set(self.marker_keys())
        if base not in taken:
            return base
        i = 1
        while f"{base}.{i:03d}" in taken:
            i += 1
        return f"{base}.{i:03d}"

    def add_marker(self, name="", position=None, rotation=None, key=None):
        marker = self.markers.add()
        marker.key = self.unique_marker_key(key or _slug(name) or "marker")
        marker.name = name or marker.key
        if position is not None:
            marker.set_position(position)
        if rotation is not None:
            marker.set_rotation(rotation)
        self.sync_marker_sockets()
        self.refresh_handles()
        return marker

    def remove_marker(self, index):
        if not 0 <= index < len(self.markers):
            return False
        self.markers.remove(index)
        self.sync_marker_sockets()
        self.refresh_handles()
        return True

    def clear_markers(self):
        self.markers.clear()
        self.sync_marker_sockets()
        self.refresh_handles()

    def marker_uses_rotation(self, key):
        marker = self.marker_by_key(key)
        return bool(marker and marker.use_rotation)

    def marker_uses_scale(self, key):
        return False  # only a Marker node wired into a Scale input scales

    # -- World values ------------------------------------------------------
    #
    # What the rest of the graph sees of a marker is its WORLD transform. A
    # marker's own values are world values too -- except on a Marker node
    # with a parent, where they are relative to the parent, like a bone's
    # (MarkerNode overrides these).

    def marker_matrix(self, marker, _seen=None):
        """The marker's world transform, as a matrix."""
        from mathutils import Euler, Matrix

        return Matrix.LocRotScale(
            Vector(marker.position), Euler(marker.rotation, "XYZ"), Vector(marker.scale)
        )

    def marker_value(self, marker, attr):
        """One part of the marker's world transform: position, rotation or scale."""
        from mathutils import Euler

        loc, rot, scale = self.marker_matrix(marker).decompose()
        if attr == "position":
            return tuple(loc)
        if attr == "rotation":
            e = rot.to_euler("XYZ", Euler(marker.rotation, "XYZ"))
            return (e.x, e.y, e.z)
        return tuple(scale)

    def apply_world(self, marker, parts):
        """Give the marker these world values ({attr: value}); the rest of
        its world transform stays as it is."""
        for attr, value in parts.items():
            if not same_vec(getattr(marker, attr), value):
                getattr(marker, "set_" + attr)(value)

    def parent_depth(self):
        """How many parents up the marker's chain goes: parents are updated first."""
        return 0

    def write_marker(self, marker, attr, value):
        """Set one of a marker's world values from code and move its handle.

        This is how a node writes through a wired marker (``write_input``).
        No rebuild: the value comes from the rig, which is already there.
        During a live pass the write waits (``deferred_marker_writes``).
        """
        if same_vec(self.marker_value(marker, attr), value):
            return False
        if _deferred is not None:
            key = (self.id_data.name, self.name, marker.key)
            _deferred.setdefault(key, [self, marker, {}])[2][attr] = tuple(value)
            return True
        self.apply_world(marker, {attr: value})
        self.push_markers_to_empties()
        return True

    def marker_rotation(self, key):
        marker = self.marker_by_key(key)
        if marker is None:
            return Vector((0.0, 0.0, 0.0))
        return Vector(marker.rotation)

    def marker_position(self, key):
        marker = self.marker_by_key(key)
        if marker is None:
            return Vector((0.0, 0.0, 0.0))
        return Vector(marker.position)

    def effective_height(self):
        """Figure height, used to size the viewport handles."""
        from ..primary_rig import DEFAULT_HEIGHT

        zs = [m.position[2] for m in self.markers]
        if not zs:
            return DEFAULT_HEIGHT
        height = max(zs) - min(zs)
        return height if height > 1e-3 else DEFAULT_HEIGHT

    def set_markers(self, values, rotations=None, push=True):
        """Write several markers at once without per-property rebuilds."""
        global _syncing_markers
        from ..tree import suspend_live_update

        _syncing_markers = True
        try:
            with suspend_live_update():
                for key, vec in values.items():
                    marker = self.marker_by_key(key)
                    if marker is not None:
                        marker.set_position(vec)
                for key, euler in (rotations or {}).items():
                    marker = self.marker_by_key(key)
                    if marker is not None:
                        marker.set_rotation(euler)
        finally:
            _syncing_markers = False
        if push:
            self.push_markers_to_empties()

    # -- Viewport handles -----------------------------------------------------

    def markers_shown(self):
        from ..primary_rig import find_marker_empties

        return bool(find_marker_empties(self))

    def refresh_handles(self):
        """Bring the viewport handles back in line with the marker list."""
        from ..primary_rig import (
            ensure_marker_empties,
            prune_marker_empties,
            tag_viewports_redraw,
        )
        from ..sync import request_visibility_refresh
        from ..tree import graph_changed

        graph_changed()  # shown or hidden: what the viewport displays changed
        request_visibility_refresh()
        try:
            # Prune unconditionally. markers_shown() asks whether any handle
            # still matches a live marker, so deleting the LAST marker makes
            # it False -- a guard on it would strand that final empty.
            prune_marker_empties(self)
            if self.markers_shown():
                ensure_marker_empties(self)
            tag_viewports_redraw()
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not refresh marker handles: %s", exc)

    def push_markers_to_empties(self):
        from ..primary_rig import find_marker_empties

        for key, obj in find_marker_empties(self).items():
            marker = self.marker_by_key(key)
            if marker is None:
                continue
            loc = Vector(marker.position)
            if (Vector(obj.location) - loc).length > _EPS:
                obj.location = loc
            if marker.use_rotation:
                rot = Vector(marker.rotation)
                if (Vector(obj.rotation_euler) - rot).length > _EPS:
                    obj.rotation_euler = rot

    def sync_from_empties(self):
        """Read dragged handles back in. Returns True if anything moved."""
        from ..primary_rig import find_marker_empties

        empties = find_marker_empties(self)
        if not empties:
            return False
        moved, turned = {}, {}
        for key, obj in empties.items():
            marker = self.marker_by_key(key)
            if marker is None:
                continue
            loc = Vector(obj.location)
            if (Vector(marker.position) - loc).length > _EPS:
                moved[key] = loc
            if marker.use_rotation:
                rot = Vector(obj.rotation_euler)
                if (Vector(marker.rotation) - rot).length > _EPS:
                    turned[key] = tuple(rot)
        if not moved and not turned:
            return False
        changes = self._with_group_followers(moved)
        if getattr(self, "symmetric", False):
            changes, turned = self._mirrored(changes, turned)
        self.set_markers(changes, rotations=turned, push=True)
        tree = self.id_data
        if tree is not None and hasattr(tree, "mark_dirty"):
            tree.mark_dirty()
        return True

    def _with_group_followers(self, changes):
        """Overridden by the Skeleton node, which has rigid landmark groups."""
        return dict(changes)

    def _mirrored(self, changes, rotations=None):
        return dict(changes), dict(rotations or {})

    def free(self):
        from ..primary_rig import remove_marker_empties

        try:
            remove_marker_empties(self)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not clean up marker handles: %s", exc)
        self.schedule_rebuild()
