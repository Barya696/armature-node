"""All node types for the Armature node tree.

The graph is a modifier stack, the way Geometry Nodes is: **Armature Input**
hands the whole rig to **Armature Output**, and every node in between reads
the bone stream, changes the bones it selects, and passes the rest through
untouched. Decompiling a rig therefore produces two nodes, not one per bone --
the rig is the input, not something the graph has to describe.

Categories:

* **Armature I/O** -- ArmatureInputNode (nothing -> Bone),
  ArmatureOutputNode (Bone -> nothing).
* **Bone** -- BoneNode, ChainNode: create bones.
* **Marker** -- MarkerNode (one draggable handle -> one bone), SkeletonNode
  (a bundle of markers, one position output each).
* **Transform** -- PositionNode, RotationNode, TransformNode, SnapNode: pose
  modifiers. They never touch rest geometry, so they cannot deform a rig they
  are layered onto.
* **Shape** -- CustomShapeNode: assigns a control widget.
* **Constraint** -- IKConstraintNode, GenericConstraintNode.

Every bone-producing node implements eval_bones(ctx) -> list[BoneDef].
Every constraint node implements eval_constraints(ctx) -> list[ConstraintDef].
"""

from collections import namedtuple

import bpy
from bpy.types import Node
from bpy.props import (
    StringProperty,
    FloatProperty,
    IntProperty,
    BoolProperty,
    FloatVectorProperty,
    EnumProperty,
    PointerProperty,
)
from mathutils import Vector

from .core import (
    TREE_IDNAME,
    BoneDef,
    ConstraintDef,
    gather_input_bones,
    gather_input_constraints,
    select_bones,
    copy_bone,
)
from .sockets import (
    RigSocket,
    ConstraintSocket,
    VectorSocket,
    RotationSocket,
    ScaleSocket,
    TransformSocket,
    marker_attrs,
)
from .widgets import PRESET_ITEMS as _widget_preset_items
from .widgets import widget_enum_items as _widget_enum_items

_EPS = 1e-5


class ArmatureNodeBase:
    """Base for all nodes in this tree; restricts them to our tree type."""

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == TREE_IDNAME

    def _multi_input(self, socket_idname, name):
        sock = self.inputs.new(socket_idname, name)
        sock.link_limit = 0  # allow multiple links
        return sock

    def schedule_rebuild(self):
        """Ask the tree to re-evaluate. Safe from restricted contexts."""
        tree = self.id_data
        try:
            if tree is not None and hasattr(tree, "mark_dirty"):
                tree.mark_dirty()
        except (ReferenceError, AttributeError):
            pass

    def free(self):
        """Blender calls this when the node is removed.

        NodeTree.update() is not a reliable deletion signal -- it does not
        fire for nodes removed through the Python API, and a node deleted
        without links changes no topology Blender bothers to report. free() is
        called for every removal, so the rebuild is scheduled from here.
        """
        self.schedule_rebuild()

    def resolve_armature(self):
        """The armature this node's graph works on, for bone-name dropdowns.

        The Armature Input's source first (that is the rig the stack is layered
        on), then the object the tree is bound to. Returns None when the graph
        builds a rig that does not exist yet, in which case the bone field
        falls back to plain text entry.
        """
        tree = self.id_data
        if tree is None:
            return None
        for node in tree.nodes:
            if node.bl_idname == "ArmatureNodesInputNode":
                src = getattr(node, "source", None)
                if src is not None and getattr(src, "type", "") == "ARMATURE":
                    return src
        # No fallback to the Output's armature_name. The target follows the
        # BINDING, never the selection or a typed name: a tree that quietly
        # retargeted itself to whichever armature was active is how a rig
        # ended up modified by a graph that was never pointed at it.
        return None

    def draw_bone_select(self, layout):
        """The Bone field every modifier node carries.

        A searchable dropdown of the rig's real bones when there is a rig to
        search, plain text otherwise. Empty means every bone on the wire.
        """
        obj = self.resolve_armature()
        row = layout.row(align=True)
        if obj is not None:
            row.prop_search(self, "bone", obj.data, "bones", text="", icon="BONE_DATA")
        else:
            row.prop(self, "bone", text="", icon="BONE_DATA")
        if not self.bone:
            layout.label(text="All bones", icon="INFO")


def _bone_select_prop(update=None):
    kwargs = dict(
        name="Bone",
        description=(
            "Bone this node affects. Empty = every bone on the wire; several "
            "can be named at once, semicolon separated"
        ),
        default="",
    )
    if update is not None:
        kwargs["update"] = update
    return StringProperty(**kwargs)


def _same_vec(a, b, eps=1e-6):
    """Equal to within float32 precision.

    Relative, not absolute: node and marker fields are stored as float32, so
    at 100 m a round trip is off by ~1e-5. An absolute 1e-6 would call that a
    change on every tick, and each write tags the tree for another depsgraph
    update -- a loop that never settles.
    """
    return all(
        abs(float(x) - float(y)) <= eps * max(1.0, abs(float(x)), abs(float(y)))
        for x, y in zip(a, b)
    )


def _write_socket_value(node, name, value, linked_too=False):
    """Set a socket's own value without scheduling a rebuild.

    Returns True when it changed. Suspended because this is the rig talking
    to the node, not an edit: rebuilding would write the same value straight
    back, and on a constrained bone that is the start of an oscillation.
    A linked socket is skipped -- its field is not what the node reads --
    unless ``linked_too``, for handing a value over as a wire comes off.
    """
    sock = node.inputs.get(name)
    if sock is None or not hasattr(sock, "default_value"):
        return False
    if sock.is_linked and not linked_too:
        return False
    value = tuple(float(v) for v in value)
    if _same_vec(sock.default_value, value):
        return False
    from .tree import suspend_live_update

    with suspend_live_update():
        sock.default_value = value
    return True


def _linked_marker(node, name):
    """(marker node, marker) wired into ``name``, or (None, None)."""
    sock = node.inputs.get(name)
    if sock is None or not sock.is_linked or not sock.links:
        return None, None
    link = sock.links[0]
    key = getattr(link.from_socket, "marker_key", "")
    source = link.from_node
    if not key or not hasattr(source, "marker_by_key"):
        return None, None
    return source, source.marker_by_key(key)


def _write_input(node, name, value, attr=None):
    """Write an input's value: its own field, or the marker wired into it.

    A wired marker *is* the field, moved: whatever the node would have written
    into its own field goes into the marker instead -- its position, rotation
    or scale, by the socket's type. That is all it takes for every live
    behaviour a node has for its fields to work through a marker too.

    ``attr`` picks the part for a socket that carries several (Transform).
    """
    marker_node, marker = _linked_marker(node, name)
    if marker is not None:
        attr = attr or marker_attrs(node.inputs[name])[0]
        return marker_node.write_marker(marker, attr, value)
    return _write_socket_value(node, name, value)


class _LiveLinkMixin:
    """A node that mirrors the one bone it poses, both ways, live.

    The graph writing a value and the user grabbing the bone both have to end
    up on the node. ``livelink`` tells them apart with a snapshot taken after
    every build: anything that differs from it was the user, and only that
    delta is folded in. Subclasses say which bone (``live_bone``), where a
    delta goes (``absorb``) and which fields are plain readouts (``readout``).

    One live node per bone: two nodes linked to the same bone would each fold
    the same grab in, and the bone would move twice on the next build.
    """

    def live_bone(self):
        return None, None

    def absorb(self, obj, pbone, delta):
        return False

    def readout(self, obj, pbone):
        return False

    def follow_live(self):
        """Pull the bone's current state into the node. True if anything changed."""
        from . import livelink

        obj, pbone = self.live_bone()
        # Edit mode leaves pose matrices stale.
        if pbone is None or obj.mode == "EDIT":
            return False
        delta = livelink.delta_since(self, obj, pbone)
        changed = False
        if delta is not None and delta.moved():
            changed = bool(self.absorb(obj, pbone, delta))
        if self.readout(obj, pbone):
            changed = True
        return changed

    def note_built(self):
        """The graph just wrote the bone: that pose is ours, not the user's."""
        from . import livelink

        obj, pbone = self.live_bone()
        if pbone is not None and obj.mode != "EDIT":
            livelink.remember(self, obj, pbone)

    def free(self):
        from . import livelink

        livelink.forget(self)
        super().free()


class _ModifierNodeBase(ArmatureNodeBase):
    """A node that reads the rig, changes a selection, passes it on.

    One value in, one value out: the whole rig. A node never receives a single
    bone -- it receives the armature as it stands at that point in the graph
    and returns a modified copy, which is what lets nodes be reordered and
    dropped onto the wire like modifiers.
    """

    #: Name of the input socket carrying the incoming rig. The Bone node calls
    #: it Parent -- the node upstream is what its bone hangs off -- but it is
    #: the same whole-rig value every other node receives.
    stream_input = "Rig"

    def init(self, context):
        self.inputs.new(RigSocket.bl_idname, self.stream_input)
        self.outputs.new(RigSocket.bl_idname, "Rig")
        self.width = 200

    def stream(self, ctx):
        return [copy_bone(b) for b in gather_input_bones(self, self.stream_input, ctx)]

    def selected(self, bones):
        return select_bones(bones, self.bone)


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

# True while markers are being written in bulk, so the per-marker callback
# does not rebuild once per component.
_syncing_markers = False


def _slug(text):
    """A key-safe token from a display name."""
    out = "".join(c if c.isalnum() else "_" for c in (text or "").strip().lower())
    return out.strip("_")


def _marker_owner(marker):
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
    from .primary_rig import tag_viewports_redraw

    node = _marker_owner(self)
    if node is not None:
        node.push_markers_to_empties()
    tag_viewports_redraw()
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


def _on_marker_name_changed(self, context):
    """Renaming a marker renames its output socket; links survive because they
    are attached to the socket, not to its name."""
    node = _marker_owner(self)
    if node is not None and hasattr(node, "sync_marker_sockets"):
        node.sync_marker_sockets()


def _on_marker_use_rotation_changed(self, context):
    """Rotation enabled/disabled on one marker: re-lock and redraw its handle,
    then rebuild -- an oriented marker also turns what it drives."""
    from .primary_rig import (
        apply_marker_locks,
        find_marker_empties,
        tag_viewports_redraw,
    )

    node = _marker_owner(self)
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

    def set_scale(self, value):
        self._set_quietly("scale", value)

    def _set_quietly(self, attr, value):
        global _syncing_markers
        was = _syncing_markers
        _syncing_markers = True
        try:
            setattr(self, attr, tuple(value))
        finally:
            _syncing_markers = was


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

    def write_marker(self, marker, attr, value):
        """Set one of a marker's values from code and move its handle.

        This is how a node writes through a wired marker (``_write_input``).
        No rebuild: the value comes from the rig, which is already there.
        """
        if _same_vec(getattr(marker, attr), value):
            return False
        {
            "position": marker.set_position,
            "rotation": marker.set_rotation,
            "scale": marker.set_scale,
        }[attr](value)
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
        from .primary_rig import DEFAULT_HEIGHT

        zs = [m.position[2] for m in self.markers]
        if not zs:
            return DEFAULT_HEIGHT
        height = max(zs) - min(zs)
        return height if height > 1e-3 else DEFAULT_HEIGHT

    def set_markers(self, values, rotations=None, push=True):
        """Write several markers at once without per-property rebuilds."""
        global _syncing_markers
        from .tree import suspend_live_update

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
        from .primary_rig import find_marker_empties

        return bool(find_marker_empties(self))

    def refresh_handles(self):
        """Bring the viewport handles back in line with the marker list."""
        from .primary_rig import (
            ensure_marker_empties,
            prune_marker_empties,
            tag_viewports_redraw,
        )

        try:
            # Prune unconditionally. markers_shown() asks whether any handle
            # still matches a live marker, so deleting the LAST marker makes
            # it False -- a guard on it would strand that final empty.
            prune_marker_empties(self)
            if self.markers_shown():
                ensure_marker_empties(self)
            tag_viewports_redraw()
        except Exception as exc:  # noqa: BLE001
            print(f"[Armature Nodes] Could not refresh marker handles: {exc}")

    def push_markers_to_empties(self):
        from .primary_rig import find_marker_empties

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
        from .primary_rig import find_marker_empties

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
        from .primary_rig import remove_marker_empties

        try:
            remove_marker_empties(self)
        except Exception as exc:  # noqa: BLE001
            print(f"[Armature Nodes] Could not clean up marker handles: {exc}")
        self.schedule_rebuild()


# ---------------------------------------------------------------------------
# Armature I/O
# ---------------------------------------------------------------------------


def _poll_armature_object(self, obj):
    return obj.type == "ARMATURE"


class ArmatureInputNode(ArmatureNodeBase, Node):
    """Head of the stack: the rig as it was before the graph touched it.

    It does **not** read the live armature. The Output writes back to that same
    object, so a live read would feed the graph its own results -- a stack that
    turns Deform off, or replaces a widget, would see the changed value next
    evaluation and could never get back to the original.

    Instead the unmodified rig is stored on the armature object itself (see
    ``store/record.py``) and this node inherits from that. Every evaluation starts
    from the same base state, so deleting a node genuinely undoes it, and
    unplugging this node cannot lose anything -- the record lives on the rig,
    not in the wire.
    """

    bl_idname = "ArmatureNodesInputNode"
    bl_label = "Armature Input"
    bl_icon = "OUTLINER_OB_ARMATURE"

    source: PointerProperty(
        name="Source", type=bpy.types.Object, poll=_poll_armature_object
    )

    def init(self, context):
        self.outputs.new(RigSocket.bl_idname, "Rig")
        self.width = 220

    def draw_buttons(self, context, layout):
        from .store import record as record_store

        layout.context_pointer_set("node", self)
        layout.prop(self, "source", text="")
        obj = self.source
        if obj is None:
            layout.label(text="Pick the rig to modify", icon="INFO")
            return
        base = record_store.read(obj)
        if base is None:
            col = layout.column(align=True)
            col.label(text="Not bound", icon="ERROR")
            col.operator("armature_nodes.bind_rig", icon="FILE_TICK")
            return
        row = layout.row(align=True)
        row.label(text=f"Bound: {len(base.bones)} bones", icon="CHECKMARK")
        row.operator("armature_nodes.capture_record", text="", icon="FILE_REFRESH")

    def eval_bones(self, ctx):
        """The bound rig, straight from its record. Never reads the armature.

        The record is the single source of truth, and the build diffs against
        that same record -- reading the live object here would make the two
        disagree, and would re-capture an already-modified rig as the original.
        """
        from . import bridge
        from .store import record as record_store

        base = record_store.read(self.source)
        if base is None:
            return []  # not bound: the Output reports it, nothing is written
        return bridge.to_bone_defs(base)


class ArmatureOutputNode(ArmatureNodeBase, Node):
    """Tail of the stack: what the graph writes back to.

    Leaving *Armature* empty targets the Armature Input's source, which is the
    normal case -- a stack that modifies a rig in place. Naming something else
    writes to that object instead, creating it if the graph builds a new rig.
    """

    bl_idname = "ArmatureNodesOutputNode"
    bl_label = "Armature Output"
    bl_icon = "ARMATURE_DATA"

    armature_name: StringProperty(name="Armature", default="")
    mode: EnumProperty(
        name="Mode",
        items=(
            (
                "MODIFY",
                "Modify",
                "Write shapes and pose onto the existing rig; bones, "
                "constraints and drivers are left alone",
            ),
            (
                "FULL",
                "Full Rig",
                "Create or rebuild bones, constraints and shapes from the graph",
            ),
        ),
        default="MODIFY",
    )

    show_markers: BoolProperty(
        name="Markers",
        description=(
            "Draw the markers of every Marker and Skeleton node feeding this "
            "output in the 3D viewport"
        ),
        default=True,
        update=lambda self, ctx: _redraw_viewports(),
    )

    def init(self, context):
        self._multi_input(RigSocket.bl_idname, "Rig")
        self.width = 200

    def target_name(self):
        """The object this output writes to.

        In **Modify** mode that is exclusively the Armature Input's source --
        ``armature_name`` is ignored, because modifying a rig the graph was
        never pointed at is never what the user meant. It applies to Full Rig
        only, where the graph builds an armature of its own and has to be able
        to name it.
        """
        if self.mode != "MODIFY" and self.armature_name:
            return self.armature_name
        obj = self.resolve_armature()
        if obj is not None:
            return obj.name
        return self.armature_name or "Armature"

    def draw_buttons(self, context, layout):
        layout.prop(self, "mode", text="")
        # The name field only means something in Full Rig mode; showing it in
        # Modify mode invites the user to type a target that is then ignored.
        if self.mode != "MODIFY":
            layout.prop(self, "armature_name", text="")
        target = self.resolve_armature()
        if target is None and self.mode == "MODIFY":
            layout.label(text="No Armature Input", icon="ERROR")
        else:
            layout.label(text=f"-> {self.target_name()}", icon="ARMATURE_DATA")
        layout.prop(self, "show_markers", toggle=True, icon="EMPTY_AXIS")

    def free(self):
        """Deleting the Output deletes what it generated.

        Only objects tagged as owned by *this* node are removed -- a rig the
        graph merely writes into is never touched.
        """
        from .build import owned_objects
        from .tree import queue_object_removal

        tree = self.id_data
        try:
            for obj in owned_objects(tree.name, self.name):
                queue_object_removal(obj.name)
        except (ReferenceError, AttributeError) as exc:
            print(f"[Armature Nodes] Could not release generated armature: {exc}")
        self.schedule_rebuild()

    def eval_bones(self, ctx):
        return gather_input_bones(self, "Rig", ctx)


# ---------------------------------------------------------------------------
# Bone
# ---------------------------------------------------------------------------


# True while a node is copying values off the rig, so the property callbacks
# do not write the same values straight back.
_syncing_bone_read = False


def _on_bone_selected(self, context):
    """Picking a bone reads its current pose off the rig.

    From then on the live link keeps the node in step (``_LiveLinkMixin``);
    this read is what makes the very first build leave the bone where it is.
    """
    if _syncing_bone_read:
        return
    try:
        self.read_from_rig()
    except Exception as exc:  # noqa: BLE001
        print(f"[Armature Nodes] Could not read bone: {exc}")
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


class BoneNode(_LiveLinkMixin, _ModifierNodeBase, Node):
    """One bone from the rig, posed.

    Pick any bone -- DEF, MCH, ORG or a control, it makes no difference to
    this node -- and it reads that bone's current world transform. Editing the
    values poses it.

    Pose only: rest geometry is never touched, so this cannot change the
    proportions of the rig and cannot fight an Edit-mode change. A component
    whose checkbox is off is left exactly as the rig has it, so a node can move
    a bone without also pinning its rotation.
    """

    bl_idname = "ArmatureNodesBoneNode"
    bl_label = "Bone"
    bl_icon = "BONE_DATA"

    bone: StringProperty(
        name="Bone",
        description="The one bone this node reads and poses",
        default="",
        update=_on_bone_selected,
    )
    bone_rotation: FloatVectorProperty(
        name="Rotation",
        description="World orientation of the bone, as an XYZ Euler",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="EULER",
    )
    bone_scale: FloatVectorProperty(
        name="Scale", size=3, default=(1.0, 1.0, 1.0), subtype="XYZ"
    )
    use_location: BoolProperty(name="Location", default=True)
    use_rotation: BoolProperty(name="Rotation", default=False)
    use_scale: BoolProperty(name="Scale", default=False)
    synced: BoolProperty(default=False, options={"HIDDEN"})

    #: The incoming rig arrives on Parent for this node.
    stream_input = "Parent"

    def init(self, context):
        super().init(context)
        self.inputs.new(VectorSocket.bl_idname, "Position")
        self._multi_input(ConstraintSocket.bl_idname, "Constraints")
        self.width = 220

    def position_socket(self):
        return self.inputs.get("Position")

    def position(self):
        """The world position this node asks for.

        The **Position socket is the location input**, whether or not a wire is
        plugged into it -- unlinked it is the value field (the Geometry Nodes
        convention), linked it follows the marker.

        This used to return None unless something was wired in, so the socket
        was drawn editable, accepted typing, and silently discarded it. The
        node also carried a separate Location field that did work, which meant
        two location inputs that disagreed.
        """
        sock = self.position_socket()
        if sock is None:
            return None
        return sock.get_value()

    def position_is_linked(self):
        sock = self.position_socket()
        return bool(sock is not None and sock.is_linked)

    def set_position(self, value):
        """Write the socket's own value, bypassing the rebuild callback."""
        sock = self.position_socket()
        if sock is None:
            return
        from .tree import suspend_live_update

        with suspend_live_update():
            sock.default_value = tuple(value)

    def linked_marker(self):
        """(node, marker) driving the Position input, or (None, None)."""
        sock = self.inputs.get("Position")
        if sock is None or not sock.is_linked or not sock.links:
            return None, None
        link = sock.links[0]
        key = getattr(link.from_socket, "marker_key", "")
        node = link.from_node
        if not key or not hasattr(node, "marker_by_key"):
            return None, None
        return node, node.marker_by_key(key)

    def read_from_rig(self):
        """Copy the selected bone's current world transform onto this node."""
        global _syncing_bone_read

        obj = self.resolve_armature()
        if obj is None or not self.bone or obj.pose is None:
            return False
        pbone = obj.pose.bones.get(self.bone)
        if pbone is None:
            return False
        loc, rot, scale = (obj.matrix_world @ pbone.matrix).decompose()
        _syncing_bone_read = True
        try:
            self.set_position(loc)
            self.bone_rotation = tuple(rot.to_euler("XYZ"))
            self.bone_scale = tuple(scale)
            self.synced = True
        finally:
            _syncing_bone_read = False
        # Move the marker onto the bone rather than the other way round. A
        # marker wired in sits wherever it was dropped, so without this the
        # bone would jump to the handle the moment it is selected; now the
        # handle lands on the bone and you drag from there.
        marker_node, marker = self.linked_marker()
        if marker is not None:
            marker.set_position(loc)
            marker_node.push_markers_to_empties()
        return True

    def live_bone(self):
        if not self.bone:
            return None, None
        obj = self.resolve_armature()
        if obj is None or obj.pose is None:
            return None, None
        return obj, obj.pose.bones.get(self.bone)

    def on_socket_edited(self, sock):
        if sock.name == "Position" and not self.use_location:
            self.use_location = True

    def marker_role(self, socket_name):
        # Location unticked, a wired marker poses nothing: it follows the bone.
        return ("absolute", self.use_location) if socket_name == "Position" else None

    def absorb(self, obj, pbone, delta):
        """Ticked components are driven: the user's move is folded into them."""
        from . import livelink

        changed = False
        if self.use_location:
            # Into the field, or through the wire into the marker.
            value = Vector(self.position() or (0.0, 0.0, 0.0)) + delta.world_loc
            changed |= _write_input(self, "Position", value)
        updates = []
        if self.use_rotation:
            value = livelink.compose(delta.world_rot, self.bone_rotation)
            if not _same_vec(value, self.bone_rotation):
                updates.append(("bone_rotation", value))
        if self.use_scale:
            _l, _r, scale = delta.now.world.decompose()
            if not _same_vec(scale, self.bone_scale):
                updates.append(("bone_scale", tuple(scale)))
        return self._set_fields(updates) or changed

    def readout(self, obj, pbone):
        """Unticked components are not driven: they just show the bone."""
        from mathutils import Euler

        loc, _rot, scale = (obj.matrix_world @ pbone.matrix).decompose()
        changed = False
        if not self.use_location and not self.position_is_linked():
            changed |= _write_socket_value(self, "Position", loc)
        updates = []
        if not self.use_rotation:
            e = (obj.matrix_world @ pbone.matrix).to_euler(
                "XYZ", Euler(self.bone_rotation, "XYZ")
            )
            if not _same_vec((e.x, e.y, e.z), self.bone_rotation):
                updates.append(("bone_rotation", (e.x, e.y, e.z)))
        if not self.use_scale and not _same_vec(scale, self.bone_scale):
            updates.append(("bone_scale", tuple(scale)))
        return self._set_fields(updates) or changed

    def _set_fields(self, updates):
        global _syncing_bone_read
        from .tree import suspend_live_update

        if not updates:
            return False
        _syncing_bone_read = True
        try:
            with suspend_live_update():
                for name, value in updates:
                    setattr(self, name, value)
                self.synced = True
        finally:
            _syncing_bone_read = False
        return True

    def follow_live_transform(self, obj=None):
        """Kept for callers that predate the live link."""
        return self.follow_live()

    def draw_buttons(self, context, layout):
        layout.context_pointer_set("node", self)
        obj = self.resolve_armature()
        row = layout.row(align=True)
        if obj is not None:
            # Every bone, whatever its role: a DEF or MCH bone is as valid a
            # thing to pose as a control.
            row.prop_search(self, "bone", obj.data, "bones", text="", icon="BONE_DATA")
        else:
            row.prop(self, "bone", text="", icon="BONE_DATA")
        row.operator("armature_nodes.bone_read_from_rig", text="", icon="FILE_REFRESH")

        if not self.bone:
            layout.label(text="Pick a bone to pose", icon="INFO")
            return
        self._draw_blockers(layout, obj)

        # Location lives on the Position socket, not here: one input, editable
        # in place when nothing is wired in, driven by a marker when something
        # is. Two location fields that disagreed is what made the socket look
        # broken.
        row = layout.row(align=True)
        row.prop(self, "use_location", text="")
        if self.position_is_linked():
            row.label(text="Position: from marker", icon="EMPTY_AXIS")
        else:
            row.label(text="Position: see input below", icon="EMPTY_AXIS")

        for flag, prop in (
            ("use_rotation", "bone_rotation"),
            ("use_scale", "bone_scale"),
        ):
            row = layout.row(align=True)
            row.prop(self, flag, text="")
            sub = row.column(align=True)
            # Unticked, the field follows the bone and is read-only: it is a
            # readout, not an input, so showing it editable would invite an
            # edit that the next rebuild throws away.
            sub.enabled = getattr(self, flag)
            sub.prop(self, prop, text="")
        layout.label(text="Unticked values follow the bone", icon="INFO")

    def _draw_blockers(self, layout, obj):
        """Say up front when Blender will refuse the pose.

        A connected bone cannot be translated (its head is pinned to the
        parent's tail) and locked channels are not writable. Blender accepts
        the write and silently drops it, so without this the node looks like
        it simply does not work.
        """
        if obj is None or obj.pose is None or not self.bone:
            return
        pbone = obj.pose.bones.get(self.bone)
        if pbone is None:
            layout.label(text="No such bone on the rig", icon="ERROR")
            return
        notes = []
        if self.use_location and pbone.bone.use_connect:
            notes.append("Connected: cannot be moved")
        if self.use_location:
            locked = [a for a, l in zip("XYZ", pbone.lock_location) if l]
            if locked:
                notes.append("Location " + "".join(locked) + " locked")
        if self.use_rotation and all(pbone.lock_rotation):
            notes.append("Rotation locked")
        if self.use_scale and all(pbone.lock_scale):
            notes.append("Scale locked")
        for note in notes:
            layout.label(text=note, icon="LOCKED")

    def eval_bones(self, ctx):
        bones = self.stream(ctx)
        if not self.bone:
            return bones
        constraints = gather_input_constraints(self, "Constraints", ctx)
        position = self.position()
        for b in bones:
            if b.name != self.bone:
                continue
            if self.use_location and position is not None:
                b.pose_location = tuple(position)
                b.pose_offset = _ZERO
                b.pose_local_offset = _ZERO
            if self.use_rotation:
                b.pose_rotation = tuple(self.bone_rotation)
                b.pose_rotation_offset = _ZERO
                b.pose_local_rotation = _ZERO
            if self.use_scale:
                b.pose_scale = tuple(self.bone_scale)
            # Constraints are pose-stack data, so they only reach the
            # armature when the Output is in Full Rig mode.
            if constraints:
                b.constraints = list(b.constraints) + constraints
            break
        return bones


class ChainNode(ArmatureNodeBase, Node):
    """Procedurally generates N connected bones (spine, finger, tail...)."""

    bl_idname = "ArmatureNodesChainNode"
    bl_label = "Chain"
    bl_icon = "CONSTRAINT_BONE"

    prefix: StringProperty(name="Prefix", default="chain")
    count: IntProperty(name="Count", default=3, min=1, max=256)
    start: FloatVectorProperty(name="Start", size=3, default=(0, 0, 0), subtype="XYZ")
    direction: FloatVectorProperty(
        name="Direction", size=3, default=(0, 0, 1), subtype="XYZ"
    )
    bone_length: FloatProperty(name="Bone Length", default=0.25, min=1e-4)
    curve: FloatProperty(
        name="Curve",
        description="Bend applied to the running direction per segment",
        default=0.0,
        subtype="ANGLE",
    )

    def init(self, context):
        self.inputs.new(RigSocket.bl_idname, "Parent")
        self._multi_input(ConstraintSocket.bl_idname, "Tip Constraints")
        self.outputs.new(RigSocket.bl_idname, "Rig")

    def draw_buttons(self, context, layout):
        layout.prop(self, "prefix", text="")
        layout.prop(self, "count")
        col = layout.column(align=True)
        col.prop(self, "start")
        col.prop(self, "direction")
        layout.prop(self, "bone_length")
        layout.prop(self, "curve")

    def eval_bones(self, ctx):
        from mathutils import Matrix

        parents = gather_input_bones(self, "Parent", ctx)
        tip_constraints = gather_input_constraints(self, "Tip Constraints", ctx)

        direction = Vector(self.direction)
        if direction.length < 1e-8:
            direction = Vector((0.0, 0.0, 1.0))
        direction.normalize()

        bones = []
        head = Vector(self.start)
        prev_name = parents[-1].name if parents else None
        for i in range(self.count):
            if self.curve != 0.0 and i > 0:
                direction = (Matrix.Rotation(self.curve, 4, "X") @ direction).normalized()
            tail = head + direction * self.bone_length
            bone = BoneDef(
                name=f"{self.prefix}.{i + 1:03d}",
                head=tuple(head),
                tail=tuple(tail),
                parent=prev_name,
                use_connect=i > 0,
            )
            bones.append(bone)
            prev_name = bone.name
            head = tail
        if bones and tip_constraints:
            bones[-1].constraints = tip_constraints
        return [copy_bone(b) for b in parents] + bones


# ---------------------------------------------------------------------------
# Marker
# ---------------------------------------------------------------------------


_MARKER_ATTRS = ("position", "rotation", "scale")
_MARKER_LABELS = {"position": "Position", "rotation": "Rotation", "scale": "Scale"}

# The live-link state a Marker node works from; see MarkerNode.link_state.
_MarkerState = namedtuple("_MarkerState", "wired obj pbone modes framed frozen")

# Prefix of MarkerNode.live_links. An empty string means the node was saved
# before links were tracked, which is not the same as "tracked, no links".
_LINKS_TAG = "links1:"


def _parse_links(text):
    """(bone, {"node<TAB>socket", ...}) from MarkerNode.live_links."""
    if not text.startswith(_LINKS_TAG):
        return "", set()
    bone, _nl, rest = text[len(_LINKS_TAG):].partition("\n")
    return bone, {p for p in rest.split("\n") if p}


def _world_part(matrix, attr, compat):
    """One of a world matrix's parts, as a marker stores it."""
    loc, rot, scale = matrix.decompose()
    if attr == "position":
        return tuple(loc)
    if attr == "rotation":
        e = rot.to_euler("XYZ", compat)
        return (e.x, e.y, e.z)
    return tuple(scale)


def _marker_field(consumer, socket_name, attr):
    """The input whose own field ``attr`` of a marker in ``socket_name`` stands
    in for. Usually that socket; for a Transform input, the matching
    Translation / Rotation / Scale field -- a Transform socket has none."""
    field_of = getattr(consumer, "marker_field", None)
    return field_of(socket_name, attr) if field_of is not None else socket_name


def _turn_between(a, b):
    """Angle between two orientations, immune to the q / -q double cover."""
    return 2.0 * _acos_w(a.rotation_difference(b))


def _handle_reading(handle):
    return [*handle.location, *handle.rotation_euler, *handle.scale]


def _remember_handle(handle):
    """Record where the handle was put, so a later difference is a drag."""
    reading = _handle_reading(handle)
    seen = handle.get("an_handle")
    if seen is None or not _same_vec(seen, reading):
        handle["an_handle"] = reading


class MarkerNode(MarkerHolderMixin, ArmatureNodeBase, Node):
    """One draggable handle in the viewport, as a position.

    Wire its *Position* output into a Bone node and dragging the handle moves
    that bone -- any bone, whatever its role. Placing a control by grabbing a
    glowing point in the viewport beats typing world coordinates, which is the
    only reason markers exist.

    It produces no bones and sits outside the Bone stream, the way a value
    node does in Geometry Nodes: it is a position, and what consumes it
    decides what that position means.

    Live, both ways, with the one bone its wire reaches (see ``follow_live``):
    wiring it in never moves the bone, grabbing the bone moves the marker, and
    dragging or typing the marker moves the bone. It works for any input that
    takes it -- Position, Rotation, the Transform node's Translation, Rotation
    and Scale -- because a wired marker is treated as the field it replaces.

    The Skeleton node does not do the wiring part: its landmarks are a layout
    to drag onto a character, and the bones go to them, not the other way.
    """

    bl_idname = "ArmatureNodesMarkerNode"
    bl_label = "Marker"
    bl_icon = "EMPTY_AXIS"

    #: The whole marker -- location, rotation and scale -- on one wire.
    output_socket = TransformSocket

    markers: bpy.props.CollectionProperty(type=SkeletonMarker)
    show_handles: BoolProperty(
        name="Handles",
        description="Show this marker in the 3D viewport",
        default=True,
        update=lambda self, ctx: self.refresh_handles(),
    )
    live_links: StringProperty(
        name="Live Links",
        description="The bone and inputs this marker fed on the last look",
        default="",
        options={"HIDDEN"},
    )

    def init(self, context):
        self.width = 180
        # Tracked from birth, so its first wire is recognised as new. A node
        # from before tracking has "" and adopts whatever it is wired to.
        self.live_links = _LINKS_TAG + "\n"
        self.add_marker(name="Marker")

    @property
    def marker(self):
        return self.markers[0] if len(self.markers) else None

    def free(self):
        # Deleted while wired: the inputs it fed take its values, as they do
        # when it is unplugged, so the bones stay where they are.
        _bone, pairs = _parse_links(self.live_links)
        self._hand_back(pairs, linked_too=True)
        super().free()

    # -- What the wires add up to ----------------------------------------------

    def wired(self):
        """(consumer, socket, attr, kind, driving) for each value it feeds.

        ``attr`` is which of the marker's values an input takes -- position,
        rotation or scale, by the socket's type; a Transform input takes all
        three, so one wire yields three entries. ``kind`` and ``driving`` come
        from the consumer (``marker_role``): "absolute" inputs set the bone's
        world value, "relative" ones offset it from rest.
        """
        out = []
        for sock in self.outputs:
            for link in sock.links:
                consumer, target = link.to_node, link.to_socket
                role_of = getattr(consumer, "marker_role", None)
                role = role_of(target.name) if role_of is not None else None
                if role is not None:
                    for attr in marker_attrs(target):
                        out.append((consumer, target.name, attr, role[0], role[1]))
        return out

    def live_bone(self, wired=None):
        """(armature, pose bone) when every wire reaches the same single bone."""
        obj = pbone = None
        for consumer, *_rest in self.wired() if wired is None else wired:
            o, pb = consumer.live_bone()
            if pb is None or (pbone is not None and pb.name != pbone.name):
                return None, None
            obj, pbone = o, pb
        return obj, pbone

    def link_state(self):
        """Everything the live link needs to know, worked out once."""
        wired = self.wired()
        obj, pbone = self.live_bone(wired)
        live = pbone is not None
        # Per value: None (not wired, no bone), "idle" (a readout of the bone),
        # "absolute" or "relative". Relative wins a tie -- its handle has to
        # be placed through a frame, and an absolute input would not care.
        modes = dict.fromkeys(_MARKER_ATTRS, "idle" if live else None)
        framed = {}
        for consumer, name, attr, kind, driving in wired:
            if kind == "relative":
                modes[attr] = "relative"
                framed.setdefault(attr, (consumer, name))
            elif driving and modes[attr] != "relative":
                modes[attr] = "absolute"
            elif modes[attr] is None:
                modes[attr] = "idle"
        return _MarkerState(
            wired, obj, pbone, modes, framed, frozen=live and obj.mode == "EDIT"
        )

    def _flags(self, state):
        """(move, turn, grow, show turn): what the handle may do and show."""
        live = state.pbone is not None
        driven = {a for a, m in state.modes.items() if m in ("absolute", "relative")}
        use_rotation = bool(self.marker and self.marker.use_rotation)
        move = not (live and state.modes["position"] == "idle")
        turn = "rotation" in driven or (use_rotation and not live)
        return move, turn, "scale" in driven, "rotation" in driven or use_rotation

    def marker_uses_rotation(self, key):
        return self._flags(self.link_state())[3]

    def marker_uses_scale(self, key):
        return self._flags(self.link_state())[2]

    def handle_locks(self, key):
        """(location, rotation, scale) locks for the handle.

        A value that is only a readout of the bone is locked: dragging it
        would be undone by the next readout.
        """
        move, turn, grow, _show = self._flags(self.link_state())
        return (not move,) * 3, (not turn,) * 3, (not grow,) * 3

    # -- Live link ------------------------------------------------------------

    def follow_live(self):
        """Keep the marker in step with the bone its wire reaches.

        * **Newly wired**: an absolute input takes the bone's current world
          value; a relative one takes the value its own field held. Either
          way the bone stays where it is. Picking another bone on the node
          re-takes the absolute values from that bone.
        * **Unplugged**: the input's field takes the marker's value, so the
          bone does not snap back to whatever the field held before.
        * **Driving**: a grab is folded into the marker by the node it drives
          (its ``absorb`` writes through the wire), so nothing is done here.
        * **Not driving**: the value is a readout and sits on the bone.

        Returns True if the marker changed.
        """
        if self.marker is None:
            return False
        self.sync_marker_sockets()  # an older node's Vector output becomes a Transform
        self.sync_from_empties()  # a drag not read back yet goes first
        state = self.link_state()
        if state.frozen:
            return False  # pose matrices are stale in Edit mode
        changed = self._track_links(state)
        if state.pbone is not None:
            changed |= self._read_idle(state)
        self._place_handle(state)
        return changed

    def _track_links(self, state):
        bone = state.pbone.name if state.pbone is not None else ""
        pairs = {f"{c.name}\t{name}" for c, name, *_ in state.wired}
        signature = _LINKS_TAG + bone + "\n" + "\n".join(sorted(pairs))
        previous = self.live_links
        if previous == signature:
            return False
        self.live_links = signature
        if not previous.startswith(_LINKS_TAG):
            return False  # first look at a node from before tracking: adopt
        old_bone, old_pairs = _parse_links(previous)
        changed = self._hand_back(old_pairs - pairs)
        if state.pbone is not None:
            fresh = pairs - old_pairs
            rebound = pairs & old_pairs if old_bone != bone else set()
            changed |= self._take_over(state, fresh, rebound)
        return changed

    def _take_over(self, state, fresh, rebound):
        """Seed the marker from what its new inputs had, so nothing jumps."""
        from mathutils import Euler

        from . import livelink

        marker = self.marker
        obj, pbone = state.obj, state.pbone
        # Before constraints: that is what gets written back.
        driven = livelink.driven_world(obj, pbone)
        changed = False
        for consumer, name, attr, kind, _driving in state.wired:
            key = f"{consumer.name}\t{name}"
            if key in fresh and kind == "relative":
                field = consumer.inputs.get(_marker_field(consumer, name, attr))
                if field is None or not hasattr(field, "default_value"):
                    continue
                value = tuple(field.default_value)
            elif key in fresh or (key in rebound and kind == "absolute"):
                value = _world_part(driven, attr, Euler(marker.rotation, "XYZ"))
            else:
                continue
            changed |= self.write_marker(marker, attr, value)
            # Its snapshot predates this: a move it had not folded in yet is
            # in the marker now, and must not be added on top.
            livelink.remember(consumer, obj, pbone)
        return changed

    def _hand_back(self, pairs, linked_too=False):
        """The inputs in ``pairs`` lost this marker: their fields take its values."""
        marker, tree = self.marker, self.id_data
        if marker is None or tree is None:
            return False
        changed = False
        for pair in pairs:
            node_name, _tab, socket_name = pair.partition("\t")
            consumer = tree.nodes.get(node_name)
            sock = consumer.inputs.get(socket_name) if consumer is not None else None
            if sock is None:
                continue
            for attr in marker_attrs(sock):
                field = _marker_field(consumer, socket_name, attr)
                value = getattr(marker, attr)
                changed |= _write_socket_value(consumer, field, value, linked_too)
        return changed

    def _read_idle(self, state):
        """Values nothing drives with show the bone, constraints and all."""
        from mathutils import Euler

        marker = self.marker
        world = state.obj.matrix_world @ state.pbone.matrix
        changed = False
        for attr in _MARKER_ATTRS:
            if state.modes[attr] != "idle":
                continue
            value = _world_part(world, attr, Euler(marker.rotation, "XYZ"))
            if not _same_vec(getattr(marker, attr), value):
                getattr(marker, "set_" + attr)(value)
                changed = True
        return changed

    # -- The handle ------------------------------------------------------------

    def _handle(self):
        from .primary_rig import find_marker_empties

        marker = self.marker
        return find_marker_empties(self).get(marker.key) if marker is not None else None

    def _frame(self, state, attr):
        """(matrix, local) placing a relative value's handle, or None."""
        if state.pbone is None or state.modes[attr] != "relative" or attr == "scale":
            return None
        consumer, name = state.framed[attr]
        return consumer.marker_frame(name, state.obj, state.pbone)

    def _handle_target(self, state):
        """Where the handle goes: (location, rotation quaternion, scale).

        An absolute value is a place, so the handle is drawn at it. A relative
        one is an offset from rest, and a handle drawn at the raw offset would
        sit near the world origin; instead it goes where the offset puts the
        bone -- rest carried by the value -- and a drag maps back the same way.
        """
        from mathutils import Euler

        marker = self.marker
        loc = Vector(marker.position)
        rot = Euler(marker.rotation, "XYZ").to_quaternion()
        frame = self._frame(state, "position")
        if frame is not None:
            matrix, local = frame
            loc = matrix @ loc if local else matrix.to_translation() + loc
        frame = self._frame(state, "rotation")
        if frame is not None:
            matrix, local = frame
            base = matrix.decompose()[1]
            rot = base @ rot if local else rot @ base
        return loc, rot, Vector(marker.scale)

    def _values_from_handle(self, state, loc, rot):
        """``_handle_target`` backwards: the values a handle pose stands for."""
        frame = self._frame(state, "position")
        if frame is not None:
            matrix, local = frame
            loc = matrix.inverted_safe() @ loc if local else loc - matrix.to_translation()
        frame = self._frame(state, "rotation")
        if frame is not None:
            matrix, local = frame
            base = matrix.decompose()[1].inverted()
            rot = base @ rot if local else rot @ base
        return loc, rot

    def _place_handle(self, state):
        from .primary_rig import set_handle_locks

        handle = self._handle()
        if handle is None or state.frozen:
            return
        move, turn, grow, show_turn = self._flags(state)
        loc, rot, scale = self._handle_target(state)
        if not _same_vec(handle.location, loc):
            handle.location = loc
        if show_turn and _turn_between(handle.rotation_euler.to_quaternion(), rot) > 1e-5:
            handle.rotation_euler = rot.to_euler("XYZ", handle.rotation_euler)
        scale = scale if grow else (1.0, 1.0, 1.0)
        if not _same_vec(handle.scale, scale):
            handle.scale = scale
        set_handle_locks(handle, (not move,) * 3, (not turn,) * 3, (not grow,) * 3, show_turn)
        _remember_handle(handle)

    def push_markers_to_empties(self):
        self._place_handle(self.link_state())

    def sync_from_empties(self):
        """Read a dragged handle back into the marker. True if it moved.

        A handle out of place is not always a drag: a relative value's handle
        sits on rest, which moves with the parent. So it is compared with
        where it was last *put* (``an_handle``), not with where the value
        says it belongs.
        """
        from mathutils import Euler

        handle = self._handle()
        marker = self.marker
        if handle is None or marker is None:
            return False
        state = self.link_state()
        if state.frozen:
            return False
        seen = handle.get("an_handle")
        if seen is None:
            self._place_handle(state)  # made before this was recorded
            return False
        if _same_vec(seen, _handle_reading(handle)):
            return False
        move, turn, grow, _show = self._flags(state)
        loc, rot = self._values_from_handle(
            state, Vector(handle.location), handle.rotation_euler.to_quaternion()
        )
        changed = False
        if move and not _same_vec(marker.position, loc):
            marker.set_position(loc)
            changed = True
        if turn:
            e = rot.to_euler("XYZ", Euler(marker.rotation, "XYZ"))
            if not _same_vec(marker.rotation, (e.x, e.y, e.z)):
                marker.set_rotation((e.x, e.y, e.z))
                changed = True
        if grow and not _same_vec(marker.scale, handle.scale):
            marker.set_scale(tuple(handle.scale))
            changed = True
        _remember_handle(handle)
        if changed:
            self.schedule_rebuild()
        return changed

    # -- UI -------------------------------------------------------------------

    def draw_buttons(self, context, layout):
        layout.context_pointer_set("node", self)
        marker = self.marker
        if marker is None:
            layout.operator(
                "armature_nodes.skeleton_add_marker", text="Add Marker", icon="ADD"
            )
            return
        row = layout.row(align=True)
        row.prop(self, "show_handles", text="", icon="HIDE_OFF" if self.show_handles else "HIDE_ON")
        row.prop(marker, "name", text="")
        op = row.operator(
            "armature_nodes.skeleton_toggle_rotation",
            text="",
            icon="ORIENTATION_GIMBAL",
            depress=marker.use_rotation,
        )
        op.marker = marker.key

        state = self.link_state()
        live = state.pbone is not None
        move, turn, grow, show_turn = self._flags(state)
        # Each value is labelled by the field it stands in for -- "Translation"
        # on a Transform node reads as the offset it is, not as a position.
        labels = {}
        for consumer, name, attr, _k, _d in state.wired:
            labels.setdefault(attr, _marker_field(consumer, name, attr))
        for attr, shown, editable in (
            ("position", True, move),
            ("rotation", live or show_turn, turn),
            ("scale", live or grow, grow),
        ):
            if not shown:
                continue
            col = layout.column(align=True)
            col.label(text=labels.get(attr, _MARKER_LABELS[attr]))
            sub = col.column(align=True)
            # A readout follows the bone; editing it would be undone.
            sub.enabled = editable
            sub.prop(marker, attr, text="")

        if state.wired:
            if live:
                layout.label(text=f"Live: {state.pbone.name}", icon="LINKED")
            else:
                layout.label(text="Live link needs one bone", icon="UNLINKED")


class SkeletonNode(MarkerHolderMixin, ArmatureNodeBase, Node):
    """A bundle of markers, one output each.

    The Principled BSDF of markers: a group of named handles with sensible
    positions, each exposed as its own socket so it can be wired wherever a
    position is wanted -- a Position node, a Snap node, a Custom Shape offset.

    A new node arrives with MediaPipe's 33 pose landmarks loaded, which is a
    usable body to drag onto a character. They are only a preset: rename them,
    delete the ones you do not want, add as many of your own as you need.

    It produces no bones itself. Markers are positions; turning one into rig
    is the job of whatever you wire it into.
    """

    bl_idname = "ArmatureNodesSkeletonNode"
    bl_label = "Skeleton"
    bl_icon = "OUTLINER_OB_ARMATURE"

    markers: bpy.props.CollectionProperty(type=SkeletonMarker)
    lock_depth: BoolProperty(
        name="Lock Depth (2D)",
        description="Handles only move in X/Z (front-view adjustment)",
        default=True,
        update=lambda self, ctx: self._relock(),
    )
    symmetric: BoolProperty(
        name="Symmetric",
        description=(
            "Right-side MediaPipe landmarks are locked and mirror the left "
            "side. Markers you added yourself have no mirror partner"
        ),
        default=True,
        update=lambda self, ctx: self._relock(),
    )
    mirror_center_x: FloatProperty(
        name="Mirror X",
        description="World X of the mirror plane used by Symmetric mode",
        default=0.0,
    )
    show_handles: BoolProperty(
        name="Handles",
        description="Show these markers in the 3D viewport",
        default=True,
        update=lambda self, ctx: self.refresh_handles(),
    )
    show_markers: BoolProperty(name="Markers", default=True)
    show_detail: BoolProperty(name="Face / Hands / Feet", default=False)
    show_advanced: BoolProperty(name="Advanced", default=False)

    def init(self, context):
        self.width = 320
        # The 33 MediaPipe landmarks are the default body: the node is useful
        # immediately, and unwanted markers are deleted rather than 33 added
        # by hand.
        self.load_mediapipe_preset()

    def _relock(self):
        from .primary_rig import apply_marker_locks, find_marker_empties

        if self.symmetric:
            self.mirror_markers("L_TO_R")
        for key, obj in find_marker_empties(self).items():
            apply_marker_locks(self, obj, key)
        _redraw_viewports()

    def load_mediapipe_preset(self, replace=True):
        """Fill the marker list with MediaPipe's 33 pose landmarks.

        A 1.8 m T-pose body, ready to drag onto the character. Markers the
        user added themselves are kept when ``replace`` is False.
        """
        from .primary_rig import LANDMARKS, LM_LABELS

        if replace:
            self.markers.clear()
        have = set(self.marker_keys())
        added = 0
        for _idx, key, _label, _side, default in LANDMARKS:
            if key in have:
                continue
            marker = self.markers.add()
            marker.key = key
            marker.name = LM_LABELS[key]
            marker.set_position(default)
            added += 1
        self.sync_marker_sockets()
        self.refresh_handles()
        return added

    # -- Symmetry and rigid groups (MediaPipe landmarks only) -----------------

    def _mirrored(self, changes, rotations=None):
        from .primary_rig import LM_MIRROR, LM_SIDE, mirror_point, mirror_rotation

        mid = float(self.mirror_center_x)
        out = dict(changes)
        rot_out = dict(rotations or {})
        for key, loc in changes.items():
            if LM_SIDE.get(key) == "L" and LM_MIRROR.get(key):
                out[LM_MIRROR[key]] = mirror_point(loc, mid)
        for key, euler in (rotations or {}).items():
            if LM_SIDE.get(key) == "L" and LM_MIRROR.get(key):
                rot_out[LM_MIRROR[key]] = mirror_rotation(euler)
        return out, rot_out

    def _with_group_followers(self, changes):
        """Face / finger / toe landmarks move rigidly with their anchor."""
        from .primary_rig import RIGID_GROUPS

        have = set(self.marker_keys())
        out = dict(changes)
        for anchor, members in RIGID_GROUPS.items():
            if anchor not in changes or anchor not in have:
                continue
            delta = changes[anchor] - self.marker_position(anchor)
            for m in members:
                if m not in changes and m in have:
                    out[m] = self.marker_position(m) + delta
        return out

    def mirror_markers(self, direction="L_TO_R"):
        from .primary_rig import LM_MIRROR, LM_SIDE, mirror_point, mirror_rotation

        src = "L" if direction == "L_TO_R" else "R"
        mid = float(self.mirror_center_x)
        keys = [
            k for k in self.marker_keys() if LM_SIDE.get(k) == src and LM_MIRROR.get(k)
        ]
        changes = {LM_MIRROR[k]: mirror_point(self.marker_position(k), mid) for k in keys}
        rotations = {
            LM_MIRROR[k]: mirror_rotation(self.marker_rotation(k))
            for k in keys
            if self.marker_uses_rotation(k)
        }
        self.set_markers(changes, rotations=rotations)

    # -- UI -------------------------------------------------------------------

    def _draw_marker_row(self, layout, index, marker):
        from .primary_rig import LM_SIDE

        enabled = not (self.symmetric and LM_SIDE.get(marker.key) == "R")
        row = layout.row(align=True)
        row.enabled = enabled
        row.prop(marker, "name", text="")
        op = row.operator(
            "armature_nodes.skeleton_toggle_rotation",
            text="",
            icon="ORIENTATION_GIMBAL",
            depress=marker.use_rotation,
        )
        op.marker = marker.key
        op = row.operator("armature_nodes.skeleton_remove_marker", text="", icon="X")
        op.index = index
        sub = layout.row(align=True)
        sub.enabled = enabled
        sub.prop(marker, "position", text="")
        if marker.use_rotation:
            sub = layout.row(align=True)
            sub.enabled = enabled
            sub.prop(marker, "rotation", text="")

    def draw_buttons(self, context, layout):
        from .primary_rig import GROUP_ANCHOR, is_landmark

        self.sync_marker_sockets()
        layout.context_pointer_set("node", self)

        col = layout.column(align=True)
        row = col.row(align=True)
        shown = self.markers_shown()
        row.operator(
            "armature_nodes.skeleton_toggle_markers",
            text=("Hide" if shown else "Show") + " Markers",
            icon="HIDE_OFF" if shown else "HIDE_ON",
        )
        row.operator("armature_nodes.skeleton_front_view", icon="VIEW_ORTHO")
        row = col.row(align=True)
        row.prop(self, "lock_depth", toggle=True)
        row.prop(self, "symmetric", toggle=True)
        row.prop(self, "show_handles", toggle=True, icon="HIDE_OFF")

        row = layout.row(align=True)
        row.operator("armature_nodes.skeleton_add_marker", text="Add Marker", icon="ADD")
        row.operator(
            "armature_nodes.skeleton_load_preset", text="MediaPipe", icon="ARMATURE_DATA"
        )
        if not self.markers:
            layout.label(text="No markers -- Add Marker, or load a preset", icon="INFO")
            return

        primary, detail = [], []
        for index, marker in enumerate(self.markers):
            if is_landmark(marker.key) and marker.key in GROUP_ANCHOR:
                detail.append((index, marker))
            else:
                primary.append((index, marker))

        row = layout.row(align=True)
        row.prop(
            self,
            "show_markers",
            icon="TRIA_DOWN" if self.show_markers else "TRIA_RIGHT",
            emboss=False,
        )
        row.label(text=str(len(self.markers)))
        if self.show_markers:
            box = layout.box()
            for index, marker in primary:
                self._draw_marker_row(box, index, marker)
            if detail:
                row = box.row(align=True)
                row.prop(
                    self,
                    "show_detail",
                    icon="TRIA_DOWN" if self.show_detail else "TRIA_RIGHT",
                    emboss=False,
                )
                if self.show_detail:
                    sub = box.box()
                    for index, marker in detail:
                        self._draw_marker_row(sub, index, marker)

        row = layout.row(align=True)
        row.prop(
            self,
            "show_advanced",
            icon="TRIA_DOWN" if self.show_advanced else "TRIA_RIGHT",
            emboss=False,
        )
        if self.show_advanced:
            box = layout.box()
            box.prop(self, "mirror_center_x")
            row = box.row(align=True)
            row.operator(
                "armature_nodes.skeleton_mirror", text="Mirror L > R"
            ).direction = "L_TO_R"
            row.operator(
                "armature_nodes.skeleton_mirror", text="Mirror R > L"
            ).direction = "R_TO_L"
            box.operator(
                "armature_nodes.skeleton_clear_markers",
                text="Clear Markers",
                icon="TRASH",
            )


def _redraw_viewports():
    from .primary_rig import tag_viewports_redraw

    tag_viewports_redraw()


# ---------------------------------------------------------------------------
# Transform (pose modifiers)
# ---------------------------------------------------------------------------


_ZERO = (0.0, 0.0, 0.0)


def _nonzero(value):
    return any(abs(v) > 1e-9 for v in value)


def _not_unit(value):
    return any(abs(v - 1.0) > 1e-9 for v in value)


def _add(a, b):
    return tuple(x + y for x, y in zip(a or _ZERO, b))


def _compose(offset, previous):
    """Rotation ``offset`` applied after ``previous``, both XYZ Euler.

    Composed as quaternions: adding Euler angles is only right for rotations
    about a single axis, and stacks of Rotation / Transform nodes are exactly
    where that breaks.
    """
    from mathutils import Euler

    q = Euler(tuple(offset), "XYZ").to_quaternion() @ Euler(
        tuple(previous or _ZERO), "XYZ"
    ).to_quaternion()
    e = q.to_euler("XYZ")
    return (e.x, e.y, e.z)


def _on_transform_select_changed(self, context):
    """A bone was picked: seed the absolute value from it, then rebuild."""
    self.seed_from_bone()
    self.schedule_rebuild()


# True while code (not the user) flips Set Position / Set Rotation, so the
# callback does not seed over a value that is being deliberately kept.
_seeding_suspended = False


def _on_use_absolute_changed(self, context):
    """Set Position / Set Rotation toggled on: start from where the bone is,
    so ticking the box never makes it jump."""
    if not _seeding_suspended:
        self.seed_from_bone()
    self.schedule_rebuild()


def _set_without_seeding(node, prop, value):
    """Flip a Set checkbox without the callback overwriting the socket.

    ``node[prop] = value`` used to be the way to skip an update callback. On
    Blender 5.x it no longer reaches the RNA property at all -- it creates a
    separate custom property of the same name and leaves the real one alone --
    so the guard has to be explicit.
    """
    global _seeding_suspended
    _seeding_suspended = True
    try:
        setattr(node, prop, value)
    finally:
        _seeding_suspended = False


def _acos_w(q):
    import math

    return math.acos(min(1.0, abs(q.w)))


def _draw_live_state(node, layout):
    """Say whether the node is mirroring a bone, and if not, why."""
    obj, pbone = node.live_bone()
    if pbone is not None:
        layout.label(text=f"Live: {pbone.name}", icon="LINKED")
    elif node.bone and ";" in node.bone:
        layout.label(text="Live link needs a single bone", icon="UNLINKED")


class _TransformNodeBase(_LiveLinkMixin, _ModifierNodeBase):
    """Shared plumbing for the Transform category.

    Each node lists its value sockets in ``_INPUTS``. ``ensure_inputs`` brings
    a node saved by an older version into line -- earlier versions used plain
    Vector sockets for rotation, which showed metres and read "90" as ninety
    radians -- so an existing graph is fixed on sight rather than needing its
    nodes deleted and re-added.
    """

    _INPUTS = ()  # (name, socket class, default or None)

    def init(self, context):
        super().init(context)
        for name, cls, default in self._INPUTS:
            sock = self.inputs.new(cls.bl_idname, name)
            if default is not None:
                sock.default_value = default

    def ensure_inputs(self):
        """Returns the names of sockets that did not exist before."""
        created = set()
        for index, (name, cls, default) in enumerate(self._INPUTS, start=1):
            sock = self.inputs.get(name)
            if sock is not None and sock.bl_idname == cls.bl_idname:
                continue
            carried = None
            if sock is not None:
                # Keep what the user typed: an old Vector-typed Rotation held
                # radians, which is exactly what the new socket stores.
                carried = tuple(getattr(sock, "default_value", ()) or ()) or None
                self.inputs.remove(sock)
            else:
                created.add(name)
            sock = self.inputs.new(cls.bl_idname, name)
            if carried is not None and len(carried) == 3:
                sock.default_value = carried
            elif default is not None:
                sock.default_value = default
            try:
                self.inputs.move(len(self.inputs) - 1, index)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass  # order is cosmetic; the socket works wherever it sits
        return created

    def socket_value(self, name, fallback=_ZERO):
        sock = self.inputs.get(name)
        if sock is None or not hasattr(sock, "get_value"):
            return tuple(fallback)
        return tuple(float(v) for v in sock.get_value())

    def socket_linked(self, name):
        sock = self.inputs.get(name)
        return bool(sock is not None and sock.is_linked)

    def single_bone(self):
        """(armature, pose bone) when exactly one bone is selected."""
        names = [n.strip() for n in self.bone.split(";") if n.strip()]
        if len(names) != 1:
            return None, None
        obj = self.resolve_armature()
        if obj is None or obj.pose is None:
            return None, None
        return obj, obj.pose.bones.get(names[0])

    def write_socket(self, name, value):
        """Set an input's value -- its field, or the marker wired into it --
        without scheduling a rebuild per axis."""
        return _write_input(self, name, value)

    def live_bone(self):
        return self.single_bone()

    def on_socket_edited(self, sock):
        """The user typed into a socket. Overridden where that means more."""

    def seed_from_bone(self):
        """Overridden by the nodes that have an absolute value to seed."""

    @staticmethod
    def top_level(bones, chosen):
        """``chosen`` minus bones whose ancestor is also chosen.

        A relative move is resolved against the rest pose, which follows the
        parent -- so moving a parent and its child by +1 X would carry the
        child +1 with its parent and then move it +1 more. Blender's own
        transform has the same problem and the same answer: with a parent and
        child both selected, only the parent is transformed and the child is
        carried. Each bone ends up moved exactly once.
        """
        names = {b.name for b in chosen}
        parents = {b.name: b.parent for b in bones}
        out = []
        for b in chosen:
            parent, seen = parents.get(b.name), set()
            carried = False
            while parent and parent not in seen:
                if parent in names:
                    carried = True
                    break
                seen.add(parent)
                parent = parents.get(parent)
            if not carried:
                out.append(b)
        return out

    def draw_buttons(self, context, layout):
        self.ensure_inputs()
        self.draw_bone_select(layout)


class PositionNode(_TransformNodeBase, Node):
    """Set Position -- move the selected bones, like the Geometry Nodes node.

    * **Position**: where the bones go, in world space. Used only when *Set
      Position* is ticked or something is wired in; otherwise the bones keep
      their location. That default is what makes a freshly added node
      harmless: it used to take (0, 0, 0) with every bone selected, so
      dropping one on the wire sent the whole rig to the origin.
    * **Offset**: added on top, along world axes. On its own it moves bones
      relative to their rest pose, so rebuilding never adds it twice.

    Setting a position replaces any offset an earlier node applied -- the
    later node wins, as in Geometry Nodes. Pose only; rest geometry is never
    touched.
    """

    bl_idname = "ArmatureNodesPositionNode"
    bl_label = "Position"
    bl_icon = "CON_LOCLIKE"

    bone: _bone_select_prop(update=_on_transform_select_changed)
    use_position: BoolProperty(
        name="Set Position",
        description=(
            "Move the bones to Position. Off, they keep their location and "
            "only Offset applies"
        ),
        default=False,
        update=_on_use_absolute_changed,
    )

    _INPUTS = (("Position", VectorSocket, None), ("Offset", VectorSocket, None))

    def uses_absolute(self):
        return self.use_position or self.socket_linked("Position")

    def ensure_inputs(self):
        created = super().ensure_inputs()
        # A node saved before Offset existed applied its Position whenever it
        # ran. If one was set, tick the box so it keeps doing so instead of
        # quietly becoming a no-op.
        if "Offset" in created and not self.socket_linked("Position"):
            if _nonzero(self.socket_value("Position")):
                # Not a plain assignment: the callback would seed Position
                # from the bone and overwrite the very value being kept.
                _set_without_seeding(self, "use_position", True)
        return created

    def seed_from_bone(self):
        from . import livelink

        if not self.use_position:
            return
        obj, pbone = self.single_bone()
        if pbone is not None:
            self.write_socket("Position", livelink.driven_world(obj, pbone).to_translation())

    def on_socket_edited(self, sock):
        # Typing a position is asking for it: take the bone over, rather than
        # leaving a field that looks like an input but only displays.
        if sock.name == "Position" and not self.use_position:
            _set_without_seeding(self, "use_position", True)

    def marker_role(self, socket_name):
        """How a marker wired into ``socket_name`` is used: (kind, driving).

        "absolute" sets the bone's world value, "relative" offsets it.
        """
        if socket_name == "Position":
            return ("absolute", True)  # a wire into Position always sets it
        if socket_name == "Offset":
            return ("relative", True)
        return None

    def marker_frame(self, socket_name, obj, pbone):
        """What a marker in Offset is measured from: (matrix, local axes)."""
        from mathutils import Matrix

        from . import livelink

        if self.uses_absolute():
            return Matrix.Translation(self.socket_value("Position")), False
        return livelink.rest_world(obj, pbone), False

    def absorb(self, obj, pbone, delta):
        """The user moved the bone: fold the move into whatever drives it.

        Writes go through ``write_socket``, which lands in a wired marker when
        there is one, so a marker follows the grab exactly as the field would.
        """
        if self.uses_absolute():
            value = Vector(self.socket_value("Position")) + delta.world_loc
            return self.write_socket("Position", value)
        if _nonzero(self.socket_value("Offset")) or self.socket_linked("Offset"):
            # Measured from rest: a parent or object move is not an offset.
            value = Vector(self.socket_value("Offset")) + delta.rel_loc
            return self.write_socket("Offset", value)
        return False

    def readout(self, obj, pbone):
        """Not driving: Position simply shows where the bone is."""
        if self.uses_absolute():
            return False
        return self.write_socket("Position", (obj.matrix_world @ pbone.matrix).to_translation())

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "use_position")
        if self.socket_linked("Position"):
            layout.label(text="Position from the wired input", icon="LINKED")
        elif self.uses_absolute() and not self.bone:
            layout.label(text="Every bone goes to one point", icon="ERROR")
        _draw_live_state(self, layout)

    def eval_bones(self, ctx):
        bones = self.stream(ctx)
        absolute = self.uses_absolute()
        position = self.socket_value("Position")
        offset = self.socket_value("Offset")
        has_offset = _nonzero(offset)
        if not absolute and not has_offset:
            return bones  # nothing asked: the node is a pass-through
        chosen = self.selected(bones)
        if absolute:
            for b in chosen:
                b.pose_location = position
                b.pose_offset = _ZERO
                b.pose_local_offset = _ZERO
        if has_offset:
            for b in self.top_level(bones, chosen):
                b.pose_offset = _add(b.pose_offset, offset)
        return bones


class RotationNode(_TransformNodeBase, Node):
    """Set Rotation -- orient the selected bones, in degrees.

    * **Rotation**: the world orientation to give the bones. Used only when
      *Set Rotation* is ticked or something is wired in (a wired marker
      supplies its own rotation). Off, the bones keep their orientation.
    * **Offset**: turned on top, along world axes, about each bone's own head.

    Setting a rotation replaces any rotation offset an earlier node applied.
    """

    bl_idname = "ArmatureNodesRotationNode"
    bl_label = "Rotation"
    bl_icon = "CON_ROTLIKE"

    bone: _bone_select_prop(update=_on_transform_select_changed)
    use_rotation: BoolProperty(
        name="Set Rotation",
        description=(
            "Give the bones this orientation. Off, they keep theirs and only "
            "Offset applies"
        ),
        default=False,
        update=_on_use_absolute_changed,
    )

    _INPUTS = (("Rotation", RotationSocket, None), ("Offset", RotationSocket, None))

    def uses_absolute(self):
        return self.use_rotation or self.socket_linked("Rotation")

    def ensure_inputs(self):
        created = super().ensure_inputs()
        if "Offset" in created and not self.socket_linked("Rotation"):
            if _nonzero(self.socket_value("Rotation")):
                _set_without_seeding(self, "use_rotation", True)
        return created

    def seed_from_bone(self):
        from . import livelink

        if not self.use_rotation:
            return
        obj, pbone = self.single_bone()
        if pbone is not None:
            e = livelink.driven_world(obj, pbone).to_euler("XYZ")
            self.write_socket("Rotation", (e.x, e.y, e.z))

    def on_socket_edited(self, sock):
        if sock.name == "Rotation" and not self.use_rotation:
            _set_without_seeding(self, "use_rotation", True)

    def marker_role(self, socket_name):
        if socket_name == "Rotation":
            return ("absolute", True)
        if socket_name == "Offset":
            return ("relative", True)
        return None

    def marker_frame(self, socket_name, obj, pbone):
        """Offset turns on top of Rotation when that is set, else of rest."""
        from mathutils import Euler

        from . import livelink

        if self.uses_absolute():
            return Euler(self.socket_value("Rotation"), "XYZ").to_matrix().to_4x4(), False
        return livelink.rest_world(obj, pbone), False

    def absorb(self, obj, pbone, delta):
        from mathutils import Euler

        from . import livelink

        if self.uses_absolute():
            # The node's own Offset sits on top of Rotation, so the user's turn
            # is taken into Rotation's frame before it is composed in.
            own = Euler(self.socket_value("Offset"), "XYZ").to_quaternion()
            spin = own.inverted() @ delta.world_rot @ own
            value = livelink.compose(spin, self.socket_value("Rotation"))
            return self.write_socket("Rotation", value)
        if _nonzero(self.socket_value("Offset")) or self.socket_linked("Offset"):
            value = livelink.compose(delta.rel_rot, self.socket_value("Offset"))
            return self.write_socket("Offset", value)
        return False

    def readout(self, obj, pbone):
        from mathutils import Euler

        if self.uses_absolute():
            return False
        current = Euler(self.socket_value("Rotation"), "XYZ")
        e = (obj.matrix_world @ pbone.matrix).to_euler("XYZ", current)
        return self.write_socket("Rotation", (e.x, e.y, e.z))

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "use_rotation")
        if self.socket_linked("Rotation"):
            layout.label(text="Rotation from the wired input", icon="LINKED")
        _draw_live_state(self, layout)

    def eval_bones(self, ctx):
        bones = self.stream(ctx)
        absolute = self.uses_absolute()
        rotation = self.socket_value("Rotation")
        offset = self.socket_value("Offset")
        has_offset = _nonzero(offset)
        if not absolute and not has_offset:
            return bones
        chosen = self.selected(bones)
        if absolute:
            for b in chosen:
                b.pose_rotation = rotation
                b.pose_rotation_offset = _ZERO
                b.pose_local_rotation = _ZERO
        if has_offset:
            for b in self.top_level(bones, chosen):
                b.pose_rotation_offset = _compose(offset, b.pose_rotation_offset)
        return bones


class TransformNode(_TransformNodeBase, Node):
    """Transform -- move, turn and scale the selected bones relative to rest.

    Like Geometry Nodes' Transform Geometry: nothing is absolute, so several
    Transform nodes stack, and each build resolves against the rest pose
    rather than the live one -- rebuilding never applies the move twice.

    **Space** decides the axes:

    * **World**: along the scene axes, whatever way the bone points.
    * **Local**: along the bone's own axes -- these are its Location and
      Rotation channels, the values in the N-panel, and they follow the
      parent the way hand-posing does.

    Rotation is about each bone's own head. Scale multiplies what earlier
    nodes set, and 1 means unchanged.

    The **Transform** input takes all three on one wire -- from a Marker, the
    handle's location, rotation and scale. While it is wired it replaces the
    three fields, which are hidden; unplugged, they take its last values.
    """

    bl_idname = "ArmatureNodesTransformNode"
    bl_label = "Transform"
    bl_icon = "ORIENTATION_GLOBAL"

    bone: _bone_select_prop()
    space: EnumProperty(
        name="Space",
        items=(
            ("WORLD", "World", "Along the scene axes"),
            ("LOCAL", "Local", "Along the bone's own axes: its Location and Rotation channels"),
        ),
        default="WORLD",
    )

    _INPUTS = (
        ("Transform", TransformSocket, None),
        ("Translation", VectorSocket, None),
        ("Rotation", RotationSocket, None),
        ("Scale", ScaleSocket, (1.0, 1.0, 1.0)),
    )

    #: Which field each part of the Transform input stands in for.
    _PARTS = (("position", "Translation"), ("rotation", "Rotation"), ("scale", "Scale"))

    def uses_transform(self):
        return self.socket_linked("Transform")

    def parts(self):
        """(translation, rotation, scale): from the Transform wire, or the fields."""
        sock = self.inputs.get("Transform")
        wired = sock.get_transform() if sock is not None else None
        if wired is not None:
            return wired
        return (
            self.socket_value("Translation"),
            self.socket_value("Rotation"),
            self.socket_value("Scale", (1.0, 1.0, 1.0)),
        )

    def write_part(self, field, value):
        """Write one part where it lives: the Transform wire, or its field."""
        if self.uses_transform():
            attr = next(a for a, f in self._PARTS if f == field)
            return _write_input(self, "Transform", value, attr)
        return self.write_socket(field, value)

    def update(self):
        """Links changed: show the three fields only while they are in use."""
        self.show_fields()

    def show_fields(self):
        hide = self.uses_transform()
        for _attr, field in self._PARTS:
            sock = self.inputs.get(field)
            if sock is not None and sock.hide != hide and not sock.is_linked:
                sock.hide = hide

    def marker_role(self, socket_name):
        # Every input here is relative to rest, and so is a marker in one.
        if socket_name in ("Transform", "Translation", "Rotation", "Scale"):
            return ("relative", True)
        return None

    def marker_field(self, socket_name, attr):
        if socket_name == "Transform":
            return dict(self._PARTS)[attr]
        return socket_name

    def marker_frame(self, socket_name, obj, pbone):
        """Rest, along world axes or the bone's own, as Space says."""
        from . import livelink

        return livelink.rest_world(obj, pbone), self.space == "LOCAL"

    def follow_live(self):
        self.show_fields()  # backstop: update() is not called for every edit
        return super().follow_live()

    def absorb(self, obj, pbone, delta):
        """Fold the user's move in, per component.

        A component the node already drives takes the delta, which is safe on
        a constrained bone because the delta never contains the node's own
        write. A component it does not drive yet takes the bone's whole
        current offset from rest instead -- otherwise a bone that was already
        hand-posed would jump back by that amount the moment the node started
        driving it.
        """
        from mathutils import Euler

        from . import livelink

        local = self.space == "LOCAL"
        now = delta.now
        t, r, s = self.parts()
        changed = False

        if local:
            loc_step, rot_step = delta.local_loc, delta.local_rot
            basis_loc, basis_rot, _ = now.basis.decompose()
            loc_whole = basis_loc
            rot_whole = basis_rot
        else:
            loc_step, rot_step = delta.rel_loc, delta.rel_rot
            wl, wr, _ = now.world.decompose()
            rl, rr, _ = now.rest.decompose()
            loc_whole = wl - rl
            rot_whole = wr @ rr.inverted()

        if loc_step.length > 1e-5:
            value = Vector(t) + loc_step if _nonzero(t) else loc_whole
            changed |= self.write_part("Translation", value)
        if 2.0 * _acos_w(rot_step) > 1e-5:
            if _nonzero(r):
                value = livelink.compose(rot_step, r)
            else:
                e = rot_whole.to_euler("XYZ", Euler(r, "XYZ"))
                value = (e.x, e.y, e.z)
            changed |= self.write_part("Rotation", value)
        if (delta.scale_ratio - Vector((1.0, 1.0, 1.0))).length > 1e-5:
            value = tuple(a * b for a, b in zip(s, delta.scale_ratio))
            changed |= self.write_part("Scale", value)
        return changed

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "space", expand=True)
        if self.uses_transform():
            layout.label(text="Translation, Rotation, Scale from the wire", icon="LINKED")
        _draw_live_state(self, layout)

    def eval_bones(self, ctx):
        bones = self.stream(ctx)
        translation, rotation, scale = self.parts()
        move, turn, grow = _nonzero(translation), _nonzero(rotation), _not_unit(scale)
        if not (move or turn or grow):
            return bones
        local = self.space == "LOCAL"
        chosen = self.selected(bones)
        # Local is channel semantics -- every bone's own Location / Rotation,
        # which do accumulate down a chain, exactly as typing the same value
        # into each bone's N-panel would. World moves each bone once.
        targets = chosen if local else self.top_level(bones, chosen)
        for b in targets:
            if move:
                if local:
                    b.pose_local_offset = _add(b.pose_local_offset, translation)
                else:
                    b.pose_offset = _add(b.pose_offset, translation)
            if turn:
                if local:
                    b.pose_local_rotation = _compose(rotation, b.pose_local_rotation)
                else:
                    b.pose_rotation_offset = _compose(rotation, b.pose_rotation_offset)
            if grow:
                base = b.pose_scale or (1.0, 1.0, 1.0)
                b.pose_scale = tuple(x * y for x, y in zip(base, scale))
        return bones


def _mesh_anchor(obj, mode, reference=None):
    """A world-space point on ``obj`` for the Snap node.

    ORIGIN   the object's own origin
    BOUNDS   centre of its bounding box
    MEDIAN   mean of its vertices
    VOLUME   volume centroid (signed tetrahedra over the triangulated mesh),
             which is the centre of mass of a solid, unlike the median
    SURFACE  closest point on the surface to ``reference``
    """
    from mathutils import Vector as V

    mw = obj.matrix_world
    if mode == "ORIGIN":
        return mw.translation.copy()
    if mode == "BOUNDS":
        corners = [V(c) for c in obj.bound_box]
        return mw @ (sum(corners, V((0, 0, 0))) / len(corners))
    mesh = getattr(obj, "data", None)
    verts = getattr(mesh, "vertices", None)
    if not verts:
        return mw.translation.copy()
    if mode == "MEDIAN":
        total = V((0.0, 0.0, 0.0))
        for v in verts:
            total += v.co
        return mw @ (total / len(verts))
    if mode == "SURFACE":
        point = V(reference) if reference is not None else mw.translation
        local = mw.inverted_safe() @ point
        try:
            hit, location, _normal, _index = obj.closest_point_on_mesh(local)
        except (RuntimeError, ValueError):
            hit = False
        if not hit:
            return mw.translation.copy()
        return mw @ location
    # VOLUME: sum signed tetrahedra (origin, a, b, c) over triangulated faces.
    mesh.calc_loop_triangles()
    total_volume = 0.0
    centroid = V((0.0, 0.0, 0.0))
    for tri in mesh.loop_triangles:
        a, b, c = (verts[i].co for i in tri.vertices)
        volume = a.cross(b).dot(c) / 6.0
        total_volume += volume
        centroid += (a + b + c) * (volume / 4.0)
    if abs(total_volume) < 1e-12:
        # Flat or open mesh: the signed volume cancels out, so fall back to
        # the median rather than dividing by ~zero.
        return _mesh_anchor(obj, "MEDIAN", reference)
    return mw @ (centroid / total_volume)


class SnapNode(_ModifierNodeBase, Node):
    """Snap the selected bones onto a mesh.

    Positions a control against real geometry instead of by eye: the object's
    origin, its bounding-box centre, the median of its vertices, its volume
    centroid, or the closest point on its surface. Like the other Transform
    nodes it writes the pose only.
    """

    bl_idname = "ArmatureNodesSnapNode"
    bl_label = "Snap"
    bl_icon = "SNAP_ON"

    bone: _bone_select_prop()
    target: PointerProperty(
        name="Target",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "MESH",
    )
    mode: EnumProperty(
        name="Snap To",
        items=(
            ("ORIGIN", "Origin", "The target object's own origin"),
            ("BOUNDS", "Bounding Box", "Centre of the target's bounding box"),
            ("MEDIAN", "Median", "Mean of the target's vertices"),
            ("VOLUME", "Volume", "Volume centroid: the centre of mass of a solid"),
            ("SURFACE", "Surface", "Closest point on the target's surface"),
        ),
        default="VOLUME",
    )
    offset: FloatVectorProperty(
        name="Offset", size=3, default=(0, 0, 0), subtype="TRANSLATION"
    )

    def draw_buttons(self, context, layout):
        self.draw_bone_select(layout)
        layout.prop(self, "target", text="")
        layout.prop(self, "mode", text="")
        layout.prop(self, "offset")

    def eval_bones(self, ctx):
        bones = self.stream(ctx)
        obj = self.target
        if obj is None or obj.type != "MESH":
            return bones
        offset = Vector(self.offset)
        for b in self.selected(bones):
            # SURFACE needs a point to be closest TO; the bone's own head is
            # the only sensible reference at evaluation time.
            reference = Vector(b.pose_location) if b.pose_location else Vector(b.head)
            try:
                anchor = _mesh_anchor(obj, self.mode, reference)
            except Exception as exc:  # noqa: BLE001
                print(f"[Armature Nodes] Snap failed on '{b.name}': {exc}")
                continue
            b.pose_location = tuple(Vector(anchor) + offset)
            b.pose_offset = _ZERO
            b.pose_local_offset = _ZERO
        return bones


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def _poll_widget_object(self, obj):
    return obj.type == "MESH"


class CustomShapeNode(_ModifierNodeBase, Node):
    """Assign a control widget to the selected bones.

    Stripped to what a modifier needs: which bone, which widget, and how the
    widget sits on it. The bone's own data comes down the wire from the
    Armature Input, so this node no longer stores a copy of it.

    The *Offset* input is a vector like Geometry Nodes' Set Position: wire a
    marker into it and the widget moves with the handle.
    """

    bl_idname = "ArmatureNodesCustomShapeNode"
    bl_label = "Custom Shape"
    bl_icon = "MESH_CIRCLE"

    bone: _bone_select_prop()
    source: EnumProperty(
        name="Source",
        items=(
            ("PRESET", "Preset", "Generate a WGT-rig_<bone> widget from a preset"),
            ("LIBRARY", "WGTS_rig", "Reuse an existing widget from WGTS_rig"),
            ("OBJECT", "Object", "Use any mesh object as the widget"),
        ),
        default="PRESET",
    )
    preset: EnumProperty(name="Preset", items=_widget_preset_items, default=1)
    library_widget: EnumProperty(name="Widget", items=_widget_enum_items)
    widget_object: PointerProperty(
        name="Widget", type=bpy.types.Object, poll=_poll_widget_object
    )
    widget_scale: FloatVectorProperty(
        name="Scale", size=3, default=(1.0, 1.0, 1.0), subtype="XYZ"
    )
    widget_rotation: FloatVectorProperty(
        name="Rotation", size=3, default=(0.0, 0.0, 0.0), subtype="EULER"
    )
    wire_width: FloatProperty(name="Wire Width", default=1.0, min=1.0, max=16.0)
    scale_to_bone_length: BoolProperty(name="Scale to Bone Length", default=True)
    show_wire: BoolProperty(name="Wireframe", default=True)
    control_only: BoolProperty(
        name="Control Only",
        description="Also turn off Deform on these bones (they are controls)",
        default=True,
    )

    def init(self, context):
        super().init(context)
        self.inputs.new(VectorSocket.bl_idname, "Offset")
        self.width = 220

    def draw_buttons(self, context, layout):
        self.draw_bone_select(layout)
        layout.prop(self, "source", text="")
        if self.source == "PRESET":
            layout.prop(self, "preset", text="")
        elif self.source == "LIBRARY":
            layout.prop(self, "library_widget", text="")
            layout.prop(self, "preset", text="Fallback")
        else:
            layout.prop(self, "widget_object", text="")
        col = layout.column(align=True)
        col.prop(self, "widget_scale", text="Scale")
        col.prop(self, "widget_rotation")
        layout.prop(self, "scale_to_bone_length")
        row = layout.row(align=True)
        row.prop(self, "show_wire")
        row.prop(self, "wire_width", text="Width")
        layout.prop(self, "control_only")

    def _widget_name(self):
        if self.source == "LIBRARY":
            return self.library_widget or ""
        if self.source == "OBJECT" and self.widget_object:
            return self.widget_object.name
        return ""  # PRESET: a per-bone WGT-rig_<bone> is generated at build

    def eval_bones(self, ctx):
        from .core import ShapeDef

        bones = self.stream(ctx)
        sock = self.inputs.get("Offset")
        offset = sock.get_value() if sock is not None else (0.0, 0.0, 0.0)
        widget = self._widget_name()
        preset = self.preset if self.source != "OBJECT" else "NONE"
        for b in self.selected(bones):
            b.shape = ShapeDef(
                widget=widget,
                preset=preset,
                scale=tuple(self.widget_scale),
                translation=tuple(offset),
                rotation=tuple(self.widget_rotation),
                wire_width=self.wire_width,
                scale_to_bone_length=self.scale_to_bone_length,
                show_wire=self.show_wire,
            )
            if self.control_only:
                b.use_deform = False
        return bones


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class IKConstraintNode(ArmatureNodeBase, Node):
    """IK constraint: target, pole target, chain length, iterations, stretch."""

    bl_idname = "ArmatureNodesIKConstraintNode"
    bl_label = "IK Constraint"
    bl_icon = "CON_KINEMATIC"

    target: PointerProperty(name="Target", type=bpy.types.Object)
    subtarget: StringProperty(name="Target Bone", default="")
    pole_target: PointerProperty(name="Pole Target", type=bpy.types.Object)
    pole_angle: FloatProperty(name="Pole Angle", default=0.0, subtype="ANGLE")
    chain_count: IntProperty(name="Chain Length", default=2, min=0)
    iterations: IntProperty(name="Iterations", default=500, min=1, max=10000)
    use_stretch: BoolProperty(name="Stretch", default=False)
    influence: FloatProperty(name="Influence", default=1.0, min=0.0, max=1.0)

    def init(self, context):
        self.outputs.new(ConstraintSocket.bl_idname, "Constraint")

    def draw_buttons(self, context, layout):
        layout.prop(self, "target")
        if self.target and self.target.type == "ARMATURE":
            layout.prop_search(self, "subtarget", self.target.data, "bones")
        layout.prop(self, "pole_target")
        if self.pole_target:
            layout.prop(self, "pole_angle")
        layout.prop(self, "chain_count")
        layout.prop(self, "iterations")
        layout.prop(self, "use_stretch")
        layout.prop(self, "influence")

    def eval_constraints(self, ctx):
        params = {
            "chain_count": self.chain_count,
            "iterations": self.iterations,
            "use_stretch": self.use_stretch,
            "influence": self.influence,
        }
        if self.target:
            params["target"] = self.target.name
            if self.subtarget:
                params["subtarget"] = self.subtarget
        if self.pole_target:
            params["pole_target"] = self.pole_target.name
            params["pole_angle"] = self.pole_angle
        return [ConstraintDef(type="IK", name="IK", params=params)]


_GENERIC_CONSTRAINT_ITEMS = (
    ("COPY_ROTATION", "Copy Rotation", ""),
    ("COPY_LOCATION", "Copy Location", ""),
    ("COPY_SCALE", "Copy Scale", ""),
    ("COPY_TRANSFORMS", "Copy Transforms", ""),
    ("LIMIT_ROTATION", "Limit Rotation", ""),
    ("LIMIT_LOCATION", "Limit Location", ""),
    ("TRACK_TO", "Track To", ""),
    ("DAMPED_TRACK", "Damped Track", ""),
    ("STRETCH_TO", "Stretch To", ""),
)

_TARGETLESS_TYPES = {"LIMIT_ROTATION", "LIMIT_LOCATION"}


class GenericConstraintNode(ArmatureNodeBase, Node):
    """One node covering the common target-based / limit constraints."""

    bl_idname = "ArmatureNodesGenericConstraintNode"
    bl_label = "Constraint"
    bl_icon = "CONSTRAINT"

    constraint_type: EnumProperty(
        name="Type", items=_GENERIC_CONSTRAINT_ITEMS, default="COPY_ROTATION"
    )
    target: PointerProperty(name="Target", type=bpy.types.Object)
    subtarget: StringProperty(name="Target Bone", default="")
    influence: FloatProperty(name="Influence", default=1.0, min=0.0, max=1.0)

    def init(self, context):
        self.outputs.new(ConstraintSocket.bl_idname, "Constraint")

    def draw_buttons(self, context, layout):
        layout.prop(self, "constraint_type", text="")
        if self.constraint_type not in _TARGETLESS_TYPES:
            layout.prop(self, "target")
            if self.target and self.target.type == "ARMATURE":
                layout.prop_search(self, "subtarget", self.target.data, "bones")
        layout.prop(self, "influence")

    def eval_constraints(self, ctx):
        params = {"influence": self.influence}
        if self.constraint_type not in _TARGETLESS_TYPES and self.target:
            params["target"] = self.target.name
            if self.subtarget:
                params["subtarget"] = self.subtarget
        return [
            ConstraintDef(
                type=self.constraint_type,
                name=self.constraint_type.replace("_", " ").title(),
                params=params,
            )
        ]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    SkeletonMarker,
    ArmatureInputNode,
    ArmatureOutputNode,
    BoneNode,
    ChainNode,
    MarkerNode,
    SkeletonNode,
    PositionNode,
    RotationNode,
    TransformNode,
    SnapNode,
    CustomShapeNode,
    IKConstraintNode,
    GenericConstraintNode,
)


def _on_node_prop_changed(self, context):
    """Any edited node value re-applies the tree to its armature (real time)."""
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


# UI-only toggles that must not trigger a rebuild.
_NO_REBUILD_PROPS = {
    "synced",
    "show_markers",
    "show_detail",
    "show_advanced",
    "show_handles",
    "show_markers",
    "markers",  # CollectionProperty: does not accept update=
    "lock_depth",
    "symmetric",
    "live_links",  # bookkeeping for the marker's live link
}


def _inject_live_update(cls):
    """Add an ``update`` callback to every property of a node class.

    Blender only calls NodeTree.update() for link/node changes, not for value
    edits, so without this a scale tweak on a Custom Shape node would do
    nothing until the next structural change. CollectionProperty is skipped
    unconditionally -- Blender's RNA does not accept ``update`` for it at all
    (registration fails outright), and per-item updates on the PropertyGroup
    itself are the correct way to react to collection edits anyway.
    """
    for name, prop in list(cls.__annotations__.items()):
        keywords = getattr(prop, "keywords", None)
        if keywords is None or name in _NO_REBUILD_PROPS:
            continue
        function = getattr(prop, "function", None)
        if function is not None and getattr(function, "__name__", "") == "CollectionProperty":
            continue
        if "update" not in keywords:
            keywords["update"] = _on_node_prop_changed


def _check_reserved_names(cls):
    """Warn when a node property shadows one of Blender's own Node members.

    Registration succeeds either way, which is what makes this worth checking:
    a node that declares ``location`` overrides the node's position in the
    editor, so it registers cleanly and then Blender's own add-node operator
    dies setting ``node.location``, and the node can never be moved. Nothing
    else catches it -- the failure surfaces as a ValueError from Blender's
    code, with no hint that an addon property is the cause.
    """
    node_rna = getattr(Node, "bl_rna", None)
    if node_rna is None:  # not running inside Blender
        return
    reserved = set(node_rna.properties.keys()) - {"bl_idname"}
    clashes = sorted(set(getattr(cls, "__annotations__", {})) & reserved)
    if clashes:
        print(
            f"[Armature Nodes] {cls.__name__} declares {', '.join(clashes)}, "
            f"which shadow bpy.types.Node properties. Rename them "
            f"(e.g. 'location' -> 'bone_location') or the node will misbehave."
        )


def register():
    for cls in classes:
        if issubclass(cls, Node):
            _check_reserved_names(cls)
            _inject_live_update(cls)
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
