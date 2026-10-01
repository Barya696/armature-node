"""Marker: the Marker node, one draggable handle in the viewport, and the
Skeleton node, a bundle of them.

A Marker node is live both ways with the bone its wire reaches (see
``MarkerNode``). A Skeleton's markers are a layout to drag onto a character --
MediaPipe's 33 pose landmarks to start with -- and the bones go to them, not
the other way.
"""

from collections import namedtuple

import bpy
from bpy.types import Node
from bpy.props import BoolProperty, FloatProperty, FloatVectorProperty, StringProperty
from mathutils import Vector

from ..core import socket_links
from ..sockets import TransformSocket, marker_attrs
from .base import (
    ArmatureNodeBase,
    redraw_viewports,
    same_vec,
    turn_angle,
    write_socket_value,
)
from .marker_base import MarkerHolderMixin, SkeletonMarker, handle_size_prop


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


def _bone_part(obj, world, attr, compat):
    """One part of a bone's world matrix, as a marker stores it.

    Location and rotation in world space. Scale in the rig's own space --
    the pose scale, 1 at rest -- because that is what a pose writes: taken
    from the world matrix, a rig whose object is scaled to 0.01 would have
    that 0.01 applied to its bones a second time.
    """
    if attr == "position":
        return tuple(world.to_translation())
    if attr == "rotation":
        e = world.decompose()[1].to_euler("XYZ", compat)
        return (e.x, e.y, e.z)
    return tuple((obj.matrix_world.inverted_safe() @ world).to_scale())


def _marker_label(consumer, socket_name, attr):
    """What the Marker node calls a value it feeds into ``socket_name``."""
    label_of = getattr(consumer, "marker_label", None)
    return label_of(socket_name, attr) if label_of is not None else socket_name


def _turn_between(a, b):
    """Angle between two orientations, immune to the q / -q double cover."""
    return turn_angle(a.rotation_difference(b))


def _handle_reading(handle):
    return [*handle.location, *handle.rotation_euler, *handle.scale]


def _remember_handle(handle):
    """Record where the handle was put, so a later difference is a drag."""
    reading = _handle_reading(handle)
    seen = handle.get("an_handle")
    if seen is None or not same_vec(seen, reading):
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
    handle_size: handle_size_prop()
    live_links: StringProperty(
        name="Live Links",
        description="The bone and inputs this marker fed on the last look",
        default="",
        options={"HIDDEN"},
    )
    parent_link: StringProperty(
        name="Parent Link",
        description="The output feeding Parent on the last look",
        default="",
        options={"HIDDEN"},
    )
    uid: StringProperty(
        name="ID",
        description="What this node's lines to other markers are attached to",
        default="",
        options={"HIDDEN"},
    )
    parent_seen: FloatVectorProperty(
        name="Parent Seen",
        description="The parent's world transform on the last look",
        size=16,
        default=(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        options={"HIDDEN"},
    )

    def init(self, context):
        from ..marker_links import new_uid

        self.width = 180
        # Tracked from birth, so its first wire is recognised as new. A node
        # from before tracking has "" and adopts whatever it is wired to.
        self.live_links = _LINKS_TAG + "\n"
        self.uid = new_uid()
        self.inputs.new(TransformSocket.bl_idname, "Parent")
        self.add_marker(name="Marker")

    def copy(self, node):
        """Shift+D, paste, or a copy of the whole tree.

        A copy next to its original needs its own ID, or the original's lines
        would be drawn to it too; the lines among the nodes copied together
        are copied once they all exist. In a tree of its own it keeps its ID,
        which is what the lines copied with that tree name.
        """
        from ..marker_links import new_uid, note_copy

        clash = any(
            n != self and getattr(n, "uid", "") == self.uid for n in self.id_data.nodes
        )
        if self.uid and not clash:
            return
        self.uid = new_uid()
        note_copy(node, self)

    @property
    def marker(self):
        return self.markers[0] if len(self.markers) else None

    def effective_height(self):
        """How big the figure is, for sizing the handle. One marker has no
        spread of its own to measure, so its rig's size, at rest."""
        from ..primary_rig import DEFAULT_HEIGHT, rig_size

        obj = self.rig_for_ui()
        size = rig_size(obj) if obj is not None else 0.0
        return size if size > 1e-3 else DEFAULT_HEIGHT

    # -- Parent and child, like bones ------------------------------------------
    #
    # Wire one marker's output into another's Parent input and the second
    # becomes the child: its values are relative to the parent, and it moves,
    # turns and scales with it, as a child bone does. What the rest of the
    # graph receives from the child is still its world transform.

    def ensure_parent_socket(self):
        """A Marker node saved before parenting existed gets its Parent input."""
        if self.inputs.get("Parent") is None:
            self.inputs.new(TransformSocket.bl_idname, "Parent")

    def _parent_source(self):
        """The output socket feeding Parent -- through reroutes and groups -- or None."""
        from ..sockets import value_source

        sock = self.inputs.get("Parent")
        if sock is None or not sock.is_linked:
            return None
        source = value_source(sock)
        return None if source is None or source == sock else source

    def parent_matrix(self, _seen=None):
        """The parent's world transform, or None when there is no parent.

        A marker parent brings its whole transform -- itself relative to its
        own parent, and so on up. Any other value brings a location. A loop
        of parents is cut where it closes.
        """
        from mathutils import Matrix

        from ..sockets import source_marker

        source = self._parent_source()
        if source is None:
            return None
        seen = set() if _seen is None else _seen
        me = (self.id_data.name, self.name)
        if me in seen:
            return None
        seen.add(me)
        node, marker = source_marker(source)
        if marker is not None:
            return node.marker_matrix(marker, seen)
        value = getattr(source, "default_value", None)
        if value is not None and len(value) == 3:
            return Matrix.Translation(Vector(value))
        return None

    def parent_depth(self):
        depth, node, seen = 0, self, set()
        from ..sockets import source_marker

        while node is not None and hasattr(node, "_parent_source"):
            key = (node.id_data.name, node.name)
            if key in seen:
                break
            seen.add(key)
            source = node._parent_source()
            if source is None:
                break
            depth += 1
            node = source_marker(source)[0]
        return depth

    def _local_matrix(self, marker):
        from mathutils import Euler, Matrix

        return Matrix.LocRotScale(
            Vector(marker.position), Euler(marker.rotation, "XYZ"), Vector(marker.scale)
        )

    def marker_matrix(self, marker, _seen=None):
        """Parent times own, as for a bone. Just its own values without a parent."""
        local = self._local_matrix(marker)
        parent = self.parent_matrix(_seen)
        return local if parent is None else parent @ local

    def apply_world(self, marker, parts):
        """Give the marker these world values, stored relative to the parent."""
        from mathutils import Euler, Matrix

        parent = self.parent_matrix()
        if parent is None:
            return super().apply_world(marker, parts)
        loc, rot, scale = self.marker_matrix(marker).decompose()
        if "position" in parts:
            loc = Vector(parts["position"])
        if "rotation" in parts:
            rot = Euler(parts["rotation"], "XYZ").to_quaternion()
        if "scale" in parts:
            scale = Vector(parts["scale"])
        self._set_local(marker, parent.inverted_safe() @ Matrix.LocRotScale(loc, rot, scale))

    def _set_local(self, marker, matrix):
        from mathutils import Euler

        loc, rot, scale = matrix.decompose()
        e = rot.to_euler("XYZ", Euler(marker.rotation, "XYZ"))
        for attr, value in (("position", tuple(loc)), ("rotation", (e.x, e.y, e.z)), ("scale", tuple(scale))):
            if not same_vec(getattr(marker, attr), value):
                getattr(marker, "set_" + attr)(value)

    def _track_parent(self, _seen=None):
        """Connecting or removing a parent keeps the marker where it is.

        Its values are re-expressed against the new parent -- the parent's
        world transform last seen standing in for a parent that is gone --
        as Blender's Ctrl+P and Alt+P do with Keep Transform. Returns True if
        they changed.

        The parent is settled first, all the way up: nodes are visited in the
        order they were made, and a child measured against a parent that has
        not yet re-expressed itself against *its* new parent would be put in
        the wrong place.
        """
        from mathutils import Matrix

        seen = set() if _seen is None else _seen
        me = (self.id_data.name, self.name)
        if me in seen:
            return False
        seen.add(me)
        source = self._parent_source()
        if source is not None and hasattr(source.node, "_track_parent"):
            source.node._track_parent(seen)
        ident = f"{source.node.id_data.name}\t{source.node.name}\t{source.identifier}" if source else ""
        current = self.parent_matrix()
        changed = False
        if ident != self.parent_link:
            old = None
            if self.parent_link:
                flat = list(self.parent_seen)
                old = Matrix([flat[0:4], flat[4:8], flat[8:12], flat[12:16]])
            for marker in self.markers:
                local = self._local_matrix(marker)
                world = local if old is None else old @ local
                self._set_local(marker, world if current is None else current.inverted_safe() @ world)
            self.parent_link = ident
            changed = True
        if current is not None:
            flat = [v for row in current for v in row]
            if not same_vec(self.parent_seen, flat):
                self.parent_seen = flat
        return changed

    def parent_position(self):
        """Where the line to this marker starts: its parent's world position."""
        parent = self.parent_matrix()
        return None if parent is None else parent.to_translation()

    def free(self):
        from ..marker_links import forget

        # Deleted while wired: the inputs it fed take its values, as they do
        # when it is unplugged, so the bones stay where they are.
        _bone, pairs = _parse_links(self.live_links)
        self._hand_back(pairs, linked_too=True)
        forget(self)  # and its lines to other markers go with it
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
            for link in socket_links(sock) if sock.is_linked else ():
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
        self.ensure_parent_socket()  # and an older node gets its Parent input
        if not self.uid:  # and an ID for its lines to other markers
            from ..marker_links import new_uid

            self.uid = new_uid()
        reparented = self._track_parent()  # before anything reads the values
        self.sync_from_empties()  # a drag not read back yet goes first
        state = self.link_state()
        if state.frozen:
            return reparented  # pose matrices are stale in Edit mode
        changed = self._track_links(state) or reparented
        if state.pbone is not None:
            from .. import livelink

            if livelink.release(self):
                # Held while another marker was moved: the rig moved this
                # bone, so the marker is read off it, as when first wired.
                changed |= self._take_over(state, {f"{c.name}\t{n}" for c, n, *_ in state.wired}, set())
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

        from .. import livelink

        marker = self.marker
        obj, pbone = state.obj, state.pbone
        # Before constraints: that is what gets written back.
        driven = livelink.driven_world(obj, pbone)
        changed = False
        for consumer, name, attr, kind, _driving in state.wired:
            key = f"{consumer.name}\t{name}"
            if key in fresh and kind == "relative":
                field = consumer.inputs.get(name)
                if field is None or not hasattr(field, "default_value"):
                    continue
                value = tuple(field.default_value)
            elif key in fresh or (key in rebound and kind == "absolute"):
                # The bone's live value: wiring the marker in takes it first --
                # less whatever the node adds on top of it (Position's Offset),
                # or the bone would move by that.
                value = _bone_part(obj, driven, attr, Euler(marker.rotation, "XYZ"))
                seed = getattr(consumer, "marker_seed", None)
                if seed is not None:
                    value = seed(name, attr, value)
            else:
                continue
            changed |= self.write_marker(marker, attr, value)
            # Its snapshot predates this: a move it had not folded in yet is
            # in the marker now, and must not be added on top.
            livelink.remember(consumer, obj, pbone)
        return changed

    def _hand_back(self, pairs, linked_too=False):
        """The inputs in ``pairs`` lost this marker: their fields take over.

        Ordinarily the field takes the marker's value. A node whose field
        means something else than the wire did -- the Transform node's world
        transform, against its relative fields -- says so in
        ``marker_unplugged``.
        """
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
            unplugged = getattr(consumer, "marker_unplugged", None)
            handled = unplugged(socket_name, marker, linked_too) if unplugged else None
            if handled is not None:
                changed |= handled
                continue
            # What the input received: the marker's world value.
            value = self.marker_value(marker, marker_attrs(sock)[0])
            changed |= write_socket_value(consumer, socket_name, value, linked_too)
        return changed

    def _read_idle(self, state):
        """Values nothing drives with show the bone, constraints and all --
        without carrying the markers parented to this one: they drive bones
        of their own, which did not move. (Read before the build it starts
        is evaluated, a shoulder's turn lags its chest's by a build; carried,
        the elbow and the hand would swing round it.)"""
        from mathutils import Euler

        from ..sockets import source_marker
        from .marker_base import hold_world

        marker = self.marker
        world = state.obj.matrix_world @ state.pbone.matrix
        changed = False
        for attr in _MARKER_ATTRS:
            if state.modes[attr] != "idle":
                continue
            value = _bone_part(state.obj, world, attr, Euler(marker.rotation, "XYZ"))
            changed |= self.write_marker(marker, attr, value)
        if changed:
            for node in self.id_data.nodes:
                source = node._parent_source() if node is not self and hasattr(node, "_parent_source") else None
                if source is not None and source_marker(source)[0] == self:
                    for child in node.markers:
                        hold_world(node, child)
        return changed

    # -- The handle ------------------------------------------------------------

    def _handle(self):
        from ..primary_rig import find_marker_empties

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
        # The marker's world transform: through its parent, if it has one.
        loc, rot, scale = self.marker_matrix(self.marker).decompose()
        frame = self._frame(state, "position")
        if frame is not None:
            matrix, local = frame
            loc = matrix @ loc if local else matrix.to_translation() + loc
        frame = self._frame(state, "rotation")
        if frame is not None:
            matrix, local = frame
            base = matrix.decompose()[1]
            rot = base @ rot if local else rot @ base
        return loc, rot, scale

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
        from ..primary_rig import set_handle_locks

        handle = self._handle()
        if handle is None or state.frozen:
            return
        move, turn, grow, show_turn = self._flags(state)
        loc, rot, scale = self._handle_target(state)
        if not same_vec(handle.location, loc):
            handle.location = loc
        if show_turn and _turn_between(handle.rotation_euler.to_quaternion(), rot) > 1e-5:
            handle.rotation_euler = rot.to_euler("XYZ", handle.rotation_euler)
        scale = scale if grow else (1.0, 1.0, 1.0)
        if not same_vec(handle.scale, scale):
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
        if same_vec(seen, _handle_reading(handle)):
            return False
        move, turn, grow, _show = self._flags(state)
        loc, rot = self._values_from_handle(
            state, Vector(handle.location), handle.rotation_euler.to_quaternion()
        )
        # The handle is in the world; a child marker keeps its values
        # relative to its parent, so they go in through apply_world.
        parts = {}
        if move and not same_vec(self.marker_value(marker, "position"), loc):
            parts["position"] = tuple(loc)
        if turn:
            e = rot.to_euler("XYZ", Euler(self.marker_value(marker, "rotation"), "XYZ"))
            if not same_vec(self.marker_value(marker, "rotation"), (e.x, e.y, e.z)):
                parts["rotation"] = (e.x, e.y, e.z)
        if grow and not same_vec(self.marker_value(marker, "scale"), handle.scale):
            parts["scale"] = tuple(handle.scale)
        if parts:
            from .. import livelink

            self.apply_world(marker, parts)
            livelink.hold_others(self)  # moved by hand: it alone drives the rig
        _remember_handle(handle)
        if parts:
            self.schedule_rebuild()
        return bool(parts)

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
        # How the handle looks: its glow colour and its size on screen.
        row = layout.row(align=True)
        row.prop(marker, "color", text="")
        row.prop(self, "handle_size", slider=True)

        state = self.link_state()
        live = state.pbone is not None
        move, turn, grow, show_turn = self._flags(state)
        # Each value is labelled by what it feeds -- "Translation" on a
        # Transform node reads as the offset it is, not as a position.
        labels = {}
        for consumer, name, attr, _k, _d in state.wired:
            labels.setdefault(attr, _marker_label(consumer, name, attr))
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

        if self._parent_source() is not None:
            # Like a child bone's Location / Rotation in the N-panel.
            layout.label(text="Relative to its parent", icon="CON_CHILDOF")
        if state.wired:
            if live:
                layout.label(text=f"Live: {state.pbone.name}", icon="LINKED")
            else:
                layout.label(text="Live link needs one bone", icon="UNLINKED")

    def draw_buttons_ext(self, context, layout):
        """The sidebar: what the node shows, and its lines to other markers."""
        from ..marker_links import partners

        self.draw_buttons(context, layout)
        others = partners(self)
        box = layout.box()
        box.label(text=f"Lines ({len(others)})", icon="IPO_LINEAR")
        if not others:
            box.label(text="Drag from the ring under the node")
        for other in others:
            row = box.row(align=True)
            name = other.marker.name if other.marker is not None else ""
            row.label(text=name or other.label or other.name, icon="EMPTY_AXIS")
            op = row.operator("armature_nodes.marker_unjoin", text="", icon="X")
            op.a, op.b = self.uid, other.uid


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
    handle_size: handle_size_prop()
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
        from ..primary_rig import apply_marker_locks, find_marker_empties

        if self.symmetric:
            self.mirror_markers("L_TO_R")
        for key, obj in find_marker_empties(self).items():
            apply_marker_locks(self, obj, key)
        redraw_viewports()

    def load_mediapipe_preset(self, replace=True):
        """Fill the marker list with MediaPipe's 33 pose landmarks.

        A 1.8 m T-pose body, ready to drag onto the character. Markers the
        user added themselves are kept when ``replace`` is False.
        """
        from ..primary_rig import LANDMARKS, LM_LABELS

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
        from ..primary_rig import LM_MIRROR, LM_SIDE, mirror_point, mirror_rotation

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
        from ..primary_rig import RIGID_GROUPS

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
        from ..primary_rig import LM_MIRROR, LM_SIDE, mirror_point, mirror_rotation

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
        from ..primary_rig import LM_SIDE

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
        from ..primary_rig import GROUP_ANCHOR, is_landmark

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
        col.prop(self, "handle_size", slider=True)

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
