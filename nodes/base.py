"""What every node is built on.

``ArmatureNodeBase`` is any node in the tree. ``ModifierNodeBase`` is one that
takes the rig in, changes the bones it selects and passes the rig on.
``LiveLinkMixin`` keeps a node and the one bone it poses in step, both ways.
The helpers write a value into a node the way the rig does: without it
counting as an edit.
"""

import math

from bpy.props import StringProperty

from ..core import TREE_IDNAME, editable, gather_input_bones, select_bones, socket_links
from ..sockets import RigSocket, marker_attrs

ZERO = (0.0, 0.0, 0.0)


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
        # Inside a node group: the rig the group is used on, when exactly one
        # uses it. That is still the binding -- found through the group node
        # -- and it is what makes a node inside a group live with its bone.
        from ..groups import unique_rig

        # No fallback to the Output's armature_name. The target follows the
        # BINDING, never the selection or a typed name: a tree that quietly
        # retargeted itself to whichever armature was active is how a rig
        # ended up modified by a graph that was never pointed at it.
        return unique_rig(tree)

    def rig_for_ui(self):
        """The rig to list bone names from, for a dropdown.

        This tree's own rig, or -- inside a node group, which has none -- the
        rig of a tree that uses the group. For display only: a group can be
        used on several rigs, so nothing that writes a rig goes through here.
        """
        obj = self.resolve_armature()
        if obj is not None:
            return obj
        from ..groups import group_users

        seen, todo = set(), list(group_users(self.id_data))
        while todo:
            tree = todo.pop()
            if tree.name in seen:
                continue
            seen.add(tree.name)
            for node in tree.nodes:
                src = getattr(node, "source", None) if node.bl_idname == "ArmatureNodesInputNode" else None
                if src is not None and getattr(src, "type", "") == "ARMATURE":
                    return src
            todo.extend(group_users(tree))
        return None

    def draw_bone_select(self, layout):
        """The Bone field every modifier node carries.

        A searchable dropdown of the rig's real bones when there is a rig to
        search, plain text otherwise. Empty means every bone on the wire.
        """
        obj = self.rig_for_ui()
        row = layout.row(align=True)
        if obj is not None:
            row.prop_search(self, "bone", obj.data, "bones", text="", icon="BONE_DATA")
        else:
            row.prop(self, "bone", text="", icon="BONE_DATA")
        if not self.bone:
            layout.label(text="All bones", icon="INFO")


def bone_select_prop(update=None):
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


def same_vec(a, b, eps=1e-6):
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


def turn_angle(q):
    """The angle a rotation turns by, immune to the q / -q double cover."""
    return 2.0 * math.acos(min(1.0, abs(q.w)))


def write_socket_value(node, name, value, linked_too=False):
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
    if same_vec(sock.default_value, value):
        return False
    from ..tree import suspend_live_update

    with suspend_live_update():
        sock.default_value = value
    return True


def _linked_marker(node, name):
    """(marker node, marker) wired into ``name``, or (None, None)."""
    sock = node.inputs.get(name)
    links = socket_links(sock) if sock is not None and sock.is_linked else ()
    if not links:
        return None, None
    link = links[0]
    key = getattr(link.from_socket, "marker_key", "")
    source = link.from_node
    if not key or not hasattr(source, "marker_by_key"):
        return None, None
    return source, source.marker_by_key(key)


def write_input(node, name, value, attr=None):
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
    return write_socket_value(node, name, value)


class LiveLinkMixin:
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
        from .. import livelink

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
        from .. import livelink

        obj, pbone = self.live_bone()
        if pbone is not None and obj.mode != "EDIT":
            livelink.remember(self, obj, pbone)

    def free(self):
        from .. import livelink

        livelink.forget(self)
        super().free()


class ModifierNodeBase(ArmatureNodeBase):
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
        """The rig coming in, as a list of this node's own. The bones in it
        are still upstream's: change one only through ``edit``."""
        return list(gather_input_bones(self, self.stream_input, ctx))

    def selected(self, bones):
        return select_bones(bones, self.bone)

    def edit(self, bones):
        """The selected bones, copied into ``bones``, ready to change."""
        return editable(bones, self.selected(bones))


def redraw_viewports():
    from ..primary_rig import tag_viewports_redraw

    tag_viewports_redraw()
