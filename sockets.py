"""Custom NodeSocket subclasses for the Armature node tree.

Three socket types, deliberately:

* **Rig** -- the stream. Carries a list of ``BoneDef``: the whole armature as
  it stands at that point in the graph, exactly like Geometry Nodes passes
  geometry from node to node. Everything between Armature Input and Armature
  Output reads this, changes some of it, and passes it on.
* **Constraint** -- a constraint definition, wired into a bone-producing node.
* **Vector / Rotation / Scale** -- a value: a position or offset in scene
  units, an orientation in degrees, or a scale factor. Unlinked, the socket is
  itself the value field (the Shader and Geometry Editor convention), so a
  node with no wire into it still has something to work with.
* **Transform** -- all three at once, on one wire: what a Marker outputs.
"""

import bpy
from bpy.types import NodeSocket
from bpy.props import FloatVectorProperty, StringProperty


def _on_socket_value_changed(self, context):
    """Typing into an unconnected socket's default_value must rebuild too.

    Node-level properties get their ``update`` callback injected by
    ``nodes.py`` at register time, but a NodeSocket is a different bpy_struct
    hierarchy that injection never touches -- without this a typed vector
    would change silently and the armature would never follow.
    """
    from .tree import is_updating

    # Only a real edit: values the rig writes into a node arrive with live
    # update suspended, and must not be mistaken for the user typing.
    if not is_updating():
        node = getattr(self, "node", None)
        if node is not None and hasattr(node, "on_socket_edited"):
            node.on_socket_edited(self)
    tree = getattr(self, "id_data", None)
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


class _SocketDrawMixin:
    def draw(self, context, layout, node, text):
        if self.is_output or self.is_linked or not hasattr(self, "default_value"):
            layout.label(text=text)
        else:
            layout.prop(self, "default_value", text=text)

    def draw_color(self, context, node):
        return self.socket_color


class RigSocket(_SocketDrawMixin, NodeSocket):
    """The whole rig.

    Carries every bone of the armature -- rest geometry, parenting, deform
    flags, constraints, widgets and any pose the graph has written so far --
    as a list of ``BoneDef``. One value, not one bone: the Armature Input puts
    the unmodified rig on the wire, each node in between returns a modified
    copy, and the Armature Output writes the last one back.

    The bl_idname still says Bone because saved files reference it by that
    name; renaming it would silently drop every link in an existing graph.
    """

    bl_idname = "ArmatureNodesBoneSocket"
    bl_label = "Rig"
    socket_color = (0.95, 0.60, 0.20, 1.0)


class ConstraintSocket(_SocketDrawMixin, NodeSocket):
    """Carries a constraint definition (type + params) for a bone."""

    bl_idname = "ArmatureNodesConstraintSocket"
    bl_label = "Constraint"
    socket_color = (0.70, 0.40, 0.95, 1.0)


class _ValueSocketMixin:
    """Shared ``get_value`` for the sockets that carry a 3-vector.

    ``_marker_attr`` names what a linked marker supplies: its position to a
    Vector socket, its orientation to a Rotation socket, its scale to a Scale
    socket -- so wiring a marker into a Rotation input turns the bone with the
    handle rather than feeding a position in as if it were angles.
    """

    _marker_attr = "position"

    def get_value(self):
        """The vector on this socket: from the link if there is one, from a
        marker when the link names one, otherwise the typed default."""
        if self.is_linked and self.links:
            link = self.links[0]
            from_sock = link.from_socket
            key = getattr(from_sock, "marker_key", "")
            node = link.from_node
            if key and hasattr(node, "marker_by_key"):
                marker = node.marker_by_key(key)
                if marker is not None:
                    return tuple(getattr(marker, self._marker_attr))
            if hasattr(from_sock, "default_value"):
                v = from_sock.default_value
                try:
                    return (float(v[0]), float(v[1]), float(v[2]))
                except (TypeError, IndexError):
                    pass
        return tuple(self.default_value)


class VectorSocket(_ValueSocketMixin, _SocketDrawMixin, NodeSocket):
    """A position or offset, in scene units.

    ``marker_key`` is set when the vector comes from a Marker or Skeleton
    output, naming which marker it is. Consumers read the marker's live value
    through it, so dragging the handle in the viewport moves whatever the wire
    feeds. It stays empty on an ordinary vector.
    """

    bl_idname = "ArmatureNodesVectorSocket"
    bl_label = "Vector"
    socket_color = (0.39, 0.39, 0.78, 1.0)

    default_value: FloatVectorProperty(
        name="Vector",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="TRANSLATION",
        update=_on_socket_value_changed,
    )
    marker_key: StringProperty(
        name="Marker Key",
        description="Marker this socket reads its value from, when it has one",
        default="",
    )


class RotationSocket(_ValueSocketMixin, _SocketDrawMixin, NodeSocket):
    """An orientation, as XYZ Euler.

    Its own type because the unit is: shown in degrees, stored in radians. A
    rotation carried on a Vector socket was labelled in metres and read "90"
    as ninety radians.
    """

    bl_idname = "ArmatureNodesRotationSocket"
    bl_label = "Rotation"
    socket_color = (0.63, 0.39, 0.78, 1.0)
    _marker_attr = "rotation"

    default_value: FloatVectorProperty(
        name="Rotation",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="EULER",
        update=_on_socket_value_changed,
    )


class ScaleSocket(_ValueSocketMixin, _SocketDrawMixin, NodeSocket):
    """A per-axis scale factor. Defaults to 1, which means "unchanged"."""

    bl_idname = "ArmatureNodesScaleSocket"
    bl_label = "Scale"
    socket_color = (0.39, 0.63, 0.78, 1.0)
    _marker_attr = "scale"

    default_value: FloatVectorProperty(
        name="Scale",
        size=3,
        default=(1.0, 1.0, 1.0),
        subtype="XYZ",
        update=_on_socket_value_changed,
    )


class TransformSocket(_SocketDrawMixin, NodeSocket):
    """Location, rotation and scale together, on one wire.

    What a Marker node outputs, and what the Transform node's Transform input
    takes -- the Geometry Nodes matrix socket, for bones. Wired into a Vector,
    Rotation or Scale input instead, that input takes its own part, as it
    always has.

    It has no field of its own. Unlinked, a Transform input is simply unused,
    and the node's Translation / Rotation / Scale fields apply.
    """

    bl_idname = "ArmatureNodesTransformSocket"
    bl_label = "Transform"
    socket_color = (0.72, 0.20, 0.52, 1.0)
    #: A linked marker supplies all three of its values through this socket.
    _marker_attrs = ("position", "rotation", "scale")

    marker_key: StringProperty(
        name="Marker Key",
        description="Marker this socket carries, when it comes from one",
        default="",
    )

    def get_transform(self):
        """(location, rotation, scale) from the wire, or None when unlinked.

        A part the source does not carry is None, and the node leaves that
        part of the bone alone: a Skeleton landmark with rotation off is a
        position, and must not turn the bone to face world zero. From any
        other value, only the part its type stands for.
        """
        if not self.is_linked or not self.links:
            return None
        link = self.links[0]
        from_sock, node = link.from_socket, link.from_node
        key = getattr(from_sock, "marker_key", "")
        if key and hasattr(node, "marker_by_key"):
            marker = node.marker_by_key(key)
            if marker is not None:
                turns = node.marker_uses_rotation(key)
                scales = node.marker_uses_scale(key)
                return (
                    tuple(marker.position),
                    tuple(marker.rotation) if turns else None,
                    tuple(marker.scale) if scales else None,
                )
        parts = dict.fromkeys(("position", "rotation", "scale"))
        value = getattr(from_sock, "default_value", None)
        if value is not None:
            parts[getattr(from_sock, "_marker_attr", "position")] = tuple(value)
        return parts["position"], parts["rotation"], parts["scale"]


def marker_attrs(sock):
    """Which of a marker's values an input takes: all three, or one."""
    return getattr(sock, "_marker_attrs", None) or (getattr(sock, "_marker_attr", "position"),)


classes = (
    RigSocket,
    ConstraintSocket,
    VectorSocket,
    RotationSocket,
    ScaleSocket,
    TransformSocket,
)


# The socket was called Bone before it carried the whole rig.
BoneSocket = RigSocket


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
