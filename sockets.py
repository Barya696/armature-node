"""Custom NodeSocket subclasses for the Armature node tree.

Three socket types, deliberately:

* **Rig** -- the stream. Carries a list of ``BoneDef``: the whole armature as
  it stands at that point in the graph, exactly like Geometry Nodes passes
  geometry from node to node. Everything between Armature Input and Armature
  Output reads this, changes some of it, and passes it on.
* **Constraint** -- a constraint definition, wired into a bone-producing node.
* **Vector** -- a position / rotation / offset value. Unlinked, the socket is
  itself the value field (the Shader and Geometry Editor convention), so a
  node with no wire into it still has something to work with.
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


class VectorSocket(_SocketDrawMixin, NodeSocket):
    """A position, rotation or offset.

    ``marker_key`` is set when the vector comes from a Skeleton node output,
    naming which marker it is. Consumers read the marker's live position
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
        description="Marker this socket reads its position from, when it has one",
        default="",
    )

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
                    return tuple(marker.position)
            if hasattr(from_sock, "default_value"):
                v = from_sock.default_value
                try:
                    return (float(v[0]), float(v[1]), float(v[2]))
                except (TypeError, IndexError):
                    pass
        return tuple(self.default_value)


classes = (
    RigSocket,
    ConstraintSocket,
    VectorSocket,
)


# The socket was called Bone before it carried the whole rig.
BoneSocket = RigSocket


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
