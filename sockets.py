"""Custom NodeSocket subclasses for the Armature node tree.

Each socket carries a kind of data and has a distinct wire colour. They are
all pure connection points: values are edited on the nodes themselves, not in
the sockets, so none of them defines a ``default_value``.
"""

import bpy
from bpy.types import NodeSocket
from bpy.props import StringProperty


class _SocketDrawMixin:
    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return self.socket_color


class BoneSocket(_SocketDrawMixin, NodeSocket):
    """Carries a single bone definition (head, tail, roll, parent link)."""

    bl_idname = "ArmatureNodesBoneSocket"
    bl_label = "Bone"
    socket_color = (0.95, 0.60, 0.20, 1.0)


class ChainSocket(_SocketDrawMixin, NodeSocket):
    """Carries an ordered list of bones (e.g. an arm or leg chain)."""

    bl_idname = "ArmatureNodesChainSocket"
    bl_label = "Chain"
    socket_color = (0.25, 0.70, 0.95, 1.0)


class MarkerSocket(_SocketDrawMixin, NodeSocket):
    """Carries ONE named marker from a Skeleton node: a world position plus an
    optional orientation.

    Deliberately not a Vector socket. A marker is an identity, not a value:
    the wire says *which* handle in the viewport drives this bone, and the
    consumer reads its live position through ``marker_key``. ``marker_key``
    rather than the socket name, because renaming a marker must not break the
    link.
    """

    bl_idname = "ArmatureNodesMarkerSocket"
    bl_label = "Marker"
    socket_color = (0.95, 0.45, 0.75, 1.0)

    marker_key: StringProperty(
        name="Marker Key",
        description="Stable identifier of the marker this socket carries",
        default="",
    )


class ConstraintSocket(_SocketDrawMixin, NodeSocket):
    """Carries a constraint definition (type + params) for a bone."""

    bl_idname = "ArmatureNodesConstraintSocket"
    bl_label = "Constraint"
    socket_color = (0.70, 0.40, 0.95, 1.0)


classes = (
    BoneSocket,
    ChainSocket,
    MarkerSocket,
    ConstraintSocket,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
