"""Custom NodeSocket subclasses for the Armature node tree.

Each socket has a distinct wire color, and value-carrying sockets expose a
default_value so unconnected inputs still work (same convention as the
built-in shader/geometry sockets).
"""

import bpy
from bpy.types import NodeSocket
from bpy.props import FloatProperty, FloatVectorProperty, BoolProperty


def _on_socket_value_changed(self, context):
    """Typing into an unconnected socket's default_value must rebuild too.

    This is the primary way most bone/chain/constraint parameters are set in
    this node system (same convention as the Shader Editor: an unlinked input
    socket is itself the value field). Node-level properties get their
    ``update`` callback injected by ``nodes.py`` at register time, but a
    NodeSocket is a different bpy_struct hierarchy that injection never
    touches -- without this, every socket-typed value was dead: it changed
    silently and the armature never followed.
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


class ArmatureSocket(_SocketDrawMixin, NodeSocket):
    """Carries a fully assembled armature (collection of bones)."""

    bl_idname = "ArmatureNodesArmatureSocket"
    bl_label = "Armature"
    socket_color = (0.90, 0.30, 0.30, 1.0)


class PoseSocket(_SocketDrawMixin, NodeSocket):
    """Carries a posed marker skeleton for retargeting onto an existing rig.

    Deliberately distinct from ChainSocket: a Chain is bone data to build,
    a Pose is bone data to drive a rig with. Keeping them apart means Blender
    itself refuses to wire Rig into Armature Output, or Skeleton into the
    Armature Input's Skeleton slot.
    """

    bl_idname = "ArmatureNodesPoseSocket"
    bl_label = "Pose"
    socket_color = (0.95, 0.85, 0.25, 1.0)


class ConstraintSocket(_SocketDrawMixin, NodeSocket):
    """Carries a constraint definition (type + params) for a bone."""

    bl_idname = "ArmatureNodesConstraintSocket"
    bl_label = "Constraint"
    socket_color = (0.70, 0.40, 0.95, 1.0)


class TransformSocket(_SocketDrawMixin, NodeSocket):
    """Carries vector/quaternion/float transform data."""

    bl_idname = "ArmatureNodesTransformSocket"
    bl_label = "Transform"
    socket_color = (0.40, 0.90, 0.50, 1.0)


class FloatSocket(_SocketDrawMixin, NodeSocket):
    bl_idname = "ArmatureNodesFloatSocket"
    bl_label = "Float"
    socket_color = (0.63, 0.63, 0.63, 1.0)

    default_value: FloatProperty(name="Value", default=0.0, update=_on_socket_value_changed)

    def get_value(self):
        if self.is_linked and self.links:
            from_sock = self.links[0].from_socket
            if hasattr(from_sock, "default_value"):
                return float(from_sock.default_value)
        return float(self.default_value)


class VectorSocket(_SocketDrawMixin, NodeSocket):
    bl_idname = "ArmatureNodesVectorSocket"
    bl_label = "Vector"
    socket_color = (0.39, 0.39, 0.78, 1.0)

    default_value: FloatVectorProperty(
        name="Vector",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="XYZ",
        update=_on_socket_value_changed,
    )

    def get_value(self):
        if self.is_linked and self.links:
            from_sock = self.links[0].from_socket
            if hasattr(from_sock, "default_value"):
                v = from_sock.default_value
                try:
                    return (float(v[0]), float(v[1]), float(v[2]))
                except (TypeError, IndexError):
                    pass
        return tuple(self.default_value)


class BoolSocket(_SocketDrawMixin, NodeSocket):
    bl_idname = "ArmatureNodesBoolSocket"
    bl_label = "Boolean"
    socket_color = (0.80, 0.65, 0.85, 1.0)

    default_value: BoolProperty(name="Boolean", default=False, update=_on_socket_value_changed)

    def get_value(self):
        if self.is_linked and self.links:
            from_sock = self.links[0].from_socket
            if hasattr(from_sock, "default_value"):
                return bool(from_sock.default_value)
        return bool(self.default_value)


classes = (
    BoneSocket,
    ChainSocket,
    ArmatureSocket,
    PoseSocket,
    ConstraintSocket,
    TransformSocket,
    FloatSocket,
    VectorSocket,
    BoolSocket,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
