"""Shape: assign a control widget to the selected bones."""

import bpy
from bpy.types import Node
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    PointerProperty,
)

from ..core import ShapeDef
from ..sockets import VectorSocket
from ..widgets import PRESET_ITEMS, widget_enum_items
from .base import ModifierNodeBase, bone_select_prop


def _poll_widget_object(self, obj):
    return obj.type == "MESH"


class CustomShapeNode(ModifierNodeBase, Node):
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

    bone: bone_select_prop()
    source: EnumProperty(
        name="Source",
        items=(
            ("PRESET", "Preset", "Generate a WGT-rig_<bone> widget from a preset"),
            ("LIBRARY", "WGTS_rig", "Reuse an existing widget from WGTS_rig"),
            ("OBJECT", "Object", "Use any mesh object as the widget"),
        ),
        default="PRESET",
    )
    preset: EnumProperty(name="Preset", items=PRESET_ITEMS, default=1)
    library_widget: EnumProperty(name="Widget", items=widget_enum_items)
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
        bones = self.stream(ctx)
        sock = self.inputs.get("Offset")
        offset = sock.get_value() if sock is not None else (0.0, 0.0, 0.0)
        widget = self._widget_name()
        preset = self.preset if self.source != "OBJECT" else "NONE"
        for b in self.edit(bones):
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
