"""Constraint: IK, and one node for the common target and limit constraints."""

import bpy
from bpy.types import Node
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from ..core import ConstraintDef
from ..sockets import ConstraintSocket
from .base import ArmatureNodeBase


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
