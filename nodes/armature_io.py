"""Armature I/O: the two ends of the stack.

Armature Input hands on the rig as it was bound; Armature Output is what the
graph writes back to.
"""

import logging

import bpy
from bpy.types import Node
from bpy.props import BoolProperty, EnumProperty, PointerProperty, StringProperty

from ..core import gather_input_bones
from ..sockets import RigSocket
from .base import ArmatureNodeBase, redraw_viewports

log = logging.getLogger(__name__)


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
        from ..store import record as record_store

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
        from .. import bridge
        from ..store import record as record_store

        base = record_store.read(self.source)
        if base is None:
            return []  # not bound: the Output reports it, nothing is written
        return bridge.to_bone_defs(base)


def _display_changed():
    """An Output's Markers toggle: the handles follow what is displayed."""
    from ..sync import request_visibility_refresh
    from ..tree import graph_changed

    graph_changed()
    request_visibility_refresh()
    redraw_viewports()


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
        update=lambda self, ctx: _display_changed(),
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
        from ..build import owned_objects
        from ..tree import queue_object_removal

        tree = self.id_data
        try:
            for obj in owned_objects(tree.name, self.name):
                queue_object_removal(obj.name)
        except (ReferenceError, AttributeError) as exc:
            log.warning("Could not release generated armature: %s", exc)
        self.schedule_rebuild()

    def eval_bones(self, ctx):
        return gather_input_bones(self, "Rig", ctx)
