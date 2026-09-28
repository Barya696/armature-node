"""Record operators: Capture, Record Switches, Restore Original, Forget.

Thin wrappers. The logic lives in ``capture`` and ``apply.pipeline``; these
exist so that every capture and every write happens inside one operator
invocation, which makes it a single undo step and gives the user somewhere to
confirm.

Capture asks first, every time. Re-capturing a rig the graph has already
modified records those modifications as the original, and there is no way back
from that -- it is the exact data loss this architecture was rebuilt to
prevent, so it is never a silent or automatic action.
"""

from dataclasses import replace

import bpy
from bpy.types import Operator
from bpy.props import StringProperty

from .. import capture
from ..apply import pipeline
from ..capture.pose_display import capture_props
from ..store import record as record_store
from ..store import touched as touched_store
from ..store import widgets_lib

__all__ = [
    "ARMATURE_NODES_OT_capture_record",
    "ARMATURE_NODES_OT_record_switches",
    "ARMATURE_NODES_OT_restore_original",
    "ARMATURE_NODES_OT_forget_record",
    "classes",
]


def _bound_armature(context):
    obj = context.active_object
    if obj is None or obj.type != "ARMATURE":
        return None
    return obj


def _sync_switch_nodes():
    """The record changed: Rigify Switch nodes list what it has now."""
    for tree in bpy.data.node_groups:
        for node in getattr(tree, "nodes", ()):
            if hasattr(node, "sync_switches"):
                node.sync_switches()


class ARMATURE_NODES_OT_capture_record(Operator):
    """Re-record this rig's current state as the original

    Use after editing the armature itself. It captures the rig as it is NOW,
    including anything the graph has already applied to it, so those changes
    become part of the new original"""

    bl_idname = "armature_nodes.capture_record"
    bl_label = "Capture Rig State"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _bound_armature(context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        obj = _bound_armature(context)
        try:
            rec, lib = capture.capture_all(obj)
        except capture.CaptureError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        record_store.write(obj, rec)
        widgets_lib.write(obj, lib)
        # What the previous build wrote describes the OLD record. Keeping it
        # would make the next build "restore" paths to values that no longer
        # mean anything.
        touched_store.clear(obj)
        _sync_switch_nodes()
        self.report(
            {"INFO"}, f"Recorded '{obj.name}': {len(rec.bones)} bones, {len(lib)} widgets"
        )
        return {"FINISHED"}


class ARMATURE_NODES_OT_record_switches(Operator):
    """Add the rig's switches (Rigify's IK/FK, pole, parents...) to a record
    made before switches were recorded. Nothing else in the record changes"""

    bl_idname = "armature_nodes.record_switches"
    bl_label = "Record Switches"
    bl_options = {"REGISTER", "UNDO"}

    rig: StringProperty(name="Rig", default="", options={"HIDDEN", "SKIP_SAVE"})

    def execute(self, context):
        obj = bpy.data.objects.get(self.rig) if self.rig else _bound_armature(context)
        record = record_store.read(obj) if obj is not None else None
        if record is None:
            self.report({"ERROR"}, "Bind the rig first")
            return {"CANCELLED"}
        # Unlike a full capture this cannot record the graph's own work: no
        # build writes a switch the record lacks, so what the rig has is its own.
        bones = {
            name: bone
            if bone.pose.props is not None
            else replace(bone, pose=replace(bone.pose, props=capture_props(obj.pose.bones.get(name))))
            for name, bone in record.bones.items()
        }
        record_store.write(obj, record.with_bones(bones))
        _sync_switch_nodes()
        count = sum(1 for bone in bones.values() if bone.pose.props)
        self.report({"INFO"}, f"Recorded the switches of {count} bones on '{obj.name}'")
        return {"FINISHED"}


class ARMATURE_NODES_OT_restore_original(Operator):
    """Put this rig back to its recorded state and forget what the graph wrote"""

    bl_idname = "armature_nodes.restore_original"
    bl_label = "Restore Original"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = _bound_armature(context)
        return obj is not None and record_store.exists(obj)

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        obj = _bound_armature(context)
        result = pipeline.restore_original(obj)
        if result.errors:
            for message in result.errors[:3]:
                self.report({"WARNING"}, message)
        self.report({"INFO"}, f"Restored '{obj.name}': {result.writes} properties")
        return {"FINISHED"}


class ARMATURE_NODES_OT_forget_record(Operator):
    """Remove this rig's record. The rig itself is left exactly as it is"""

    bl_idname = "armature_nodes.forget_record"
    bl_label = "Forget Record"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = _bound_armature(context)
        return obj is not None and record_store.exists(obj)

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        obj = _bound_armature(context)
        record_store.forget(obj)
        widgets_lib.forget(obj)
        touched_store.clear(obj)
        self.report({"INFO"}, f"'{obj.name}' is no longer bound")
        return {"FINISHED"}


classes = (
    ARMATURE_NODES_OT_capture_record,
    ARMATURE_NODES_OT_record_switches,
    ARMATURE_NODES_OT_restore_original,
    ARMATURE_NODES_OT_forget_record,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
