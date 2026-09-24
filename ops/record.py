"""Record operators: Capture, Restore Original, Forget.

Thin wrappers. The logic lives in ``capture`` and ``apply.pipeline``; these
exist so that every capture and every write happens inside one operator
invocation, which makes it a single undo step and gives the user somewhere to
confirm.

Capture asks first, every time. Re-capturing a rig the graph has already
modified records those modifications as the original, and there is no way back
from that -- it is the exact data loss this architecture was rebuilt to
prevent, so it is never a silent or automatic action.
"""

import bpy
from bpy.types import Operator

from .. import capture
from ..apply import pipeline
from ..store import record as record_store
from ..store import touched as touched_store
from ..store import widgets_lib

__all__ = [
    "ARMATURE_OT_capture_record",
    "ARMATURE_OT_restore_original",
    "ARMATURE_OT_forget_record",
    "classes",
]


def _bound_armature(context):
    obj = context.active_object
    if obj is None or obj.type != "ARMATURE":
        return None
    return obj


class ARMATURE_OT_capture_record(Operator):
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
        self.report(
            {"INFO"}, f"Recorded '{obj.name}': {len(rec.bones)} bones, {len(lib)} widgets"
        )
        return {"FINISHED"}


class ARMATURE_OT_restore_original(Operator):
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


class ARMATURE_OT_forget_record(Operator):
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
    ARMATURE_OT_capture_record,
    ARMATURE_OT_restore_original,
    ARMATURE_OT_forget_record,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
