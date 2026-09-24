"""Bind Rig -- the one explicit capture.

Binding is the moment a rig becomes the database: its complete state is read
and written to ``an_rig_record``, and from then on the Armature Input inherits
from that record and nothing re-reads the object.

It refuses to overwrite an existing record. Re-capturing is a separate,
confirmed operation (``ops/record.py``) because capturing a rig a graph has
already modified records the modifications as the original -- which is exactly
the data loss this architecture exists to prevent.
"""

import bpy
from bpy.types import Operator

from .. import capture
from ..store import record as record_store
from ..store import widgets_lib

__all__ = ["ARMATURE_OT_bind_rig", "classes", "bind"]


def bind(obj, force=False):
    """Capture ``obj`` and store the record. Returns (bones, widgets)."""
    if record_store.exists(obj) and not force:
        raise RuntimeError("already bound; use Capture Rig State to re-record")
    rec, lib = capture.capture_all(obj)
    record_store.write(obj, rec)
    widgets_lib.write(obj, lib)
    return len(rec.bones), len(lib)


class ARMATURE_OT_bind_rig(Operator):
    """Record this armature as the original, so a graph can modify it safely"""

    bl_idname = "armature_nodes.bind_rig"
    bl_label = "Bind Rig"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "ARMATURE"

    def execute(self, context):
        obj = context.active_object
        try:
            bones, widgets = bind(obj)
        except RuntimeError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        except capture.CaptureError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Bound '{obj.name}': {bones} bones, {widgets} widgets")
        return {"FINISHED"}


classes = (ARMATURE_OT_bind_rig,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
