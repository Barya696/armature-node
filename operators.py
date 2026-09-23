"""Operators: Build Armature (forward) and Decompile to Nodes (reverse)."""

import bpy
from bpy.types import Operator
from bpy.props import BoolProperty, StringProperty

from .core import TREE_IDNAME
from .build import build_armature_from_tree
from .decompile import decompile_armature_to_tree


def _active_armature_tree(context):
    space = context.space_data
    if (
        space is not None
        and space.type == "NODE_EDITOR"
        and space.tree_type == TREE_IDNAME
        and space.node_tree is not None
    ):
        return space.node_tree
    return None


class ARMATURE_OT_build_from_nodes(Operator):
    """Compile the active armature node tree into a real armature.

    Wrapped in a single undo step ("Build" is one undo, not dozens).
    """

    bl_idname = "armature_nodes.build"
    bl_label = "Build Armature"
    bl_options = {"REGISTER", "UNDO"}

    tree_name: StringProperty(
        name="Tree",
        description="Node tree to build; defaults to the editor's active tree",
        default="",
    )

    @classmethod
    def poll(cls, context):
        return _active_armature_tree(context) is not None or bool(
            [t for t in bpy.data.node_groups if t.bl_idname == TREE_IDNAME]
        )

    def execute(self, context):
        tree = None
        if self.tree_name:
            tree = bpy.data.node_groups.get(self.tree_name)
        if tree is None:
            tree = _active_armature_tree(context)
        if tree is None:
            self.report({"ERROR"}, "No Armature node tree to build")
            return {"CANCELLED"}

        try:
            obj = build_armature_from_tree(tree, strict=True)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        bpy.ops.ed.undo_push(message="Build Armature from Nodes")
        if obj is None:
            self.report({"INFO"}, f"'{tree.name}' produces no armature")
            return {"FINISHED"}
        self.report(
            {"INFO"},
            f"Built '{obj.name}' ({len(obj.data.bones)} bones) from '{tree.name}'",
        )
        return {"FINISHED"}


class ARMATURE_OT_decompile_to_nodes(Operator):
    """Decompile an existing armature into an equivalent node graph"""

    bl_idname = "armature_nodes.decompile"
    bl_label = "Decompile Armature to Nodes"
    bl_options = {"REGISTER", "UNDO"}

    armature_name: StringProperty(
        name="Armature",
        description=(
            "Armature object to decompile; defaults to the selected/active "
            "armature in the 3D viewport"
        ),
        default="",
    )

    shapes_only: BoolProperty(
        name="Custom Shapes Only",
        description=(
            "Do not rebuild bones or constraints. Reference the armature via "
            "an Armature Input node and emit only Custom Shape nodes, so the "
            "existing rig (e.g. Rigify output) stays untouched"
        ),
        default=False,
    )

    def _resolve_armature(self, context):
        if self.armature_name:
            obj = bpy.data.objects.get(self.armature_name)
            if obj and obj.type == "ARMATURE":
                return obj
        obj = context.active_object
        if obj and obj.type == "ARMATURE":
            return obj
        for obj in context.selected_objects:
            if obj.type == "ARMATURE":
                return obj
        return None

    def execute(self, context):
        obj = self._resolve_armature(context)
        if obj is None:
            self.report({"ERROR"}, "Select an armature object to decompile")
            return {"CANCELLED"}

        # Reuse the armature's bound tree. This is the same model as the
        # Shader Editor: selecting a different rig changes the displayed tree,
        # while explicit Decompile refreshes the current rig's tree in place.
        from .sync import resync_tree_from_armature

        try:
            tree = resync_tree_from_armature(obj, shapes_only=self.shapes_only)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        space = context.space_data
        if space is not None and space.type == "NODE_EDITOR":
            space.tree_type = TREE_IDNAME
            space.node_tree = tree
            space.pin = False

        what = "custom shapes" if self.shapes_only else "nodes"
        bpy.ops.ed.undo_push(message=f"Decompile Armature to {what.title()}")
        self.report(
            {"INFO"},
            f"Decompiled '{obj.name}' {what} into node tree '{tree.name}' "
            f"({len(tree.nodes)} nodes)",
        )
        return {"FINISHED"}


def _first_node_editor_space(context):
    """Return an open Node Editor space (prefer the current one), or None."""
    space = context.space_data
    if space is not None and space.type == "NODE_EDITOR":
        return space
    screen = context.screen
    if screen is None:
        return None
    for area in screen.areas:
        if area.type == "NODE_EDITOR":
            return area.spaces.active
    return None


class ARMATURE_OT_convert_rig(Operator):
    """Convert the active armature into a fully dynamic Armature Nodes graph.

    Like Object > Convert > Mesh for curves: every bone, chain, constraint and
    custom shape becomes a node that OWNS the rig, so editing or deleting a
    node updates this armature in place instead of building a new one."""

    bl_idname = "armature_nodes.convert_rig"
    bl_label = "Convert to Armature Nodes"
    bl_options = {"REGISTER", "UNDO"}

    armature_name: StringProperty(
        name="Armature",
        description="Armature object to convert; defaults to the active armature",
        default="",
    )
    open_editor: BoolProperty(
        name="Show in Node Editor",
        description="Display the generated graph in an open Node Editor",
        default=True,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is not None and obj.type == "ARMATURE":
            return True
        return any(o.type == "ARMATURE" for o in context.selected_objects)

    def _resolve_armature(self, context):
        if self.armature_name:
            obj = bpy.data.objects.get(self.armature_name)
            if obj and obj.type == "ARMATURE":
                return obj
        obj = context.active_object
        if obj and obj.type == "ARMATURE":
            return obj
        for obj in context.selected_objects:
            if obj.type == "ARMATURE":
                return obj
        return None

    def execute(self, context):
        obj = self._resolve_armature(context)
        if obj is None:
            self.report({"ERROR"}, "Select an armature object to convert")
            return {"CANCELLED"}

        from .sync import resync_tree_from_armature
        from .build import tag_owner, find_output_node

        try:
            tree = resync_tree_from_armature(obj, shapes_only=False, full=True)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        output = find_output_node(tree)
        if output is not None:
            tag_owner(obj, tree, output)
        obj.armature_nodes_tree = tree
        tree.live_update = True
        tree.is_dirty = False
        tree.last_error = ""

        if self.open_editor:
            space = _first_node_editor_space(context)
            if space is not None:
                space.tree_type = TREE_IDNAME
                space.node_tree = tree
                space.pin = False

        bpy.ops.ed.undo_push(message="Convert to Armature Nodes")
        self.report(
            {"INFO"},
            f"Converted '{obj.name}' into dynamic node tree '{tree.name}' "
            f"({len(tree.nodes)} nodes)",
        )
        return {"FINISHED"}


class ARMATURE_OT_capture_baseline(Operator):
    """Re-record the rig's current state as the graph's starting point.

    Use it after editing the armature itself -- adding bones, changing the
    rest pose. Note it captures whatever the rig looks like NOW, including
    anything this graph has already applied to it, so the modifications become
    part of the new base state"""

    bl_idname = "armature_nodes.capture_baseline"
    bl_label = "Capture Rig State"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        node = getattr(context, "node", None)
        return node is not None and getattr(node, "source", None) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        from . import baseline

        obj = context.node.source
        count = baseline.store(obj)
        if not count:
            self.report({"WARNING"}, "Nothing to capture")
            return {"CANCELLED"}
        context.node.id_data.mark_dirty()
        self.report({"INFO"}, f"Stored {count} bones on '{obj.name}'")
        return {"FINISHED"}


class ARMATURE_OT_bone_read_from_rig(Operator):
    """Re-read this bone's current transform off the rig into the node"""

    bl_idname = "armature_nodes.bone_read_from_rig"
    bl_label = "Read From Rig"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        node = getattr(context, "node", None)
        return node is not None and hasattr(node, "read_from_rig")

    def execute(self, context):
        node = context.node
        if not node.read_from_rig():
            self.report({"WARNING"}, "No such bone on the rig")
            return {"CANCELLED"}
        node.id_data.mark_dirty()
        self.report({"INFO"}, f"Read '{node.bone}' from the rig")
        return {"FINISHED"}


classes = (
    ARMATURE_OT_capture_baseline,
    ARMATURE_OT_bone_read_from_rig,
    ARMATURE_OT_build_from_nodes,
    ARMATURE_OT_decompile_to_nodes,
    ARMATURE_OT_convert_rig,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
