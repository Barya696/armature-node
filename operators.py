"""Operators: Build (graph -> rig), Convert (rig -> graph), Read From Rig."""

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator

from .build import build_armature_from_tree
from .core import TREE_IDNAME


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


def _armature(context, name=""):
    """The named armature, else the active one, else a selected one."""
    if name:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.type == "ARMATURE":
            return obj
    obj = context.active_object
    if obj is not None and obj.type == "ARMATURE":
        return obj
    return next((o for o in context.selected_objects if o.type == "ARMATURE"), None)


class ARMATURE_NODES_OT_build(Operator):
    """Apply an Armature node tree to its armature now"""

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
        return _active_armature_tree(context) is not None or any(
            t.bl_idname == TREE_IDNAME for t in bpy.data.node_groups
        )

    def execute(self, context):
        tree = bpy.data.node_groups.get(self.tree_name) if self.tree_name else None
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
        if obj is None:
            self.report({"INFO"}, f"'{tree.name}' produces no armature")
        else:
            self.report({"INFO"}, f"Built '{obj.name}' ({len(obj.data.bones)} bones) from '{tree.name}'")
        return {"FINISHED"}


def _first_node_editor_space(context):
    """An open Node Editor space (the current one first), or None."""
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


class ARMATURE_NODES_OT_convert_rig(Operator):
    """Give the active armature an Armature Nodes graph: Armature Input
    straight into Armature Output, ready for nodes in between"""

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
        description="Display the graph in an open Node Editor",
        default=True,
    )

    @classmethod
    def poll(cls, context):
        return _armature(context) is not None

    def execute(self, context):
        from .build import find_output_node, tag_owner
        from .sync import resync_tree_from_armature

        obj = _armature(context, self.armature_name)
        if obj is None:
            self.report({"ERROR"}, "Select an armature object to convert")
            return {"CANCELLED"}
        # Modify, not Full Rig: the graph is two nodes that modify the rig as
        # it is, rather than one per bone rebuilding it.
        tree = resync_tree_from_armature(obj, full=False)
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
        self.report({"INFO"}, f"'{obj.name}' is now driven by '{tree.name}'")
        return {"FINISHED"}


class ARMATURE_NODES_OT_bone_read_from_rig(Operator):
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
        return {"FINISHED"}


classes = (
    ARMATURE_NODES_OT_build,
    ARMATURE_NODES_OT_convert_rig,
    ARMATURE_NODES_OT_bone_read_from_rig,
)

register, unregister = bpy.utils.register_classes_factory(classes)
