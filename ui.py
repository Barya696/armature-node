"""Editor UI: Shift+A node categories, header buttons, N-panel sidebar."""

import bpy
from bpy.types import Panel

import nodeitems_utils
from nodeitems_utils import NodeCategory, NodeItem

from .core import TREE_IDNAME


class ArmatureNodeCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        return (
            context.space_data is not None
            and context.space_data.type == "NODE_EDITOR"
            and context.space_data.tree_type == TREE_IDNAME
        )


def _group_items(context):
    from .groups import add_menu_items

    return add_menu_items(context) if context is not None else []


NODE_CATEGORIES = [
    ArmatureNodeCategory(
        "ARMATURE_NODES_IO",
        "Armature I/O",
        items=[
            NodeItem("ArmatureNodesInputNode"),
            NodeItem("ArmatureNodesOutputNode"),
        ],
    ),
    ArmatureNodeCategory(
        "ARMATURE_NODES_MARKER",
        "Marker",
        items=[
            NodeItem("ArmatureNodesMarkerNode"),
            NodeItem("ArmatureNodesSkeletonNode"),
        ],
    ),
    ArmatureNodeCategory(
        "ARMATURE_NODES_BONE",
        "Bone",
        items=[
            NodeItem("ArmatureNodesBoneNode"),
            NodeItem("ArmatureNodesChainNode"),
        ],
    ),
    ArmatureNodeCategory(
        "ARMATURE_NODES_TRANSFORM",
        "Transform",
        items=[
            NodeItem("ArmatureNodesPositionNode"),
            NodeItem("ArmatureNodesRotationNode"),
            NodeItem("ArmatureNodesTransformNode"),
            NodeItem("ArmatureNodesSnapNode"),
        ],
    ),
    ArmatureNodeCategory(
        "ARMATURE_NODES_SHAPE",
        "Shape",
        items=[
            NodeItem("ArmatureNodesCustomShapeNode"),
        ],
    ),
    ArmatureNodeCategory(
        "ARMATURE_NODES_CONSTRAINTS",
        "Constraint",
        items=[
            NodeItem("ArmatureNodesIKConstraintNode"),
            NodeItem("ArmatureNodesGenericConstraintNode"),
        ],
    ),
    # Filled when the menu opens: the groups that exist change as you work.
    ArmatureNodeCategory("ARMATURE_NODES_GROUP", "Group", items=_group_items),
]


def _is_armature_node_editor(context):
    space = context.space_data
    return (
        space is not None
        and space.type == "NODE_EDITOR"
        and space.tree_type == TREE_IDNAME
    )


class ARMATURE_NODES_PT_sidebar(Panel):
    bl_label = "Armature Nodes"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Armature"

    @classmethod
    def poll(cls, context):
        return _is_armature_node_editor(context)

    def draw(self, context):
        layout = self.layout
        tree = context.space_data.node_tree

        if tree is not None:
            layout.separator()
            layout.prop(tree, "live_update")
            if tree.is_dirty and not tree.live_update:
                layout.label(text="Live Update is disabled", icon="INFO")
            if tree.last_error:
                box = layout.box()
                box.alert = True
                box.label(text="Live update failed:", icon="ERROR")
                box.label(text=tree.last_error)


def _active_armature(context):
    obj = context.active_object
    if obj is not None and obj.type == "ARMATURE":
        return obj
    return None


def _draw_convert_button(layout, context, text="Convert to Nodes"):
    obj = _active_armature(context)
    row = layout.row(align=True)
    row.enabled = obj is not None
    row.operator("armature_nodes.convert_rig", text=text, icon="NODETREE")


def draw_view3d_header(self, context):
    """Top bar of the 3D Viewport, next to the tool settings."""
    if _active_armature(context) is None:
        return
    layout = self.layout
    layout.separator_spacer()
    _draw_convert_button(layout, context)


def draw_node_editor_header(self, context):
    """Header of the Armature node editor."""
    if not _is_armature_node_editor(context):
        return
    layout = self.layout
    layout.separator()
    _draw_convert_button(layout, context)


class ARMATURE_NODES_PT_view3d_tool(Panel):
    """Sits in the viewport sidebar Tool tab, right below Active Tool."""

    bl_label = "Armature Nodes"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Tool"
    bl_order = 1

    @classmethod
    def poll(cls, context):
        return _active_armature(context) is not None

    def draw(self, context):
        layout = self.layout
        obj = _active_armature(context)
        tree = getattr(obj, "armature_nodes_tree", None)
        col = layout.column(align=True)
        col.operator(
            "armature_nodes.convert_rig",
            text="Convert to Armature Nodes",
            icon="NODETREE",
        )
        if tree is not None:
            col.separator()
            col.label(text=f"Tree: {tree.name}", icon="NODETREE")
            col.prop(tree, "live_update")
            if tree.last_error:
                box = col.box()
                box.alert = True
                box.label(text=tree.last_error, icon="ERROR")


classes = (ARMATURE_NODES_PT_sidebar, ARMATURE_NODES_PT_view3d_tool)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    nodeitems_utils.register_node_categories("ARMATURE_NODES", NODE_CATEGORIES)
    bpy.types.VIEW3D_HT_header.append(draw_view3d_header)
    bpy.types.NODE_HT_header.append(draw_node_editor_header)


def unregister():
    bpy.types.NODE_HT_header.remove(draw_node_editor_header)
    bpy.types.VIEW3D_HT_header.remove(draw_view3d_header)
    nodeitems_utils.unregister_node_categories("ARMATURE_NODES")
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
