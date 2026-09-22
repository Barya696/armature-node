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


class ARMATURE_OT_refresh_shape_info(Operator):
    """Re-read the controlled bone's parent, children, constraints and
    head/tail from the armature into this Custom Shape node"""

    bl_idname = "armature_nodes.refresh_shape_info"
    bl_label = "Refresh Bone Info"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        node = getattr(context, "node", None)
        return node is not None and hasattr(node, "refresh_bone_info")

    def execute(self, context):
        node = context.node
        if not node.controlled_bone:
            self.report({"WARNING"}, "Set a single bone name on the node first")
            return {"CANCELLED"}
        if not node.refresh_bone_info():
            self.report(
                {"WARNING"},
                f"Bone '{node.controlled_bone}' not found on any armature "
                "(link an Armature Input node upstream)",
            )
            return {"CANCELLED"}
        return {"FINISHED"}


class ARMATURE_OT_read_widget(Operator):
    """Read the widget's form (mesh vertices/edges) and display settings
    into this Custom Shape node so it can be recreated without the original"""

    bl_idname = "armature_nodes.read_widget"
    bl_label = "Read Widget Form"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        node = getattr(context, "node", None)
        return node is not None and hasattr(node, "read_widget")

    def execute(self, context):
        node = context.node
        # Best source: the controlled bone on a live armature (also captures
        # scale/translation/rotation as currently set on the pose bone).
        obj = node.find_armature()
        pbone = None
        if obj is not None and node.controlled_bone:
            pbone = obj.pose.bones.get(node.controlled_bone)
        if pbone is not None and pbone.custom_shape is not None:
            ok = node.read_widget(pbone=pbone)
        else:
            ok = node.read_widget()
        if not ok:
            self.report(
                {"WARNING"},
                "No widget mesh to read: pick a WGTS_rig widget or object first",
            )
            return {"CANCELLED"}
        from .snapshot import describe_geometry

        self.report(
            {"INFO"},
            f"Read '{node.widget_name}': {describe_geometry(node.widget_geometry)}",
        )
        return {"FINISHED"}


class ARMATURE_OT_snapshot_armature(Operator):
    """Store a full copy of the source armature (bones, constraints, widgets)
    on this Armature Input node so the graph survives deleting the original"""

    bl_idname = "armature_nodes.snapshot_armature"
    bl_label = "Snapshot Armature"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        node = getattr(context, "node", None)
        return node is not None and hasattr(node, "take_snapshot")

    def execute(self, context):
        node = context.node
        if not node.take_snapshot():
            self.report({"WARNING"}, "Set a source armature first")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Snapshot of '{node.snapshot_name}' stored ({len(node.snapshot)} bones)",
        )
        return {"FINISHED"}


class ARMATURE_OT_copy_bone_transform(Operator):
    """Copy this bone's world Position and Scale to the clipboard"""

    bl_idname = "armature_nodes.copy_bone_transform"
    bl_label = "Copy Position/Scale"

    @classmethod
    def poll(cls, context):
        node = getattr(context, "node", None)
        return node is not None and hasattr(node, "world_transform_clipboard_text")

    def execute(self, context):
        node = context.node
        context.window_manager.clipboard = node.world_transform_clipboard_text()
        self.report({"INFO"}, f"Copied transform of '{node.controlled_bone}'")
        return {"FINISHED"}


class ARMATURE_OT_paste_bone_transform(Operator):
    """Paste a world Position/Scale from the clipboard onto this bone (pose
    mode). Accepts the Copy format, Blender vector copies and any x, y, z"""

    bl_idname = "armature_nodes.paste_bone_transform"
    bl_label = "Paste Position/Scale"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        node = getattr(context, "node", None)
        return node is not None and hasattr(node, "set_world_transform")

    def execute(self, context):
        node = context.node
        if not node.controlled_bone:
            self.report({"WARNING"}, "Set a single bone name on the node first")
            return {"CANCELLED"}
        pos, rot, scl = node.parse_world_transform_text(
            context.window_manager.clipboard
        )
        if pos is None and rot is None and scl is None:
            self.report({"WARNING"}, "Clipboard has no 'x, y, z' values to paste")
            return {"CANCELLED"}
        obj = node.find_armature()
        if obj is not None and obj.mode == "EDIT":
            self.report(
                {"WARNING"},
                "Position/Rotation/Scale are pose values: leave Edit mode first",
            )
            return {"CANCELLED"}
        node.set_world_transform(position=pos, rotation=rot, scale=scl)
        return {"FINISHED"}


def _override_node(context):
    node = getattr(context, "node", None)
    if node is not None and hasattr(node, "bone_overrides"):
        return node
    return None


class ARMATURE_OT_add_bone_override(Operator):
    """Add a manual skeleton bone -> control bone pairing, which takes
    priority over the automatic match for that bone"""

    bl_idname = "armature_nodes.add_bone_override"
    bl_label = "Add Bone Override"
    bl_options = {"REGISTER", "UNDO"}

    skeleton_bone: StringProperty(name="Skeleton Bone", default="")

    @classmethod
    def poll(cls, context):
        return _override_node(context) is not None

    def execute(self, context):
        node = _override_node(context)
        node.add_override(self.skeleton_bone)
        node.show_overrides = True
        return {"FINISHED"}


class ARMATURE_OT_remove_bone_override(Operator):
    """Remove this manual bone pairing"""

    bl_idname = "armature_nodes.remove_bone_override"
    bl_label = "Remove Bone Override"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(name="Index", default=-1)

    @classmethod
    def poll(cls, context):
        return _override_node(context) is not None

    def execute(self, context):
        node = _override_node(context)
        if not 0 <= self.index < len(node.bone_overrides):
            return {"CANCELLED"}
        node.bone_overrides.remove(self.index)
        node.id_data.mark_dirty()
        return {"FINISHED"}


class ARMATURE_OT_fill_bone_overrides(Operator):
    """Add a row for every bone the automatic match resolved, so any pairing
    can be re-pointed at a different control"""

    bl_idname = "armature_nodes.fill_bone_overrides"
    bl_label = "Fill Overrides From Matches"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _override_node(context) is not None

    def execute(self, context):
        from .core import EvalContext

        node = _override_node(context)
        added = node.fill_overrides_from_matches(EvalContext())
        if not added:
            self.report(
                {"WARNING"},
                "Nothing to fill: wire a Primary Rig into Skeleton and set a source rig",
            )
            return {"CANCELLED"}
        node.show_overrides = True
        self.report({"INFO"}, f"Added {added} override rows")
        return {"FINISHED"}


classes = (
    ARMATURE_OT_build_from_nodes,
    ARMATURE_OT_decompile_to_nodes,
    ARMATURE_OT_convert_rig,
    ARMATURE_OT_refresh_shape_info,
    ARMATURE_OT_read_widget,
    ARMATURE_OT_snapshot_armature,
    ARMATURE_OT_copy_bone_transform,
    ARMATURE_OT_paste_bone_transform,
    ARMATURE_OT_add_bone_override,
    ARMATURE_OT_remove_bone_override,
    ARMATURE_OT_fill_bone_overrides,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
