"""The Human Skeleton: a default node group of markers that pose a Rigify
rig, and where those markers go on a character.

Thirteen Marker nodes -- pelvis, chest, head, clavicles, elbows, hands,
knees, feet -- each drive the Rigify control that does that job, through a
Transform node, and are parented like the bones they stand for. A Rigify
Switch turns the poles on, so the elbow and knee markers steer the bend; a
Wrap Markers node fits the whole skeleton onto a character mesh.

Shift+A > Group > Human Skeleton adds it, and so does the Wrap panel in the
3D Viewport sidebar; the group is made the first time it is asked for.
Wired into a Rigify rig's tree, every marker takes its control's place, so
nothing moves until the skeleton is wrapped onto a mesh.

``landmarks`` says where each of its markers sits on a standing character
-- by Rigify's own human, measured, and at the tips of the arms for the
hands -- which is how Auto Pairs pairs them without a click.
"""

import bpy
import numpy as np
from bpy.types import Operator
from mathutils import Vector

PRESET = "human_skeleton"  # the tag on the group, whatever it gets called
NAME = "Human Skeleton"
_CENTER, _LEFT, _RIGHT = (1.0, 0.78, 0.25), (0.25, 0.65, 1.0), (1.0, 0.38, 0.35)

# (marker, Rigify control, parent marker, where it sits on a standing human
# one unit tall -- (across from the middle, back, up), Rigify's human
# measured -- and how Auto Pairs finds it: at that height, at the tip of the
# arm, or not at all (a pole target, carried by the skeleton).
_BODY = (
    ("Pelvis", "torso", None, (0.0, 0.018, 0.544), "height"),
    ("Chest", "chest", "Pelvis", (0.0, 0.0, 0.65), "height"),
    ("Head", "head", "Chest", (0.0, -0.013, 0.899), "height"),
)
_LIMBS = (
    ("Clavicle", "shoulder", "Chest", (0.009, -0.035, 0.809), "height"),
    ("Elbow", "upper_arm_ik_target", "Clavicle", (0.225, 0.1, 0.73), None),
    ("Hand", "hand_ik", "Elbow", (0.336, 0.025, 0.657), "tip"),
    ("Foot", "foot_ik", None, (0.05, 0.008, 0.035), "height"),
    ("Knee", "thigh_ik_target", "Foot", (0.05, -0.06, 0.28), None),
)
# The shoulder joint and how far down the arm the wrist is, from the shoulder
# to the fingertips: where Auto Pairs looks for a hand.
_SHOULDER = (0.099, 0.799)
_WRIST_ALONG = 0.734


def spec():
    """[(marker, control, parent, where, how, side)]: each side of a limb
    its own marker, the right one mirrored."""
    out = [(name, control, parent, where, how, 0) for name, control, parent, where, how in _BODY]
    for side, suffix in ((1, "L"), (-1, "R")):
        for name, control, parent, where, how in _LIMBS:
            parent = parent if parent in ("Chest", None) else f"{parent}.{suffix}"
            out.append((f"{name}.{suffix}", f"{control}.{suffix}", parent, where, how, side))
    return out


# ---------------------------------------------------------------------------
# The group
# ---------------------------------------------------------------------------

# Frames laid out like the body seen from the front, chained into the rig in
# this order: (title, markers, x of the first column, y).
_ROWS = (
    ("Spine & Head", ("Pelvis", "Chest", "Head"), 0.0, 0.0),
    ("Left Arm", ("Clavicle.L", "Elbow.L", "Hand.L"), 1000.0, -900.0),
    ("Right Arm", ("Clavicle.R", "Elbow.R", "Hand.R"), -1000.0, -900.0),
    ("Left Leg", ("Foot.L", "Knee.L"), 1000.0, -1800.0),
    ("Right Leg", ("Foot.R", "Knee.R"), -700.0, -1800.0),
)


def human_skeleton_group():
    """The Human Skeleton group, made the first time it is asked for."""
    for tree in bpy.data.node_groups:
        if tree.get("an_preset") == PRESET:
            return tree
    return _build()


def _build():
    from .groups import new_group_tree
    from .marker_links import join
    from .nodes.wrap import WRAP_NODE
    from .tree import suspend_live_update

    with suspend_live_update():
        group = new_group_tree(NAME)
        group["an_preset"] = PRESET
        gin = next(n for n in group.nodes if n.bl_idname == "NodeGroupInput")
        gout = next(n for n in group.nodes if n.bl_idname == "NodeGroupOutput")
        group.links.remove(group.links[0])

        # Rigify's poles on: without them the elbow and knee targets steer
        # nothing. Kept by the switch list once it meets the rig's record.
        switch = group.nodes.new("ArmatureNodesRigifySwitchNode")
        pole = switch.switches.add()
        pole.name, pole.kind, pole.flag, pole.use = "pole_vector", "BOOL", True, True
        wrap = group.nodes.new(WRAP_NODE)

        markers, moves = {}, {}
        for name, control, parent, where, how, side in spec():
            marker = group.nodes.new("ArmatureNodesMarkerNode")
            marker.name = marker.label = name
            m = marker.markers[0]
            m.name = name
            m.color = _CENTER if not side else _LEFT if side > 0 else _RIGHT
            m.set_position((side * where[0] * 1.8 if side else 0.0, where[1] * 1.8, where[2] * 1.8))
            m.wrap_role = "INSIDE" if how else "FREE"
            move = group.nodes.new("ArmatureNodesTransformNode")
            move.name = move.label = f"{name} ({control})"
            move.bone = control
            group.links.new(marker.outputs[0], move.inputs["Transform"])
            markers[name], moves[name] = marker, move
        for name, _control, parent, _where, _how, _side in spec():
            if parent:
                group.links.new(markers[parent].outputs[0], markers[name].inputs["Parent"])
                markers[name]._track_parent()  # values relative, world kept
        for side in ("L", "R"):
            join(markers["Pelvis"], markers[f"Knee.{side}"])  # the thighs

        previous = gin.outputs[0]
        group.links.new(previous, switch.inputs["Rig"])
        previous = switch.outputs["Rig"]
        for title, names, x0, y in _ROWS:
            frame = group.nodes.new("NodeFrame")
            frame.name = frame.label = title
            for col, name in enumerate(names):
                x = x0 + col * 300.0
                markers[name].location, moves[name].location = (x, y), (x - 10.0, y - 520.0)
                markers[name].parent = moves[name].parent = frame
                group.links.new(previous, moves[name].inputs["Rig"])
                previous = moves[name].outputs["Rig"]
        group.links.new(previous, gout.inputs[0])
        gin.location, switch.location, wrap.location = (-1500.0, 0.0), (-1200.0, 0.0), (-1200.0, 450.0)
        gout.location = (1900.0, -900.0)
    return group


def add_to_tree(tree):
    """A group node running the Human Skeleton in ``tree``, on the rig's
    wire from the Armature Input to the Output. Returns it."""
    from .groups import GROUP_NODE, sync_group_sockets

    src = next((n for n in tree.nodes if n.bl_idname == "ArmatureNodesInputNode"), None)
    out = next((n for n in tree.nodes if n.bl_idname == "ArmatureNodesOutputNode"), None)
    node = tree.nodes.new(GROUP_NODE)
    node.node_tree = human_skeleton_group()
    sync_group_sockets(node)
    if src is not None and out is not None:
        node.location = ((src.location.x + out.location.x) / 2.0, min(src.location.y, out.location.y) - 300.0)
        tree.links.new(src.outputs["Rig"], node.inputs[0])
        tree.links.new(node.outputs[0], out.inputs["Rig"])  # the Output chains its wires
    return node


# ---------------------------------------------------------------------------
# Where the markers go on a character
# ---------------------------------------------------------------------------


def landmarks(mesh):
    """{marker: world point} -- where each marker that Auto Pairs finds sits
    on ``mesh`` (a ``nodes.wrap.MeshTarget``), a character standing on its
    lowest point, facing -Y, its middle the mesh's own YZ plane.

    Heights are shares of the character's height, floor to the top of the
    head, as on Rigify's human; a point is taken halfway through the body
    there, front to back, measured just above it -- where the limb comes in:
    below an ankle is the foot, reaching forward. A hand is found down the
    arm, most of the way from the shoulder to the fingertip furthest from it,
    so it lands at the wrist whether the arms are out (a T-pose) or down (an
    A-pose).
    """
    from .nodes.wrap import _mirrored

    points = mesh.points
    floor = float(points[:, 2].min())
    height = max(float(points[:, 2].max()) - floor, 1e-6)
    middle = (Vector(mesh.center) + _mirrored(mesh.frame, mesh.center)).x / 2.0

    def depth(x, z):
        """Front to back, the middle of the body just above (x, z)."""
        up = points[:, 2] - z
        near = points[(up > 0.02 * height) & (up < 0.06 * height) & (np.abs(points[:, 0] - x) < 0.1 * height)]
        return float(near[:, 1].mean()) if len(near) else float(mesh.center.y)

    out = {}
    for name, _control, _parent, where, how, side in spec():
        if how == "height":
            x, z = middle + side * where[0] * height, floor + where[2] * height
        elif how == "tip":
            shoulder = np.array((middle + side * _SHOULDER[0] * height, floor + _SHOULDER[1] * height))
            arm = points[(side * (points[:, 0] - middle) > 0.16 * height) & (points[:, 2] > floor + 0.3 * height)]
            if not len(arm):
                continue
            tip = arm[np.argmax(np.linalg.norm(arm[:, (0, 2)] - shoulder, axis=1)), (0, 2)]
            x, z = shoulder + (tip - shoulder) * _WRIST_ALONG
        else:
            continue
        out[name] = Vector((x, depth(x, z), z))
    return out


class ARMATURE_NODES_OT_add_human_skeleton(Operator):
    """Add the Human Skeleton: thirteen markers that pose a Rigify rig, and a
    Wrap Markers node to fit them onto a character"""

    bl_idname = "armature_nodes.add_human_skeleton"
    bl_label = "Human Skeleton"
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        group = human_skeleton_group()
        space = context.space_data
        if space is not None and space.type == "NODE_EDITOR" and space.edit_tree is not None:
            # From Shift+A: under the mouse, and grabbed, like any node added.
            return bpy.ops.node.add_node(
                "INVOKE_DEFAULT",
                type="ArmatureNodesGroupNode",
                use_transform=True,
                settings=[{"name": "node_tree", "value": f"bpy.data.node_groups[{group.name!r}]"}],
            )
        return self.execute(context)

    def execute(self, context):
        from .nodes.wrap import selected_rig_tree

        tree = selected_rig_tree(context)
        if tree is None:
            self.report({"WARNING"}, "Select a rig with an Armature Nodes tree first")
            return {"CANCELLED"}
        add_to_tree(tree)
        self.report({"INFO"}, f"Added the {NAME} to '{tree.name}'")
        return {"FINISHED"}


classes = (ARMATURE_NODES_OT_add_human_skeleton,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
