"""The Human Skeleton: a default node group of markers that pose a Rigify
rig, and where those markers go on a character.

Sixteen Marker nodes on the joints of a stick figure -- pelvis, chest, neck,
head, shoulders, elbows, hands, hips, knees, feet -- all one colour, joined
the way the figure is drawn and parented like the bones. Each moves the
Rigify control that does its joint's job, through a Transform or Position
node: the torso, chest, neck and head, the shoulders, the IK hands and feet;
an elbow or a knee moves its limb's pole target, which stands well behind or
in front of it, and a hip rides with the pelvis. A Rigify Switch turns the
poles on; a Wrap Markers node fits the whole skeleton onto a character mesh.

Shift+A > Group > Human Skeleton adds it, and so does the Wrap panel in the
3D Viewport sidebar; the group is made the first time it is asked for.
Wired into a Rigify rig's tree, every marker takes its joint's place on the
rig, so nothing moves until the skeleton is wrapped onto a mesh.

``landmarks`` says where each of its markers sits on a standing character
-- by Rigify's own human, measured, and down the arms for the elbows and the
hands -- which is how Auto Pairs pairs them without a click.
"""

import bpy
import numpy as np
from bpy.types import Operator
from mathutils import Vector

# The tag on the group, whatever it gets called. A new skeleton, a new tag:
# a file keeps the group it was made with.
PRESET = "human_skeleton_3"
NAME = "Human Skeleton"
_COLOR = (0.947, 0.212, 0.006)

# (marker, the Rigify control it moves -- None: it rides with its parent --
# where on the rig it sits when not on that control's head: (bone, how far
# from its head to its tail) -- its parent marker, and where Auto Pairs finds
# it on a standing human one unit tall: (across from the middle, back, up),
# Rigify's human measured; for an arm, how far down it, from the shoulder to
# the fingertips; None, not at all -- a hip, where the body is all pelvis, is
# carried by the skeleton instead.
_SHOULDER = (0.099, 0.014, 0.799)
_BODY = (
    ("Pelvis", "torso", None, None, (0.0, 0.018, 0.544)),
    ("Chest", "chest", None, "Pelvis", (0.0, 0.0, 0.65)),
    ("Neck", "neck", None, "Chest", (0.0, 0.006, 0.837)),
    ("Head", "head", ("head", 0.5), "Neck", (0.0, -0.013, 0.95)),
)
_LIMBS = (
    ("Shoulder", "shoulder", ("ORG-upper_arm", 0.0), "Chest", _SHOULDER),
    ("Elbow", "upper_arm_ik_target", ("ORG-forearm", 0.0), "Shoulder", 0.382),
    ("Hand", "hand_ik", None, "Elbow", 0.734),
    ("Hip", None, None, "Pelvis", None),
    ("Knee", "thigh_ik_target", ("ORG-shin", 0.0), "Hip", (0.049, -0.015, 0.271)),
    ("Foot", "foot_ik", None, None, (0.05, 0.008, 0.035)),
)
# Markers that turn their control too, through a Rotation node: the head,
# which a Position node only moves.
_TURNS = {"Head"}
# The lines that are not a marker and its parent: the collarbones and shins.
# (A hip moves no control: its knee, a child, is what shows it -- a marker is
# drawn when it feeds the output.)
_JOINS = (("Neck", "Shoulder"), ("Knee", "Foot"))
# Where the markers stand until they meet a rig: a T-pose, drawn by hand.
# The left side; the right is its mirror image.
_POSE = {
    "Pelvis": (0.0, -0.004, 0.777), "Chest": (0.0, 0.008, 1.012), "Neck": (0.0, 0.0, 1.259),
    "Head": (0.0, 0.028, 1.486), "Shoulder": (0.15, 0.0, 1.249), "Elbow": (0.368, 0.014, 1.227),
    "Hand": (0.592, 0.0, 1.203), "Hip": (0.093, -0.004, 0.767), "Knee": (0.111, -0.017, 0.389),
    "Foot": (0.126, -0.007, -0.016),
}


def spec():
    """[(marker, control, seed, parent, where, side)]: each side of a limb
    its own marker, the right one mirrored."""
    body = {name for name, *_rest in _BODY} | {None}
    out = [(name, control, seed, parent, where, 0) for name, control, seed, parent, where in _BODY]
    for side, suffix in ((1, "L"), (-1, "R")):
        for name, control, seed, parent, where in _LIMBS:
            out.append((
                f"{name}.{suffix}",
                control and f"{control}.{suffix}",
                seed and (f"{seed[0]}.{suffix}", seed[1]),
                parent if parent in body else f"{parent}.{suffix}",
                where,
                side,
            ))
    return out


# ---------------------------------------------------------------------------
# The group
# ---------------------------------------------------------------------------

# Frames laid out like the body seen from the front, chained into the rig in
# this order: (title, markers, x of the first column, y).
_ROWS = (
    ("Spine & Head", ("Pelvis", "Chest", "Neck", "Head"), 0.0, 0.0),
    ("Left Arm", ("Shoulder.L", "Elbow.L", "Hand.L"), 1000.0, -900.0),
    ("Right Arm", ("Shoulder.R", "Elbow.R", "Hand.R"), -1000.0, -900.0),
    ("Left Leg", ("Hip.L", "Knee.L", "Foot.L"), 1000.0, -1800.0),
    ("Right Leg", ("Hip.R", "Knee.R", "Foot.R"), -1000.0, -1800.0),
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
        for name, control, seed, _parent, where, side in spec():
            marker = group.nodes.new("ArmatureNodesMarkerNode")
            marker.name = marker.label = name
            m = marker.markers[0]
            m.name, m.color = name, _COLOR
            m.wrap_role = "INSIDE" if where is not None else "FREE"
            x, y, z = _POSE[name.split(".")[0]]
            m.set_position((x * (side or 1), y, z))
            markers[name] = marker
            if control is None:
                continue
            # On a joint the control is not on: a Position node, seeded there.
            move = group.nodes.new("ArmatureNodesPositionNode" if seed else "ArmatureNodesTransformNode")
            move.name = move.label = f"{name} ({control})"
            move.bone = control
            if seed:
                move.seed_bone, move.seed_at = seed
            group.links.new(marker.outputs[0], move.inputs["Position" if seed else "Transform"])
            moves[name] = [move]
            if name in _TURNS:
                turn = group.nodes.new("ArmatureNodesRotationNode")
                turn.name = turn.label = f"{name} turn ({control})"
                turn.bone = control
                group.links.new(marker.outputs[0], turn.inputs["Rotation"])
                moves[name].append(turn)
        for name, _control, _seed, parent, _where, _side in spec():
            if parent:
                group.links.new(markers[parent].outputs[0], markers[name].inputs["Parent"])
                markers[name]._track_parent()  # values relative, world kept
        for a, b in _JOINS:
            for side in ("L", "R"):
                join(markers[a if a in markers else f"{a}.{side}"], markers[f"{b}.{side}"])

        previous = gin.outputs[0]
        group.links.new(previous, switch.inputs["Rig"])
        previous = switch.outputs["Rig"]
        for title, names, x0, y in _ROWS:
            frame = group.nodes.new("NodeFrame")
            frame.name = frame.label = title
            for col, name in enumerate(names):
                x = x0 + col * 300.0
                markers[name].location, markers[name].parent = (x, y), frame
                for row, move in enumerate(moves.get(name, ())):
                    move.location, move.parent = (x - 10.0, y - 520.0 - 330.0 * row), frame
                    group.links.new(previous, move.inputs["Rig"])
                    previous = move.outputs["Rig"]
        group.links.new(previous, gout.inputs[0])
        gin.location, switch.location, wrap.location = (-1500.0, 0.0), (-1200.0, 0.0), (-1200.0, 450.0)
        gout.location = (1900.0, -900.0)
        for node in group.nodes:
            node.select = False  # opened, the group is not all selected
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
    if src is not None:
        seat_hips(node.node_tree, getattr(src, "source", None))
    return node


def seat_hips(group, rig):
    """Each hip marker on ``rig``'s hip joint. A hip moves no control, so
    nothing seeds it the way the others are, and the drawing's offset from
    the pelvis fits one size of figure only: it is set against the pelvis
    marker as that will stand -- on the torso control."""
    if rig is None or rig.type != "ARMATURE" or rig.pose.bones.get("torso") is None:
        return
    pose = rig.pose.bones
    frame = (rig.matrix_world @ pose["torso"].matrix).inverted_safe()
    for node in group.nodes:
        marker = node.markers[0] if node.bl_idname == "ArmatureNodesMarkerNode" and len(node.markers) else None
        thigh = pose.get(f"DEF-thigh{marker.name[3:]}") if marker is not None and marker.name.startswith("Hip.") else None
        if thigh is not None:
            marker.set_position(tuple(frame @ (rig.matrix_world @ thigh.head)))


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
    below an ankle is the foot, reaching forward. An elbow and a hand are
    found down the arm, from the shoulder to the fingertip furthest from it
    -- a little over a third of the way, and most of it -- so they land on
    the elbow and the wrist whether the arms are out (a T-pose) or down (an
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
    for name, _control, _seed, _parent, where, side in spec():
        if where is None:
            continue
        if isinstance(where, tuple):  # at its height
            x, z = middle + side * where[0] * height, floor + where[2] * height
        else:  # down the arm
            shoulder = np.array((middle + side * _SHOULDER[0] * height, floor + _SHOULDER[2] * height))
            arm = points[(side * (points[:, 0] - middle) > 0.16 * height) & (points[:, 2] > floor + 0.3 * height)]
            if not len(arm):
                continue
            tip = arm[np.argmax(np.linalg.norm(arm[:, (0, 2)] - shoulder, axis=1)), (0, 2)]
            x, z = shoulder + (tip - shoulder) * where
        out[name] = Vector((x, depth(x, z), z))
    return out


class ARMATURE_NODES_OT_add_human_skeleton(Operator):
    """Add the Human Skeleton: sixteen markers on the joints of a stick figure
    that pose a Rigify rig, and a Wrap Markers node to fit them onto a
    character"""

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
