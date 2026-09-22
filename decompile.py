"""Reverse evaluation: decompile an existing armature into a node graph.

- Walks edit-bone data (head/tail/roll/parent/connected).
- Detects straight, evenly-spaced connected runs of bones and collapses
  them into a single ChainNode instead of N BoneNodes.
- Emits constraint nodes from pose-bone constraints, wired to the right
  bone/chain nodes.
- Lays out generated nodes on a hierarchy-depth grid.

One-shot operator, not a continuous mirror: forward-live plus reverse-live
simultaneously is a correctness minefield, so v1 keeps them separate.
"""

import bpy
from mathutils import Vector

CHAIN_MIN_BONES = 3
DIR_TOLERANCE = 1e-4  # squared-distance tolerance on normalized directions
LEN_TOLERANCE = 1e-4

NODE_X_SPACING = 320
NODE_Y_SPACING = 220


class _BoneInfo:
    def __init__(self, bone):
        self.name = bone.name
        self.head = Vector(bone.head_local)
        self.tail = Vector(bone.tail_local)
        self.parent = bone.parent.name if bone.parent else None
        self.use_connect = bone.use_connect
        self.children = []
        self.constraints = []  # filled from pose bones
        self.shape = None  # ShapeDef if the pose bone has a custom shape
        # Roll is only available on edit bones; captured separately.
        self.roll = 0.0

    @property
    def direction(self):
        d = self.tail - self.head
        return d.normalized() if d.length > 1e-9 else Vector((0, 0, 1))

    @property
    def length(self):
        return (self.tail - self.head).length


def _read_armature(obj):
    """Read bones (with rolls via a temporary edit-mode trip) + constraints."""
    prev_active = bpy.context.view_layer.objects.active
    prev_mode = bpy.context.mode

    bones = {}
    for b in obj.data.bones:
        bones[b.name] = _BoneInfo(b)
    for b in obj.data.bones:
        if b.parent:
            bones[b.parent.name].children.append(bones[b.name])

    # Rolls require edit mode.
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for eb in obj.data.edit_bones:
            if eb.name in bones:
                bones[eb.name].roll = eb.roll
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    from .core import ShapeDef

    for pbone in obj.pose.bones:
        if pbone.name in bones:
            bones[pbone.name].constraints = list(pbone.constraints)
            if pbone.custom_shape is not None:
                bones[pbone.name].shape = ShapeDef(
                    widget=pbone.custom_shape.name,
                    preset="NONE",
                    scale=tuple(pbone.custom_shape_scale_xyz),
                    translation=tuple(pbone.custom_shape_translation),
                    rotation=tuple(pbone.custom_shape_rotation_euler),
                    wire_width=getattr(pbone, "custom_shape_wire_width", 1.0),
                    scale_to_bone_length=pbone.use_custom_shape_bone_size,
                    show_wire=pbone.bone.show_wire,
                )

    if prev_active is not None:
        bpy.context.view_layer.objects.active = prev_active
    if prev_mode == "POSE" and prev_active and prev_active.type == "ARMATURE":
        bpy.ops.object.mode_set(mode="POSE")

    return bones


def _find_chain_runs(bones):
    """Split the hierarchy into runs collapsible into ChainNodes.

    A run qualifies when every link is connected, single-child, collinear,
    evenly spaced, and no bone except the last carries constraints.
    Returns (chains, singles): chains is a list of lists of _BoneInfo,
    singles is a list of _BoneInfo emitted as individual BoneNodes.
    """
    consumed = set()
    chains = []
    singles = []

    roots = [b for b in bones.values() if b.parent is None]
    stack = list(roots)
    while stack:
        bone = stack.pop()
        if bone.name in consumed:
            continue

        run = [bone]
        current = bone
        while True:
            if len(current.children) != 1:
                break
            child = current.children[0]
            if not child.use_connect:
                break
            if current.constraints:  # mid-run constraints block collapsing
                break
            if (child.direction - current.direction).length_squared > DIR_TOLERANCE:
                break
            if abs(child.length - current.length) > LEN_TOLERANCE:
                break
            run.append(child)
            current = child

        if len(run) >= CHAIN_MIN_BONES:
            chains.append(run)
            consumed.update(b.name for b in run)
            stack.extend(run[-1].children)
        else:
            singles.append(bone)
            consumed.add(bone.name)
            stack.extend(bone.children)

    return chains, singles


def _depth_of(bone, bones):
    depth = 0
    current = bone
    while current.parent is not None:
        depth += 1
        current = bones[current.parent]
    return depth


def _make_constraint_nodes(tree, con_list, target_socket, base_loc):
    """Emit constraint nodes for a pose bone's constraints, wired in."""
    from .nodes import IKConstraintNode, GenericConstraintNode, _GENERIC_CONSTRAINT_ITEMS

    generic_types = {item[0] for item in _GENERIC_CONSTRAINT_ITEMS}
    y_offset = 0
    for con in con_list:
        if con.type == "IK":
            node = tree.nodes.new(IKConstraintNode.bl_idname)
            node.target = con.target
            node.subtarget = getattr(con, "subtarget", "") or ""
            node.pole_target = con.pole_target
            node.pole_angle = con.pole_angle
            node.chain_count = con.chain_count
            node.iterations = con.iterations
            node.use_stretch = con.use_stretch
            node.influence = con.influence
        elif con.type in generic_types:
            node = tree.nodes.new(GenericConstraintNode.bl_idname)
            node.constraint_type = con.type
            node.target = getattr(con, "target", None)
            node.subtarget = getattr(con, "subtarget", "") or ""
            node.influence = con.influence
        else:
            print(f"[Armature Nodes] Skipped unsupported constraint type: {con.type}")
            continue
        node.location = (base_loc[0] - NODE_X_SPACING, base_loc[1] - y_offset)
        y_offset += NODE_Y_SPACING
        tree.links.new(node.outputs["Constraint"], target_socket)


def _bone_in_rig_layers(pbone):
    """True when the bone lives in a bone collection (4.x) or layer (<4.0)
    that is visible, i.e. it is part of the rig's user-facing layers."""
    bone = pbone.bone
    collections = getattr(bone, "collections", None)
    if collections is not None:
        if len(collections) == 0:
            return True  # unassigned bones are always shown
        return any(getattr(c, "is_visible", True) for c in collections)
    layers = getattr(bone, "layers", None)
    arm_layers = getattr(pbone.id_data, "layers", None)
    if layers is not None and arm_layers is not None:
        return any(a and b for a, b in zip(layers, arm_layers))
    return True


def iter_control_bones(obj):
    """Yield pose bones that are control bones: they carry a custom shape and
    sit in the rig's layers. Works for any armature, not only Rigify output."""
    for pb in obj.pose.bones:
        if pb.custom_shape is None:
            continue
        if not _bone_in_rig_layers(pb):
            continue
        yield pb


def is_generated_rig(obj):
    """True for any armature that has custom-shaped control bones in its
    rig layers. Such a rig is never rebuilt as Bone/Chain nodes; only its
    custom shapes are decompiled. A metarig (no custom shapes) is not a rig."""
    if obj.pose is None:
        return False
    return any(True for _ in iter_control_bones(obj))


def _shape_key(shape):
    return (
        shape.widget,
        tuple(round(v, 5) for v in shape.scale),
        tuple(round(v, 5) for v in shape.translation),
        tuple(round(v, 5) for v in shape.rotation),
        round(shape.wire_width, 3),
        shape.scale_to_bone_length,
        shape.show_wire,
    )


def _configure_shape_node(node, shape):
    from .widgets import WGT_PREFIX

    widget_obj = bpy.data.objects.get(shape.widget)
    if widget_obj is not None and widget_obj.name.startswith(WGT_PREFIX):
        node.source = "LIBRARY"
        try:
            node.library_widget = widget_obj.name
        except TypeError:
            node.source = "OBJECT"
            node.widget_object = widget_obj
    else:
        node.source = "OBJECT"
        node.widget_object = widget_obj
    node.preset = "NONE"
    node.scale = shape.scale
    node.translation = shape.translation
    node.rotation = shape.rotation
    node.wire_width = max(1.0, shape.wire_width)
    node.scale_to_bone_length = shape.scale_to_bone_length
    node.show_wire = shape.show_wire
    # Existing rigs decide deform themselves; do not override it.
    node.control_only = False
    # Capture the widget's actual mesh so it can be recreated later.
    if widget_obj is not None:
        node.read_widget(obj=widget_obj)


def _decompile_rig_shapes(obj, tree):
    """Existing rig: emit ONLY Custom Shape nodes, no bones.

    Graph: one Custom Shape node per control bone, carrying that bone's full
    info (name, position, scale + parent, children, constraints, widget mesh)
    -> Armature Output (Custom Shapes Only), bound to the rig by name.

    No Armature Input node. It used to sit in front of the Custom Shape nodes
    and re-read the live rig on every evaluation, which made the rig -- not
    the graph -- the source of truth: an edit on a node was overwritten by
    the armature's current state on the next rebuild, so the graph looked
    frozen. Each Custom Shape node already stores everything about its bone,
    so nothing upstream is needed to drive the rig.
    """
    from .core import ShapeDef
    from .nodes import CustomShapeNode, ArmatureOutputNode

    tree.nodes.clear()

    output = tree.nodes.new(ArmatureOutputNode.bl_idname)
    output.armature_name = obj.name
    output.mode = "SHAPES_ONLY"
    output.location = (NODE_X_SPACING + 80, 0)

    # Order nodes by hierarchy depth then name so parents sit above children.
    def _depth(pb):
        d, b = 0, pb.bone
        while b.parent is not None:
            d += 1
            b = b.parent
        return d

    control_bones = sorted(iter_control_bones(obj), key=lambda pb: (_depth(pb), pb.name))

    y = 0
    for pbone in control_bones:
        shape = ShapeDef(
            widget=pbone.custom_shape.name,
            preset="NONE",
            scale=tuple(pbone.custom_shape_scale_xyz),
            translation=tuple(pbone.custom_shape_translation),
            rotation=tuple(pbone.custom_shape_rotation_euler),
            wire_width=getattr(pbone, "custom_shape_wire_width", 1.0),
            scale_to_bone_length=pbone.use_custom_shape_bone_size,
            show_wire=pbone.bone.show_wire,
        )
        node = tree.nodes.new(CustomShapeNode.bl_idname)
        _configure_shape_node(node, shape)
        node.sync_bone_info(pbone)  # bone data + reads the widget's mesh
        node.label = pbone.name
        node.location = (0, -y)
        y += NODE_Y_SPACING
        tree.links.new(node.outputs["Bones"], output.inputs["Bones"])

    # The graph now owns these controllers (not the rig itself, so deleting
    # the Output node never deletes a Rigify rig). Record them so the very
    # first live update can already strip a widget whose node was deleted.
    from .build import SHAPED_BONES_KEY

    obj[SHAPED_BONES_KEY] = sorted(pb.name for pb in control_bones)
    if hasattr(obj, "armature_nodes_tree"):
        obj.armature_nodes_tree = tree

    return tree


def decompile_armature_to_tree(obj, tree, shapes_only=False, full=False):
    """Populate `tree` with nodes equivalent to armature object `obj`.

    Generated rigs (Rigify output) are NOT rebuilt as Bone/Chain nodes by
    default: they decompile to Armature Input + Custom Shape nodes only.

    Pass ``shapes_only=True`` to force that Custom-Shapes-Only graph for ANY
    armature (hand-built or generated), so the existing bones, constraints
    and drivers are referenced rather than reconstructed.

    Pass ``full=True`` ("Convert to Armature Nodes") to force the fully
    dynamic Bone/Chain/Constraint/Custom Shape graph for ANY armature, even
    one with custom-shaped control bones. The graph then owns the armature:
    editing or deleting nodes updates the original object in place.
    """
    from .nodes import BoneNode, ChainNode, CustomShapeNode, ArmatureOutputNode
    from .build import tag_owner

    if shapes_only or (not full and is_generated_rig(obj)):
        return _decompile_rig_shapes(obj, tree)

    bones = _read_armature(obj)
    chains, singles = _find_chain_runs(bones)

    tree.nodes.clear()

    # name -> (node, output socket) for the producer of each bone
    producers = {}
    # Track a y-cursor per depth column so nodes stack instead of overlap.
    y_cursor = {}

    def place(node, depth):
        y = y_cursor.get(depth, 0)
        node.location = (depth * NODE_X_SPACING, -y)
        y_cursor[depth] = y + NODE_Y_SPACING

    # Singles as BoneNodes.
    for bone in singles:
        node = tree.nodes.new(BoneNode.bl_idname)
        node.bone_name = bone.name
        node.head = tuple(bone.head)
        node.tail = tuple(bone.tail)
        node.roll = bone.roll
        node.use_connect = bone.use_connect
        depth = _depth_of(bone, bones)
        place(node, depth)
        producers[bone.name] = (node, node.outputs["Bone"])
        if bone.constraints:
            _make_constraint_nodes(
                tree, bone.constraints, node.inputs["Constraints"], node.location
            )

    # Collapsed runs as ChainNodes.
    for run in chains:
        first, last = run[0], run[-1]
        node = tree.nodes.new(ChainNode.bl_idname)
        # Longest common prefix of the run's bone names, trimmed.
        import os

        prefix = os.path.commonprefix([b.name for b in run]).rstrip("._-0")
        node.prefix = prefix or first.name
        node.count = len(run)
        node.start = tuple(first.head)
        node.direction = tuple(first.direction)
        node.bone_length = first.length
        depth = _depth_of(first, bones)
        place(node, depth)
        out_sock = node.outputs["Chain"]
        for b in run:
            producers[b.name] = (node, out_sock)
        if last.constraints:
            _make_constraint_nodes(
                tree, last.constraints, node.inputs["Tip Constraints"], node.location
            )

    # Parent wiring between producer nodes.
    for bone in singles:
        if bone.parent and bone.parent in producers:
            parent_node, parent_sock = producers[bone.parent]
            node = producers[bone.name][0]
            if parent_node is not node:
                tree.links.new(parent_sock, node.inputs["Parent"])
    for run in chains:
        first = run[0]
        if first.parent and first.parent in producers:
            parent_node, parent_sock = producers[first.parent]
            node = producers[first.name][0]
            if parent_node is not node:
                tree.links.new(parent_sock, node.inputs["Parent"])

    max_depth = max(y_cursor.keys(), default=0)

    # Custom shapes: bones with a widget get routed through a CustomShapeNode
    # before reaching the output. Bones sharing the same widget + transform
    # share one node. Since forward evaluation emits full lineages and the
    # output dedupes keeping the LAST occurrence, shape nodes are wired into
    # the output AFTER the plain leaf sockets so the shaped copies win.
    shape_column = max_depth + 1
    shape_nodes = {}  # key -> CustomShapeNode
    shaped_producer_sockets = set()

    for bone in bones.values():
        if bone.shape is None:
            continue
        key = _shape_key(bone.shape)
        node = shape_nodes.get(key)
        if node is None:
            node = tree.nodes.new(CustomShapeNode.bl_idname)
            _configure_shape_node(node, bone.shape)
            place(node, shape_column)
            shape_nodes[key] = node
        # Restrict the shape to exactly the bones that had it, since the
        # producer socket may carry a whole chain/lineage.
        names = node.filtered_names()
        names.add(bone.name)
        node.bone_filter = ";".join(sorted(names))
        prod_node, prod_sock = producers[bone.name]
        sock_id = (prod_node.name, prod_sock.identifier)
        if sock_id not in shaped_producer_sockets:
            tree.links.new(prod_sock, node.inputs["Bones"])
            shaped_producer_sockets.add(sock_id)

    # Output node collects every LEAF producer. In the forward model each
    # node emits its full parent lineage, so wiring the leaves gives the
    # output node the entire hierarchy (deduplicated by bone name).
    output = tree.nodes.new(ArmatureOutputNode.bl_idname)
    output.armature_name = obj.name
    output.mode = "FULL"
    output_column = shape_column + 1 if shape_nodes else shape_column
    output.location = (output_column * NODE_X_SPACING, 0)

    # Claim the source armature. Without this the forward build treats the
    # existing object as foreign, renames the output to "<name>.001" and
    # builds a duplicate instead of updating the rig the graph came from.
    tag_owner(obj, tree, output)
    if hasattr(obj, "armature_nodes_tree"):
        obj.armature_nodes_tree = tree

    leaf_sockets = []
    seen = set()
    for bone in bones.values():
        if bone.children:
            continue  # not a leaf
        node, sock = producers[bone.name]
        if node.name in seen:
            continue
        seen.add(node.name)
        leaf_sockets.append(sock)
    for sock in leaf_sockets:
        tree.links.new(sock, output.inputs["Bones"])
    for node in shape_nodes.values():
        tree.links.new(node.outputs["Bones"], output.inputs["Bones"])

    return tree
