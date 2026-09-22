"""All node types for the Armature node tree.

Forward building blocks: BoneNode, ChainNode, MirrorNode, ParentNode,
DeformGroupNode, constraint nodes, ArmatureOutputNode.
Reverse entry point: ArmatureInputNode.

Every bone-producing node implements eval_bones(ctx) -> list[BoneDef].
Every constraint node implements eval_constraints(ctx) -> list[ConstraintDef].
"""

import bpy
from bpy.types import Node
from bpy.props import (
    StringProperty,
    FloatProperty,
    IntProperty,
    BoolProperty,
    FloatVectorProperty,
    EnumProperty,
    PointerProperty,
)
from mathutils import Vector

from .core import (
    TREE_IDNAME,
    BoneDef,
    ConstraintDef,
    gather_input_bones,
    gather_input_constraints,
    copy_bone,
)
from .sockets import (
    BoneSocket,
    ChainSocket,
    ConstraintSocket,
    PoseSocket,
)
from .widgets import PRESET_ITEMS as _widget_preset_items
from .widgets import widget_enum_items as _widget_enum_items


class ArmatureNodeBase:
    """Base for all nodes in this tree; restricts them to our tree type."""

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == TREE_IDNAME

    def _multi_input(self, socket_idname, name):
        sock = self.inputs.new(socket_idname, name)
        sock.link_limit = 0  # allow multiple links
        return sock

    def schedule_rebuild(self):
        """Ask the tree to re-evaluate. Safe from restricted contexts."""
        tree = self.id_data
        try:
            if tree is not None and hasattr(tree, "mark_dirty"):
                tree.mark_dirty()
        except (ReferenceError, AttributeError):
            pass

    def free(self):
        """Blender calls this when the node is removed.

        NodeTree.update() is not a reliable deletion signal -- it does not
        fire for nodes removed through the Python API, and a node deleted
        without links changes no topology Blender bothers to report. free() is
        called for every removal, so the rebuild is scheduled from here.
        """
        self.schedule_rebuild()


# ---------------------------------------------------------------------------
# Bones / Chains
# ---------------------------------------------------------------------------


class BoneNode(ArmatureNodeBase, Node):
    """Defines one bone: head, tail, roll, connected flag, parent input."""

    bl_idname = "ArmatureNodesBoneNode"
    bl_label = "Bone"
    bl_icon = "BONE_DATA"

    bone_name: StringProperty(name="Name", default="Bone")
    head: FloatVectorProperty(name="Head", size=3, default=(0.0, 0.0, 0.0), subtype="XYZ")
    tail: FloatVectorProperty(name="Tail", size=3, default=(0.0, 0.0, 0.5), subtype="XYZ")
    roll: FloatProperty(name="Roll", default=0.0, subtype="ANGLE")
    use_connect: BoolProperty(name="Connected", default=False)

    def init(self, context):
        self.inputs.new(BoneSocket.bl_idname, "Parent")
        self._multi_input(ConstraintSocket.bl_idname, "Constraints")
        self.outputs.new(BoneSocket.bl_idname, "Bone")

    def draw_buttons(self, context, layout):
        layout.prop(self, "bone_name")
        col = layout.column(align=True)
        col.prop(self, "head")
        col.prop(self, "tail")
        layout.prop(self, "roll")
        layout.prop(self, "use_connect")

    def eval_bones(self, ctx):
        parents = gather_input_bones(self, "Parent", ctx)
        constraints = gather_input_constraints(self, "Constraints", ctx)
        bone = BoneDef(
            name=self.bone_name or self.name,
            head=tuple(self.head),
            tail=tuple(self.tail),
            roll=self.roll,
            use_connect=self.use_connect,
            constraints=constraints,
        )
        if parents:
            bone.parent = parents[-1].name
            if self.use_connect:
                bone.head = tuple(parents[-1].tail)
        # Emit parents too so the output node receives the whole lineage.
        return [copy_bone(b) for b in parents] + [bone]


class ChainNode(ArmatureNodeBase, Node):
    """Procedurally generates N connected bones (spine, finger, tail...)."""

    bl_idname = "ArmatureNodesChainNode"
    bl_label = "Chain"
    bl_icon = "CONSTRAINT_BONE"

    prefix: StringProperty(name="Prefix", default="chain")
    count: IntProperty(name="Count", default=3, min=1, max=256)
    start: FloatVectorProperty(name="Start", size=3, default=(0.0, 0.0, 0.0), subtype="XYZ")
    direction: FloatVectorProperty(
        name="Direction", size=3, default=(0.0, 0.0, 1.0), subtype="XYZ"
    )
    bone_length: FloatProperty(name="Bone Length", default=0.25, min=0.0001)
    curve: FloatProperty(
        name="Curve",
        description="Per-bone bend applied around the X axis, in radians",
        default=0.0,
        subtype="ANGLE",
    )

    def init(self, context):
        self.inputs.new(BoneSocket.bl_idname, "Parent")
        self._multi_input(ConstraintSocket.bl_idname, "Tip Constraints")
        self.outputs.new(ChainSocket.bl_idname, "Chain")

    def draw_buttons(self, context, layout):
        layout.prop(self, "prefix")
        layout.prop(self, "count")
        col = layout.column(align=True)
        col.prop(self, "start")
        col.prop(self, "direction")
        layout.prop(self, "bone_length")
        layout.prop(self, "curve")

    def eval_bones(self, ctx):
        parents = gather_input_bones(self, "Parent", ctx)
        tip_constraints = gather_input_constraints(self, "Tip Constraints", ctx)

        direction = Vector(self.direction)
        if direction.length < 1e-8:
            direction = Vector((0.0, 0.0, 1.0))
        direction.normalize()

        bones = []
        head = Vector(self.start)
        prev_name = parents[-1].name if parents else None
        for i in range(self.count):
            if self.curve != 0.0 and i > 0:
                # Bend the running direction around global X per segment.
                from mathutils import Matrix

                direction = (Matrix.Rotation(self.curve, 4, "X") @ direction).normalized()
            tail = head + direction * self.bone_length
            bone = BoneDef(
                name=f"{self.prefix}.{i + 1:03d}",
                head=tuple(head),
                tail=tuple(tail),
                parent=prev_name,
                use_connect=i > 0,
            )
            bones.append(bone)
            prev_name = bone.name
            head = tail

        if bones and tip_constraints:
            bones[-1].constraints.extend(tip_constraints)
        return [copy_bone(b) for b in parents] + bones


class MirrorNode(ArmatureNodeBase, Node):
    """Outputs a mirrored copy of the input bones across an axis."""

    bl_idname = "ArmatureNodesMirrorNode"
    bl_label = "Mirror"
    bl_icon = "MOD_MIRROR"

    axis: EnumProperty(
        name="Axis",
        items=(
            ("X", "X", "Mirror across the YZ plane"),
            ("Y", "Y", "Mirror across the XZ plane"),
            ("Z", "Z", "Mirror across the XY plane"),
        ),
        default="X",
    )
    keep_original: BoolProperty(
        name="Keep Original",
        description="Output both the original and the mirrored bones",
        default=True,
    )

    def init(self, context):
        self.inputs.new(ChainSocket.bl_idname, "Bones")
        self.outputs.new(ChainSocket.bl_idname, "Bones")

    def draw_buttons(self, context, layout):
        layout.prop(self, "axis")
        layout.prop(self, "keep_original")

    @staticmethod
    def _flip_name(name):
        try:
            flipped = bpy.utils.flip_name(name)
        except AttributeError:
            flipped = name
        if flipped == name:
            flipped = name + ".mirror"
        return flipped

    def eval_bones(self, ctx):
        source = gather_input_bones(self, "Bones", ctx)
        idx = {"X": 0, "Y": 1, "Z": 2}[self.axis]
        source_names = {b.name for b in source}

        mirrored = []
        for b in source:
            m = copy_bone(b)
            m.name = self._flip_name(b.name)
            head = list(m.head)
            tail = list(m.tail)
            head[idx] = -head[idx]
            tail[idx] = -tail[idx]
            m.head = tuple(head)
            m.tail = tuple(tail)
            m.roll = -m.roll
            if m.parent and m.parent in source_names:
                m.parent = self._flip_name(m.parent)
            mirrored.append(m)

        if self.keep_original:
            return [copy_bone(b) for b in source] + mirrored
        return mirrored


class ParentNode(ArmatureNodeBase, Node):
    """Explicit hierarchy wiring when it is not implicit from chains."""

    bl_idname = "ArmatureNodesParentNode"
    bl_label = "Parent"
    bl_icon = "FILE_PARENT"

    use_connect: BoolProperty(name="Connected", default=False)

    def init(self, context):
        self.inputs.new(BoneSocket.bl_idname, "Parent")
        self._multi_input(ChainSocket.bl_idname, "Children")
        self.outputs.new(ChainSocket.bl_idname, "Bones")

    def draw_buttons(self, context, layout):
        layout.prop(self, "use_connect")

    def eval_bones(self, ctx):
        parents = gather_input_bones(self, "Parent", ctx)
        children = [copy_bone(b) for b in gather_input_bones(self, "Children", ctx)]
        if parents:
            parent_name = parents[-1].name
            child_names = {b.name for b in children}
            for b in children:
                # Only re-parent the roots of the incoming child set.
                if b.parent is None or b.parent not in child_names:
                    b.parent = parent_name
                    b.use_connect = self.use_connect
                    if self.use_connect:
                        b.head = tuple(parents[-1].tail)
        return [copy_bone(b) for b in parents] + children


class DeformGroupNode(ArmatureNodeBase, Node):
    """Marks bones as deform vs control-only; sets envelope params."""

    bl_idname = "ArmatureNodesDeformGroupNode"
    bl_label = "Deform Group"
    bl_icon = "MOD_VERTEX_WEIGHT"

    use_deform: BoolProperty(name="Deform", default=True)
    envelope_distance: FloatProperty(name="Envelope Distance", default=0.25, min=0.0)
    envelope_weight: FloatProperty(name="Envelope Weight", default=1.0, min=0.0)

    def init(self, context):
        self.inputs.new(ChainSocket.bl_idname, "Bones")
        self.outputs.new(ChainSocket.bl_idname, "Bones")

    def draw_buttons(self, context, layout):
        layout.prop(self, "use_deform")
        col = layout.column(align=True)
        col.enabled = self.use_deform
        col.prop(self, "envelope_distance")
        col.prop(self, "envelope_weight")

    def eval_bones(self, ctx):
        bones = [copy_bone(b) for b in gather_input_bones(self, "Bones", ctx)]
        for b in bones:
            b.use_deform = self.use_deform
            b.envelope_distance = self.envelope_distance
            b.envelope_weight = self.envelope_weight
        return bones


def _poll_widget_object(self, obj):
    return obj.type == "MESH"


# True while a node is being refreshed FROM the armature, so the property
# callbacks do not write the same values straight back to the bone.
_syncing_transform = False

_TRANSFORM_EPS = 1e-5


def _set_syncing(flag):
    global _syncing_transform
    _syncing_transform = flag


def _on_world_transform_changed(self, context):
    """User edited Position/Scale on a Custom Shape node: pose the bone."""
    if _syncing_transform:
        return
    try:
        self.apply_world_transform()
    except Exception as exc:  # noqa: BLE001
        print(f"[Armature Nodes] Could not apply world transform: {exc}")


def _on_rest_transform_changed(self, context):
    """User edited Head/Tail/Roll on a Custom Shape node.

    In Edit mode the edit bone is moved directly (live, no rebuild). In any
    other mode the rest geometry can only change through a rebuild, so fall
    back to the normal live-update path.
    """
    if _syncing_transform:
        return
    try:
        if self.apply_rest_transform():
            return
    except Exception as exc:  # noqa: BLE001
        print(f"[Armature Nodes] Could not apply rest transform: {exc}")
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


class CustomShapeNode(ArmatureNodeBase, Node):
    """Assigns a control widget (custom shape) from the WGTS_rig collection.

    Marks the incoming bones as control bones: non-deforming, with a
    ``WGT-rig_*`` wire mesh as their pose-mode display shape. The widget is
    either an existing object picked from WGTS_rig, or generated on build
    from a Rigify-style preset (circle, cube, sphere, ...).
    """

    bl_idname = "ArmatureNodesCustomShapeNode"
    bl_label = "Custom Shape"
    bl_icon = "MESH_CIRCLE"

    source: EnumProperty(
        name="Source",
        items=(
            ("PRESET", "Preset", "Generate a WGT-rig_<bone> widget per bone from a preset"),
            ("LIBRARY", "WGTS_rig", "Reuse one existing widget from the WGTS_rig collection"),
            ("OBJECT", "Object", "Use any mesh object as the widget"),
        ),
        default="PRESET",
    )
    preset: EnumProperty(name="Preset", items=_widget_preset_items, default=1)
    library_widget: EnumProperty(name="Widget", items=_widget_enum_items)
    widget_object: PointerProperty(
        name="Widget", type=bpy.types.Object, poll=_poll_widget_object
    )

    scale: FloatVectorProperty(name="Scale", size=3, default=(1.0, 1.0, 1.0), subtype="XYZ")
    translation: FloatVectorProperty(
        name="Translation", size=3, default=(0.0, 0.0, 0.0), subtype="TRANSLATION"
    )
    rotation: FloatVectorProperty(
        name="Rotation", size=3, default=(0.0, 0.0, 0.0), subtype="EULER"
    )
    wire_width: FloatProperty(name="Wire Width", default=1.0, min=1.0, max=16.0)
    scale_to_bone_length: BoolProperty(name="Scale to Bone Length", default=True)
    show_wire: BoolProperty(name="Wireframe", default=True)
    control_only: BoolProperty(
        name="Control Only",
        description="Also turn off Deform on these bones (they are controls, not deformers)",
        default=True,
    )
    bone_filter: StringProperty(
        name="Bone",
        description=(
            "Bone(s) this widget controls (semicolon separated for several). "
            "Empty = every incoming bone. Set automatically when decompiling "
            "a generated rig"
        ),
        default="",
    )

    # -- The controlled bone itself -----------------------------------------
    # Captured from the armature when the node is created by the decompiler
    # (or via Refresh). This makes the node OWN the bone: if nothing arrives
    # from upstream (e.g. the original rig was deleted) the node emits the
    # bone from this data, so the graph alone can rebuild the rig.
    # Head/Tail/Roll are the EDIT-mode (rest) transform, in armature space.
    # They follow the edit bone live while in Edit mode and move it when
    # edited there; outside Edit mode a change schedules a rebuild.
    bone_head: FloatVectorProperty(
        name="Head",
        description="Rest position of the bone head (Edit mode, armature space)",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="XYZ",
        update=_on_rest_transform_changed,
    )
    bone_tail: FloatVectorProperty(
        name="Tail",
        description="Rest position of the bone tail (Edit mode, armature space)",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="XYZ",
        update=_on_rest_transform_changed,
    )
    bone_roll: FloatProperty(
        name="Roll",
        default=0.0,
        subtype="ANGLE",
        update=_on_rest_transform_changed,
    )
    bone_parent: StringProperty(name="Parent", default="")
    bone_use_connect: BoolProperty(name="Connected", default=False)
    bone_use_deform: BoolProperty(name="Deform", default=False)
    bone_children: StringProperty(
        name="Children", description="Semicolon separated", default=""
    )
    # Full constraint stack as JSON: [{'type','name','params'}...]. Object
    # targets are stored by name and resolved when building.
    bone_constraints_json: StringProperty(name="Constraints", default="")
    bone_info_synced: BoolProperty(default=False, options={"HIDDEN"})

    # -- The widget's actual form ------------------------------------------
    # Mesh read off the widget object: {'verts': [...], 'edges': [...]}.
    # Used to recreate the exact widget when WGTS_rig no longer has it.
    widget_name: StringProperty(
        name="Widget Name",
        description="Object name the widget is (re)created under, e.g. WGT-rig_hand.L",
        default="",
    )
    widget_geometry_json: StringProperty(name="Widget Geometry", default="")

    # -- Live world-space transform of the controlled bone ------------------
    # Measured from the WORLD origin: armature object transform x current pose
    # (x rest geometry). Kept in sync from the viewport on every depsgraph
    # update and written back to the bone when edited here. These have their
    # own update callback so _inject_live_update leaves them alone: changing
    # them moves the bone directly instead of rebuilding the rig.
    world_position: FloatVectorProperty(
        name="Position",
        description=(
            "Location of the bone in world space (measured from the world "
            "origin, including the armature object transform and current pose)"
        ),
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="TRANSLATION",
        update=_on_world_transform_changed,
    )
    world_rotation: FloatVectorProperty(
        name="Rotation",
        description=(
            "Orientation of the bone in world space, as an XYZ Euler "
            "(measured from the world origin, including the armature object "
            "transform and current pose)"
        ),
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="EULER",
        update=_on_world_transform_changed,
    )
    world_scale: FloatVectorProperty(
        name="Scale",
        description="Scale of the bone in world space",
        size=3,
        default=(1.0, 1.0, 1.0),
        subtype="XYZ",
        update=_on_world_transform_changed,
    )
    transform_synced: BoolProperty(default=False, options={"HIDDEN"})

    show_bone_info: BoolProperty(name="Bone Info", default=False)
    show_widget_options: BoolProperty(name="Widget", default=False)

    def init(self, context):
        self.inputs.new(ChainSocket.bl_idname, "Bones")
        self.outputs.new(ChainSocket.bl_idname, "Bones")
        self.width = 240

    def filtered_names(self):
        return {n.strip() for n in self.bone_filter.split(";") if n.strip()}

    @property
    def controlled_bone(self):
        """The single bone name this node controls, or '' when it is a
        multi-bone filter / passthrough node."""
        names = [n.strip() for n in self.bone_filter.split(";") if n.strip()]
        return names[0] if len(names) == 1 else ""

    @property
    def widget_geometry(self):
        from .snapshot import loads

        return loads(self.widget_geometry_json)

    @property
    def stored_constraints(self):
        from .snapshot import loads

        return loads(self.bone_constraints_json, []) or []

    # -- Sync from a live armature -------------------------------------------

    def sync_bone_info(self, pbone):
        """Record everything about the controlled pose bone on this node."""
        from .snapshot import bone_roll, dumps, serialize_constraint

        bone = pbone.bone
        self.bone_filter = pbone.name
        rest = self.measure_rest_transform(obj=pbone.id_data)
        if rest is not None:
            head, tail, roll = rest
        else:
            head, tail, roll = bone.head_local, bone.tail_local, bone_roll(bone)
        _set_syncing(True)
        try:
            self.bone_head = tuple(head)
            self.bone_tail = tuple(tail)
            self.bone_roll = roll
        finally:
            _set_syncing(False)
        self.bone_parent = bone.parent.name if bone.parent else ""
        self.bone_use_connect = bone.use_connect
        self.bone_use_deform = bone.use_deform
        self.bone_children = ";".join(c.name for c in bone.children)
        self.bone_constraints_json = dumps(
            [serialize_constraint(c) for c in pbone.constraints]
        )
        self.bone_info_synced = True
        if not self.label:
            self.label = pbone.name
        if pbone.custom_shape is not None:
            self.read_widget(pbone)
        self.sync_world_transform(obj=pbone.id_data)

    # -- Live world-space transform ------------------------------------------

    def _pose_bone(self, obj=None):
        """(armature object, pose bone) for the controlled bone, or (None, None)."""
        name = self.controlled_bone
        if not name:
            return None, None
        # The caller passes the tree's armature, which is not necessarily the
        # one holding THIS node's bone -- resolve our own when it is not.
        if not self._armature_has_bone(obj, name):
            obj = self.find_armature()
        if obj is None or obj.pose is None:
            return None, None
        return obj, obj.pose.bones.get(name)

    def measure_world_transform(self, obj=None):
        """Return (position, rotation_euler, scale) of the controlled POSE bone
        measured from the world origin, or None when the bone cannot be found.

        This is a pose-mode value only: the full pose-space matrix
        (object x pose) is decomposed. In Edit mode the pose matrices are
        stale, so nothing is measured and the node keeps its last values.
        """
        obj, pbone = self._pose_bone(obj)
        if pbone is None or obj.mode == "EDIT":
            return None
        loc, rot, scl = (obj.matrix_world @ pbone.matrix).decompose()
        return loc, rot.to_euler("XYZ"), scl

    def sync_world_transform(self, obj=None):
        """Refresh Position/Scale FROM the armature. Returns True if they
        changed. Never triggers a rebuild or a write-back to the bone."""
        global _syncing_transform
        from .tree import suspend_live_update

        measured = self.measure_world_transform(obj)
        if measured is None:
            return False
        pos, rot, scl = measured
        if (
            self.transform_synced
            and (Vector(self.world_position) - pos).length < _TRANSFORM_EPS
            and (Vector(self.world_rotation) - Vector(rot)).length < _TRANSFORM_EPS
            and (Vector(self.world_scale) - scl).length < _TRANSFORM_EPS
        ):
            return False
        _syncing_transform = True
        try:
            with suspend_live_update():
                self.world_position = tuple(pos)
                self.world_rotation = (rot.x, rot.y, rot.z)
                self.world_scale = tuple(scl)
                self.transform_synced = True
        finally:
            _syncing_transform = False
        return True

    def apply_world_transform(self, obj=None):
        """Write Position/Rotation/Scale TO the POSE bone: place the controlled
        bone at ``world_position`` / ``world_rotation`` / ``world_scale`` in
        world space. This is a pose-mode transform (like grabbing/rotating the
        bone in Pose mode); the rest geometry in Edit mode is never touched.
        Returns True when a bone was moved."""
        from mathutils import Matrix, Euler

        obj, pbone = self._pose_bone(obj)
        if pbone is None:
            return False
        if obj.mode == "EDIT":
            print(
                "[Armature Nodes] Position/Rotation/Scale are pose values; "
                "leave Edit mode to move the bone"
            )
            return False
        target_pos = Vector(self.world_position)
        target_rot = Euler(self.world_rotation, "XYZ")
        target_scl = Vector(self.world_scale)
        inv_world = obj.matrix_world.inverted_safe()

        world = obj.matrix_world @ pbone.matrix
        cur_loc, cur_rot, cur_scl = world.decompose()
        if (
            (cur_loc - target_pos).length < _TRANSFORM_EPS
            and (Vector(cur_rot.to_euler("XYZ")) - Vector(target_rot)).length
            < _TRANSFORM_EPS
            and (cur_scl - target_scl).length < _TRANSFORM_EPS
        ):
            return False
        new_world = Matrix.LocRotScale(target_pos, target_rot, target_scl)
        pbone.matrix = inv_world @ new_world
        return True

    # -- Live rest (Edit mode) transform --------------------------------------

    def _edit_bone(self, obj=None):
        """(armature object, edit bone) while the armature is in Edit mode,
        otherwise (obj, None)."""
        name = self.controlled_bone
        if not name:
            return None, None
        if not self._armature_has_bone(obj, name):
            obj = self.find_armature()
        if obj is None or obj.mode != "EDIT" or not obj.data.is_editmode:
            return obj, None
        return obj, obj.data.edit_bones.get(name)

    def measure_rest_transform(self, obj=None):
        """Return (head, tail, roll) of the controlled bone's REST geometry in
        armature space, or None when the bone cannot be found.

        In Edit mode this is read live from the edit bone (so the node follows
        the bone while it is grabbed); in other modes from the stored rest
        data of the armature.
        """
        from .snapshot import bone_roll

        obj, ebone = self._edit_bone(obj)
        if obj is None:
            return None
        if ebone is not None:
            return Vector(ebone.head), Vector(ebone.tail), float(ebone.roll)
        bone = obj.data.bones.get(self.controlled_bone)
        if bone is None:
            return None
        return Vector(bone.head_local), Vector(bone.tail_local), bone_roll(bone)

    def sync_rest_transform(self, obj=None):
        """Refresh Head/Tail/Roll FROM the armature (live in Edit mode).
        Returns True if they changed. Never rebuilds or writes back."""
        from .tree import suspend_live_update

        measured = self.measure_rest_transform(obj)
        if measured is None:
            return False
        head, tail, roll = measured
        if (
            self.bone_info_synced
            and (Vector(self.bone_head) - head).length < _TRANSFORM_EPS
            and (Vector(self.bone_tail) - tail).length < _TRANSFORM_EPS
            and abs(self.bone_roll - roll) < _TRANSFORM_EPS
        ):
            return False
        _set_syncing(True)
        try:
            with suspend_live_update():
                self.bone_head = tuple(head)
                self.bone_tail = tuple(tail)
                self.bone_roll = roll
                self.bone_info_synced = True
        finally:
            _set_syncing(False)
        return True

    def apply_rest_transform(self, obj=None):
        """Write Head/Tail/Roll TO the edit bone. Only possible while the
        armature is in Edit mode; returns False otherwise so the caller can
        fall back to a rebuild."""
        obj, ebone = self._edit_bone(obj)
        if ebone is None:
            return False
        head = Vector(self.bone_head)
        tail = Vector(self.bone_tail)
        if (
            (Vector(ebone.head) - head).length < _TRANSFORM_EPS
            and (Vector(ebone.tail) - tail).length < _TRANSFORM_EPS
            and abs(ebone.roll - self.bone_roll) < _TRANSFORM_EPS
        ):
            return True
        if (tail - head).length < _TRANSFORM_EPS:
            return True  # zero-length bones are rejected by Blender; skip
        # A connected bone's head is pinned to its parent's tail: moving the
        # head means moving the parent's tail.
        if ebone.use_connect and ebone.parent is not None:
            ebone.parent.tail = head
        else:
            ebone.head = head
        ebone.tail = tail
        ebone.roll = self.bone_roll
        return True

    def set_world_transform(self, position=None, rotation=None, scale=None):
        """Set Position/Rotation/Scale in ONE step and push them to the pose
        bone once. Used by Paste so the bone does not get several partial
        component updates (which is what makes hover Ctrl+V inconsistent)."""
        global _syncing_transform
        from .tree import suspend_live_update

        _syncing_transform = True
        try:
            with suspend_live_update():
                if position is not None:
                    self.world_position = tuple(position)
                if rotation is not None:
                    self.world_rotation = tuple(rotation)
                if scale is not None:
                    self.world_scale = tuple(scale)
        finally:
            _syncing_transform = False
        return self.apply_world_transform()

    def world_transform_clipboard_text(self):
        """Position/Rotation/Scale serialised for the clipboard as
        ``pos: x, y, z | rot: x, y, z | scale: x, y, z``."""
        p = self.world_position
        r = self.world_rotation
        s = self.world_scale
        return (
            f"pos: {p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f} | "
            f"rot: {r[0]:.6f}, {r[1]:.6f}, {r[2]:.6f} | "
            f"scale: {s[0]:.6f}, {s[1]:.6f}, {s[2]:.6f}"
        )

    @staticmethod
    def parse_world_transform_text(text):
        """Parse clipboard text into (position, rotation, scale); any may be
        None.

        Accepts the format written by Copy, Blender's own vector copy
        (``[x, y, z]`` / ``Vector((x, y, z))``) and any loose ``x, y, z`` /
        ``x y z`` triplet (treated as a position). A 9-number bare list is
        read as position, rotation, scale in order; a 6-number list as
        position and scale (rotation omitted, backwards compatible).
        """
        import re

        if not text:
            return None, None, None
        text = text.strip()
        low = text.lower()
        nums = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

        def triplet(chunk):
            vals = [float(v) for v in nums.findall(chunk)]
            return Vector(vals[:3]) if len(vals) >= 3 else None

        if "pos" in low or "rot" in low or "scale" in low:
            pos = rot = scl = None
            for part in re.split(r"[|;\n]", text):
                lp = part.lower()
                if "pos" in lp or "loc" in lp:
                    pos = triplet(part)
                elif "rot" in lp:
                    rot = triplet(part)
                elif "scale" in lp or "scl" in lp:
                    scl = triplet(part)
            return pos, rot, scl
        vals = [float(v) for v in nums.findall(text)]
        if len(vals) >= 9:
            return Vector(vals[:3]), Vector(vals[3:6]), Vector(vals[6:9])
        if len(vals) >= 6:
            return Vector(vals[:3]), None, Vector(vals[3:6])
        if len(vals) >= 3:
            return Vector(vals[:3]), None, None
        return None, None, None

    def read_widget(self, pbone=None, obj=None):
        """Read the widget's form (mesh verts/edges) + display settings.

        Reads from the pose bone's custom shape when given, otherwise from an
        explicit mesh object, otherwise from whatever this node currently
        points at (library widget / object / stored name).
        """
        from .snapshot import dumps, mesh_geometry

        if pbone is not None and pbone.custom_shape is not None:
            obj = pbone.custom_shape
            self.scale = tuple(pbone.custom_shape_scale_xyz)
            self.translation = tuple(pbone.custom_shape_translation)
            self.rotation = tuple(pbone.custom_shape_rotation_euler)
            self.scale_to_bone_length = pbone.use_custom_shape_bone_size
            self.show_wire = pbone.bone.show_wire
            self.wire_width = float(getattr(pbone, "custom_shape_wire_width", 1.0))
        if obj is None:
            name = self._widget_name() or self.widget_name
            obj = bpy.data.objects.get(name) if name else None
        if obj is None or obj.type != "MESH":
            return False
        self.widget_name = obj.name
        self.widget_geometry_json = dumps(mesh_geometry(obj))
        self.source = "LIBRARY"
        try:
            self.library_widget = obj.name
        except TypeError:
            pass  # not in WGTS_rig; widget_name still resolves it at build
        return True

    def copy(self, node):
        """Keep the widget shape intact when Blender duplicates/pastes the node.

        ``library_widget`` is a dynamic EnumProperty, and Blender stores such
        enums by index — so copy/paste could silently repoint it at a different
        ``WGTS_rig`` mesh (or clear it), which is why some shapes (e.g. sphere,
        gear) appeared to change on paste. The plain-string ``widget_name`` and
        ``widget_geometry_json`` copy reliably by value, so re-pin the enum from
        ``widget_name`` (falling back to clearing it so ``_widget_name()`` uses
        the reliable name / stored geometry)."""
        # Blender normally copies RNA properties, but dynamic enums and ID
        # pointers are not portable across node-tree paste. Reassert every
        # serialized shape value explicitly before resolving the mesh.
        self.source = node.source
        self.preset = node.preset
        self.widget_name = node.widget_name
        self.widget_geometry_json = node.widget_geometry_json
        self.scale = tuple(node.scale)
        self.translation = tuple(node.translation)
        self.rotation = tuple(node.rotation)
        self.wire_width = node.wire_width
        self.scale_to_bone_length = node.scale_to_bone_length
        self.show_wire = node.show_wire
        self.control_only = node.control_only
        if self.source != "LIBRARY":
            return
        # Resolve the intended shape from the ORIGINAL node (``node``): the
        # stored name if it has one, else whatever its dropdown pointed at.
        # Nodes whose shape was picked from the dropdown never had
        # ``read_widget`` run, so ``widget_name`` can be empty there.
        name = node.widget_name or node.library_widget or self.widget_name
        if not name:
            return
        source_obj = bpy.data.objects.get(name)
        if source_obj is not None and source_obj.type == "MESH":
            # Re-read from the actual object (not its string name), but keep
            # the source node's serialized payload authoritative for paste.
            self.read_widget(obj=source_obj)
            self.widget_name = name
            self.widget_geometry_json = node.widget_geometry_json or self.widget_geometry_json
        else:
            # Cross-file paste: the WGTS_rig mesh is not here. Keep the stable
            # name and embedded geometry; never trust the dynamic enum index.
            self.widget_name = name
            self.widget_geometry_json = node.widget_geometry_json or self.widget_geometry_json
        # The enum is only a UI selector. Clearing it forces build/decompile
        # to use the stable name and embedded mesh instead of a shifted index.
        try:
            self.library_widget = ""
        except (TypeError, ValueError):
            pass

    @staticmethod
    def _armature_has_bone(obj, name):
        """True when ``obj`` is an armature carrying ``name``.

        Edit mode is checked first: a bone added or renamed in the current
        edit session only exists in ``edit_bones`` until the mode is left.
        """
        if obj is None or getattr(obj, "type", "") != "ARMATURE":
            return False
        if not name:
            return True
        data = obj.data
        if obj.mode == "EDIT" and data.is_editmode:
            if data.edit_bones.get(name) is not None:
                return True
        return data.bones.get(name) is not None

    def _upstream_armature(self, node=None, seen=None):
        """First live source armature reachable by walking the inputs up.

        Walks the whole upstream graph, not just the directly linked node:
        a Custom Shape sitting behind a Mirror / Parent / another Custom
        Shape still has to find the Armature Input feeding the chain.
        """
        node = self if node is None else node
        if seen is None:
            seen = set()
        if node.name in seen:
            return None  # graphs can be re-entered through several sockets
        seen.add(node.name)
        for sock in node.inputs:
            for link in sock.links:
                if not link.is_valid:
                    continue
                up = link.from_node
                # Only the Armature Input node's ``source`` is an object;
                # Custom Shape's own ``source`` is an enum string, which the
                # type check rejects.
                src = getattr(up, "source", None)
                if getattr(src, "type", "") == "ARMATURE":
                    return src
                found = self._upstream_armature(up, seen)
                if found is not None:
                    return found
        return None

    def find_armature(self):
        """Armature that owns the controlled bone.

        Resolved the same way the live sync does it (``sync.armature_for_tree``
        -- the Armature Input's source, else the Armature Output's object), so
        that editing Position/Rotation/Scale or Head/Tail on this node writes
        to the rig the tree is actually bound to. Without this a forward graph
        (an Output node and no Input node) resolved nothing here, and the
        scene-wide scan below would pose whichever armature happened to own a
        bone of the same name first.
        """
        from .sync import armature_for_tree

        name = self.controlled_bone
        tree = self.id_data
        if tree is not None:
            obj = armature_for_tree(tree)
            if self._armature_has_bone(obj, name):
                return obj
        obj = self._upstream_armature()
        if self._armature_has_bone(obj, name):
            return obj
        if name:
            for obj in bpy.data.objects:
                if obj.type == "ARMATURE" and obj.pose and name in obj.pose.bones:
                    return obj
        return None

    def refresh_bone_info(self):
        obj = self.find_armature()
        name = self.controlled_bone
        if obj is None or not name:
            return False
        pbone = obj.pose.bones.get(name)
        if pbone is None:
            return False
        self.sync_bone_info(pbone)
        return True

    # -- UI --------------------------------------------------------------------

    def _split_list(self, value):
        return [v.strip() for v in value.split(";") if v.strip()]

    def draw_buttons(self, context, layout):
        from .snapshot import describe_geometry

        layout.context_pointer_set("node", self)

        # Visible, adjustable part: which bone, and where it is in the WORLD
        # (live: follows the viewport, and moves the bone when edited).
        box = layout.box()
        header = box.row(align=True)
        header.prop(self, "bone_filter", text="", icon="BONE_DATA")
        header.operator("armature_nodes.copy_bone_transform", text="", icon="COPYDOWN")
        header.operator("armature_nodes.paste_bone_transform", text="", icon="PASTEDOWN")
        obj = self.find_armature() if self.controlled_bone else None
        in_edit = obj is not None and obj.mode == "EDIT"

        # Pose mode: world-space Position/Scale of the posed bone.
        pose_col = box.column(align=True)
        pose_col.label(text="Pose (world)", icon="POSE_HLT")
        pose_col.enabled = bool(self.controlled_bone) and not in_edit
        pose_col.prop(self, "world_position", text="Position")
        pose_col.prop(self, "world_rotation", text="Rotation")
        pose_col.prop(self, "world_scale", text="Scale")

        # Edit mode: rest Head/Tail of the bone in armature space.
        rest_col = box.column(align=True)
        rest_col.label(text="Rest (edit)", icon="EDITMODE_HLT")
        rest_col.enabled = bool(self.controlled_bone)
        rest_col.prop(self, "bone_head", text="Head")
        rest_col.prop(self, "bone_tail", text="Tail")

        if self.controlled_bone:
            if obj is None:
                box.label(text="No armature found for this bone", icon="INFO")
            elif in_edit:
                box.label(text="Edit mode: Head/Tail are live", icon="INFO")
            else:
                box.label(
                    text="Pose mode: Position/Rotation/Scale are live", icon="INFO"
                )

        # Background: everything else we know about the controlled bone.
        row = layout.row(align=True)
        row.prop(
            self,
            "show_bone_info",
            icon="TRIA_DOWN" if self.show_bone_info else "TRIA_RIGHT",
            emboss=False,
        )
        if self.show_bone_info:
            info = layout.box()
            if not self.bone_info_synced:
                info.label(text="Not synced from an armature yet", icon="INFO")
            col = info.column(align=True)
            col.prop(self, "bone_roll")
            row = info.row(align=True)
            row.prop(self, "bone_use_connect")
            row.prop(self, "bone_use_deform")
            info.prop(self, "bone_parent", icon="CON_CHILDOF")
            children = self._split_list(self.bone_children)
            info.label(
                text=f"Children: {len(children)}" if children else "Children: (none)",
                icon="GROUP_BONE",
            )
            for child in children:
                info.label(text=child, icon="BLANK1")
            cons = self.stored_constraints
            info.label(
                text=f"Constraints: {len(cons)}" if cons else "Constraints: (none)",
                icon="CONSTRAINT_BONE",
            )
            for con in cons:
                params = con.get("params", {})
                text = f"{con.get('name', con['type'])} ({con['type']})"
                target = params.get("target")
                if target:
                    text += f" -> {target}"
                    if params.get("subtarget"):
                        text += f":{params['subtarget']}"
                info.label(text=text, icon="BLANK1")

        # Widget form / source / display options.
        row = layout.row(align=True)
        row.prop(
            self,
            "show_widget_options",
            icon="TRIA_DOWN" if self.show_widget_options else "TRIA_RIGHT",
            emboss=False,
        )
        if self.show_widget_options:
            wgt = layout.box()
            geo = self.widget_geometry
            if geo:
                wgt.label(text=self.widget_name or "(unnamed widget)", icon="MESH_DATA")
                wgt.label(text=describe_geometry(geo), icon="BLANK1")
            else:
                wgt.label(text="Widget form not read yet", icon="INFO")
            wgt.prop(self, "source", text="")
            if self.source == "PRESET":
                wgt.prop(self, "preset", text="")
            elif self.source == "LIBRARY":
                wgt.prop(self, "library_widget", text="")
                wgt.prop(self, "preset", text="Fallback")
            else:
                wgt.prop(self, "widget_object", text="")
            # Widget placement relative to the bone (Blender's custom shape
            # offset/scale/rotation), as opposed to the bone's world transform.
            col = wgt.column(align=True)
            col.prop(self, "translation", text="Offset")
            col.prop(self, "scale", text="Widget Scale")
            col.prop(self, "rotation")
            wgt.prop(self, "scale_to_bone_length")
            row = wgt.row(align=True)
            row.prop(self, "show_wire")
            row.prop(self, "wire_width", text="Width")
            wgt.prop(self, "control_only")

    def _widget_name(self):
        if self.source == "LIBRARY":
            return self.library_widget or self.widget_name or ""
        if self.source == "OBJECT" and self.widget_object:
            return self.widget_object.name
        return self.widget_name or ""  # PRESET without a stored name: per-bone WGT-rig_<bone>

    def _own_bone(self):
        """BoneDef for the controlled bone, from the stored snapshot."""
        from .snapshot import constraint_from_dict

        name = self.controlled_bone
        if not name or not self.bone_info_synced:
            return None
        return BoneDef(
            name=name,
            head=tuple(self.bone_head),
            tail=tuple(self.bone_tail),
            roll=self.bone_roll,
            parent=self.bone_parent or None,
            use_connect=self.bone_use_connect,
            use_deform=self.bone_use_deform,
            constraints=[constraint_from_dict(c) for c in self.stored_constraints],
        )

    def eval_bones(self, ctx):
        from .core import ShapeDef

        bones = [copy_bone(b) for b in gather_input_bones(self, "Bones", ctx)]
        names = self.filtered_names()

        # The node owns its controlled bone. If it did not come from upstream
        # (rig deleted, input unlinked, ...) emit it from the stored data; if
        # it did, the node's editable bone values override the upstream copy
        # so tweaking Head/Tail/Parent here changes the armature in real time.
        own = self._own_bone()
        if own is not None:
            for b in bones:
                if b.name == own.name:
                    b.head, b.tail, b.roll = own.head, own.tail, own.roll
                    b.parent = own.parent
                    b.use_connect = own.use_connect
                    b.use_deform = own.use_deform
                    if own.constraints:
                        b.constraints = own.constraints
                    break
            else:
                bones.append(own)

        # What the widget IS depends on Source, so a change of Source or
        # Preset always resolves to a different form (otherwise the stored
        # decompiled name + captured geometry would win forever and the node
        # would look frozen):
        #   PRESET  -> per-bone graph-owned WGT-rig_<bone> built from the preset
        #   LIBRARY -> the chosen WGTS_rig object as-is (geometry only used to
        #              recreate it if it went missing; preset is the fallback)
        #   OBJECT  -> any mesh object as-is
        if self.source == "PRESET":
            widget = ""
            preset = self.preset
            geometry = None
        elif self.source == "LIBRARY":
            widget = self.library_widget or self.widget_name or ""
            preset = self.preset
            geometry = self.widget_geometry
        else:
            widget = self.widget_object.name if self.widget_object else (self.widget_name or "")
            preset = "NONE"
            geometry = self.widget_geometry
        for b in bones:
            if names and b.name not in names:
                continue
            b.shape = ShapeDef(
                widget=widget,
                preset=preset,
                scale=tuple(self.scale),
                translation=tuple(self.translation),
                rotation=tuple(self.rotation),
                wire_width=self.wire_width,
                scale_to_bone_length=self.scale_to_bone_length,
                show_wire=self.show_wire,
                geometry=geometry,
            )
            if self.control_only:
                b.use_deform = False
        return bones


# ---------------------------------------------------------------------------
# Primary Rig (MediaPipe-style 33-point marker skeleton)
# ---------------------------------------------------------------------------


_syncing_landmarks = False


def _on_landmark_changed(self, context):
    """A landmark was typed on the node: move its handle and rebuild."""
    if _syncing_landmarks:
        return
    from .primary_rig import tag_viewports_redraw

    self.push_landmarks_to_empties()
    tag_viewports_redraw()
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


def _on_marker_lock_changed(self, context):
    """Lock Depth / Symmetric toggled: re-apply locks to the handles."""
    from .primary_rig import apply_marker_locks, find_marker_empties, tag_viewports_redraw

    if self.symmetric:
        self.mirror_markers("L_TO_R")
    for key, obj in find_marker_empties(self).items():
        apply_marker_locks(self, obj, key)
    tag_viewports_redraw()


def _on_marker_rotation_toggled(self, context):
    """Rotation enabled/disabled on one landmark: re-lock and redraw its
    handle, then rebuild (roll and retarget twist both change)."""
    from .primary_rig import apply_marker_locks, find_marker_empties, tag_viewports_redraw

    for key, obj in find_marker_empties(self).items():
        apply_marker_locks(self, obj, key)
        if self.marker_uses_rotation(key):
            obj.rotation_euler = self.marker_rotation(key)
    tag_viewports_redraw()
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


def _on_overlay_changed(self, context):
    from .primary_rig import tag_viewports_redraw

    tag_viewports_redraw()


def _landmark_prop(key):
    from .primary_rig import LM_DEFAULTS, LM_INDEX, LM_LABELS

    return FloatVectorProperty(
        name=LM_LABELS[key],
        description=(
            f"MediaPipe landmark {LM_INDEX[key]} ({LM_LABELS[key]}), world space. "
            "Drag its handle in the viewport or type a value here."
        ),
        size=3,
        subtype="TRANSLATION",
        default=LM_DEFAULTS[key],
        update=_on_landmark_changed,
    )


def _landmark_rotation_prop(key):
    from .primary_rig import LM_LABELS

    return FloatVectorProperty(
        name=f"{LM_LABELS[key]} Rotation",
        description=(
            f"Orientation of the {LM_LABELS[key]} landmark. Supplies the roll "
            "of the bone starting here and its twist when retargeting"
        ),
        size=3,
        subtype="EULER",
        default=(0.0, 0.0, 0.0),
        update=_on_landmark_changed,
    )


def _landmark_use_rotation_prop(key):
    from .primary_rig import LM_LABELS

    return BoolProperty(
        name=f"{LM_LABELS[key]} Rotation",
        description=(
            "Adjust this landmark's rotation as well as its position. Off by "
            "default: the handle is position-only until this is enabled"
        ),
        default=False,
        update=_on_marker_rotation_toggled,
    )


class PrimaryRigNode(ArmatureNodeBase, Node):
    """Marker skeleton placed from MediaPipe's 33 pose landmarks.

    The landmarks are drawn in the viewport as a MediaPipe-style skeleton
    (orange left, cyan right) with one draggable handle per point. Position is
    always adjustable; rotation can be switched on per landmark, on any of the
    33, and then feeds bone roll and the retarget twist.

    Two outputs: *Skeleton* is the bones themselves (wire it into an Armature
    Output), *Rig* hands the same skeleton to an Armature Input node, which
    matches it onto an existing Rigify rig's controls and drives them.
    """

    bl_idname = "ArmatureNodesPrimaryRigNode"
    bl_label = "Primary Rig"
    bl_icon = "OUTLINER_OB_ARMATURE"

    lock_depth: BoolProperty(
        name="Lock Depth (2D)",
        description="Handles only move in X/Z (front-view adjustment)",
        default=True,
        update=_on_marker_lock_changed,
    )
    symmetric: BoolProperty(
        name="Symmetric",
        description="Right-side landmarks are locked and mirror the left side",
        default=True,
        update=_on_marker_lock_changed,
    )
    mirror_center_x: FloatProperty(
        name="Mirror X",
        description="World X of the mirror plane used by Symmetric mode",
        default=0.0,
    )
    show_skeleton: BoolProperty(
        name="Skeleton",
        description="Draw the landmark skeleton in the 3D viewport",
        default=True,
        update=_on_overlay_changed,
    )
    show_markers: BoolProperty(name="Landmarks", default=False)
    show_detail: BoolProperty(name="Face / Hands / Feet", default=False)
    show_advanced: BoolProperty(name="Advanced", default=False)

    def init(self, context):
        self.inputs.new(BoneSocket.bl_idname, "Parent")
        self.outputs.new(ChainSocket.bl_idname, "Skeleton")
        self.outputs.new(PoseSocket.bl_idname, "Rig")
        self.width = 320

    def ensure_sockets(self):
        """Migrate nodes saved before the Skeleton / Rig split."""
        old = self.outputs.get("Bones")
        if old is not None and self.outputs.get("Skeleton") is None:
            old.name = "Skeleton"
        if self.outputs.get("Skeleton") is None:
            self.outputs.new(ChainSocket.bl_idname, "Skeleton")
        if self.outputs.get("Rig") is None:
            self.outputs.new(PoseSocket.bl_idname, "Rig")

    def free(self):
        from .primary_rig import remove_marker_empties

        try:
            remove_marker_empties(self)
        except Exception as exc:  # noqa: BLE001
            print(f"[Armature Nodes] Could not clean up landmark handles: {exc}")
        self.schedule_rebuild()

    # -- Landmark access -------------------------------------------------------

    def landmark(self, key):
        from .primary_rig import LM_PROP

        return Vector(getattr(self, LM_PROP[key]))

    def landmarks(self):
        from .primary_rig import LM_KEYS

        return {k: self.landmark(k) for k in LM_KEYS}

    def marker_uses_rotation(self, key):
        from .primary_rig import LM_USE_ROT_PROP

        return bool(getattr(self, LM_USE_ROT_PROP[key], False))

    def marker_rotation(self, key):
        from .primary_rig import LM_ROT_PROP

        return Vector(getattr(self, LM_ROT_PROP[key]))

    def marker_rotations(self):
        """Euler per landmark, only for the landmarks with rotation enabled."""
        from .primary_rig import LM_KEYS

        return {
            k: tuple(self.marker_rotation(k))
            for k in LM_KEYS
            if self.marker_uses_rotation(k)
        }

    def set_landmarks(self, values, rotations=None, push=True):
        """Write several landmarks at once without per-property rebuilds."""
        global _syncing_landmarks
        from .primary_rig import LM_PROP, LM_ROT_PROP
        from .tree import suspend_live_update

        _syncing_landmarks = True
        try:
            with suspend_live_update():
                for key, vec in values.items():
                    setattr(self, LM_PROP[key], tuple(vec))
                for key, euler in (rotations or {}).items():
                    setattr(self, LM_ROT_PROP[key], tuple(euler))
        finally:
            _syncing_landmarks = False
        if push:
            self.push_landmarks_to_empties()

    def effective_height(self):
        from .primary_rig import DEFAULT_HEIGHT

        zs = [v.z for v in self.landmarks().values()]
        height = max(zs) - min(zs)
        return height if height > 1e-3 else DEFAULT_HEIGHT

    def center_x(self):
        """Mirror plane for Symmetric mode."""
        return float(self.mirror_center_x)

    def _mirrored(self, changes, rotations=None):
        """Add the right-side mirrors of every left-side entry."""
        from .primary_rig import LM_MIRROR, LM_SIDE, mirror_point, mirror_rotation

        mid = self.center_x()
        out = dict(changes)
        rot_out = dict(rotations or {})
        for key, loc in changes.items():
            if LM_SIDE[key] == "L":
                out[LM_MIRROR[key]] = mirror_point(loc, mid)
        for key, euler in (rotations or {}).items():
            if LM_SIDE[key] == "L":
                rot_out[LM_MIRROR[key]] = mirror_rotation(euler)
        return out, rot_out

    def _with_group_followers(self, changes):
        """Face / finger / toe landmarks move rigidly with their anchor."""
        from .primary_rig import RIGID_GROUPS

        out = dict(changes)
        for anchor, members in RIGID_GROUPS.items():
            if anchor not in changes:
                continue
            delta = changes[anchor] - self.landmark(anchor)
            for m in members:
                if m not in changes:
                    out[m] = self.landmark(m) + delta
        return out

    # -- Viewport handles ------------------------------------------------------

    def markers_shown(self):
        from .primary_rig import find_marker_empties

        return bool(find_marker_empties(self))

    def push_landmarks_to_empties(self):
        from .primary_rig import find_marker_empties

        for key, obj in find_marker_empties(self).items():
            loc = self.landmark(key)
            if (Vector(obj.location) - loc).length > _TRANSFORM_EPS:
                obj.location = loc
            if self.marker_uses_rotation(key):
                rot = self.marker_rotation(key)
                if (Vector(obj.rotation_euler) - rot).length > _TRANSFORM_EPS:
                    obj.rotation_euler = rot

    def sync_from_empties(self):
        """Read dragged handles back into the node. Returns True if anything moved."""
        from .primary_rig import find_marker_empties

        empties = find_marker_empties(self)
        if not empties:
            return False
        moved, turned = {}, {}
        for key, obj in empties.items():
            loc = Vector(obj.location)
            if (self.landmark(key) - loc).length > _TRANSFORM_EPS:
                moved[key] = loc
            if self.marker_uses_rotation(key):
                rot = Vector(obj.rotation_euler)
                if (self.marker_rotation(key) - rot).length > _TRANSFORM_EPS:
                    turned[key] = tuple(rot)
        if not moved and not turned:
            return False
        changes = self._with_group_followers(moved)
        if self.symmetric:
            changes, turned = self._mirrored(changes, turned)
        self.set_landmarks(changes, rotations=turned, push=True)
        tree = self.id_data
        if tree is not None and hasattr(tree, "mark_dirty"):
            tree.mark_dirty()
        return True

    def mirror_markers(self, direction="L_TO_R"):
        from .primary_rig import (
            LM_KEYS,
            LM_MIRROR,
            LM_SIDE,
            mirror_point,
            mirror_rotation,
        )

        src = "L" if direction == "L_TO_R" else "R"
        mid = self.center_x()
        changes = {
            LM_MIRROR[k]: mirror_point(self.landmark(k), mid)
            for k in LM_KEYS
            if LM_SIDE[k] == src
        }
        rotations = {
            LM_MIRROR[k]: mirror_rotation(self.marker_rotation(k))
            for k in LM_KEYS
            if LM_SIDE[k] == src and self.marker_uses_rotation(k)
        }
        self.set_landmarks(changes, rotations=rotations)

    # -- UI -------------------------------------------------------------------

    def _draw_landmark_rows(self, layout, keys):
        from .primary_rig import LM_LABELS, LM_PROP, LM_ROT_PROP, LM_SIDE

        for key in keys:
            enabled = not (self.symmetric and LM_SIDE[key] == "R")
            use_rot = self.marker_uses_rotation(key)
            row = layout.row(align=True)
            row.enabled = enabled
            row.prop(self, LM_PROP[key], text=LM_LABELS[key])
            op = row.operator(
                "armature_nodes.primary_rig_toggle_rotation",
                text="",
                icon="ORIENTATION_GIMBAL",
                depress=use_rot,
            )
            op.marker = key
            if use_rot:
                sub = layout.row(align=True)
                sub.enabled = enabled
                sub.prop(self, LM_ROT_PROP[key], text="")

    def draw_buttons(self, context, layout):
        from .primary_rig import GROUP_ANCHOR, PRIMARY_KEYS

        self.ensure_sockets()
        layout.context_pointer_set("node", self)

        col = layout.column(align=True)
        row = col.row(align=True)
        shown = self.markers_shown()
        row.operator(
            "armature_nodes.primary_rig_toggle_markers",
            text=("Hide" if shown else "Show") + " Landmarks",
            icon="HIDE_OFF" if shown else "HIDE_ON",
        )
        row.operator("armature_nodes.primary_rig_front_view", icon="VIEW_ORTHO")
        row = col.row(align=True)
        row.prop(self, "lock_depth", toggle=True)
        row.prop(self, "symmetric", toggle=True)
        row.prop(self, "show_skeleton", toggle=True, icon="ARMATURE_DATA")

        row = layout.row(align=True)
        row.prop(self, "show_markers", icon="TRIA_DOWN" if self.show_markers else "TRIA_RIGHT", emboss=False)
        if self.show_markers:
            box = layout.box()
            self._draw_landmark_rows(box, PRIMARY_KEYS)
            row = box.row(align=True)
            row.prop(self, "show_detail", icon="TRIA_DOWN" if self.show_detail else "TRIA_RIGHT", emboss=False)
            if self.show_detail:
                self._draw_landmark_rows(box.box(), tuple(GROUP_ANCHOR))

        row = layout.row(align=True)
        row.prop(self, "show_advanced", icon="TRIA_DOWN" if self.show_advanced else "TRIA_RIGHT", emboss=False)
        if self.show_advanced:
            box = layout.box()
            box.prop(self, "mirror_center_x")
            row = box.row(align=True)
            row.operator("armature_nodes.primary_rig_mirror", text="Mirror L > R").direction = "L_TO_R"
            row.operator("armature_nodes.primary_rig_mirror", text="Mirror R > L").direction = "R_TO_L"

    # -- Evaluation -----------------------------------------------------------

    def _skeleton_bones(self):
        from .primary_rig import primary_rig_bones

        landmarks = self.landmarks()
        floor_z = min(v.z for v in landmarks.values())
        return primary_rig_bones(
            landmarks,
            self.effective_height(),
            floor_z,
            rotations=self.marker_rotations(),
        )

    def eval_bones(self, ctx):
        """Skeleton output: the marker skeleton, optionally re-rooted."""
        parents = gather_input_bones(self, "Parent", ctx)
        bones = self._skeleton_bones()
        if parents:
            root = parents[-1].name
            for b in bones:
                if b.parent is None:
                    b.parent = root
        return [copy_bone(b) for b in parents] + bones

    def eval_pose(self, ctx):
        """Rig output: the skeleton as a pose to drive an existing rig with.

        No parent re-rooting here -- the rig owns its own hierarchy; only the
        bone transforms are handed over.
        """
        return self._skeleton_bones()


def _add_landmark_properties(cls):
    """Position, rotation and rotation-enabled properties per landmark."""
    from .primary_rig import LM_KEYS, LM_PROP, LM_ROT_PROP, LM_USE_ROT_PROP

    for key in LM_KEYS:
        cls.__annotations__[LM_PROP[key]] = _landmark_prop(key)
        cls.__annotations__[LM_ROT_PROP[key]] = _landmark_rotation_prop(key)
        cls.__annotations__[LM_USE_ROT_PROP[key]] = _landmark_use_rotation_prop(key)


_add_landmark_properties(PrimaryRigNode)

# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class IKConstraintNode(ArmatureNodeBase, Node):
    """IK constraint: target, pole target, chain length, iterations, stretch."""

    bl_idname = "ArmatureNodesIKConstraintNode"
    bl_label = "IK Constraint"
    bl_icon = "CON_KINEMATIC"

    target: PointerProperty(name="Target", type=bpy.types.Object)
    subtarget: StringProperty(
        name="Target Bone",
        description="If the target is an armature, the bone to track",
        default="",
    )
    pole_target: PointerProperty(name="Pole Target", type=bpy.types.Object)
    pole_angle: FloatProperty(name="Pole Angle", default=0.0, subtype="ANGLE")
    chain_count: IntProperty(
        name="Chain Length",
        description="How many bones are included in the IK effect (0 = all)",
        default=2,
        min=0,
    )
    iterations: IntProperty(name="Iterations", default=500, min=1, max=10000)
    use_stretch: BoolProperty(name="Stretch", default=False)
    influence: FloatProperty(name="Influence", default=1.0, min=0.0, max=1.0)

    def init(self, context):
        self.outputs.new(ConstraintSocket.bl_idname, "Constraint")

    def draw_buttons(self, context, layout):
        layout.prop(self, "target")
        if self.target and self.target.type == "ARMATURE":
            layout.prop_search(self, "subtarget", self.target.data, "bones")
        layout.prop(self, "pole_target")
        if self.pole_target:
            layout.prop(self, "pole_angle")
        layout.prop(self, "chain_count")
        layout.prop(self, "iterations")
        layout.prop(self, "use_stretch")
        layout.prop(self, "influence")

    def eval_constraints(self, ctx):
        params = {
            "chain_count": self.chain_count,
            "iterations": self.iterations,
            "use_stretch": self.use_stretch,
            "influence": self.influence,
        }
        if self.target:
            params["target"] = self.target.name
            if self.subtarget:
                params["subtarget"] = self.subtarget
        if self.pole_target:
            params["pole_target"] = self.pole_target.name
            params["pole_angle"] = self.pole_angle
        return [ConstraintDef(type="IK", name="IK", params=params)]


_GENERIC_CONSTRAINT_ITEMS = (
    ("COPY_ROTATION", "Copy Rotation", ""),
    ("COPY_LOCATION", "Copy Location", ""),
    ("COPY_SCALE", "Copy Scale", ""),
    ("COPY_TRANSFORMS", "Copy Transforms", ""),
    ("LIMIT_ROTATION", "Limit Rotation", ""),
    ("LIMIT_LOCATION", "Limit Location", ""),
    ("TRACK_TO", "Track To", ""),
    ("DAMPED_TRACK", "Damped Track", ""),
    ("STRETCH_TO", "Stretch To", ""),
)

_TARGETLESS_TYPES = {"LIMIT_ROTATION", "LIMIT_LOCATION"}


class GenericConstraintNode(ArmatureNodeBase, Node):
    """One node covering the common target-based / limit constraints."""

    bl_idname = "ArmatureNodesGenericConstraintNode"
    bl_label = "Constraint"
    bl_icon = "CONSTRAINT"

    constraint_type: EnumProperty(
        name="Type", items=_GENERIC_CONSTRAINT_ITEMS, default="COPY_ROTATION"
    )
    target: PointerProperty(name="Target", type=bpy.types.Object)
    subtarget: StringProperty(name="Target Bone", default="")
    influence: FloatProperty(name="Influence", default=1.0, min=0.0, max=1.0)

    def init(self, context):
        self.outputs.new(ConstraintSocket.bl_idname, "Constraint")

    def draw_buttons(self, context, layout):
        layout.prop(self, "constraint_type", text="")
        if self.constraint_type not in _TARGETLESS_TYPES:
            layout.prop(self, "target")
            if self.target and self.target.type == "ARMATURE":
                layout.prop_search(self, "subtarget", self.target.data, "bones")
        layout.prop(self, "influence")

    def eval_constraints(self, ctx):
        params = {"influence": self.influence}
        if self.constraint_type not in _TARGETLESS_TYPES and self.target:
            params["target"] = self.target.name
            if self.subtarget:
                params["subtarget"] = self.subtarget
        return [
            ConstraintDef(
                type=self.constraint_type,
                name=self.constraint_type.replace("_", " ").title(),
                params=params,
            )
        ]


# ---------------------------------------------------------------------------
# Armature I/O
# ---------------------------------------------------------------------------


class ArmatureOutputNode(ArmatureNodeBase, Node):
    """Terminal node: the Build operator evaluates from here."""

    bl_idname = "ArmatureNodesOutputNode"
    bl_label = "Armature Output"
    bl_icon = "ARMATURE_DATA"

    armature_name: StringProperty(name="Armature Name", default="Armature")
    mode: EnumProperty(
        name="Mode",
        items=(
            ("FULL", "Full Rig", "Create or update bones, constraints and shapes"),
            (
                "SHAPES_ONLY",
                "Custom Shapes Only",
                "Only assign custom shapes on an existing armature (e.g. a "
                "generated Rigify rig); bones, constraints and drivers are left alone",
            ),
        ),
        default="FULL",
    )

    def init(self, context):
        self._multi_input(ChainSocket.bl_idname, "Bones")
        self.width = 200

    def draw_buttons(self, context, layout):
        layout.prop(self, "armature_name")
        layout.prop(self, "mode", text="")

    def free(self):
        """Deleting the Output node deletes what it generated.

        Only objects tagged as owned by *this* node are removed -- a rig the
        user built and the graph merely writes into (Custom Shapes Only mode,
        or an object that already existed under this name) is never touched.
        """
        from .build import owned_objects
        from .tree import queue_object_removal

        tree = self.id_data
        try:
            for obj in owned_objects(tree.name, self.name):
                queue_object_removal(obj.name)
        except (ReferenceError, AttributeError) as exc:
            print(f"[Armature Nodes] Could not release generated armature: {exc}")
        self.schedule_rebuild()

    def eval_bones(self, ctx):
        return gather_input_bones(self, "Bones", ctx)


def _poll_armature_object(self, obj):
    return obj.type == "ARMATURE"


def _skeleton_bone_items(self, context):
    """Every bone the marker skeleton can emit, for the override dropdown."""
    from .primary_rig import skeleton_key_map

    items = [("", "Skeleton Bone...", "")]
    for name in sorted(skeleton_key_map()):
        items.append((name, name, f"Override the control matched to {name}"))
    return items


def _on_override_changed(self, context):
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


class BoneOverride(bpy.types.PropertyGroup):
    """One manual skeleton-bone -> control-bone pairing.

    Takes priority over the automatic match table for that bone, whether or
    not the automatic match found something: the table's first choice is not
    always the control you want driven.
    """

    skeleton_bone: EnumProperty(
        name="Skeleton Bone",
        description="Marker-skeleton bone whose match is being overridden",
        items=_skeleton_bone_items,
        update=_on_override_changed,
    )
    target_bone: StringProperty(
        name="Control Bone",
        description="Bone on the rig to drive instead of the automatic match",
        default="",
        update=_on_override_changed,
    )
    enabled: BoolProperty(
        name="Enabled",
        description="Use this override",
        default=True,
        update=_on_override_changed,
    )


class ArmatureInputNode(ArmatureNodeBase, Node):
    """Entry point for the reverse direction: references a source armature."""

    bl_idname = "ArmatureNodesInputNode"
    bl_label = "Armature Input"
    bl_icon = "OUTLINER_OB_ARMATURE"

    source: PointerProperty(
        name="Source", type=bpy.types.Object, poll=_poll_armature_object
    )
    # Full serialized copy of the source armature (every bone, constraint and
    # widget mesh). Taken when decompiling so the graph keeps working after
    # the source object is deleted.
    snapshot_json: StringProperty(name="Snapshot", default="")
    snapshot_name: StringProperty(name="Snapshot Of", default="")
    use_snapshot: BoolProperty(
        name="Prefer Snapshot",
        description=(
            "Evaluate from the stored snapshot even when the source object "
            "exists (the snapshot is always used when it does not)"
        ),
        default=False,
    )

    retarget_mode: EnumProperty(
        name="Drive Mode",
        description=(
            "Which control chain the marker skeleton drives. FK maps one "
            "skeleton bone onto one FK control; IK drives the hand/foot IK "
            "controls and places the elbow/knee pole targets"
        ),
        items=(
            ("FK", "FK", "Drive the FK controls (upper_arm_fk, thigh_fk, ...)"),
            ("IK", "IK", "Drive the IK controls and pole targets"),
        ),
        default="FK",
    )
    retarget_enabled: BoolProperty(
        name="Drive Rig",
        description=(
            "Apply the wired-in marker skeleton to this rig's controls on "
            "every update"
        ),
        default=True,
    )
    retarget_switch_ikfk: BoolProperty(
        name="Set IK/FK Switch",
        description=(
            "Also set Rigify's IK/FK slider on each limb so the driven chain "
            "is the one the rig follows"
        ),
        default=True,
    )
    bone_overrides: bpy.props.CollectionProperty(type=BoneOverride)
    show_overrides: BoolProperty(name="Bone Overrides", default=False)
    match_report: StringProperty(name="Match", default="", options={"HIDDEN"})

    def init(self, context):
        self.outputs.new(ChainSocket.bl_idname, "Bones")
        self.inputs.new(PoseSocket.bl_idname, "Skeleton")
        self.width = 220

    def ensure_sockets(self):
        """Migrate nodes saved before the Skeleton input existed."""
        if self.inputs.get("Skeleton") is None:
            self.inputs.new(PoseSocket.bl_idname, "Skeleton")

    @property
    def snapshot(self):
        from .snapshot import loads

        return loads(self.snapshot_json, []) or []

    def take_snapshot(self, obj=None):
        from .snapshot import dumps, serialize_armature

        obj = obj or self.source
        if obj is None or obj.type != "ARMATURE":
            return False
        self.snapshot_json = dumps(serialize_armature(obj))
        self.snapshot_name = obj.name
        return True

    # -- Retarget (Skeleton input -> this rig's controls) ---------------------

    @property
    def overrides(self):
        """skeleton bone name -> control bone name, for the enabled rows."""
        return {
            ov.skeleton_bone: ov.target_bone
            for ov in self.bone_overrides
            if ov.enabled and ov.skeleton_bone and ov.target_bone
        }

    def add_override(self, skeleton_bone="", target_bone=""):
        item = self.bone_overrides.add()
        if skeleton_bone:
            item.skeleton_bone = skeleton_bone
        item.target_bone = target_bone
        return item

    def fill_overrides_from_matches(self, ctx):
        """Create a row for every bone the auto-match resolved, so any of them
        can be re-pointed -- not just the ones that failed."""
        from .retarget import match_bones

        obj = self.source
        bones = self.pose_bones_in(ctx)
        if obj is None or not bones:
            return 0
        existing = {ov.skeleton_bone for ov in self.bone_overrides}
        matches, _unmatched = match_bones(obj, bones, self.retarget_mode, {})
        added = 0
        for bdef, pbone, _key in matches:
            if bdef.name in existing:
                continue
            self.add_override(bdef.name, pbone.name)
            added += 1
        return added

    def pose_bones_in(self, ctx):
        """BoneDefs from every Primary Rig wired into the Skeleton input."""
        sock = self.inputs.get("Skeleton")
        if sock is None:
            return []
        bones = []
        for link in sock.links:
            if not link.is_valid:
                continue
            node = link.from_node
            if hasattr(node, "eval_pose"):
                bones.extend(node.eval_pose(ctx))
        return bones

    def apply_retarget(self, ctx):
        """Drive ``self.source`` from the wired-in marker skeleton.

        This is what makes the Rig output the default rigging path: it runs
        automatically at the end of every build / live update, with no
        operator to press.
        """
        from .retarget import apply_pose

        if not self.retarget_enabled:
            return 0
        obj = self.source
        if obj is None or obj.type != "ARMATURE":
            return 0
        bones = self.pose_bones_in(ctx)
        if not bones:
            return 0
        applied, unmatched = apply_pose(
            obj,
            bones,
            mode=self.retarget_mode,
            overrides=self.overrides,
            set_switches=self.retarget_switch_ikfk,
        )
        report = f"{applied}/{applied + len(unmatched)} bones"
        if unmatched:
            report += " | unmatched: " + ", ".join(unmatched)
        if self.match_report != report:
            self.match_report = report
        return applied

    def draw_buttons(self, context, layout):
        self.ensure_sockets()
        layout.context_pointer_set("node", self)
        layout.prop(self, "source")
        sock = self.inputs.get("Skeleton")
        if sock is not None and sock.is_linked:
            box = layout.box()
            row = box.row(align=True)
            row.prop(self, "retarget_enabled", text="")
            row.prop(self, "retarget_mode", expand=True)
            box.prop(self, "retarget_switch_ikfk")
            if self.match_report:
                icon = "ERROR" if "unmatched" in self.match_report else "CHECKMARK"
                box.label(text=self.match_report.split(" | ")[0], icon=icon)
                if "unmatched" in self.match_report:
                    col = box.column(align=True)
                    col.scale_y = 0.8
                    for name in self.match_report.split("unmatched: ")[1].split(", "):
                        row = col.row(align=True)
                        row.label(text=name, icon="DOT")
                        op = row.operator(
                            "armature_nodes.add_bone_override", text="", icon="ADD"
                        )
                        op.skeleton_bone = name.split(" (")[0]
            self._draw_overrides(box)
        snap = self.snapshot
        row = layout.row(align=True)
        if snap:
            row.label(
                text=f"Snapshot: {self.snapshot_name} ({len(snap)} bones)",
                icon="CHECKMARK",
            )
        else:
            row.label(text="No snapshot stored", icon="INFO")
        if snap and self.source:
            layout.prop(self, "use_snapshot")

    def _draw_overrides(self, layout):
        row = layout.row(align=True)
        row.prop(
            self,
            "show_overrides",
            icon="TRIA_DOWN" if self.show_overrides else "TRIA_RIGHT",
            emboss=False,
        )
        row.label(text=str(len(self.bone_overrides)) if self.bone_overrides else "")
        if not self.show_overrides:
            return
        box = layout.box()
        obj = self.source
        for index, ov in enumerate(self.bone_overrides):
            row = box.row(align=True)
            row.prop(ov, "enabled", text="")
            sub = row.row(align=True)
            sub.enabled = ov.enabled
            sub.prop(ov, "skeleton_bone", text="")
            if obj is not None and obj.type == "ARMATURE":
                # Searchable dropdown of the rig's actual bones.
                sub.prop_search(ov, "target_bone", obj.data, "bones", text="")
            else:
                sub.prop(ov, "target_bone", text="")
            row.operator(
                "armature_nodes.remove_bone_override", text="", icon="X"
            ).index = index
        row = box.row(align=True)
        row.operator("armature_nodes.add_bone_override", text="Add", icon="ADD")
        row.operator(
            "armature_nodes.fill_bone_overrides", text="Fill From Matches", icon="COPYDOWN"
        )

    def copy(self, node):
        """When Blender copies/pastes this node, retain the serialized rig but
        drop the live object pointer. The pasted graph then becomes an
        independent rig definition and live update creates a new armature."""
        self.source = None
        self.use_snapshot = True

    def _bones_from_snapshot(self):
        from .snapshot import bonedef_from_dict

        return [bonedef_from_dict(d) for d in self.snapshot]

    def _bones_from_source(self):
        from .snapshot import bone_roll, mesh_geometry
        from .core import ShapeDef

        pose = self.source.pose
        bones = []
        for b in self.source.data.bones:
            bdef = BoneDef(
                name=b.name,
                head=tuple(b.head_local),
                tail=tuple(b.tail_local),
                roll=bone_roll(b),
                parent=b.parent.name if b.parent else None,
                use_connect=b.use_connect,
                use_deform=b.use_deform,
                envelope_distance=b.envelope_distance,
                envelope_weight=b.envelope_weight,
            )
            # Keep existing widgets so a shapes-only pass does not strip them.
            pbone = pose.bones.get(b.name) if pose else None
            if pbone is not None and pbone.custom_shape is not None:
                bdef.shape = ShapeDef(
                    widget=pbone.custom_shape.name,
                    preset="NONE",
                    scale=tuple(pbone.custom_shape_scale_xyz),
                    translation=tuple(pbone.custom_shape_translation),
                    rotation=tuple(pbone.custom_shape_rotation_euler),
                    wire_width=getattr(pbone, "custom_shape_wire_width", 1.0),
                    scale_to_bone_length=pbone.use_custom_shape_bone_size,
                    show_wire=b.show_wire,
                    geometry=mesh_geometry(pbone.custom_shape),
                )
            bones.append(bdef)
        return bones

    def eval_bones(self, ctx):
        """Forward evaluation: the live source armature's bones, or the
        stored snapshot when the source is gone (or preferred)."""
        has_source = bool(self.source) and self.source.type == "ARMATURE"
        if not has_source or (self.use_snapshot and self.snapshot_json):
            return self._bones_from_snapshot()
        return self._bones_from_source()


classes = (
    BoneOverride,
    BoneNode,
    ChainNode,
    MirrorNode,
    ParentNode,
    DeformGroupNode,
    CustomShapeNode,
    PrimaryRigNode,
    IKConstraintNode,
    GenericConstraintNode,
    ArmatureOutputNode,
    ArmatureInputNode,
)


def _on_node_prop_changed(self, context):
    """Any edited node value re-applies the tree to its armature (real time)."""
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


# UI-only toggles that must not trigger a rebuild.
_NO_REBUILD_PROPS = {
    "show_bone_info",
    "show_widget_options",
    "bone_info_synced",
    "transform_synced",
    "show_markers",
    "show_detail",
    "show_advanced",
    "show_overrides",
    "show_skeleton",
    "bone_overrides",  # CollectionProperty: does not accept update=
    "lock_depth",
    "symmetric",
    "match_report",
}


def _inject_live_update(cls):
    """Add an ``update`` callback to every property of a node class.

    Blender only calls NodeTree.update() for link/node changes, not for value
    edits, so without this a scale tweak on a Custom Shape node would do
    nothing until the next structural change. CollectionProperty is skipped
    unconditionally -- Blender's RNA does not accept ``update`` for it at all
    (registration fails outright), and per-item updates on the PropertyGroup
    itself are the correct way to react to collection edits anyway.
    """
    for name, prop in list(cls.__annotations__.items()):
        keywords = getattr(prop, "keywords", None)
        if keywords is None or name in _NO_REBUILD_PROPS:
            continue
        function = getattr(prop, "function", None)
        if function is not None and getattr(function, "__name__", "") == "CollectionProperty":
            continue
        if "update" not in keywords:
            keywords["update"] = _on_node_prop_changed


def register():
    for cls in classes:
        if issubclass(cls, Node):
            _inject_live_update(cls)
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
