"""All node types for the Armature node tree.

The graph is a modifier stack, the way Geometry Nodes is: **Armature Input**
hands the whole rig to **Armature Output**, and every node in between reads
the bone stream, changes the bones it selects, and passes the rest through
untouched. Decompiling a rig therefore produces two nodes, not one per bone --
the rig is the input, not something the graph has to describe.

Categories:

* **Armature I/O** -- ArmatureInputNode (nothing -> Bone),
  ArmatureOutputNode (Bone -> nothing).
* **Bone** -- BoneNode, ChainNode: create bones.
* **Marker** -- MarkerNode (one draggable handle -> one bone), SkeletonNode
  (a bundle of markers, one position output each).
* **Transform** -- PositionNode, RotationNode, TransformNode, SnapNode: pose
  modifiers. They never touch rest geometry, so they cannot deform a rig they
  are layered onto.
* **Shape** -- CustomShapeNode: assigns a control widget.
* **Constraint** -- IKConstraintNode, GenericConstraintNode.

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
    select_bones,
    copy_bone,
)
from .sockets import BoneSocket, ConstraintSocket, VectorSocket
from .widgets import PRESET_ITEMS as _widget_preset_items
from .widgets import widget_enum_items as _widget_enum_items

_EPS = 1e-5


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

    def resolve_armature(self):
        """The armature this node's graph works on, for bone-name dropdowns.

        The Armature Input's source first (that is the rig the stack is layered
        on), then the object the tree is bound to. Returns None when the graph
        builds a rig that does not exist yet, in which case the bone field
        falls back to plain text entry.
        """
        tree = self.id_data
        if tree is None:
            return None
        for node in tree.nodes:
            if node.bl_idname == "ArmatureNodesInputNode":
                src = getattr(node, "source", None)
                if src is not None and getattr(src, "type", "") == "ARMATURE":
                    return src
        for node in tree.nodes:
            if node.bl_idname == "ArmatureNodesOutputNode":
                obj = bpy.data.objects.get(node.armature_name)
                if obj is not None and obj.type == "ARMATURE":
                    return obj
        return None

    def draw_bone_select(self, layout):
        """The Bone field every modifier node carries.

        A searchable dropdown of the rig's real bones when there is a rig to
        search, plain text otherwise. Empty means every bone on the wire.
        """
        obj = self.resolve_armature()
        row = layout.row(align=True)
        if obj is not None:
            row.prop_search(self, "bone", obj.data, "bones", text="", icon="BONE_DATA")
        else:
            row.prop(self, "bone", text="", icon="BONE_DATA")
        if not self.bone:
            layout.label(text="All bones", icon="INFO")


def _bone_select_prop():
    return StringProperty(
        name="Bone",
        description=(
            "Bone this node affects. Empty = every bone on the wire; several "
            "can be named at once, semicolon separated"
        ),
        default="",
    )


class _ModifierNodeBase(ArmatureNodeBase):
    """A node that reads the bone stream, changes a selection, passes it on."""

    def init(self, context):
        self.inputs.new(BoneSocket.bl_idname, "Bone")
        self.outputs.new(BoneSocket.bl_idname, "Bone")
        self.width = 200

    def stream(self, ctx):
        return [copy_bone(b) for b in gather_input_bones(self, "Bone", ctx)]

    def selected(self, bones):
        return select_bones(bones, self.bone)


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

# True while markers are being written in bulk, so the per-marker callback
# does not rebuild once per component.
_syncing_markers = False


def _slug(text):
    """A key-safe token from a display name."""
    out = "".join(c if c.isalnum() else "_" for c in (text or "").strip().lower())
    return out.strip("_")


def _marker_owner(marker):
    """The node a marker belongs to.

    A PropertyGroup only knows its owning ID (the node tree), not the node, so
    the node is found by identity. A tree holds a handful of marker nodes at
    most and this only runs on an edit, so the scan is cheaper than keeping a
    back-reference correct across copy / paste / rename.
    """
    tree = marker.id_data
    if tree is None:
        return None
    for node in tree.nodes:
        if not hasattr(node, "markers"):
            continue
        for m in node.markers:
            if m == marker:
                return node
    return None


def _on_marker_changed(self, context):
    """A marker position/rotation was typed: move its handle and rebuild."""
    if _syncing_markers:
        return
    from .primary_rig import tag_viewports_redraw

    node = _marker_owner(self)
    if node is not None:
        node.push_markers_to_empties()
    tag_viewports_redraw()
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


def _on_marker_name_changed(self, context):
    """Renaming a marker renames its output socket; links survive because they
    are attached to the socket, not to its name."""
    node = _marker_owner(self)
    if node is not None and hasattr(node, "sync_marker_sockets"):
        node.sync_marker_sockets()


def _on_marker_use_rotation_changed(self, context):
    """Rotation enabled/disabled on one marker: re-lock and redraw its handle,
    then rebuild -- an oriented marker also turns what it drives."""
    from .primary_rig import (
        apply_marker_locks,
        find_marker_empties,
        tag_viewports_redraw,
    )

    node = _marker_owner(self)
    if node is None:
        return
    for key, obj in find_marker_empties(node).items():
        apply_marker_locks(node, obj, key)
        marker = node.marker_by_key(key)
        if marker is not None and marker.use_rotation:
            obj.rotation_euler = tuple(marker.rotation)
    tag_viewports_redraw()
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


class SkeletonMarker(bpy.types.PropertyGroup):
    """One marker: a world position, optionally oriented.

    ``key`` is the stable identity -- the output socket and the viewport
    handle are both bound to it, so it is assigned once and never changes.
    ``name`` is only the label and is free to be edited.
    """

    key: StringProperty(name="Key", default="", options={"HIDDEN"})
    name: StringProperty(
        name="Name",
        description="Label for this marker; also names its output socket",
        default="Marker",
        update=_on_marker_name_changed,
    )
    position: FloatVectorProperty(
        name="Position",
        description="World position of the marker handle",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="TRANSLATION",
        update=_on_marker_changed,
    )
    rotation: FloatVectorProperty(
        name="Rotation",
        description="Orientation of the marker, when rotation is enabled on it",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="EULER",
        update=_on_marker_changed,
    )
    use_rotation: BoolProperty(
        name="Rotation",
        description=(
            "Adjust this marker's rotation as well as its position. Off by "
            "default: the handle is position-only until this is enabled"
        ),
        default=False,
        update=_on_marker_use_rotation_changed,
    )

    def set_position(self, value):
        """Write without firing the per-marker rebuild callback.

        Callers that set many markers at once (a preset, a mirror, a handle
        drag) push to the viewport and mark the tree dirty themselves; letting
        each component fire would rebuild the rig dozens of times per edit.
        """
        global _syncing_markers
        was = _syncing_markers
        _syncing_markers = True
        try:
            self.position = tuple(value)
        finally:
            _syncing_markers = was

    def set_rotation(self, value):
        global _syncing_markers
        was = _syncing_markers
        _syncing_markers = True
        try:
            self.rotation = tuple(value)
        finally:
            _syncing_markers = was


class MarkerHolderMixin:
    """Shared marker list, viewport handles and sockets.

    Both the single Marker node and the Skeleton node hold markers and both
    draw handles through ``primary_rig``, which asks a node for ``markers``
    and ``marker_keys()``. Keeping that in one place is what lets the handle,
    lock, mirror and overlay code stay ignorant of which node it is serving.
    """

    def marker_keys(self):
        return [m.key for m in self.markers if m.key]

    def marker_by_key(self, key):
        for m in self.markers:
            if m.key == key:
                return m
        return None

    def unique_marker_key(self, base):
        """A key not already used on this node. Keys are stable identifiers:
        sockets and handles are bound to them, so they never change once
        assigned -- renaming a marker only changes its label."""
        base = base or "marker"
        taken = set(self.marker_keys())
        if base not in taken:
            return base
        i = 1
        while f"{base}.{i:03d}" in taken:
            i += 1
        return f"{base}.{i:03d}"

    def add_marker(self, name="", position=None, rotation=None, key=None):
        marker = self.markers.add()
        marker.key = self.unique_marker_key(key or _slug(name) or "marker")
        marker.name = name or marker.key
        if position is not None:
            marker.set_position(position)
        if rotation is not None:
            marker.set_rotation(rotation)
        self.sync_marker_sockets()
        self.refresh_handles()
        return marker

    def remove_marker(self, index):
        if not 0 <= index < len(self.markers):
            return False
        self.markers.remove(index)
        self.sync_marker_sockets()
        self.refresh_handles()
        return True

    def clear_markers(self):
        self.markers.clear()
        self.sync_marker_sockets()
        self.refresh_handles()

    def marker_uses_rotation(self, key):
        marker = self.marker_by_key(key)
        return bool(marker and marker.use_rotation)

    def marker_rotation(self, key):
        marker = self.marker_by_key(key)
        if marker is None:
            return Vector((0.0, 0.0, 0.0))
        return Vector(marker.rotation)

    def marker_position(self, key):
        marker = self.marker_by_key(key)
        if marker is None:
            return Vector((0.0, 0.0, 0.0))
        return Vector(marker.position)

    def effective_height(self):
        """Figure height, used to size the viewport handles."""
        from .primary_rig import DEFAULT_HEIGHT

        zs = [m.position[2] for m in self.markers]
        if not zs:
            return DEFAULT_HEIGHT
        height = max(zs) - min(zs)
        return height if height > 1e-3 else DEFAULT_HEIGHT

    def set_markers(self, values, rotations=None, push=True):
        """Write several markers at once without per-property rebuilds."""
        global _syncing_markers
        from .tree import suspend_live_update

        _syncing_markers = True
        try:
            with suspend_live_update():
                for key, vec in values.items():
                    marker = self.marker_by_key(key)
                    if marker is not None:
                        marker.set_position(vec)
                for key, euler in (rotations or {}).items():
                    marker = self.marker_by_key(key)
                    if marker is not None:
                        marker.set_rotation(euler)
        finally:
            _syncing_markers = False
        if push:
            self.push_markers_to_empties()

    # -- Viewport handles -----------------------------------------------------

    def markers_shown(self):
        from .primary_rig import find_marker_empties

        return bool(find_marker_empties(self))

    def refresh_handles(self):
        """Bring the viewport handles back in line with the marker list."""
        from .primary_rig import (
            ensure_marker_empties,
            prune_marker_empties,
            tag_viewports_redraw,
        )

        try:
            # Prune unconditionally. markers_shown() asks whether any handle
            # still matches a live marker, so deleting the LAST marker makes
            # it False -- a guard on it would strand that final empty.
            prune_marker_empties(self)
            if self.markers_shown():
                ensure_marker_empties(self)
            tag_viewports_redraw()
        except Exception as exc:  # noqa: BLE001
            print(f"[Armature Nodes] Could not refresh marker handles: {exc}")

    def push_markers_to_empties(self):
        from .primary_rig import find_marker_empties

        for key, obj in find_marker_empties(self).items():
            marker = self.marker_by_key(key)
            if marker is None:
                continue
            loc = Vector(marker.position)
            if (Vector(obj.location) - loc).length > _EPS:
                obj.location = loc
            if marker.use_rotation:
                rot = Vector(marker.rotation)
                if (Vector(obj.rotation_euler) - rot).length > _EPS:
                    obj.rotation_euler = rot

    def sync_from_empties(self):
        """Read dragged handles back in. Returns True if anything moved."""
        from .primary_rig import find_marker_empties

        empties = find_marker_empties(self)
        if not empties:
            return False
        moved, turned = {}, {}
        for key, obj in empties.items():
            marker = self.marker_by_key(key)
            if marker is None:
                continue
            loc = Vector(obj.location)
            if (Vector(marker.position) - loc).length > _EPS:
                moved[key] = loc
            if marker.use_rotation:
                rot = Vector(obj.rotation_euler)
                if (Vector(marker.rotation) - rot).length > _EPS:
                    turned[key] = tuple(rot)
        if not moved and not turned:
            return False
        changes = self._with_group_followers(moved)
        if getattr(self, "symmetric", False):
            changes, turned = self._mirrored(changes, turned)
        self.set_markers(changes, rotations=turned, push=True)
        tree = self.id_data
        if tree is not None and hasattr(tree, "mark_dirty"):
            tree.mark_dirty()
        return True

    def _with_group_followers(self, changes):
        """Overridden by the Skeleton node, which has rigid landmark groups."""
        return dict(changes)

    def _mirrored(self, changes, rotations=None):
        return dict(changes), dict(rotations or {})

    def free(self):
        from .primary_rig import remove_marker_empties

        try:
            remove_marker_empties(self)
        except Exception as exc:  # noqa: BLE001
            print(f"[Armature Nodes] Could not clean up marker handles: {exc}")
        self.schedule_rebuild()


# ---------------------------------------------------------------------------
# Armature I/O
# ---------------------------------------------------------------------------


def _poll_armature_object(self, obj):
    return obj.type == "ARMATURE"


class ArmatureInputNode(ArmatureNodeBase, Node):
    """Head of the stack: the rig as it is now.

    Reads every bone of ``source`` -- rest geometry, parenting, deform flags
    and existing widgets -- and puts them on the wire. Nothing upstream, by
    design: this IS the upstream.

    It re-reads the live armature on every evaluation, which is exactly right
    for a modifier stack: the nodes downstream are the change, so the rig is
    the base state, not something the graph has to remember.
    """

    bl_idname = "ArmatureNodesInputNode"
    bl_label = "Armature Input"
    bl_icon = "OUTLINER_OB_ARMATURE"

    source: PointerProperty(
        name="Source", type=bpy.types.Object, poll=_poll_armature_object
    )

    def init(self, context):
        self.outputs.new(BoneSocket.bl_idname, "Bone")
        self.width = 200

    def draw_buttons(self, context, layout):
        layout.prop(self, "source", text="")
        if self.source is None:
            layout.label(text="Pick the rig to modify", icon="INFO")

    def eval_bones(self, ctx):
        from .core import ShapeDef, bone_roll

        obj = self.source
        if obj is None or obj.type != "ARMATURE":
            return []
        pose = obj.pose
        bones = []
        for b in obj.data.bones:
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
            # Carry existing widgets, so a stack that only adds a Position
            # node does not strip every control shape off the rig.
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
                )
            bones.append(bdef)
        return bones


class ArmatureOutputNode(ArmatureNodeBase, Node):
    """Tail of the stack: what the graph writes back to.

    Leaving *Armature* empty targets the Armature Input's source, which is the
    normal case -- a stack that modifies a rig in place. Naming something else
    writes to that object instead, creating it if the graph builds a new rig.
    """

    bl_idname = "ArmatureNodesOutputNode"
    bl_label = "Armature Output"
    bl_icon = "ARMATURE_DATA"

    armature_name: StringProperty(name="Armature", default="")
    mode: EnumProperty(
        name="Mode",
        items=(
            (
                "MODIFY",
                "Modify",
                "Write shapes and pose onto the existing rig; bones, "
                "constraints and drivers are left alone",
            ),
            (
                "FULL",
                "Full Rig",
                "Create or rebuild bones, constraints and shapes from the graph",
            ),
        ),
        default="MODIFY",
    )

    def init(self, context):
        self._multi_input(BoneSocket.bl_idname, "Bone")
        self.width = 200

    def target_name(self):
        """The object this output writes to."""
        if self.armature_name:
            return self.armature_name
        obj = self.resolve_armature()
        return obj.name if obj is not None else "Armature"

    def draw_buttons(self, context, layout):
        layout.prop(self, "mode", text="")
        layout.prop(self, "armature_name", text="")
        if not self.armature_name:
            layout.label(text=f"-> {self.target_name()}", icon="ARMATURE_DATA")

    def free(self):
        """Deleting the Output deletes what it generated.

        Only objects tagged as owned by *this* node are removed -- a rig the
        graph merely writes into is never touched.
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
        return gather_input_bones(self, "Bone", ctx)


# ---------------------------------------------------------------------------
# Bone
# ---------------------------------------------------------------------------


# True while a node is copying values off the rig, so the property callbacks
# do not write the same values straight back.
_syncing_bone_read = False


def _on_bone_selected(self, context):
    """Picking a bone reads its current pose off the rig.

    Read once, on selection, not continuously: in a modifier stack the bone's
    position is an *output* of the graph, so a live read-back would race the
    pose this node itself writes. After the read the node's values are what
    the bone follows, and Read From Rig re-syncs on demand.
    """
    if _syncing_bone_read:
        return
    try:
        self.read_from_rig()
    except Exception as exc:  # noqa: BLE001
        print(f"[Armature Nodes] Could not read bone: {exc}")
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


class BoneNode(_ModifierNodeBase, Node):
    """One bone from the rig, posed.

    Pick any bone -- DEF, MCH, ORG or a control, it makes no difference to
    this node -- and it reads that bone's current world transform. Editing the
    values poses it.

    Pose only: rest geometry is never touched, so this cannot change the
    proportions of the rig and cannot fight an Edit-mode change. A component
    whose checkbox is off is left exactly as the rig has it, so a node can move
    a bone without also pinning its rotation.
    """

    bl_idname = "ArmatureNodesBoneNode"
    bl_label = "Bone"
    bl_icon = "BONE_DATA"

    bone: StringProperty(
        name="Bone",
        description="The one bone this node reads and poses",
        default="",
        update=_on_bone_selected,
    )
    location: FloatVectorProperty(
        name="Location",
        description="World position of the bone",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="TRANSLATION",
    )
    rotation: FloatVectorProperty(
        name="Rotation",
        description="World orientation of the bone, as an XYZ Euler",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="EULER",
    )
    scale: FloatVectorProperty(
        name="Scale", size=3, default=(1.0, 1.0, 1.0), subtype="XYZ"
    )
    use_location: BoolProperty(name="Location", default=True)
    use_rotation: BoolProperty(name="Rotation", default=False)
    use_scale: BoolProperty(name="Scale", default=False)
    synced: BoolProperty(default=False, options={"HIDDEN"})

    def init(self, context):
        super().init(context)
        self.inputs.new(BoneSocket.bl_idname, "Parent")
        self._multi_input(ConstraintSocket.bl_idname, "Constraints")
        self.width = 220

    def read_from_rig(self):
        """Copy the selected bone's current world transform onto this node."""
        global _syncing_bone_read

        obj = self.resolve_armature()
        if obj is None or not self.bone or obj.pose is None:
            return False
        pbone = obj.pose.bones.get(self.bone)
        if pbone is None:
            return False
        loc, rot, scale = (obj.matrix_world @ pbone.matrix).decompose()
        _syncing_bone_read = True
        try:
            self.location = tuple(loc)
            self.rotation = tuple(rot.to_euler("XYZ"))
            self.scale = tuple(scale)
            self.synced = True
        finally:
            _syncing_bone_read = False
        return True

    def draw_buttons(self, context, layout):
        layout.context_pointer_set("node", self)
        obj = self.resolve_armature()
        row = layout.row(align=True)
        if obj is not None:
            # Every bone, whatever its role: a DEF or MCH bone is as valid a
            # thing to pose as a control.
            row.prop_search(self, "bone", obj.data, "bones", text="", icon="BONE_DATA")
        else:
            row.prop(self, "bone", text="", icon="BONE_DATA")
        row.operator("armature_nodes.bone_read_from_rig", text="", icon="FILE_REFRESH")

        if not self.bone:
            layout.label(text="Pick a bone to pose", icon="INFO")
            return
        if not self.synced:
            layout.label(text="Not read from the rig yet", icon="INFO")
        for flag, prop in (
            ("use_location", "location"),
            ("use_rotation", "rotation"),
            ("use_scale", "scale"),
        ):
            row = layout.row(align=True)
            row.prop(self, flag, text="")
            sub = row.column(align=True)
            sub.enabled = getattr(self, flag)
            sub.prop(self, prop, text="")

    def eval_bones(self, ctx):
        bones = self.stream(ctx)
        if not self.bone:
            return bones
        parents = gather_input_bones(self, "Parent", ctx)
        constraints = gather_input_constraints(self, "Constraints", ctx)
        for b in bones:
            if b.name != self.bone:
                continue
            if self.use_location:
                b.pose_location = tuple(self.location)
            if self.use_rotation:
                b.pose_rotation = tuple(self.rotation)
            if self.use_scale:
                b.pose_scale = tuple(self.scale)
            # Re-parenting and constraints are rest/pose-stack data, so they
            # only reach the armature when the Output is in Full Rig mode.
            if parents:
                b.parent = parents[-1].name
            if constraints:
                b.constraints = list(b.constraints) + constraints
            break
        return bones


class ChainNode(ArmatureNodeBase, Node):
    """Procedurally generates N connected bones (spine, finger, tail...)."""

    bl_idname = "ArmatureNodesChainNode"
    bl_label = "Chain"
    bl_icon = "CONSTRAINT_BONE"

    prefix: StringProperty(name="Prefix", default="chain")
    count: IntProperty(name="Count", default=3, min=1, max=256)
    start: FloatVectorProperty(name="Start", size=3, default=(0, 0, 0), subtype="XYZ")
    direction: FloatVectorProperty(
        name="Direction", size=3, default=(0, 0, 1), subtype="XYZ"
    )
    bone_length: FloatProperty(name="Bone Length", default=0.25, min=1e-4)
    curve: FloatProperty(
        name="Curve",
        description="Bend applied to the running direction per segment",
        default=0.0,
        subtype="ANGLE",
    )

    def init(self, context):
        self.inputs.new(BoneSocket.bl_idname, "Parent")
        self._multi_input(ConstraintSocket.bl_idname, "Tip Constraints")
        self.outputs.new(BoneSocket.bl_idname, "Bone")

    def draw_buttons(self, context, layout):
        layout.prop(self, "prefix", text="")
        layout.prop(self, "count")
        col = layout.column(align=True)
        col.prop(self, "start")
        col.prop(self, "direction")
        layout.prop(self, "bone_length")
        layout.prop(self, "curve")

    def eval_bones(self, ctx):
        from mathutils import Matrix

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
            bones[-1].constraints = tip_constraints
        return [copy_bone(b) for b in parents] + bones


# ---------------------------------------------------------------------------
# Marker
# ---------------------------------------------------------------------------


class MarkerNode(MarkerHolderMixin, ArmatureNodeBase, Node):
    """One draggable handle in the viewport that defines a bone.

    The marker is the bone's head; the bone runs from there along *Direction*
    for *Length*. Placing a bone by dragging a handle beats typing head and
    tail coordinates, which is the whole reason this node exists.
    """

    bl_idname = "ArmatureNodesMarkerNode"
    bl_label = "Marker"
    bl_icon = "EMPTY_AXIS"

    markers: bpy.props.CollectionProperty(type=SkeletonMarker)
    bone_name: StringProperty(name="Name", default="marker")
    length: FloatProperty(name="Length", default=0.25, min=1e-4)
    direction: FloatVectorProperty(
        name="Direction", size=3, default=(0, 0, 1), subtype="XYZ"
    )
    use_deform: BoolProperty(name="Deform", default=True)

    def init(self, context):
        self.inputs.new(BoneSocket.bl_idname, "Parent")
        self._multi_input(ConstraintSocket.bl_idname, "Constraints")
        self.outputs.new(BoneSocket.bl_idname, "Bone")
        self.width = 200
        self.add_marker(name="Marker")

    @property
    def marker(self):
        return self.markers[0] if len(self.markers) else None

    def sync_marker_sockets(self):
        """The Marker node has no per-marker socket: it outputs the bone it
        builds, not the position. Present so the shared marker code can call
        it unconditionally."""
        return

    def draw_buttons(self, context, layout):
        layout.context_pointer_set("node", self)
        marker = self.marker
        if marker is None:
            layout.operator(
                "armature_nodes.skeleton_add_marker", text="Add Marker", icon="ADD"
            )
            return
        row = layout.row(align=True)
        shown = self.markers_shown()
        row.operator(
            "armature_nodes.skeleton_toggle_markers",
            text=("Hide" if shown else "Show") + " Handle",
            icon="HIDE_OFF" if shown else "HIDE_ON",
        )
        op = row.operator(
            "armature_nodes.skeleton_toggle_rotation",
            text="",
            icon="ORIENTATION_GIMBAL",
            depress=marker.use_rotation,
        )
        op.marker = marker.key
        layout.prop(self, "bone_name", text="")
        layout.prop(marker, "position", text="")
        if marker.use_rotation:
            layout.prop(marker, "rotation", text="")
        col = layout.column(align=True)
        col.prop(self, "direction")
        col.prop(self, "length")
        layout.prop(self, "use_deform")

    def eval_bones(self, ctx):
        parents = gather_input_bones(self, "Parent", ctx)
        constraints = gather_input_constraints(self, "Constraints", ctx)
        marker = self.marker
        if marker is None:
            return [copy_bone(b) for b in parents]
        head = Vector(marker.position)
        direction = Vector(self.direction)
        if direction.length < 1e-8:
            direction = Vector((0.0, 0.0, 1.0))
        tail = head + direction.normalized() * self.length
        bone = BoneDef(
            name=self.bone_name or self.name,
            head=tuple(head),
            tail=tuple(tail),
            use_deform=self.use_deform,
            constraints=constraints,
        )
        if marker.use_rotation:
            bone.roll = float(Vector(marker.rotation).y)
        if parents:
            bone.parent = parents[-1].name
        return [copy_bone(b) for b in parents] + [bone]


class SkeletonNode(MarkerHolderMixin, ArmatureNodeBase, Node):
    """A bundle of markers, one output each.

    The Principled BSDF of markers: a group of named handles with sensible
    positions, each exposed as its own socket so it can be wired wherever a
    position is wanted -- a Position node, a Snap node, a Custom Shape offset.

    A new node arrives with MediaPipe's 33 pose landmarks loaded, which is a
    usable body to drag onto a character. They are only a preset: rename them,
    delete the ones you do not want, add as many of your own as you need.

    It produces no bones itself. Markers are positions; turning one into rig
    is the job of whatever you wire it into.
    """

    bl_idname = "ArmatureNodesSkeletonNode"
    bl_label = "Skeleton"
    bl_icon = "OUTLINER_OB_ARMATURE"

    markers: bpy.props.CollectionProperty(type=SkeletonMarker)
    lock_depth: BoolProperty(
        name="Lock Depth (2D)",
        description="Handles only move in X/Z (front-view adjustment)",
        default=True,
        update=lambda self, ctx: self._relock(),
    )
    symmetric: BoolProperty(
        name="Symmetric",
        description=(
            "Right-side MediaPipe landmarks are locked and mirror the left "
            "side. Markers you added yourself have no mirror partner"
        ),
        default=True,
        update=lambda self, ctx: self._relock(),
    )
    mirror_center_x: FloatProperty(
        name="Mirror X",
        description="World X of the mirror plane used by Symmetric mode",
        default=0.0,
    )
    show_skeleton: BoolProperty(
        name="Draw",
        description="Draw the markers in the 3D viewport",
        default=True,
        update=lambda self, ctx: _redraw_viewports(),
    )
    show_markers: BoolProperty(name="Markers", default=True)
    show_detail: BoolProperty(name="Face / Hands / Feet", default=False)
    show_advanced: BoolProperty(name="Advanced", default=False)

    def init(self, context):
        self.width = 320
        # The 33 MediaPipe landmarks are the default body: the node is useful
        # immediately, and unwanted markers are deleted rather than 33 added
        # by hand.
        self.load_mediapipe_preset()

    def _relock(self):
        from .primary_rig import apply_marker_locks, find_marker_empties

        if self.symmetric:
            self.mirror_markers("L_TO_R")
        for key, obj in find_marker_empties(self).items():
            apply_marker_locks(self, obj, key)
        _redraw_viewports()

    def load_mediapipe_preset(self, replace=True):
        """Fill the marker list with MediaPipe's 33 pose landmarks.

        A 1.8 m T-pose body, ready to drag onto the character. Markers the
        user added themselves are kept when ``replace`` is False.
        """
        from .primary_rig import LANDMARKS, LM_LABELS

        if replace:
            self.markers.clear()
        have = set(self.marker_keys())
        added = 0
        for _idx, key, _label, _side, default in LANDMARKS:
            if key in have:
                continue
            marker = self.markers.add()
            marker.key = key
            marker.name = LM_LABELS[key]
            marker.set_position(default)
            added += 1
        self.sync_marker_sockets()
        self.refresh_handles()
        return added

    def sync_marker_sockets(self):
        """One Vector output per marker, bound by ``marker_key``.

        Sockets are matched and renamed rather than rebuilt, because a link is
        attached to the socket itself: dropping and re-adding one would break
        every wire. New markers append, so inserting in the middle of the list
        leaves socket order behind list order -- harmless, and the alternative
        costs links.
        """
        wanted = [(m.key, m.name or m.key) for m in self.markers if m.key]
        wanted_keys = {k for k, _n in wanted}
        for sock in list(self.outputs):
            if sock.marker_key not in wanted_keys:
                self.outputs.remove(sock)
        existing = {s.marker_key: s for s in self.outputs}
        for key, label in wanted:
            sock = existing.get(key)
            if sock is None:
                sock = self.outputs.new(VectorSocket.bl_idname, label)
                sock.marker_key = key
            elif sock.name != label:
                sock.name = label

    # -- Symmetry and rigid groups (MediaPipe landmarks only) -----------------

    def _mirrored(self, changes, rotations=None):
        from .primary_rig import LM_MIRROR, LM_SIDE, mirror_point, mirror_rotation

        mid = float(self.mirror_center_x)
        out = dict(changes)
        rot_out = dict(rotations or {})
        for key, loc in changes.items():
            if LM_SIDE.get(key) == "L" and LM_MIRROR.get(key):
                out[LM_MIRROR[key]] = mirror_point(loc, mid)
        for key, euler in (rotations or {}).items():
            if LM_SIDE.get(key) == "L" and LM_MIRROR.get(key):
                rot_out[LM_MIRROR[key]] = mirror_rotation(euler)
        return out, rot_out

    def _with_group_followers(self, changes):
        """Face / finger / toe landmarks move rigidly with their anchor."""
        from .primary_rig import RIGID_GROUPS

        have = set(self.marker_keys())
        out = dict(changes)
        for anchor, members in RIGID_GROUPS.items():
            if anchor not in changes or anchor not in have:
                continue
            delta = changes[anchor] - self.marker_position(anchor)
            for m in members:
                if m not in changes and m in have:
                    out[m] = self.marker_position(m) + delta
        return out

    def mirror_markers(self, direction="L_TO_R"):
        from .primary_rig import LM_MIRROR, LM_SIDE, mirror_point, mirror_rotation

        src = "L" if direction == "L_TO_R" else "R"
        mid = float(self.mirror_center_x)
        keys = [
            k for k in self.marker_keys() if LM_SIDE.get(k) == src and LM_MIRROR.get(k)
        ]
        changes = {LM_MIRROR[k]: mirror_point(self.marker_position(k), mid) for k in keys}
        rotations = {
            LM_MIRROR[k]: mirror_rotation(self.marker_rotation(k))
            for k in keys
            if self.marker_uses_rotation(k)
        }
        self.set_markers(changes, rotations=rotations)

    # -- UI -------------------------------------------------------------------

    def _draw_marker_row(self, layout, index, marker):
        from .primary_rig import LM_SIDE

        enabled = not (self.symmetric and LM_SIDE.get(marker.key) == "R")
        row = layout.row(align=True)
        row.enabled = enabled
        row.prop(marker, "name", text="")
        op = row.operator(
            "armature_nodes.skeleton_toggle_rotation",
            text="",
            icon="ORIENTATION_GIMBAL",
            depress=marker.use_rotation,
        )
        op.marker = marker.key
        op = row.operator("armature_nodes.skeleton_remove_marker", text="", icon="X")
        op.index = index
        sub = layout.row(align=True)
        sub.enabled = enabled
        sub.prop(marker, "position", text="")
        if marker.use_rotation:
            sub = layout.row(align=True)
            sub.enabled = enabled
            sub.prop(marker, "rotation", text="")

    def draw_buttons(self, context, layout):
        from .primary_rig import GROUP_ANCHOR, is_landmark

        self.sync_marker_sockets()
        layout.context_pointer_set("node", self)

        col = layout.column(align=True)
        row = col.row(align=True)
        shown = self.markers_shown()
        row.operator(
            "armature_nodes.skeleton_toggle_markers",
            text=("Hide" if shown else "Show") + " Markers",
            icon="HIDE_OFF" if shown else "HIDE_ON",
        )
        row.operator("armature_nodes.skeleton_front_view", icon="VIEW_ORTHO")
        row = col.row(align=True)
        row.prop(self, "lock_depth", toggle=True)
        row.prop(self, "symmetric", toggle=True)
        row.prop(self, "show_skeleton", toggle=True, icon="ARMATURE_DATA")

        row = layout.row(align=True)
        row.operator("armature_nodes.skeleton_add_marker", text="Add Marker", icon="ADD")
        row.operator(
            "armature_nodes.skeleton_load_preset", text="MediaPipe", icon="ARMATURE_DATA"
        )
        if not self.markers:
            layout.label(text="No markers -- Add Marker, or load a preset", icon="INFO")
            return

        primary, detail = [], []
        for index, marker in enumerate(self.markers):
            if is_landmark(marker.key) and marker.key in GROUP_ANCHOR:
                detail.append((index, marker))
            else:
                primary.append((index, marker))

        row = layout.row(align=True)
        row.prop(
            self,
            "show_markers",
            icon="TRIA_DOWN" if self.show_markers else "TRIA_RIGHT",
            emboss=False,
        )
        row.label(text=str(len(self.markers)))
        if self.show_markers:
            box = layout.box()
            for index, marker in primary:
                self._draw_marker_row(box, index, marker)
            if detail:
                row = box.row(align=True)
                row.prop(
                    self,
                    "show_detail",
                    icon="TRIA_DOWN" if self.show_detail else "TRIA_RIGHT",
                    emboss=False,
                )
                if self.show_detail:
                    sub = box.box()
                    for index, marker in detail:
                        self._draw_marker_row(sub, index, marker)

        row = layout.row(align=True)
        row.prop(
            self,
            "show_advanced",
            icon="TRIA_DOWN" if self.show_advanced else "TRIA_RIGHT",
            emboss=False,
        )
        if self.show_advanced:
            box = layout.box()
            box.prop(self, "mirror_center_x")
            row = box.row(align=True)
            row.operator(
                "armature_nodes.skeleton_mirror", text="Mirror L > R"
            ).direction = "L_TO_R"
            row.operator(
                "armature_nodes.skeleton_mirror", text="Mirror R > L"
            ).direction = "R_TO_L"
            box.operator(
                "armature_nodes.skeleton_clear_markers",
                text="Clear Markers",
                icon="TRASH",
            )


def _redraw_viewports():
    from .primary_rig import tag_viewports_redraw

    tag_viewports_redraw()


# ---------------------------------------------------------------------------
# Transform (pose modifiers)
# ---------------------------------------------------------------------------


class PositionNode(_ModifierNodeBase, Node):
    """Set the world position of the selected bones (Set Position).

    A pose transform: the bone is moved the way grabbing it in Pose mode
    would, and its rest geometry is untouched -- so this cannot deform the
    proportions of the rig it is layered onto, and a custom shape follows
    because Blender draws widgets at the posed bone.
    """

    bl_idname = "ArmatureNodesPositionNode"
    bl_label = "Position"
    bl_icon = "CON_LOCLIKE"

    bone: _bone_select_prop()

    def init(self, context):
        super().init(context)
        self.inputs.new(VectorSocket.bl_idname, "Position")

    def draw_buttons(self, context, layout):
        self.draw_bone_select(layout)

    def eval_bones(self, ctx):
        bones = self.stream(ctx)
        sock = self.inputs.get("Position")
        position = sock.get_value() if sock is not None else (0.0, 0.0, 0.0)
        for b in self.selected(bones):
            b.pose_location = tuple(position)
        return bones


class RotationNode(_ModifierNodeBase, Node):
    """Set the world orientation of the selected bones."""

    bl_idname = "ArmatureNodesRotationNode"
    bl_label = "Rotation"
    bl_icon = "CON_ROTLIKE"

    bone: _bone_select_prop()

    def init(self, context):
        super().init(context)
        sock = self.inputs.new(VectorSocket.bl_idname, "Rotation")
        sock.default_value = (0.0, 0.0, 0.0)

    def draw_buttons(self, context, layout):
        self.draw_bone_select(layout)

    def eval_bones(self, ctx):
        bones = self.stream(ctx)
        sock = self.inputs.get("Rotation")
        rotation = sock.get_value() if sock is not None else (0.0, 0.0, 0.0)
        for b in self.selected(bones):
            b.pose_rotation = tuple(rotation)
        return bones


class TransformNode(_ModifierNodeBase, Node):
    """Offset the selected bones (Transform Geometry).

    Translation and rotation are *deltas* added to whatever pose the bone
    already has, so several Transform nodes stack. Scale is absolute, because
    a control's scale is a value rather than a displacement.
    """

    bl_idname = "ArmatureNodesTransformNode"
    bl_label = "Transform"
    bl_icon = "ORIENTATION_GLOBAL"

    bone: _bone_select_prop()
    scale: FloatVectorProperty(
        name="Scale", size=3, default=(1.0, 1.0, 1.0), subtype="XYZ"
    )
    use_scale: BoolProperty(
        name="Set Scale",
        description="Off leaves the control's scale alone",
        default=False,
    )

    def init(self, context):
        super().init(context)
        self.inputs.new(VectorSocket.bl_idname, "Translation")
        self.inputs.new(VectorSocket.bl_idname, "Rotation")

    def draw_buttons(self, context, layout):
        self.draw_bone_select(layout)
        row = layout.row(align=True)
        row.prop(self, "use_scale", text="")
        sub = row.row(align=True)
        sub.enabled = self.use_scale
        sub.prop(self, "scale", text="")

    def eval_bones(self, ctx):
        bones = self.stream(ctx)
        t_sock = self.inputs.get("Translation")
        r_sock = self.inputs.get("Rotation")
        translation = Vector(t_sock.get_value() if t_sock else (0, 0, 0))
        rotation = Vector(r_sock.get_value() if r_sock else (0, 0, 0))
        for b in self.selected(bones):
            # Fold into an absolute target when one is already set, so a
            # Position node followed by a Transform node reads as "put it
            # there, then nudge it" rather than the two fighting.
            if b.pose_location is not None:
                b.pose_location = tuple(Vector(b.pose_location) + translation)
            else:
                b.pose_offset = tuple(Vector(b.pose_offset) + translation)
            if b.pose_rotation is not None:
                b.pose_rotation = tuple(Vector(b.pose_rotation) + rotation)
            else:
                b.pose_rotation_offset = tuple(
                    Vector(b.pose_rotation_offset) + rotation
                )
            if self.use_scale:
                b.pose_scale = tuple(self.scale)
        return bones


def _mesh_anchor(obj, mode, reference=None):
    """A world-space point on ``obj`` for the Snap node.

    ORIGIN   the object's own origin
    BOUNDS   centre of its bounding box
    MEDIAN   mean of its vertices
    VOLUME   volume centroid (signed tetrahedra over the triangulated mesh),
             which is the centre of mass of a solid, unlike the median
    SURFACE  closest point on the surface to ``reference``
    """
    from mathutils import Vector as V

    mw = obj.matrix_world
    if mode == "ORIGIN":
        return mw.translation.copy()
    if mode == "BOUNDS":
        corners = [V(c) for c in obj.bound_box]
        return mw @ (sum(corners, V((0, 0, 0))) / len(corners))
    mesh = getattr(obj, "data", None)
    verts = getattr(mesh, "vertices", None)
    if not verts:
        return mw.translation.copy()
    if mode == "MEDIAN":
        total = V((0.0, 0.0, 0.0))
        for v in verts:
            total += v.co
        return mw @ (total / len(verts))
    if mode == "SURFACE":
        point = V(reference) if reference is not None else mw.translation
        local = mw.inverted_safe() @ point
        try:
            hit, location, _normal, _index = obj.closest_point_on_mesh(local)
        except (RuntimeError, ValueError):
            hit = False
        if not hit:
            return mw.translation.copy()
        return mw @ location
    # VOLUME: sum signed tetrahedra (origin, a, b, c) over triangulated faces.
    mesh.calc_loop_triangles()
    total_volume = 0.0
    centroid = V((0.0, 0.0, 0.0))
    for tri in mesh.loop_triangles:
        a, b, c = (verts[i].co for i in tri.vertices)
        volume = a.cross(b).dot(c) / 6.0
        total_volume += volume
        centroid += (a + b + c) * (volume / 4.0)
    if abs(total_volume) < 1e-12:
        # Flat or open mesh: the signed volume cancels out, so fall back to
        # the median rather than dividing by ~zero.
        return _mesh_anchor(obj, "MEDIAN", reference)
    return mw @ (centroid / total_volume)


class SnapNode(_ModifierNodeBase, Node):
    """Snap the selected bones onto a mesh.

    Positions a control against real geometry instead of by eye: the object's
    origin, its bounding-box centre, the median of its vertices, its volume
    centroid, or the closest point on its surface. Like the other Transform
    nodes it writes the pose only.
    """

    bl_idname = "ArmatureNodesSnapNode"
    bl_label = "Snap"
    bl_icon = "SNAP_ON"

    bone: _bone_select_prop()
    target: PointerProperty(
        name="Target",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "MESH",
    )
    mode: EnumProperty(
        name="Snap To",
        items=(
            ("ORIGIN", "Origin", "The target object's own origin"),
            ("BOUNDS", "Bounding Box", "Centre of the target's bounding box"),
            ("MEDIAN", "Median", "Mean of the target's vertices"),
            ("VOLUME", "Volume", "Volume centroid: the centre of mass of a solid"),
            ("SURFACE", "Surface", "Closest point on the target's surface"),
        ),
        default="VOLUME",
    )
    offset: FloatVectorProperty(
        name="Offset", size=3, default=(0, 0, 0), subtype="TRANSLATION"
    )

    def draw_buttons(self, context, layout):
        self.draw_bone_select(layout)
        layout.prop(self, "target", text="")
        layout.prop(self, "mode", text="")
        layout.prop(self, "offset")

    def eval_bones(self, ctx):
        bones = self.stream(ctx)
        obj = self.target
        if obj is None or obj.type != "MESH":
            return bones
        offset = Vector(self.offset)
        for b in self.selected(bones):
            # SURFACE needs a point to be closest TO; the bone's own head is
            # the only sensible reference at evaluation time.
            reference = Vector(b.pose_location) if b.pose_location else Vector(b.head)
            try:
                anchor = _mesh_anchor(obj, self.mode, reference)
            except Exception as exc:  # noqa: BLE001
                print(f"[Armature Nodes] Snap failed on '{b.name}': {exc}")
                continue
            b.pose_location = tuple(Vector(anchor) + offset)
        return bones


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def _poll_widget_object(self, obj):
    return obj.type == "MESH"


class CustomShapeNode(_ModifierNodeBase, Node):
    """Assign a control widget to the selected bones.

    Stripped to what a modifier needs: which bone, which widget, and how the
    widget sits on it. The bone's own data comes down the wire from the
    Armature Input, so this node no longer stores a copy of it.

    The *Offset* input is a vector like Geometry Nodes' Set Position: wire a
    marker into it and the widget moves with the handle.
    """

    bl_idname = "ArmatureNodesCustomShapeNode"
    bl_label = "Custom Shape"
    bl_icon = "MESH_CIRCLE"

    bone: _bone_select_prop()
    source: EnumProperty(
        name="Source",
        items=(
            ("PRESET", "Preset", "Generate a WGT-rig_<bone> widget from a preset"),
            ("LIBRARY", "WGTS_rig", "Reuse an existing widget from WGTS_rig"),
            ("OBJECT", "Object", "Use any mesh object as the widget"),
        ),
        default="PRESET",
    )
    preset: EnumProperty(name="Preset", items=_widget_preset_items, default=1)
    library_widget: EnumProperty(name="Widget", items=_widget_enum_items)
    widget_object: PointerProperty(
        name="Widget", type=bpy.types.Object, poll=_poll_widget_object
    )
    widget_scale: FloatVectorProperty(
        name="Scale", size=3, default=(1.0, 1.0, 1.0), subtype="XYZ"
    )
    widget_rotation: FloatVectorProperty(
        name="Rotation", size=3, default=(0.0, 0.0, 0.0), subtype="EULER"
    )
    wire_width: FloatProperty(name="Wire Width", default=1.0, min=1.0, max=16.0)
    scale_to_bone_length: BoolProperty(name="Scale to Bone Length", default=True)
    show_wire: BoolProperty(name="Wireframe", default=True)
    control_only: BoolProperty(
        name="Control Only",
        description="Also turn off Deform on these bones (they are controls)",
        default=True,
    )

    def init(self, context):
        super().init(context)
        self.inputs.new(VectorSocket.bl_idname, "Offset")
        self.width = 220

    def draw_buttons(self, context, layout):
        self.draw_bone_select(layout)
        layout.prop(self, "source", text="")
        if self.source == "PRESET":
            layout.prop(self, "preset", text="")
        elif self.source == "LIBRARY":
            layout.prop(self, "library_widget", text="")
            layout.prop(self, "preset", text="Fallback")
        else:
            layout.prop(self, "widget_object", text="")
        col = layout.column(align=True)
        col.prop(self, "widget_scale", text="Scale")
        col.prop(self, "widget_rotation")
        layout.prop(self, "scale_to_bone_length")
        row = layout.row(align=True)
        row.prop(self, "show_wire")
        row.prop(self, "wire_width", text="Width")
        layout.prop(self, "control_only")

    def _widget_name(self):
        if self.source == "LIBRARY":
            return self.library_widget or ""
        if self.source == "OBJECT" and self.widget_object:
            return self.widget_object.name
        return ""  # PRESET: a per-bone WGT-rig_<bone> is generated at build

    def eval_bones(self, ctx):
        from .core import ShapeDef

        bones = self.stream(ctx)
        sock = self.inputs.get("Offset")
        offset = sock.get_value() if sock is not None else (0.0, 0.0, 0.0)
        widget = self._widget_name()
        preset = self.preset if self.source != "OBJECT" else "NONE"
        for b in self.selected(bones):
            b.shape = ShapeDef(
                widget=widget,
                preset=preset,
                scale=tuple(self.widget_scale),
                translation=tuple(offset),
                rotation=tuple(self.widget_rotation),
                wire_width=self.wire_width,
                scale_to_bone_length=self.scale_to_bone_length,
                show_wire=self.show_wire,
            )
            if self.control_only:
                b.use_deform = False
        return bones


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class IKConstraintNode(ArmatureNodeBase, Node):
    """IK constraint: target, pole target, chain length, iterations, stretch."""

    bl_idname = "ArmatureNodesIKConstraintNode"
    bl_label = "IK Constraint"
    bl_icon = "CON_KINEMATIC"

    target: PointerProperty(name="Target", type=bpy.types.Object)
    subtarget: StringProperty(name="Target Bone", default="")
    pole_target: PointerProperty(name="Pole Target", type=bpy.types.Object)
    pole_angle: FloatProperty(name="Pole Angle", default=0.0, subtype="ANGLE")
    chain_count: IntProperty(name="Chain Length", default=2, min=0)
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
# Registration
# ---------------------------------------------------------------------------

classes = (
    SkeletonMarker,
    ArmatureInputNode,
    ArmatureOutputNode,
    BoneNode,
    ChainNode,
    MarkerNode,
    SkeletonNode,
    PositionNode,
    RotationNode,
    TransformNode,
    SnapNode,
    CustomShapeNode,
    IKConstraintNode,
    GenericConstraintNode,
)


def _on_node_prop_changed(self, context):
    """Any edited node value re-applies the tree to its armature (real time)."""
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


# UI-only toggles that must not trigger a rebuild.
_NO_REBUILD_PROPS = {
    "synced",
    "show_markers",
    "show_detail",
    "show_advanced",
    "show_skeleton",
    "markers",  # CollectionProperty: does not accept update=
    "lock_depth",
    "symmetric",
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
