"""Bone: pose one bone of the rig (Bone), or generate new ones (Chain)."""

import logging

from bpy.types import Node
from bpy.props import (
    BoolProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    StringProperty,
)
from mathutils import Vector

from ..core import (
    BoneDef,
    editable,
    gather_input_bones,
    gather_input_constraints,
    socket_links,
)
from ..sockets import ConstraintSocket, RigSocket, VectorSocket
from .base import (
    ZERO,
    ArmatureNodeBase,
    LiveLinkMixin,
    ModifierNodeBase,
    same_vec,
    write_input,
    write_socket_value,
)

log = logging.getLogger(__name__)


# True while a node is copying values off the rig, so the property callbacks
# do not write the same values straight back.
_syncing_bone_read = False


def _on_bone_selected(self, context):
    """Picking a bone reads its current pose off the rig.

    From then on the live link keeps the node in step (``LiveLinkMixin``);
    this read is what makes the very first build leave the bone where it is.
    """
    if _syncing_bone_read:
        return
    try:
        self.read_from_rig()
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read bone: %s", exc)
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


class BoneNode(LiveLinkMixin, ModifierNodeBase, Node):
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
    bone_rotation: FloatVectorProperty(
        name="Rotation",
        description="World orientation of the bone, as an XYZ Euler",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype="EULER",
    )
    bone_scale: FloatVectorProperty(
        name="Scale", size=3, default=(1.0, 1.0, 1.0), subtype="XYZ"
    )
    use_location: BoolProperty(name="Location", default=True)
    use_rotation: BoolProperty(name="Rotation", default=False)
    use_scale: BoolProperty(name="Scale", default=False)
    synced: BoolProperty(default=False, options={"HIDDEN"})

    #: The incoming rig arrives on Parent for this node.
    stream_input = "Parent"

    def init(self, context):
        super().init(context)
        self.inputs.new(VectorSocket.bl_idname, "Position")
        self._multi_input(ConstraintSocket.bl_idname, "Constraints")
        self.width = 220

    def position_socket(self):
        return self.inputs.get("Position")

    def position(self):
        """The world position this node asks for.

        The **Position socket is the location input**, whether or not a wire is
        plugged into it -- unlinked it is the value field (the Geometry Nodes
        convention), linked it follows the marker.

        This used to return None unless something was wired in, so the socket
        was drawn editable, accepted typing, and silently discarded it. The
        node also carried a separate Location field that did work, which meant
        two location inputs that disagreed.
        """
        sock = self.position_socket()
        if sock is None:
            return None
        return sock.get_value()

    def position_is_linked(self):
        sock = self.position_socket()
        return bool(sock is not None and sock.is_linked)

    def set_position(self, value):
        """Write the socket's own value, bypassing the rebuild callback."""
        sock = self.position_socket()
        if sock is None:
            return
        from ..tree import suspend_live_update

        with suspend_live_update():
            sock.default_value = tuple(value)

    def linked_marker(self):
        """(node, marker) driving the Position input, or (None, None)."""
        sock = self.inputs.get("Position")
        links = socket_links(sock) if sock is not None and sock.is_linked else ()
        if not links:
            return None, None
        link = links[0]
        key = getattr(link.from_socket, "marker_key", "")
        node = link.from_node
        if not key or not hasattr(node, "marker_by_key"):
            return None, None
        return node, node.marker_by_key(key)

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
            self.set_position(loc)
            self.bone_rotation = tuple(rot.to_euler("XYZ"))
            self.bone_scale = tuple(scale)
            self.synced = True
        finally:
            _syncing_bone_read = False
        # Move the marker onto the bone rather than the other way round. A
        # marker wired in sits wherever it was dropped, so without this the
        # bone would jump to the handle the moment it is selected; now the
        # handle lands on the bone and you drag from there.
        marker_node, marker = self.linked_marker()
        if marker is not None:
            marker_node.write_marker(marker, "position", tuple(loc))
        return True

    def live_bone(self):
        if not self.bone:
            return None, None
        obj = self.resolve_armature()
        if obj is None or obj.pose is None:
            return None, None
        return obj, obj.pose.bones.get(self.bone)

    def on_socket_edited(self, sock):
        if sock.name == "Position" and not self.use_location:
            self.use_location = True

    def marker_role(self, socket_name):
        # Location unticked, a wired marker poses nothing: it follows the bone.
        return ("absolute", self.use_location) if socket_name == "Position" else None

    def absorb(self, obj, pbone, delta):
        """Ticked components are driven: the user's move is folded into them."""
        from .. import livelink

        changed = False
        if self.use_location:
            # Into the field, or through the wire into the marker.
            value = Vector(self.position() or (0.0, 0.0, 0.0)) + delta.world_loc
            changed |= write_input(self, "Position", value)
        updates = []
        if self.use_rotation:
            value = livelink.compose(delta.world_rot, self.bone_rotation)
            if not same_vec(value, self.bone_rotation):
                updates.append(("bone_rotation", value))
        if self.use_scale:
            _l, _r, scale = delta.now.world.decompose()
            if not same_vec(scale, self.bone_scale):
                updates.append(("bone_scale", tuple(scale)))
        return self._set_fields(updates) or changed

    def readout(self, obj, pbone):
        """Unticked components are not driven: they just show the bone."""
        from mathutils import Euler

        loc, _rot, scale = (obj.matrix_world @ pbone.matrix).decompose()
        changed = False
        if not self.use_location and not self.position_is_linked():
            changed |= write_socket_value(self, "Position", loc)
        updates = []
        if not self.use_rotation:
            e = (obj.matrix_world @ pbone.matrix).to_euler(
                "XYZ", Euler(self.bone_rotation, "XYZ")
            )
            if not same_vec((e.x, e.y, e.z), self.bone_rotation):
                updates.append(("bone_rotation", (e.x, e.y, e.z)))
        if not self.use_scale and not same_vec(scale, self.bone_scale):
            updates.append(("bone_scale", tuple(scale)))
        return self._set_fields(updates) or changed

    def _set_fields(self, updates):
        global _syncing_bone_read
        from ..tree import suspend_live_update

        if not updates:
            return False
        _syncing_bone_read = True
        try:
            with suspend_live_update():
                for name, value in updates:
                    setattr(self, name, value)
                self.synced = True
        finally:
            _syncing_bone_read = False
        return True

    def draw_buttons(self, context, layout):
        layout.context_pointer_set("node", self)
        obj = self.rig_for_ui()
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
        self._draw_blockers(layout, obj)

        # Location lives on the Position socket, not here: one input, editable
        # in place when nothing is wired in, driven by a marker when something
        # is. Two location fields that disagreed is what made the socket look
        # broken.
        row = layout.row(align=True)
        row.prop(self, "use_location", text="")
        if self.position_is_linked():
            row.label(text="Position: from marker", icon="EMPTY_AXIS")
        else:
            row.label(text="Position: see input below", icon="EMPTY_AXIS")

        for flag, prop in (
            ("use_rotation", "bone_rotation"),
            ("use_scale", "bone_scale"),
        ):
            row = layout.row(align=True)
            row.prop(self, flag, text="")
            sub = row.column(align=True)
            # Unticked, the field follows the bone and is read-only: it is a
            # readout, not an input, so showing it editable would invite an
            # edit that the next rebuild throws away.
            sub.enabled = getattr(self, flag)
            sub.prop(self, prop, text="")
        layout.label(text="Unticked values follow the bone", icon="INFO")

    def _draw_blockers(self, layout, obj):
        """Say up front when Blender will refuse the pose.

        A connected bone cannot be translated (its head is pinned to the
        parent's tail) and locked channels are not writable. Blender accepts
        the write and silently drops it, so without this the node looks like
        it simply does not work.
        """
        if obj is None or obj.pose is None or not self.bone:
            return
        pbone = obj.pose.bones.get(self.bone)
        if pbone is None:
            layout.label(text="No such bone on the rig", icon="ERROR")
            return
        notes = []
        if self.use_location and pbone.bone.use_connect:
            notes.append("Connected: cannot be moved")
        if self.use_location:
            locked = [a for a, l in zip("XYZ", pbone.lock_location) if l]
            if locked:
                notes.append("Location " + "".join(locked) + " locked")
        if self.use_rotation and all(pbone.lock_rotation):
            notes.append("Rotation locked")
        if self.use_scale and all(pbone.lock_scale):
            notes.append("Scale locked")
        for note in notes:
            layout.label(text=note, icon="LOCKED")

    def eval_bones(self, ctx):
        bones = self.stream(ctx)
        if not self.bone:
            return bones
        constraints = gather_input_constraints(self, "Constraints", ctx)
        position = self.position()
        mine = [b for b in bones if b.name == self.bone][:1]
        for b in editable(bones, mine):
            if self.use_location and position is not None:
                b.pose_location = tuple(position)
                b.pose_offset = ZERO
                b.pose_local_offset = ZERO
            if self.use_rotation:
                b.pose_rotation = tuple(self.bone_rotation)
                b.pose_rotation_offset = ZERO
                b.pose_local_rotation = ZERO
            if self.use_scale:
                b.pose_scale = tuple(self.bone_scale)
            # Constraints are pose-stack data, so they only reach the
            # armature when the Output is in Full Rig mode.
            if constraints:
                b.constraints = list(b.constraints) + constraints
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
        self.inputs.new(RigSocket.bl_idname, "Parent")
        self._multi_input(ConstraintSocket.bl_idname, "Tip Constraints")
        self.outputs.new(RigSocket.bl_idname, "Rig")

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
        return list(parents) + bones  # the parents pass on unchanged
