"""Transform: Position, Rotation, Transform and Snap, the pose modifiers.

They write the pose only and never touch rest geometry, so they cannot deform
a rig they are layered onto.
"""

import logging

import bpy
from bpy.types import Node
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from mathutils import Vector

from ..sockets import (
    RotationSocket,
    ScaleSocket,
    TransformSocket,
    VectorSocket,
    marker_attrs,
)
from .base import (
    ZERO,
    LiveLinkMixin,
    ModifierNodeBase,
    bone_select_prop,
    turn_angle,
    write_input,
    write_socket_value,
)
from .marker_base import marker_owner

log = logging.getLogger(__name__)


def _nonzero(value):
    return any(abs(v) > 1e-9 for v in value)


def _not_unit(value):
    return any(abs(v - 1.0) > 1e-9 for v in value)


def _add(a, b):
    return tuple(x + y for x, y in zip(a or ZERO, b))


def _compose(offset, previous):
    """Rotation ``offset`` applied after ``previous``, both XYZ Euler.

    Composed as quaternions: adding Euler angles is only right for rotations
    about a single axis, and stacks of Rotation / Transform nodes are exactly
    where that breaks.
    """
    from mathutils import Euler

    q = Euler(tuple(offset), "XYZ").to_quaternion() @ Euler(
        tuple(previous or ZERO), "XYZ"
    ).to_quaternion()
    e = q.to_euler("XYZ")
    return (e.x, e.y, e.z)


def _on_transform_select_changed(self, context):
    """A bone was picked: seed the absolute value from it, then rebuild."""
    self.seed_from_bone()
    self.schedule_rebuild()


# True while code (not the user) flips Set Position / Set Rotation, so the
# callback does not seed over a value that is being deliberately kept.
_seeding_suspended = False


def _on_use_absolute_changed(self, context):
    """Set Position / Set Rotation toggled on: start from where the bone is,
    so ticking the box never makes it jump."""
    if not _seeding_suspended:
        self.seed_from_bone()
    self.schedule_rebuild()


def _set_without_seeding(node, prop, value):
    """Flip a Set checkbox without the callback overwriting the socket.

    ``node[prop] = value`` used to be the way to skip an update callback. On
    Blender 5.x it no longer reaches the RNA property at all -- it creates a
    separate custom property of the same name and leaves the real one alone --
    so the guard has to be explicit.
    """
    global _seeding_suspended
    _seeding_suspended = True
    try:
        setattr(node, prop, value)
    finally:
        _seeding_suspended = False


def _draw_live_state(node, layout):
    """Say whether the node is mirroring a bone, and if not, why."""
    obj, pbone = node.live_bone()
    if pbone is not None:
        layout.label(text=f"Live: {pbone.name}", icon="LINKED")
    elif node.bone and ";" in node.bone:
        layout.label(text="Live link needs a single bone", icon="UNLINKED")


class _TransformNodeBase(LiveLinkMixin, ModifierNodeBase):
    """Shared plumbing for the Transform category.

    Each node lists its value sockets in ``_INPUTS``. ``ensure_inputs`` brings
    a node saved by an older version into line -- earlier versions used plain
    Vector sockets for rotation, which showed metres and read "90" as ninety
    radians -- so an existing graph is fixed on sight rather than needing its
    nodes deleted and re-added.
    """

    _INPUTS = ()  # (name, socket class, default or None)

    def init(self, context):
        from ..tree import suspend_live_update

        super().init(context)
        # Suspended: a socket's default is the node setting itself up, not
        # the user typing -- which would turn a Set toggle on (Scale's 1).
        with suspend_live_update():
            for name, cls, default in self._INPUTS:
                sock = self.inputs.new(cls.bl_idname, name)
                if default is not None:
                    sock.default_value = default

    def ensure_inputs(self):
        """Returns the names of sockets that did not exist before."""
        created = set()
        for index, (name, cls, default) in enumerate(self._INPUTS, start=1):
            sock = self.inputs.get(name)
            if sock is not None and sock.bl_idname == cls.bl_idname:
                continue
            carried = None
            if sock is not None:
                # Keep what the user typed: an old Vector-typed Rotation held
                # radians, which is exactly what the new socket stores.
                carried = tuple(getattr(sock, "default_value", ()) or ()) or None
                self.inputs.remove(sock)
            else:
                created.add(name)
            sock = self.inputs.new(cls.bl_idname, name)
            from ..tree import suspend_live_update

            with suspend_live_update():  # not an edit, as in init()
                if carried is not None and len(carried) == 3:
                    sock.default_value = carried
                elif default is not None:
                    sock.default_value = default
            try:
                self.inputs.move(len(self.inputs) - 1, index)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass  # order is cosmetic; the socket works wherever it sits
        return created

    def socket_value(self, name, fallback=ZERO):
        sock = self.inputs.get(name)
        if sock is None or not hasattr(sock, "get_value"):
            return tuple(fallback)
        return tuple(float(v) for v in sock.get_value())

    def socket_linked(self, name):
        sock = self.inputs.get(name)
        return bool(sock is not None and sock.is_linked)

    def single_bone(self):
        """(armature, pose bone) when exactly one bone is selected."""
        names = [n.strip() for n in self.bone.split(";") if n.strip()]
        if len(names) != 1:
            return None, None
        obj = self.resolve_armature()
        if obj is None or obj.pose is None:
            return None, None
        return obj, obj.pose.bones.get(names[0])

    def write_socket(self, name, value):
        """Set an input's value -- its field, or the marker wired into it --
        without scheduling a rebuild per axis."""
        return write_input(self, name, value)

    def live_bone(self):
        return self.single_bone()

    def on_socket_edited(self, sock):
        """The user typed into a socket. Overridden where that means more."""

    def seed_from_bone(self):
        """Overridden by the nodes that have an absolute value to seed."""

    @staticmethod
    def top_level(bones, chosen):
        """``chosen`` minus bones whose ancestor is also chosen.

        A relative move is resolved against the rest pose, which follows the
        parent -- so moving a parent and its child by +1 X would carry the
        child +1 with its parent and then move it +1 more. Blender's own
        transform has the same problem and the same answer: with a parent and
        child both selected, only the parent is transformed and the child is
        carried. Each bone ends up moved exactly once.
        """
        names = {b.name for b in chosen}
        parents = {b.name: b.parent for b in bones}
        out = []
        for b in chosen:
            parent, seen = parents.get(b.name), set()
            carried = False
            while parent and parent not in seen:
                if parent in names:
                    carried = True
                    break
                seen.add(parent)
                parent = parents.get(parent)
            if not carried:
                out.append(b)
        return out

    def draw_buttons(self, context, layout):
        self.ensure_inputs()
        self.draw_bone_select(layout)


class PositionNode(_TransformNodeBase, Node):
    """Set Position -- move the selected bones, like the Geometry Nodes node.

    * **Position**: where the bones go, in world space. Used only when *Set
      Position* is ticked or something is wired in; otherwise the bones keep
      their location. That default is what makes a freshly added node
      harmless: it used to take (0, 0, 0) with every bone selected, so
      dropping one on the wire sent the whole rig to the origin.
    * **Offset**: added on top, along world axes. On its own it moves bones
      relative to their rest pose, so rebuilding never adds it twice.

    Setting a position replaces any offset an earlier node applied -- the
    later node wins, as in Geometry Nodes. Pose only; rest geometry is never
    touched.

    A marker wired into Position takes the bone's place -- or, given a seed
    bone, a point along that one, the Offset taking up the difference so
    nothing moves: the Human Skeleton's elbow sits on the elbow and moves
    the pole target standing well behind it.
    """

    bl_idname = "ArmatureNodesPositionNode"
    bl_label = "Position"
    bl_icon = "CON_LOCLIKE"

    bone: bone_select_prop(update=_on_transform_select_changed)
    use_position: BoolProperty(
        name="Set Position",
        description=(
            "Move the bones to Position. Off, they keep their location and "
            "only Offset applies"
        ),
        default=False,
        update=_on_use_absolute_changed,
    )
    # Where a wired marker is seeded instead: this far from the head of
    # this bone to its tail.
    seed_bone: StringProperty(name="Seed Bone", options={"HIDDEN"})
    seed_at: FloatProperty(name="Seed At", min=0.0, max=1.0, options={"HIDDEN"})

    _INPUTS = (("Position", VectorSocket, None), ("Offset", VectorSocket, None))

    def uses_absolute(self):
        return self.use_position or self.socket_linked("Position")

    def ensure_inputs(self):
        created = super().ensure_inputs()
        # A node saved before Offset existed applied its Position whenever it
        # ran. If one was set, tick the box so it keeps doing so instead of
        # quietly becoming a no-op.
        if "Offset" in created and not self.socket_linked("Position"):
            if _nonzero(self.socket_value("Position")):
                # Not a plain assignment: the callback would seed Position
                # from the bone and overwrite the very value being kept.
                _set_without_seeding(self, "use_position", True)
        return created

    def seed_from_bone(self):
        from .. import livelink

        if not self.use_position:
            return
        obj, pbone = self.single_bone()
        if pbone is not None:
            here = livelink.driven_world(obj, pbone).to_translation()
            self.write_socket("Position", self.marker_seed("Position", "position", here))

    def marker_seed(self, socket_name, attr, value):
        """The Position that keeps the bone at ``value``: this node adds its
        own Offset on top, so taking the bone's place as it is would move it
        by the Offset -- or, with a seed bone, the point on it, the Offset
        becoming the way from there to the bone."""
        if socket_name != "Position" or attr != "position":
            return value
        obj, _pbone = self.single_bone()
        seed = obj.pose.bones.get(self.seed_bone) if obj is not None and self.seed_bone else None
        if seed is None:
            return tuple(Vector(value) - Vector(self.socket_value("Offset")))
        at = obj.matrix_world @ seed.head.lerp(seed.tail, self.seed_at)
        self.write_socket("Offset", tuple(Vector(value) - at))
        return tuple(at)

    def on_socket_edited(self, sock):
        # Typing a position is asking for it: take the bone over, rather than
        # leaving a field that looks like an input but only displays.
        if sock.name == "Position" and not self.use_position:
            _set_without_seeding(self, "use_position", True)

    def marker_role(self, socket_name):
        """How a marker wired into ``socket_name`` is used: (kind, driving).

        "absolute" sets the bone's world value, "relative" offsets it.
        """
        if socket_name == "Position":
            return ("absolute", True)  # a wire into Position always sets it
        if socket_name == "Offset":
            return ("relative", True)
        return None

    def marker_frame(self, socket_name, obj, pbone):
        """What a marker in Offset is measured from: (matrix, local axes)."""
        from mathutils import Matrix

        from .. import livelink

        if self.uses_absolute():
            return Matrix.Translation(self.socket_value("Position")), False
        return livelink.rest_world(obj, pbone), False

    def absorb(self, obj, pbone, delta):
        """The user moved the bone: fold the move into whatever drives it.

        Writes go through ``write_socket``, which lands in a wired marker when
        there is one, so a marker follows the grab exactly as the field would.
        """
        if self.uses_absolute():
            value = Vector(self.socket_value("Position")) + delta.world_loc
            return self.write_socket("Position", value)
        if _nonzero(self.socket_value("Offset")) or self.socket_linked("Offset"):
            # Measured from rest: a parent or object move is not an offset.
            value = Vector(self.socket_value("Offset")) + delta.rel_loc
            return self.write_socket("Offset", value)
        return False

    def readout(self, obj, pbone):
        """Not driving: Position simply shows where the bone is."""
        if self.uses_absolute():
            return False
        return self.write_socket("Position", (obj.matrix_world @ pbone.matrix).to_translation())

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "use_position")
        if self.socket_linked("Position"):
            layout.label(text="Position from the wired input", icon="LINKED")
        elif self.uses_absolute() and not self.bone:
            layout.label(text="Every bone goes to one point", icon="ERROR")
        _draw_live_state(self, layout)

    def eval_bones(self, ctx):
        bones = self.stream(ctx)
        absolute = self.uses_absolute()
        position = self.socket_value("Position")
        offset = self.socket_value("Offset")
        has_offset = _nonzero(offset)
        if not absolute and not has_offset:
            return bones  # nothing asked: the node is a pass-through
        chosen = self.edit(bones)
        if absolute:
            for b in chosen:
                b.pose_location = position
                b.pose_offset = ZERO
                b.pose_local_offset = ZERO
        if has_offset:
            for b in self.top_level(bones, chosen):
                b.pose_offset = _add(b.pose_offset, offset)
        return bones


class RotationNode(_TransformNodeBase, Node):
    """Set Rotation -- orient the selected bones, in degrees.

    * **Rotation**: the world orientation to give the bones. Used only when
      *Set Rotation* is ticked or something is wired in (a wired marker
      supplies its own rotation). Off, the bones keep their orientation.
    * **Offset**: turned on top, along world axes, about each bone's own head.

    Setting a rotation replaces any rotation offset an earlier node applied.
    """

    bl_idname = "ArmatureNodesRotationNode"
    bl_label = "Rotation"
    bl_icon = "CON_ROTLIKE"

    bone: bone_select_prop(update=_on_transform_select_changed)
    use_rotation: BoolProperty(
        name="Set Rotation",
        description=(
            "Give the bones this orientation. Off, they keep theirs and only "
            "Offset applies"
        ),
        default=False,
        update=_on_use_absolute_changed,
    )

    _INPUTS = (("Rotation", RotationSocket, None), ("Offset", RotationSocket, None))

    def uses_absolute(self):
        return self.use_rotation or self.socket_linked("Rotation")

    def ensure_inputs(self):
        created = super().ensure_inputs()
        if "Offset" in created and not self.socket_linked("Rotation"):
            if _nonzero(self.socket_value("Rotation")):
                _set_without_seeding(self, "use_rotation", True)
        return created

    def seed_from_bone(self):
        from .. import livelink

        if not self.use_rotation:
            return
        obj, pbone = self.single_bone()
        if pbone is not None:
            e = livelink.driven_world(obj, pbone).to_euler("XYZ")
            self.write_socket("Rotation", self.marker_seed("Rotation", "rotation", (e.x, e.y, e.z)))

    def marker_seed(self, socket_name, attr, value):
        """The Rotation that keeps the bone turned as ``value``: this node's
        own Offset turns it further, so that is taken back out first."""
        from mathutils import Euler

        if socket_name == "Rotation" and attr == "rotation":
            own = Euler(self.socket_value("Offset"), "XYZ").to_quaternion()
            q = own.inverted() @ Euler(value, "XYZ").to_quaternion()
            e = q.to_euler("XYZ", Euler(value, "XYZ"))
            return (e.x, e.y, e.z)
        return value

    def on_socket_edited(self, sock):
        if sock.name == "Rotation" and not self.use_rotation:
            _set_without_seeding(self, "use_rotation", True)

    def marker_role(self, socket_name):
        if socket_name == "Rotation":
            return ("absolute", True)
        if socket_name == "Offset":
            return ("relative", True)
        return None

    def marker_frame(self, socket_name, obj, pbone):
        """Offset turns on top of Rotation when that is set, else of rest."""
        from mathutils import Euler

        from .. import livelink

        if self.uses_absolute():
            return Euler(self.socket_value("Rotation"), "XYZ").to_matrix().to_4x4(), False
        return livelink.rest_world(obj, pbone), False

    def absorb(self, obj, pbone, delta):
        from mathutils import Euler

        from .. import livelink

        if self.uses_absolute():
            # The node's own Offset sits on top of Rotation, so the user's turn
            # is taken into Rotation's frame before it is composed in.
            own = Euler(self.socket_value("Offset"), "XYZ").to_quaternion()
            spin = own.inverted() @ delta.world_rot @ own
            value = livelink.compose(spin, self.socket_value("Rotation"))
            return self.write_socket("Rotation", value)
        if _nonzero(self.socket_value("Offset")) or self.socket_linked("Offset"):
            value = livelink.compose(delta.rel_rot, self.socket_value("Offset"))
            return self.write_socket("Offset", value)
        return False

    def readout(self, obj, pbone):
        from mathutils import Euler

        if self.uses_absolute():
            return False
        current = Euler(self.socket_value("Rotation"), "XYZ")
        e = (obj.matrix_world @ pbone.matrix).to_euler("XYZ", current)
        return self.write_socket("Rotation", (e.x, e.y, e.z))

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "use_rotation")
        if self.socket_linked("Rotation"):
            layout.label(text="Rotation from the wired input", icon="LINKED")
        _draw_live_state(self, layout)

    def eval_bones(self, ctx):
        bones = self.stream(ctx)
        absolute = self.uses_absolute()
        rotation = self.socket_value("Rotation")
        offset = self.socket_value("Offset")
        has_offset = _nonzero(offset)
        if not absolute and not has_offset:
            return bones
        chosen = self.edit(bones)
        if absolute:
            for b in chosen:
                b.pose_rotation = rotation
                b.pose_rotation_offset = ZERO
                b.pose_local_rotation = ZERO
        if has_offset:
            for b in self.top_level(bones, chosen):
                b.pose_rotation_offset = _compose(offset, b.pose_rotation_offset)
        return bones


#: The Transform node's fields, and the toggle that sets each.
_TRANSFORM_FLAGS = {"Location": "use_location", "Rotation": "use_rotation", "Scale": "use_scale"}


def _on_transform_set_toggled(field):
    """Update for one of the Transform node's Set toggles.

    Turned on, the value starts from where the bone is, so turning it on
    never moves the bone.
    """

    def update(self, context):
        if not _seeding_suspended and getattr(self, _TRANSFORM_FLAGS[field]):
            self.seed_from_bone((field,))
        self.schedule_rebuild()

    return update


def _on_transform_space_changed(self, context):
    """The values mean something else in the other space: re-read the bone."""
    self.seed_from_bone()
    self.schedule_rebuild()


def _unset_local(bone, part):
    """``part`` is no longer a local channel set outright."""
    bone.pose_local_set = tuple(p for p in bone.pose_local_set if p != part)


class TransformNode(_TransformNodeBase, Node):
    """Transform -- the bone's location, rotation and scale, live.

    Pick a bone and the fields fill with where it is. **Space** says in what
    terms:

    * **World**: its world Location and Rotation.
    * **Local**: its own channels -- the Location and Rotation in the
      N-panel, which follow the parent the way hand-posing does.

    Scale is the bone's own in both, 1 at rest -- never the armature object's.

    Nothing is applied until asked. A value you have not set is a readout
    that follows the bone, so a fresh node changes nothing. Type a value, or
    turn on its Set toggle, and the node sets that part of the bone; turning
    it on starts from where the bone is, so it never jumps. A later node that
    sets the same part wins.

    The **Transform** input is the whole world transform on one wire. Wire a
    Marker into it and the marker first takes the bone's live Location,
    Rotation and Scale -- nothing moves -- and from then on the handle and the
    bone follow each other. While it is wired the fields are hidden and Space
    does not apply. Unplugged, the fields take over, set, holding the bone
    where the wire left it.
    """

    bl_idname = "ArmatureNodesTransformNode"
    bl_label = "Transform"
    bl_icon = "ORIENTATION_GLOBAL"

    bone: bone_select_prop(update=_on_transform_select_changed)
    space: EnumProperty(
        name="Space",
        items=(
            ("WORLD", "World", "The bone's world location and rotation"),
            ("LOCAL", "Local", "The bone's own Location and Rotation channels, as in the N-panel"),
        ),
        default="WORLD",
        update=_on_transform_space_changed,
    )
    use_location: BoolProperty(
        name="Location",
        description="Set the bone's location to the value below. Off, the field shows where the bone is",
        default=False,
        update=_on_transform_set_toggled("Location"),
    )
    use_rotation: BoolProperty(
        name="Rotation",
        description="Set the bone's rotation to the value below. Off, the field shows the bone's rotation",
        default=False,
        update=_on_transform_set_toggled("Rotation"),
    )
    use_scale: BoolProperty(
        name="Scale",
        description="Set the bone's scale to the value below. Off, the field shows the bone's scale",
        default=False,
        update=_on_transform_set_toggled("Scale"),
    )
    layout_version: IntProperty(name="Layout", default=0, options={"HIDDEN"})

    _INPUTS = (
        ("Transform", TransformSocket, None),
        ("Location", VectorSocket, None),
        ("Rotation", RotationSocket, None),
        ("Scale", ScaleSocket, (1.0, 1.0, 1.0)),
    )
    _FIELDS = ("Location", "Rotation", "Scale")
    #: How the Marker node labels the parts of the Transform wire.
    _WORLD_LABELS = {"position": "Location", "rotation": "Rotation", "scale": "Scale"}
    #: 2: the fields are the bone's own values. Before, offsets from rest.
    _LAYOUT = 2

    def init(self, context):
        super().init(context)
        self.layout_version = self._LAYOUT

    def ensure_inputs(self):
        # Location was called Translation while it was an offset. Renamed in
        # place, it keeps its wires; migrate() converts the value.
        old = self.inputs.get("Translation")
        if old is not None and self.inputs.get("Location") is None:
            old.name = "Location"
        return super().ensure_inputs()

    def migrate(self):
        """Bring a node saved when the fields were offsets from rest up to date.

        Such a node applied every value that was not zero. Here a value is the
        bone's own and is applied when set. So each part it applied is set,
        and re-read from the bone -- which is exactly where those offsets put
        it. Nothing moves.
        """
        if self.layout_version >= self._LAYOUT:
            return
        obj, pbone = self.single_bone()
        if obj is not None and obj.mode == "EDIT":
            return  # the pose is stale in Edit mode: read it on the way out
        self.ensure_inputs()
        applied = [
            field
            for field, used in (
                ("Location", _nonzero(self.socket_value("Location"))),
                ("Rotation", _nonzero(self.socket_value("Rotation"))),
                ("Scale", _not_unit(self.socket_value("Scale", (1.0, 1.0, 1.0)))),
            )
            if used or self.socket_linked(field)
        ]
        if pbone is not None:
            for field in applied:
                _set_without_seeding(self, _TRANSFORM_FLAGS[field], True)
            self.seed_from_bone(applied)
        elif applied:
            # Several bones cannot share one set of values the way they could
            # share an offset: leave them unset rather than pile every bone
            # onto one point.
            log.warning(
                "'%s' moved several bones by an offset; "
                "Transform now sets values, so those moves were dropped",
                self.name,
            )
        self.layout_version = self._LAYOUT

    # -- Values ----------------------------------------------------------------

    def uses_transform(self):
        return self.socket_linked("Transform")

    def wired_transform(self):
        """(location, rotation, scale) in world space from the Transform wire,
        None for a part the source does not carry; None when unwired."""
        sock = self.inputs.get("Transform")
        return sock.get_transform() if sock is not None else None

    def drives(self, field):
        """Whether the node sets this part: toggled on, or a wire into it."""
        return getattr(self, _TRANSFORM_FLAGS[field]) or self.socket_linked(field)

    def fields(self):
        return (
            self.socket_value("Location"),
            self.socket_value("Rotation"),
            self.socket_value("Scale", (1.0, 1.0, 1.0)),
        )

    def bone_values(self, obj, pbone, evaluated=False):
        """(location, rotation, scale) of the bone, as the fields hold them.

        Before constraints by default: the values to set so that nothing
        moves. ``evaluated`` gives what the bone shows, constraints and all,
        for a readout. Local is the channels either way, as the N-panel shows.
        """
        from mathutils import Euler

        from .. import livelink

        if evaluated:
            world = obj.matrix_world @ pbone.matrix
        else:
            world = livelink.driven_world(obj, pbone)
        if self.space == "LOCAL":
            loc, rot, _s = pbone.matrix_basis.decompose()
        else:
            loc, rot, _s = world.decompose()
        e = rot.to_euler("XYZ", Euler(self.socket_value("Rotation"), "XYZ"))
        scale = (obj.matrix_world.inverted_safe() @ world).to_scale()
        return tuple(loc), (e.x, e.y, e.z), tuple(scale)

    def seed_from_bone(self, fields=None):
        """Fill ``fields`` (all by default) with where the bone is now."""
        if self.uses_transform():
            return
        obj, pbone = self.single_bone()
        if pbone is None or obj.mode == "EDIT":
            return
        values = dict(zip(self._FIELDS, self.bone_values(obj, pbone)))
        for field in fields or self._FIELDS:
            self.write_socket(field, values[field])

    def on_socket_edited(self, sock):
        # Typing a value is asking for it: that part is now set.
        flag = _TRANSFORM_FLAGS.get(sock.name)
        if flag and not getattr(self, flag):
            _set_without_seeding(self, flag, True)

    def update(self):
        """Links changed: show the three fields only while they are in use."""
        self.show_fields()

    def show_fields(self):
        hide = self.uses_transform()
        for field in self._FIELDS:
            sock = self.inputs.get(field)
            if sock is not None and sock.hide != hide and not sock.is_linked:
                sock.hide = hide

    # -- Markers ---------------------------------------------------------------

    def marker_role(self, socket_name):
        if socket_name == "Transform":
            return ("absolute", True)  # the bone's world transform
        if socket_name in _TRANSFORM_FLAGS:
            # A wired field is set, like a typed one. In Local space its
            # values are channels -- offsets from rest along the bone -- so
            # the marker's handle is placed through the rest frame.
            return ("relative" if self.space == "LOCAL" else "absolute", True)
        return None

    def marker_label(self, socket_name, attr):
        return self._WORLD_LABELS[attr] if socket_name == "Transform" else socket_name

    def marker_frame(self, socket_name, obj, pbone):
        """Local channels are measured from rest, along the bone's own axes."""
        from .. import livelink

        return livelink.rest_world(obj, pbone), True

    def marker_unplugged(self, socket_name, marker, linked_too=False):
        """A wire came off: the fields take over, holding the bone.

        Set, not left as readouts: a part nothing sets goes back to the rig's
        own pose, and the bone would jump there.
        """
        if socket_name == "Transform":
            for field in self._FIELDS:
                _set_without_seeding(self, _TRANSFORM_FLAGS[field], True)
            obj, pbone = self.single_bone()
            if pbone is None:
                return False
            changed = False
            for field, value in zip(self._FIELDS, self.bone_values(obj, pbone)):
                changed |= write_socket_value(self, field, value, linked_too)
            return changed
        flag = _TRANSFORM_FLAGS.get(socket_name)
        if flag is None:
            return None
        _set_without_seeding(self, flag, True)
        attr = marker_attrs(self.inputs[socket_name])[0]
        owner = marker_owner(marker)
        value = owner.marker_value(marker, attr) if owner is not None else getattr(marker, attr)
        return write_socket_value(self, socket_name, value, linked_too)

    # -- Live link -------------------------------------------------------------

    def follow_live(self):
        self.migrate()
        self.show_fields()  # backstop: update() is not called for every edit
        return super().follow_live()

    def absorb(self, obj, pbone, delta):
        """The user moved the bone: the parts the node sets take the move.

        Only the move -- never the pose the node wrote -- so a constrained
        bone does not chase itself. Parts it does not set are readouts.
        """
        from .. import livelink

        wired = self.wired_transform()
        if wired is not None:
            return self._absorb_world(wired, delta)
        local = self.space == "LOCAL"
        loc_step = delta.local_loc if local else delta.world_loc
        rot_step = delta.local_rot if local else delta.world_rot
        loc, rot, scale = self.fields()
        changed = False
        if self.drives("Location") and loc_step.length > 1e-5:
            changed |= self.write_socket("Location", Vector(loc) + loc_step)
        if self.drives("Rotation") and turn_angle(rot_step) > 1e-5:
            changed |= self.write_socket("Rotation", livelink.compose(rot_step, rot))
        if self.drives("Scale") and (delta.scale_ratio - Vector((1.0, 1.0, 1.0))).length > 1e-5:
            value = tuple(a * b for a, b in zip(scale, delta.scale_ratio))
            changed |= self.write_socket("Scale", value)
        return changed

    def _absorb_world(self, wired, delta):
        """A move of the bone, into the world transform on the wire."""
        from .. import livelink

        loc, rot, scale = wired
        changed = False
        if loc is not None and delta.world_loc.length > 1e-5:
            value = Vector(loc) + delta.world_loc
            changed |= write_input(self, "Transform", value, "position")
        if rot is not None and turn_angle(delta.world_rot) > 1e-5:
            value = livelink.compose(delta.world_rot, rot)
            changed |= write_input(self, "Transform", value, "rotation")
        if scale is not None and (delta.scale_ratio - Vector((1.0, 1.0, 1.0))).length > 1e-5:
            value = tuple(a * b for a, b in zip(scale, delta.scale_ratio))
            changed |= write_input(self, "Transform", value, "scale")
        return changed

    def readout(self, obj, pbone):
        """Parts the node does not set show where the bone is."""
        if self.uses_transform():
            return False
        changed = False
        for field, value in zip(self._FIELDS, self.bone_values(obj, pbone, evaluated=True)):
            if not self.drives(field):
                changed |= write_socket_value(self, field, value)
        return changed

    # -- UI and evaluation -------------------------------------------------------

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        if self.uses_transform():
            layout.label(text="World transform from the wire", icon="LINKED")
        else:
            layout.prop(self, "space", expand=True)
            row = layout.row(align=True)
            row.label(text="Set")
            for field in self._FIELDS:
                row.prop(self, _TRANSFORM_FLAGS[field], toggle=True)
            names = [n for n in self.bone.split(";") if n.strip()]
            if self.space == "WORLD" and self.drives("Location") and len(names) != 1:
                layout.label(text="Every bone goes to one point", icon="ERROR")
        _draw_live_state(self, layout)

    def eval_bones(self, ctx):
        bones = self.stream(ctx)
        wired = self.wired_transform()
        if wired is not None:
            return self._set_world(bones, wired)
        loc, rot, scale = self.fields()
        parts = (
            loc if self.drives("Location") else None,
            rot if self.drives("Rotation") else None,
            scale if self.drives("Scale") else None,
        )
        if all(p is None for p in parts):
            return bones  # nothing set: a fresh node changes nothing
        if self.space == "LOCAL":
            return self._set_local(bones, parts)
        return self._set_world(bones, parts)

    def _set_world(self, bones, parts):
        """World values, set outright. As with Set Position, a set value
        replaces any offset an earlier node applied."""
        loc, rot, scale = parts
        for b in self.edit(bones):
            if loc is not None:
                b.pose_location = tuple(loc)
                b.pose_offset = ZERO
                b.pose_local_offset = ZERO
                _unset_local(b, "location")
            if rot is not None:
                b.pose_rotation = tuple(rot)
                b.pose_rotation_offset = ZERO
                b.pose_local_rotation = ZERO
                _unset_local(b, "rotation")
            if scale is not None:
                b.pose_scale = tuple(scale)
        return bones

    def _set_local(self, bones, parts):
        """The bones' own channels, set outright -- exactly as typing the same
        values into each bone's N-panel would."""
        loc, rot, scale = parts
        for b in self.edit(bones):
            local_set = set(b.pose_local_set)
            if loc is not None:
                b.pose_local_offset = tuple(loc)
                b.pose_location = None
                b.pose_offset = ZERO
                local_set.add("location")
            if rot is not None:
                b.pose_local_rotation = tuple(rot)
                b.pose_rotation = None
                b.pose_rotation_offset = ZERO
                local_set.add("rotation")
            if scale is not None:
                b.pose_scale = tuple(scale)
            b.pose_local_set = tuple(sorted(local_set))
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


class SnapNode(ModifierNodeBase, Node):
    """Snap the selected bones onto a mesh.

    Positions a control against real geometry instead of by eye: the object's
    origin, its bounding-box centre, the median of its vertices, its volume
    centroid, or the closest point on its surface. Like the other Transform
    nodes it writes the pose only.
    """

    bl_idname = "ArmatureNodesSnapNode"
    bl_label = "Snap"
    bl_icon = "SNAP_ON"

    bone: bone_select_prop()
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
        for b in self.edit(bones):
            # SURFACE needs a point to be closest TO; the bone's own head is
            # the only sensible reference at evaluation time.
            reference = Vector(b.pose_location) if b.pose_location else Vector(b.head)
            try:
                anchor = _mesh_anchor(obj, self.mode, reference)
            except Exception as exc:  # noqa: BLE001
                log.warning("Snap failed on '%s': %s", b.name, exc)
                continue
            b.pose_location = tuple(Vector(anchor) + offset)
            b.pose_offset = ZERO
            b.pose_local_offset = ZERO
        return bones
