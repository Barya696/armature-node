"""The rig as plain data.

Every type here is a frozen dataclass and this module imports nothing from
``bpy``. That is the point: the whole model is unit-testable without Blender,
and a node can only ever hand back a *new* record rather than reaching into a
live armature.

Frozen buys immutability of the fields themselves. Nested ``dict`` values
(constraint props, pose locks, bone colours) are still technically mutable, so
the convention is that nothing mutates a record in place -- ``model.ops``
returns rebuilt records instead. Sequence fields are tuples so equality is
value-based and deterministic, which is what ``model.diff`` relies on.

Pose transforms are never *captured* -- posing is what the graph does, and a
record that remembered the live pose would fight the Transform nodes on every
build. They do exist on the type, as :class:`TransformDef`, so that the graph
setting a pose flows through the same diff / restore machinery as every other
field; see that class for why.
"""

from dataclasses import dataclass, field, replace
from typing import Optional

__all__ = [
    "RestDef",
    "MembershipDef",
    "DisplayDef",
    "PoseDef",
    "ConstraintDef",
    "TransformDef",
    "BoneDef",
    "ArmatureDef",
    "SourceDef",
    "RigRecord",
    "WidgetLib",
    "replace",
]


@dataclass(frozen=True)
class RestDef:
    """Edit-bone geometry and the flags that live on ``Bone``."""

    head: tuple = (0.0, 0.0, 0.0)
    tail: tuple = (0.0, 0.0, 1.0)
    roll: float = 0.0
    parent: Optional[str] = None
    connect: bool = False
    deform: bool = True
    inherit_scale: str = "FULL"
    use_local_location: bool = True
    envelope_distance: float = 0.25
    envelope_weight: float = 1.0
    head_radius: float = 0.1
    tail_radius: float = 0.05


@dataclass(frozen=True)
class MembershipDef:
    """Which bone collections (4.x) or layers (3.6) the bone belongs to.

    ``layers`` stays ``None`` on a 4.x capture and ``collections`` stays empty
    on a 3.6 one, so a record says which world it came from instead of
    guessing.
    """

    collections: tuple = ()
    layers: Optional[tuple] = None


@dataclass(frozen=True)
class DisplayDef:
    """How the bone is drawn: its custom shape and that shape's placement."""

    shape: str = ""
    preset: str = "NONE"  # generator preset when the widget object is missing
    scale: tuple = (1.0, 1.0, 1.0)
    translation: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = (0.0, 0.0, 0.0)
    wire_width: float = 1.0
    use_bone_size: bool = True
    override: str = ""  # custom_shape_transform, by bone name
    show_wire: bool = False
    color: Optional[dict] = None


@dataclass(frozen=True)
class PoseDef:
    """Pose-bone settings that are rig definition, not animation."""

    rotation_mode: str = "QUATERNION"
    locks: dict = field(default_factory=dict)
    ik: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ConstraintDef:
    """One pose constraint.

    ``props`` is a generic RNA dump: every writable property of the constraint
    by name, with ID pointers stored as object names. Generic because a
    hard-coded attribute list silently drops whatever it does not know about,
    which is how the previous version lost constraint settings.
    """

    type: str
    name: str = ""
    props: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TransformDef:
    """The world-space pose the *graph* asks for. ``None`` means "not set".

    The spec excludes pose transforms from the record, and capture honours
    that: this section is always empty on a captured record. It exists on the
    type so that posing flows through the same diff / touched / restore
    machinery as everything else -- the graph sets it, ``diff`` reports it as
    a change, ``apply`` writes it, and the next build restores it to ``None``
    when the node that set it is gone, which clears the pose.

    Without it the graph would need a side channel for posing, and posing
    would be the one effect that a deleted node could not undo.
    """

    location: Optional[tuple] = None
    rotation: Optional[tuple] = None  # euler XYZ, radians
    scale: Optional[tuple] = None
    # Relative moves, resolved at apply time against the bone's REST pose
    # (identity basis, following its parent's current pose) -- never against
    # the live pose, which would re-add the offset on every build and drift.
    # World offsets are along global axes; local ones are the bone's own
    # Location / Rotation channels, exactly as in the N-panel.
    offset: Optional[tuple] = None
    rotation_offset: Optional[tuple] = None  # euler XYZ, radians
    local_offset: Optional[tuple] = None
    local_rotation: Optional[tuple] = None  # euler XYZ, radians

    def is_empty(self):
        """True when the graph asks nothing of this bone's pose."""
        return all(
            v is None
            for v in (
                self.location, self.rotation, self.scale, self.offset,
                self.rotation_offset, self.local_offset, self.local_rotation,
            )
        )


@dataclass(frozen=True)
class BoneDef:
    name: str
    rest: RestDef = field(default_factory=RestDef)
    membership: MembershipDef = field(default_factory=MembershipDef)
    display: DisplayDef = field(default_factory=DisplayDef)
    pose: PoseDef = field(default_factory=PoseDef)
    transform: TransformDef = field(default_factory=TransformDef)
    constraints: tuple = ()


@dataclass(frozen=True)
class ArmatureDef:
    display_type: str = "OCTAHEDRAL"
    show_in_front: bool = False
    pose_position: str = "POSE"
    collections: tuple = ()


@dataclass(frozen=True)
class SourceDef:
    """Provenance, for diagnostics and for spotting a record from another rig."""

    object: str = ""
    armature: str = ""
    captured_at: str = ""
    blender: str = ""


@dataclass(frozen=True)
class RigRecord:
    """A complete rig, self-sufficient enough to rebuild it."""

    version: int = 2
    source: SourceDef = field(default_factory=SourceDef)
    armature: ArmatureDef = field(default_factory=ArmatureDef)
    bones: dict = field(default_factory=dict)  # name -> BoneDef

    def bone(self, name):
        return self.bones.get(name)

    def names(self):
        """Bone names in insertion order, which capture keeps parent-first."""
        return list(self.bones)

    def with_bones(self, bones):
        return replace(self, bones=bones)


@dataclass(frozen=True)
class WidgetLib:
    """Widget meshes by object name.

    Stored separately from the record because geometry is bulky and changes on
    a different cadence: the record is read on every evaluation, the library
    only when a widget object has to be rebuilt.
    """

    widgets: dict = field(default_factory=dict)  # name -> {verts, edges, faces}

    def get(self, name):
        return self.widgets.get(name)

    def __contains__(self, name):
        return name in self.widgets

    def __len__(self):
        return len(self.widgets)
