"""Translate between the old node graph and the new record.

The node graph still speaks ``core.BoneDef`` -- a flat bone with a ``shape``
and pose fields -- while the store and the apply pipeline speak
``model.RigRecord``. Until ``graph/`` replaces the old nodes, this is the seam
between them, and it is deliberately the *only* place the two vocabularies
meet.

The round trip must be an identity: ``overlay(base, to_bone_defs(base))``
has to equal ``base``, or every build would diff against itself and rewrite
the whole rig. The tests assert exactly that.
"""

from dataclasses import replace

from .core import BoneDef as GraphBone
from .core import ConstraintDef as GraphConstraint
from .core import ShapeDef as GraphShape
from .model.types import ConstraintDef, DisplayDef, RestDef, TransformDef

__all__ = ["to_bone_defs", "overlay"]


def _shape_from_display(display):
    """A graph ShapeDef from a record DisplayDef, or None when unshaped."""
    if not display.shape and display.preset in ("", "NONE"):
        return None
    return GraphShape(
        widget=display.shape,
        preset=display.preset,
        scale=tuple(display.scale),
        translation=tuple(display.translation),
        rotation=tuple(display.rotation),
        wire_width=display.wire_width,
        scale_to_bone_length=display.use_bone_size,
        show_wire=display.show_wire,
    )


def to_bone_defs(record):
    """A record as the flat bone list the old graph expects."""
    out = []
    for name, bone in record.bones.items():
        rest = bone.rest
        out.append(
            GraphBone(
                name=name,
                head=tuple(rest.head),
                tail=tuple(rest.tail),
                roll=rest.roll,
                parent=rest.parent,
                use_connect=rest.connect,
                use_deform=rest.deform,
                envelope_distance=rest.envelope_distance,
                envelope_weight=rest.envelope_weight,
                constraints=[
                    GraphConstraint(type=c.type, name=c.name, params=dict(c.props))
                    for c in bone.constraints
                ],
                shape=_shape_from_display(bone.display),
            )
        )
    return out


def _display_from_shape(base_display, shape):
    """Fold a graph ShapeDef back into a record DisplayDef.

    ``override`` and ``color`` have no representation in the graph's ShapeDef,
    so they are carried over from the record untouched -- dropping to the
    dataclass default would silently clear a bone colour on every build.
    """
    if shape is None:
        # The graph left this bone unshaped. Only the widget itself is
        # cleared; placement values stay, so re-adding a shape does not also
        # reset its scale.
        return replace(base_display, shape="", preset="NONE")
    return replace(
        base_display,
        shape=shape.widget or "",
        preset=shape.preset or "NONE",
        scale=tuple(shape.scale),
        translation=tuple(shape.translation),
        rotation=tuple(shape.rotation),
        wire_width=float(shape.wire_width),
        use_bone_size=bool(shape.scale_to_bone_length),
        show_wire=bool(shape.show_wire),
    )


def _transform_from_graph(bone):
    """The graph's pose request as a record TransformDef.

    The old graph carries absolute values plus accumulated offsets; the record
    stores one resolved world value per component. An offset with no absolute
    part is left for the pipeline to resolve against the live pose, so it is
    only folded in when there is something to fold it into.
    """
    location = getattr(bone, "pose_location", None)
    rotation = getattr(bone, "pose_rotation", None)
    scale = getattr(bone, "pose_scale", None)
    offset = tuple(getattr(bone, "pose_offset", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
    rot_offset = tuple(
        getattr(bone, "pose_rotation_offset", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)
    )
    if location is not None and any(offset):
        location = tuple(a + b for a, b in zip(location, offset))
    if rotation is not None and any(rot_offset):
        rotation = tuple(a + b for a, b in zip(rotation, rot_offset))
    return TransformDef(
        location=tuple(location) if location is not None else None,
        rotation=tuple(rotation) if rotation is not None else None,
        scale=tuple(scale) if scale is not None else None,
    )


def overlay(base, bone_defs):
    """Fold what the graph produced onto ``base``, returning a new record.

    Bones the graph does not mention keep their recorded state, and bones it
    invents are ignored -- Modify mode cannot add bones, and the diff reports
    them as ``added`` so the Output can say so.
    """
    if not bone_defs:
        return base  # an empty stack is the identity, not a teardown

    bones = dict(base.bones)
    for gb in bone_defs:
        recorded = base.bones.get(gb.name)
        if recorded is None:
            continue
        bones[gb.name] = replace(
            recorded,
            rest=RestDef(
                head=tuple(gb.head),
                tail=tuple(gb.tail),
                roll=float(gb.roll),
                parent=gb.parent,
                connect=bool(gb.use_connect),
                deform=bool(gb.use_deform),
                # The graph has no vocabulary for these, so they stay as
                # recorded rather than reverting to dataclass defaults.
                inherit_scale=recorded.rest.inherit_scale,
                use_local_location=recorded.rest.use_local_location,
                envelope_distance=float(gb.envelope_distance),
                envelope_weight=float(gb.envelope_weight),
                head_radius=recorded.rest.head_radius,
                tail_radius=recorded.rest.tail_radius,
            ),
            display=_display_from_shape(recorded.display, gb.shape),
            transform=_transform_from_graph(gb),
            constraints=tuple(
                ConstraintDef(type=c.type, name=c.name, props=dict(c.params))
                for c in (gb.constraints or ())
            ),
        )
    return replace(base, bones=bones)
