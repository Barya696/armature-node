"""Bone collection (4.x) / layer (3.6) membership -> MembershipDef."""

from .. import compat
from ..model.types import MembershipDef

__all__ = ["capture_membership"]


def capture_membership(bone):
    """Which collections or layers a bone belongs to.

    Exactly one of the two is populated, so the record records which Blender
    it came from instead of leaving the reader to guess.
    """
    return MembershipDef(
        collections=compat.bone_collection_names(bone),
        layers=compat.bone_layers(bone),
    )
