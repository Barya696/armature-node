"""Bone collection membership -> MembershipDef."""

from ..model.types import MembershipDef

__all__ = ["capture_membership"]


def capture_membership(bone):
    """The bone collections a bone belongs to."""
    return MembershipDef(collections=tuple(c.name for c in bone.collections))
