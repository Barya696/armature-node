"""Write bone collection (4.x) / layer (3.6) membership."""

from .. import compat

__all__ = ["apply_membership"]


def apply_membership(armature, bone, leaf, value, writer):
    """Write one membership leaf: ``collections`` or ``layers``."""
    if leaf == "collections":
        writer.count(compat.set_bone_collections(armature, bone, value or ()))
    elif leaf == "layers":
        writer.count(compat.set_bone_layers(bone, value))
