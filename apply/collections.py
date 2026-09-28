"""Write bone collection membership."""

__all__ = ["apply_membership"]


def apply_membership(armature, bone, leaf, value, writer):
    """Write one membership leaf. Only ``collections`` is written: ``layers``
    is the Blender 3.6 equivalent, kept in old records, with nowhere to go."""
    if leaf == "collections":
        writer.count(_set_collections(armature, bone, value or ()))


def _collection(armature, name):
    """The named bone collection, created if missing."""
    for coll in armature.collections_all:
        if coll.name == name:
            return coll
    return armature.collections.new(name)


def _set_collections(armature, bone, names):
    """Put ``bone`` in exactly ``names``. True when that changed anything."""
    wanted = set(names)
    current = {c.name for c in bone.collections}
    if wanted == current:
        return False
    for coll in list(bone.collections):
        if coll.name not in wanted:
            coll.unassign(bone)
    for name in wanted - current:
        _collection(armature, name).assign(bone)
    return True
