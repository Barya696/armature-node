"""Write rest geometry. The only part of apply that needs Edit mode.

``head``/``tail``/``roll``/``parent``/``connect`` live on ``EditBone`` and
cannot be set any other way; the rest of the flags live on ``Bone`` and are
written in Object mode, which is cheaper. Splitting them means a build that
only changes a widget never enters Edit mode at all.
"""

__all__ = ["EDIT_PATHS", "OBJECT_PATHS", "apply_edit", "apply_flags"]

#: Paths that require Edit mode.
EDIT_PATHS = frozenset({"rest.head", "rest.tail", "rest.roll", "rest.parent",
                        "rest.connect"})

#: Rest paths writable without Edit mode.
OBJECT_PATHS = frozenset({"rest.deform", "rest.inherit_scale",
                          "rest.use_local_location", "rest.envelope_distance",
                          "rest.envelope_weight", "rest.head_radius",
                          "rest.tail_radius"})

_LEAF_TO_BONE_ATTR = {
    "deform": "use_deform",
    "inherit_scale": "inherit_scale",
    "use_local_location": "use_local_location",
    "envelope_distance": "envelope_distance",
    "envelope_weight": "envelope_weight",
    "head_radius": "head_radius",
    "tail_radius": "tail_radius",
}


def apply_edit(edit_bones, name, values, writer):
    """Write Edit-mode rest values for one bone. ``values`` is {leaf: value}."""
    ebone = edit_bones.get(name)
    if ebone is None:
        return
    # Parent first: connecting a bone is meaningless without one, and
    # Blender clamps use_connect when the parent is absent.
    if "parent" in values:
        parent_name = values["parent"]
        parent = edit_bones.get(parent_name) if parent_name else None
        if ebone.parent is not parent:
            ebone.parent = parent
            writer.count()
    for leaf in ("head", "tail"):
        if leaf in values:
            writer.set(ebone, leaf, values[leaf])
    if "roll" in values:
        writer.set(ebone, "roll", values["roll"])
    if "connect" in values:
        writer.set(ebone, "use_connect", bool(values["connect"]))


def apply_flags(bone, values, writer):
    """Write the rest flags that do not need Edit mode."""
    for leaf, value in values.items():
        attr = _LEAF_TO_BONE_ATTR.get(leaf)
        if attr is not None:
            writer.set(bone, attr, value)
