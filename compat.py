"""RNA differences between Blender 3.6 and 4.x / 5.x.

Every version-dependent access goes through here, so the capture and apply
modules read as one story and the shims are in one place to delete when 3.6
support is dropped.

Probed against 5.2, not assumed:

* ``Bone.layers`` is gone from 4.0 onward; ``Bone.collections`` replaces it.
* ``Bone.color`` and ``PoseBone.color`` only exist from 4.0.
* ``show_in_front`` lives on the **Object**, never on the Armature.
* ``custom_shape_wire_width`` only exists from 4.0.
"""

import bpy

__all__ = [
    "has_prop",
    "HAS_COLLECTIONS",
    "HAS_BONE_COLOR",
    "HAS_WIRE_WIDTH",
    "bone_collection_names",
    "bone_layers",
    "set_bone_collections",
    "set_bone_layers",
    "armature_collection_names",
    "ensure_collection",
    "bone_color",
    "set_bone_color",
    "wire_width",
    "set_wire_width",
]


def has_prop(rna_type, name):
    """True when this build's RNA carries the property.

    ``hasattr`` on an RNA *class* does not answer this -- properties are not
    class attributes -- so the check has to go through ``bl_rna``.
    """
    try:
        return name in rna_type.bl_rna.properties
    except AttributeError:  # pragma: no cover - non-RNA type
        return False


HAS_COLLECTIONS = has_prop(bpy.types.Bone, "collections")
HAS_BONE_COLOR = has_prop(bpy.types.Bone, "color")
HAS_WIRE_WIDTH = has_prop(bpy.types.PoseBone, "custom_shape_wire_width")


# --- collections / layers ---------------------------------------------------


def bone_collection_names(bone):
    """Names of the collections a bone belongs to (4.x+), else ``()``."""
    if not HAS_COLLECTIONS:
        return ()
    return tuple(c.name for c in bone.collections)


def bone_layers(bone):
    """The 3.6 layer mask, or ``None`` on a build that has no layers."""
    if HAS_COLLECTIONS or not has_prop(type(bone), "layers"):
        return None
    return tuple(bool(v) for v in bone.layers)


def armature_collection_names(armature):
    if not HAS_COLLECTIONS:
        return ()
    source = getattr(armature, "collections_all", None) or armature.collections
    return tuple(c.name for c in source)


def ensure_collection(armature, name):
    """The named bone collection, created if missing. ``None`` pre-4.0."""
    if not HAS_COLLECTIONS:
        return None
    existing = armature.collections_all if hasattr(armature, "collections_all") else armature.collections
    for coll in existing:
        if coll.name == name:
            return coll
    return armature.collections.new(name)


def set_bone_collections(armature, bone, names):
    """Put ``bone`` in exactly ``names``. No-op pre-4.0."""
    if not HAS_COLLECTIONS:
        return False
    wanted = set(names or ())
    current = {c.name for c in bone.collections}
    if wanted == current:
        return False
    for coll in list(bone.collections):
        if coll.name not in wanted:
            coll.unassign(bone)
    for name in wanted - current:
        coll = ensure_collection(armature, name)
        if coll is not None:
            coll.assign(bone)
    return True


def set_bone_layers(bone, value):
    """Write the 3.6 layer mask. No-op on a build that has no layers."""
    if value is None or HAS_COLLECTIONS or not has_prop(type(bone), "layers"):
        return False
    current = tuple(bool(v) for v in bone.layers)
    wanted = tuple(bool(v) for v in value)
    if current == wanted:
        return False
    bone.layers = wanted
    return True


# --- colour -----------------------------------------------------------------


def bone_color(bone):
    """Bone colour as a plain dict, or ``None``.

    Only the palette and, for a custom palette, the three colours: the rest of
    the ``color`` struct is derived and would produce spurious diffs.
    """
    if not HAS_BONE_COLOR:
        return None
    color = getattr(bone, "color", None)
    if color is None or color.palette == "DEFAULT":
        return None
    out = {"palette": color.palette}
    if color.palette == "CUSTOM":
        custom = color.custom
        out["custom"] = {
            "normal": list(custom.normal),
            "select": list(custom.select),
            "active": list(custom.active),
        }
    return out


def set_bone_color(bone, data):
    if not HAS_BONE_COLOR:
        return False
    color = getattr(bone, "color", None)
    if color is None:
        return False
    if not data:
        if color.palette == "DEFAULT":
            return False
        color.palette = "DEFAULT"
        return True
    palette = data.get("palette", "DEFAULT")
    changed = color.palette != palette
    color.palette = palette
    custom = data.get("custom")
    if palette == "CUSTOM" and custom:
        for key in ("normal", "select", "active"):
            if key in custom:
                setattr(color.custom, key, custom[key])
        changed = True
    return changed


# --- misc -------------------------------------------------------------------


def wire_width(pose_bone):
    return float(getattr(pose_bone, "custom_shape_wire_width", 1.0)) if HAS_WIRE_WIDTH else 1.0


def set_wire_width(pose_bone, value):
    if not HAS_WIRE_WIDTH:
        return False
    if abs(pose_bone.custom_shape_wire_width - value) < 1e-6:
        return False
    pose_bone.custom_shape_wire_width = value
    return True
