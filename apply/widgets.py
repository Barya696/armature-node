"""Rebuild widget objects from stored geometry.

The v1 failure this exists to prevent: widgets were stored by object name, so
when a ``WGT-*`` object went missing the shape could not be rebuilt --- and the
code that was supposed to restore the rig cleared the custom shape instead,
stripping every control off the armature.

Two rules follow from that:

* A widget named in the record is recreated from the library when it is gone.
* A widget that cannot be rebuilt (no stored geometry) is **left alone**. A
  missing shape is a broken rig; silently clearing it is a destroyed one.
"""

import bpy

__all__ = ["WIDGET_COLLECTION", "ensure_widget", "widget_collection"]

WIDGET_COLLECTION = "WGTS_rig"


def widget_collection(create=True):
    """The hidden collection widget objects live in.

    Excluded from the view layer so the widget meshes themselves are never
    drawn --- only the bones that use them as custom shapes.
    """
    coll = bpy.data.collections.get(WIDGET_COLLECTION)
    if coll is None:
        if not create:
            return None
        coll = bpy.data.collections.new(WIDGET_COLLECTION)
    scene = bpy.context.scene
    if scene is not None and coll.name not in scene.collection.children:
        try:
            scene.collection.children.link(coll)
        except RuntimeError:
            pass  # already linked somewhere in the tree
        layer = bpy.context.view_layer.layer_collection.children.get(coll.name)
        if layer is not None:
            layer.exclude = True
    return coll


def ensure_widget(name, library):
    """The widget object called ``name``, rebuilt from ``library`` if missing.

    Returns the object, or ``None`` when it does not exist and cannot be
    rebuilt --- the caller must then leave the bone's shape untouched.
    """
    if not name:
        return None
    existing = bpy.data.objects.get(name)
    if existing is not None:
        return existing if existing.type == "MESH" else None

    geo = library.get(name) if library is not None else None
    if not geo or not geo.get("verts"):
        return None

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        [tuple(v) for v in geo.get("verts", ())],
        [tuple(e) for e in geo.get("edges", ())],
        [tuple(f) for f in geo.get("faces", ())],
    )
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.hide_render = True
    coll = widget_collection()
    if coll is not None:
        coll.objects.link(obj)
    return obj
