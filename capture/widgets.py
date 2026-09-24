"""Widget meshes -> geometry, so a rig can be restored without its WGT objects.

The v1 implementation stored only the widget's object name. When that object
went missing the shape could not be rebuilt, and the code that was supposed to
restore the rig instead cleared every custom shape. Geometry is the thing
itself; a name is a pointer into a scene that may no longer have it.

Captured in the widget object's own local space: ``apply`` recreates the mesh
datablock, and the pose bone's custom-shape transform positions it.
"""

from ..model.types import WidgetLib

__all__ = ["capture_mesh", "capture_widgets"]


def capture_mesh(obj):
    """``{"verts", "edges", "faces"}`` for a mesh object, or ``None``."""
    if obj is None or obj.type != "MESH" or obj.data is None:
        return None
    mesh = obj.data
    return {
        "verts": [tuple(v.co) for v in mesh.vertices],
        "edges": [tuple(e.vertices) for e in mesh.edges],
        "faces": [tuple(p.vertices) for p in mesh.polygons],
    }


def capture_widgets(obj):
    """Every distinct custom shape used by ``obj``, as a WidgetLib.

    Deduplicated by name: a Rigify rig points dozens of bones at a handful of
    shared widgets, and storing one copy each keeps the library small.
    """
    if obj is None or obj.pose is None:
        return WidgetLib()
    seen = {}
    for pbone in obj.pose.bones:
        shape = pbone.custom_shape
        if shape is None or shape.name in seen:
            continue
        geo = capture_mesh(shape)
        if geo is not None:
            seen[shape.name] = geo
    return WidgetLib(widgets=seen)
