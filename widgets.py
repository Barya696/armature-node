"""Control-widget library following the Rigify ``WGTS_rig`` convention.

Rigify stores every bone custom shape as a wireframe mesh object named
``WGT-rig_<bone>`` inside a ``WGTS_rig`` collection that is hidden / excluded
from the view layer. This module:

- Resolves the WGTS_rig collection (creating and hiding it on demand).
- Lists the widgets currently inside it (for the node's dropdown).
- Generates the common Rigify presets (circle, cube, sphere, ...) as meshes
  into that collection when a referenced widget does not exist yet.

Presets are generated in bone space: +Y along the bone, unit length, so
``use_custom_shape_bone_size`` scales them to the bone automatically.
"""

import math

import bpy
from mathutils import Matrix, Vector

WGTS_COLLECTION = "WGTS_rig"
WGT_PREFIX = "WGT-rig_"

PRESET_ITEMS = (
    ("NONE", "None (Existing Only)", "Do not generate; only use an existing widget"),
    ("CIRCLE", "Circle", "Flat ring at the bone head (Rigify FK controls)"),
    ("CIRCLE_MID", "Circle (Mid Bone)", "Ring centred halfway along the bone"),
    ("CUBE", "Cube", "Wire cube centred at the bone head"),
    ("SPHERE", "Sphere", "Three orthogonal rings (Rigify IK/pole)"),
    ("BONE", "Bone", "Octahedral bone-like outline"),
    ("LINE", "Line", "Straight line along the bone"),
    ("SQUARE", "Square", "Flat square at the bone head"),
    ("DIAMOND", "Diamond", "Small pyramid pair (Rigify pivot/pole)"),
    ("ROOT", "Root", "Rigify root widget (arrowed ring)"),
    ("GEAR", "Gear", "Cog outline (Rigify property/switch controls)"),
)


# ---------------------------------------------------------------------------
# Collection / lookup
# ---------------------------------------------------------------------------


def get_widget_collection(create=True):
    """Return the WGTS_rig collection, creating and hiding it if needed."""
    coll = bpy.data.collections.get(WGTS_COLLECTION)
    if coll is None:
        if not create:
            return None
        coll = bpy.data.collections.new(WGTS_COLLECTION)
        bpy.context.scene.collection.children.link(coll)
    coll.hide_viewport = False  # object-level hiding is handled per widget
    coll.hide_render = True
    # Exclude from the active view layer the way Rigify does.
    layer_coll = _find_layer_collection(bpy.context.view_layer.layer_collection, coll)
    if layer_coll is not None:
        layer_coll.exclude = True
    return coll


def _find_layer_collection(layer_coll, target):
    if layer_coll.collection == target:
        return layer_coll
    for child in layer_coll.children:
        found = _find_layer_collection(child, target)
        if found is not None:
            return found
    return None


def widget_object_name(bone_name):
    return f"{WGT_PREFIX}{bone_name}"


def list_widget_objects():
    """All mesh objects currently in WGTS_rig (or named WGT-rig_* anywhere)."""
    coll = bpy.data.collections.get(WGTS_COLLECTION)
    found = {}
    if coll is not None:
        for obj in coll.all_objects:
            if obj.type == "MESH":
                found[obj.name] = obj
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name.startswith(WGT_PREFIX):
            found.setdefault(obj.name, obj)
    return [found[k] for k in sorted(found)]


def widget_enum_items(self, context):
    """EnumProperty items callback: existing widgets in WGTS_rig."""
    items = [("", "Pick Widget...", "")]
    for obj in list_widget_objects():
        label = obj.name[len(WGT_PREFIX):] if obj.name.startswith(WGT_PREFIX) else obj.name
        items.append((obj.name, label, obj.name))
    return items


# ---------------------------------------------------------------------------
# Mesh generators (bone space: Y along bone, unit length)
# ---------------------------------------------------------------------------


def _ring(radius, segments, axis="Y", offset=0.0):
    """Vertices + edge loop of a circle perpendicular to ``axis``."""
    verts = []
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        c, s = math.cos(a) * radius, math.sin(a) * radius
        if axis == "Y":
            verts.append((c, offset, s))
        elif axis == "X":
            verts.append((offset, c, s))
        else:
            verts.append((c, s, offset))
    edges = [(i, (i + 1) % segments) for i in range(segments)]
    return verts, edges


def _merge(*parts):
    verts, edges = [], []
    for v, e in parts:
        base = len(verts)
        verts.extend(v)
        edges.extend((a + base, b + base) for a, b in e)
    return verts, edges


def _cube(size=0.5, center=(0.0, 0.0, 0.0)):
    s = size
    cx, cy, cz = center
    verts = [
        (cx - s, cy - s, cz - s), (cx + s, cy - s, cz - s),
        (cx + s, cy + s, cz - s), (cx - s, cy + s, cz - s),
        (cx - s, cy - s, cz + s), (cx + s, cy - s, cz + s),
        (cx + s, cy + s, cz + s), (cx - s, cy + s, cz + s),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return verts, edges


def _bone_shape():
    # Octahedron: head at origin, tail at y=1, waist at y=0.1
    w = 0.1
    verts = [
        (0, 0, 0), (0, 1, 0),
        (w, w, w), (-w, w, w), (-w, w, -w), (w, w, -w),
    ]
    edges = [
        (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 3), (3, 4), (4, 5), (5, 2),
    ]
    return verts, edges


def _diamond(size=0.15, center_y=0.0):
    s = size
    verts = [
        (0, center_y + s, 0), (0, center_y - s, 0),
        (s, center_y, 0), (-s, center_y, 0), (0, center_y, s), (0, center_y, -s),
    ]
    edges = [
        (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 4), (4, 3), (3, 5), (5, 2),
    ]
    return verts, edges


def _root():
    ring_v, ring_e = _ring(1.0, 32, axis="Z")
    arrows_v, arrows_e = [], []
    for ax, ay in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        base = len(arrows_v)
        px, py = ax * 1.15, ay * 1.15
        tx, ty = ax * 1.45, ay * 1.45
        # perpendicular
        qx, qy = -ay * 0.15, ax * 0.15
        arrows_v.extend([(px + qx, py + qy, 0), (tx, ty, 0), (px - qx, py - qy, 0)])
        arrows_e.extend([(base, base + 1), (base + 1, base + 2)])
    return _merge((ring_v, ring_e), (arrows_v, arrows_e))


def _gear(teeth=8, r_in=0.45, r_out=0.6):
    verts, edges = [], []
    n = teeth * 4
    for i in range(n):
        a = 2.0 * math.pi * i / n
        r = r_out if (i % 4) in (1, 2) else r_in
        verts.append((math.cos(a) * r, 0.0, math.sin(a) * r))
        edges.append((i, (i + 1) % n))
    return verts, edges


def generate_preset_geometry(preset):
    """Return (verts, edges) for a preset, or None for NONE / unknown."""
    if preset == "CIRCLE":
        return _ring(0.5, 32, axis="Y")
    if preset == "CIRCLE_MID":
        return _ring(0.5, 32, axis="Y", offset=0.5)
    if preset == "CUBE":
        return _cube(0.5)
    if preset == "SPHERE":
        return _merge(
            _ring(0.5, 32, axis="X"), _ring(0.5, 32, axis="Y"), _ring(0.5, 32, axis="Z")
        )
    if preset == "BONE":
        return _bone_shape()
    if preset == "LINE":
        return [(0, 0, 0), (0, 1, 0)], [(0, 1)]
    if preset == "SQUARE":
        s = 0.5
        return [(-s, 0, -s), (s, 0, -s), (s, 0, s), (-s, 0, s)], [(0, 1), (1, 2), (2, 3), (3, 0)]
    if preset == "DIAMOND":
        return _diamond(0.15)
    if preset == "ROOT":
        return _root()
    if preset == "GEAR":
        return _gear()
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Custom property stamped on every widget the graph generates, recording what
# it was built from ("PRESET:CUBE" / "GEOMETRY"). Widgets without it belong
# to Rigify or the user and are never rewritten.
WIDGET_SOURCE_KEY = "an_widget_source"


def _source_tag(preset, geometry):
    if geometry and geometry.get("verts"):
        return "GEOMETRY"
    return f"PRESET:{preset}"


def _build_geometry(preset, geometry):
    """(verts, edges, faces) from captured geometry, else from the preset."""
    if geometry and geometry.get("verts"):
        return (
            [tuple(v) for v in geometry["verts"]],
            [tuple(e) for e in geometry.get("edges", [])],
            [tuple(f) for f in geometry.get("faces", [])],
        )
    generated = generate_preset_geometry(preset)
    if generated is not None:
        return generated[0], generated[1], []
    return None


def _fill_mesh(mesh, verts, edges, faces):
    mesh.clear_geometry()
    mesh.from_pydata([Vector(v) for v in verts], edges, faces)
    mesh.update()


def is_graph_widget(obj):
    return obj is not None and obj.get(WIDGET_SOURCE_KEY) is not None


def ensure_widget(name, preset="NONE", geometry=None):
    """Return the widget object called ``name``, matching the requested form.

    * Missing: create it -- first from captured ``geometry`` ({'verts',
      'edges'} read off the original widget when decompiling), then from
      ``preset``.
    * Exists and was generated by the graph from a different preset:
      regenerate its mesh IN PLACE so every bone using it updates live.
    * Exists but is not ours (Rigify / hand-made): returned untouched.

    Returns None when it cannot be resolved.
    """
    wanted = _source_tag(preset, geometry)
    obj = bpy.data.objects.get(name)
    if obj is not None:
        current = obj.get(WIDGET_SOURCE_KEY)
        if current is None or current == wanted or wanted == "GEOMETRY" or preset == "NONE":
            return obj
        geo = _build_geometry(preset, None)
        if geo is not None and obj.type == "MESH":
            _fill_mesh(obj.data, *geo)
            obj[WIDGET_SOURCE_KEY] = wanted
        return obj

    geo = _build_geometry(preset, geometry)
    if geo is None:
        return None

    verts, edges, faces = geo
    mesh = bpy.data.meshes.new(name)
    _fill_mesh(mesh, verts, edges, faces)
    obj = bpy.data.objects.new(name, mesh)
    obj.display_type = "WIRE"
    obj.hide_render = True
    obj[WIDGET_SOURCE_KEY] = wanted

    coll = get_widget_collection(create=True)
    coll.objects.link(obj)

    # Put the widget back where it lived in the world (global frame). The
    # mesh data is untouched so the shape drawn on the bone is identical; only
    # the object's own placement is restored instead of collapsing to 0/0/0.
    matrix = geometry.get("matrix") if geometry else None
    if matrix and len(matrix) == 4:
        try:
            obj.matrix_world = Matrix([tuple(row) for row in matrix])
        except (TypeError, ValueError):
            pass
    return obj


def resolve_widget_for_bone(shape_def, bone_name):
    """Pick the widget object for a bone from its ShapeDef.

    Order: explicit widget name -> per-bone ``WGT-rig_<bone>`` (generated
    from the preset if needed) -> None.
    """
    if shape_def is None:
        return None
    geometry = shape_def.geometry
    if shape_def.widget:
        return ensure_widget(shape_def.widget, shape_def.preset, geometry)
    if geometry or shape_def.preset != "NONE":
        name = widget_object_name(bone_name)
        existing = bpy.data.objects.get(name)
        if existing is not None and not is_graph_widget(existing) and shape_def.preset != "NONE":
            # Rigify already owns WGT-rig_<bone>. Never overwrite its mesh;
            # give the preset its own graph-owned widget instead.
            name = f"{WGT_PREFIX}{bone_name}.an"
        return ensure_widget(name, shape_def.preset, geometry)
    return None
