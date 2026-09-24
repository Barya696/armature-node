"""Full Rig writers, moved here so apply/ is the only thing that writes.

These build an armature from scratch -- create the object, lay out edit bones,
then attach constraints and custom shapes. They are the legacy Full Rig path,
untouched in behaviour, relocated because the layering rule is "only apply/
assigns to a bone" and a rule with an exception is not a rule.

They do not yet go through the record, diff or touched set: Full Rig owns its
armature outright, so there is nothing to restore to. Porting them onto the
pipeline is step 3 work.
"""

import bpy

from ..core import unique_names  # noqa: F401  (kept for the legacy call shape)

def _ensure_object_mode():
    if bpy.context.mode != "OBJECT" and bpy.context.active_object:
        bpy.ops.object.mode_set(mode="OBJECT")


def _target_collection():
    # bpy.context.collection is None inside timer callbacks; fall back to the
    # scene root collection so live updates can still link new objects.
    collection = bpy.context.collection
    if collection is None:
        collection = bpy.context.scene.collection
    return collection


def _get_or_create_armature_object(name):
    obj = bpy.data.objects.get(name)
    if obj is not None and obj.type == "ARMATURE":
        return obj, False
    arm_data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, arm_data)
    _target_collection().objects.link(obj)
    return obj, True


def _activate(obj):
    """Make ``obj`` the active, visible, selected object so mode_set polls true."""
    view_layer = bpy.context.view_layer
    if obj.name not in view_layer.objects:
        # Object exists but is not in this view layer (e.g. excluded
        # collection): link it to the scene root so it can be edited.
        bpy.context.scene.collection.objects.link(obj)
    try:
        if obj.hide_get():
            obj.hide_set(False)
    except RuntimeError:
        pass
    obj.hide_viewport = False
    view_layer.objects.active = obj
    obj.select_set(True)


def _edit_mode_pass(obj, bone_defs):
    """Create/update edit bones from BoneDefs, matched by name."""
    _activate(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        edit_bones = obj.data.edit_bones
        wanted = {b.name for b in bone_defs}

        # Remove bones no longer produced by the graph.
        for eb in list(edit_bones):
            if eb.name not in wanted:
                edit_bones.remove(eb)

        # Create/update all bones first (positions), then parent links,
        # so parenting never references a bone that does not exist yet.
        for b in bone_defs:
            eb = edit_bones.get(b.name)
            if eb is None:
                eb = edit_bones.new(b.name)
            eb.head = b.head
            eb.tail = b.tail
            eb.roll = b.roll
            eb.use_deform = b.use_deform
            eb.envelope_distance = b.envelope_distance
            eb.envelope_weight = b.envelope_weight

        for b in bone_defs:
            eb = edit_bones[b.name]
            if b.parent and b.parent in edit_bones:
                eb.parent = edit_bones[b.parent]
                eb.use_connect = b.use_connect
            else:
                eb.parent = None
                eb.use_connect = False
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def _resolve_target(name, self_obj):
    """Constraint targets are stored by object name. A rig rebuilt from a
    snapshot may carry a different object name, so targets that pointed at
    the original armature are redirected to the armature being built."""
    # The generic capture stores an unset POINTER as None, where the old
    # hard-coded one simply omitted the key -- and objects.get(None) raises.
    if not name:
        return None
    obj = bpy.data.objects.get(name)
    if obj is None and self_obj is not None:
        return self_obj
    return obj


def _apply_constraint(pose_bone, cdef, self_obj=None):
    try:
        con = pose_bone.constraints.new(cdef.type)
    except TypeError as exc:
        print(f"[Armature Nodes] Cannot create constraint {cdef.type}: {exc}")
        return
    if cdef.name:
        con.name = cdef.name
    params = dict(cdef.params)
    # Set 'targets' collection (Armature constraint) separately.
    targets = params.pop("targets", None)
    # Object pointers first so dependent props (subtarget) validate.
    for key in ("target", "pole_target", "space_object"):
        if key in params:
            value = _resolve_target(params.pop(key), self_obj)
            if value is not None:
                try:
                    setattr(con, key, value)
                except (AttributeError, TypeError) as exc:
                    print(f"[Armature Nodes] Skipped constraint param {key}: {exc}")
    for key, value in params.items():
        if isinstance(value, list):
            value = tuple(value)
        try:
            setattr(con, key, value)
        except (AttributeError, TypeError, ValueError) as exc:
            print(f"[Armature Nodes] Skipped constraint param {key}: {exc}")
    if targets and hasattr(con, "targets"):
        for t in targets:
            tgt = con.targets.new()
            tgt.target = _resolve_target(t.get("target", ""), self_obj)
            tgt.subtarget = t.get("subtarget", "")
            tgt.weight = t.get("weight", 1.0)


def _apply_shape(pbone, bdef):
    """Assign (or clear) the custom shape widget on a pose bone."""
    # The top-level widgets.py, not apply/widgets.py: this legacy path still
    # resolves widgets by preset/name, while the record-driven pipeline
    # rebuilds them from stored geometry.
    from ..widgets import resolve_widget_for_bone

    shape = bdef.shape
    widget = resolve_widget_for_bone(shape, bdef.name) if shape else None
    if widget is None:
        pbone.custom_shape = None
        pbone.bone.show_wire = False
        return
    pbone.custom_shape = widget
    pbone.custom_shape_scale_xyz = shape.scale
    pbone.custom_shape_translation = shape.translation
    pbone.custom_shape_rotation_euler = shape.rotation
    pbone.use_custom_shape_bone_size = shape.scale_to_bone_length
    pbone.bone.show_wire = shape.show_wire
    if hasattr(pbone, "custom_shape_wire_width"):  # Blender 4.x+
        pbone.custom_shape_wire_width = shape.wire_width


def _pose_mode_pass(obj, bone_defs):
    """Clear graph-managed constraints and apply constraints + custom shapes."""
    _activate(obj)
    bpy.ops.object.mode_set(mode="POSE")
    try:
        defs_by_name = {b.name: b for b in bone_defs}
        for pbone in obj.pose.bones:
            bdef = defs_by_name.get(pbone.name)
            if bdef is None:
                continue
            for con in list(pbone.constraints):
                pbone.constraints.remove(con)
            for cdef in bdef.constraints:
                _apply_constraint(pbone, cdef, self_obj=obj)
            _apply_shape(pbone, bdef)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def _geometry_differs(bone, bdef, eps=1e-6):
    """True when a BoneDef's rest geometry/hierarchy no longer matches the
    armature bone -- i.e. the user edited it on a Custom Shape node."""
    if (Vector(bdef.head) - bone.head_local).length > eps:
        return True
    if (Vector(bdef.tail) - bone.tail_local).length > eps:
        return True
    parent = bone.parent.name if bone.parent else None
    if (bdef.parent or None) != parent:
        return True
    return bone.use_connect != bdef.use_connect or bone.use_deform != bdef.use_deform


def _edit_geometry_pass(obj, bone_defs):
    """Move/re-parent ONLY the given existing bones (no create/remove)."""
    _activate(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        edit_bones = obj.data.edit_bones
        for b in bone_defs:
            eb = edit_bones.get(b.name)
            if eb is None:
                continue
            eb.head = b.head
            eb.tail = b.tail
            eb.roll = b.roll
            eb.use_deform = b.use_deform
            if b.parent and b.parent in edit_bones:
                eb.parent = edit_bones[b.parent]
                eb.use_connect = b.use_connect
            else:
                eb.parent = None
                eb.use_connect = False
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


