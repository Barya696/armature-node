"""Build test rigs in Blender.

Rigify is not installed in every environment (it is not enabled in this one),
so the fixtures construct a rig by hand that exercises the same things a
generated rig does: shared widgets across several bones, bone collections,
constraints with object targets, non-default display values and deform flags.
"""

import bpy


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_widget(name, size=1.0):
    """A small wire mesh, like a ``WGT-rig_*`` control shape."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        [(-size, -size, 0), (size, -size, 0), (size, size, 0), (-size, size, 0)],
        [(0, 1), (1, 2), (2, 3), (3, 0)],
        [],
    )
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def make_rig(name="rig", n_bones=4):
    """An armature with widgets, collections, constraints and odd settings."""
    arm = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, arm)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode="EDIT")
    previous = None
    for i in range(n_bones):
        eb = arm.edit_bones.new("root" if i == 0 else f"bone.{i:03d}")
        eb.head = (0.0, 0.0, float(i))
        eb.tail = (0.0, 0.0, float(i) + 0.8)
        eb.roll = 0.1 * i
        if previous is not None:
            eb.parent = previous
            eb.use_connect = True
        previous = eb
    bpy.ops.object.mode_set(mode="OBJECT")

    # Two widgets shared across bones, as a real rig does.
    widgets = [make_widget("WGT-rig_circle"), make_widget("WGT-rig_box", 0.5)]
    target = bpy.data.objects.new("target", None)
    bpy.context.scene.collection.objects.link(target)

    for i, pbone in enumerate(obj.pose.bones):
        bone = pbone.bone
        bone.use_deform = i % 2 == 0
        bone.envelope_distance = 0.2 + 0.01 * i
        if i:
            pbone.custom_shape = widgets[i % 2]
            pbone.custom_shape_scale_xyz = (1.0 + i, 1.0, 1.0)
            pbone.custom_shape_translation = (0.0, 0.05 * i, 0.0)
            pbone.custom_shape_rotation_euler = (0.0, 0.0, 0.1 * i)
            pbone.use_custom_shape_bone_size = i % 2 == 0
            bone.show_wire = True
            if hasattr(pbone, "custom_shape_wire_width"):
                pbone.custom_shape_wire_width = 1.0 + i
        pbone.rotation_mode = "XYZ" if i % 2 else "QUATERNION"
        pbone.lock_location = (True, False, i % 2 == 0)
        if i == n_bones - 1:
            con = pbone.constraints.new("COPY_TRANSFORMS")
            con.target = target
            con.influence = 0.5
            con.subtarget = "root"

    # Bone collections (4.x+); on 3.6 this is simply skipped.
    if hasattr(arm, "collections"):
        controls = arm.collections.new("Controls")
        for pbone in list(obj.pose.bones)[1:]:
            controls.assign(pbone.bone)

    arm.display_type = "WIRE"
    obj.show_in_front = True
    return obj


_REGISTERED = False


def ensure_registered():
    """Register the addon once per Blender session.

    ``read_factory_settings`` resets the scene but not the class registry, so
    registering per test raises "already registered". Checking bpy.types by
    bl_idname does not work either -- it is keyed by CLASS name -- so a plain
    module flag is the honest guard.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    import armature_nodes

    armature_nodes.register()
    _REGISTERED = True


def make_tree(obj, connected=True):
    """A two-node stack targeting ``obj``: (tree, input_node, output_node).

    Uses the registered Armature node tree, so links are real and
    ``tree.links.remove`` in a test does what it does in the editor.
    """
    ensure_registered()
    tree = bpy.data.node_groups.new(f"{obj.name} Nodes", "ArmatureNodeTreeType")
    src = tree.nodes.new("ArmatureNodesInputNode")
    src.source = obj
    out = tree.nodes.new("ArmatureNodesOutputNode")
    if connected:
        tree.links.new(src.outputs["Rig"], out.inputs["Rig"])
    return tree, src, out


def flush(tree, obj=None):
    """Run one build.

    Step 2 has no ``graph/`` yet, so the stack is the identity: the target is
    the record itself. That is precisely the case the acceptance tests need --
    "an Output with nothing connected equals the record" -- and it exercises
    the whole restore-then-apply path. Step 3 replaces the target with
    ``graph.evaluate(tree)``; nothing else here changes.
    """
    from armature_nodes.apply import pipeline
    from armature_nodes.store import record as record_store
    from armature_nodes.store import touched as touched_store

    if obj is None:
        for node in tree.nodes:
            if node.bl_idname == "ArmatureNodesInputNode" and node.source:
                obj = node.source
                break
    if obj is None:
        return None
    base = record_store.read(obj)
    if base is None:
        return None
    return pipeline.apply(obj, base, base, touched_store.read(obj))


def snapshot(obj):
    """A comparable dump of everything the record is supposed to preserve.

    Deliberately independent of ``capture`` -- comparing a capture against
    itself would prove nothing. This reads the live armature directly, so a
    pass means the viewport really did come back.
    """
    from armature_nodes.store import record as record_store

    shapes = {}
    deform = {}
    widget_names = set()
    widget_geometry = {}
    for pbone in obj.pose.bones:
        bone = pbone.bone
        shape = pbone.custom_shape
        deform[bone.name] = bone.use_deform
        shapes[bone.name] = (
            shape.name if shape else None,
            tuple(round(v, 5) for v in pbone.custom_shape_scale_xyz),
            tuple(round(v, 5) for v in pbone.custom_shape_translation),
            tuple(round(v, 5) for v in pbone.custom_shape_rotation_euler),
            round(getattr(pbone, "custom_shape_wire_width", 1.0), 5),
            pbone.use_custom_shape_bone_size,
            bone.show_wire,
            pbone.rotation_mode,
            tuple(pbone.lock_location),
            tuple(sorted(c.name for c in bone.collections))
            if hasattr(bone, "collections")
            else (),
            tuple(
                (c.type, c.name, round(c.influence, 5),
                 c.target.name if getattr(c, "target", None) else None)
                for c in pbone.constraints
            ),
        )
        if shape is not None:
            widget_names.add(shape.name)
            widget_geometry[shape.name] = (
                len(shape.data.vertices),
                len(shape.data.edges),
                tuple(tuple(round(v, 5) for v in vert.co) for vert in shape.data.vertices),
            )
    return {
        "shapes": shapes,
        "deform": deform,
        "widget_names": tuple(sorted(widget_names)),
        "widget_geometry": widget_geometry,
        "record": record_store.raw(obj),
    }
