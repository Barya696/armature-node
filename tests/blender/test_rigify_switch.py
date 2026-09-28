"""The Rigify Switch node: a rig's own switches, set from the graph.

Rigify is not enabled here, so the fixture rig is given switches the way a
generated rig has them: custom properties on a bone, a parent switch with
named choices, and a driver that reads one. What matters is the same --
set from the node, recorded at Bind, put back when the node goes.
"""

import bpy

import fixtures

SWITCHES = {"IK_FK": 0.0, "pole_vector": False, "IK_parent": 1}


def _rig():
    """The fixture rig with Rigify-style switches on bone.001 and bone.002,
    and a driver reading bone.001's IK/FK into a constraint's influence."""
    obj = fixtures.make_rig()
    holder = obj.pose.bones["bone.001"]
    for key, value in SWITCHES.items():
        holder[key] = value
    holder.id_properties_ui("IK_FK").update(min=0.0, max=1.0)
    holder.id_properties_ui("IK_parent").update(
        items=[("P0", "None", "", 0, 0), ("P1", "Root", "", 0, 1), ("P2", "Torso", "", 0, 2)]
    )
    # Fewer parents to choose from, as a Rigify leg has fewer than an arm.
    other = obj.pose.bones["bone.002"]
    other["pole_vector"] = False
    other["IK_parent"] = 1
    other.id_properties_ui("IK_parent").update(
        items=[("P0", "None", "", 0, 0), ("P1", "Root", "", 0, 1)]
    )
    con = obj.pose.bones["bone.003"].constraints[0]
    driver = con.driver_add("influence").driver
    driver.type = "SCRIPTED"
    var = driver.variables.new()
    var.targets[0].id = obj
    var.targets[0].data_path = 'pose.bones["bone.001"]["IK_FK"]'
    driver.expression = var.name
    bpy.context.view_layer.update()
    return obj


def _wire(obj, bone):
    tree, src, out = fixtures.make_tree(obj)
    for link in list(tree.links):
        tree.links.remove(link)
    node = tree.nodes.new("ArmatureNodesRigifySwitchNode")
    node.bone = bone
    tree.links.new(src.outputs["Rig"], node.inputs["Rig"])
    tree.links.new(node.outputs["Rig"], out.inputs["Rig"])
    return tree, node


def _setup(bone="bone.001"):
    from armature_nodes.ops.bind import bind

    obj = _rig()
    bind(obj)
    tree, node = _wire(obj, bone)
    return obj, tree, node


def _build(tree):
    from armature_nodes.build import build_armature_from_tree

    build_armature_from_tree(tree)
    bpy.context.view_layer.update()


def _item(node, name):
    return next(item for item in node.switches if item.name == name)


def test_bind_records_the_switches():
    from armature_nodes.ops.bind import bind
    from armature_nodes.store import record as record_store

    obj = _rig()
    bind(obj)
    record = record_store.read(obj)
    props = record.bones["bone.001"].pose.props
    assert props == SWITCHES
    assert type(props["pole_vector"]) is bool and type(props["IK_parent"]) is int
    assert record.bones["root"].pose.props == {}, "a bone without switches has none, not 'unknown'"


def test_the_node_lists_the_bones_switches():
    """In Rigify's own panel order: IK / FK, Pole, then the parents."""
    _obj, _tree, node = _setup()
    names = [item.name for item in node.switches]
    assert names == list(SWITCHES), names
    assert not any(item.use for item in node.switches), "a fresh node sets nothing"


def test_a_switch_is_set_and_deleting_the_node_puts_it_back():
    obj, tree, node = _setup()
    item = _item(node, "pole_vector")
    item.use = True
    item.flag = True
    _build(tree)
    assert obj.pose.bones["bone.001"]["pole_vector"] is True

    tree.nodes.remove(node)
    _build(tree)
    value = obj.pose.bones["bone.001"]["pole_vector"]
    assert value is False, f"recorded Off, got {value!r}"


def test_an_empty_bone_field_sets_every_bone_that_has_the_switch():
    obj, tree, node = _setup(bone="")
    item = _item(node, "pole_vector")
    assert item.targets.split(";") == ["bone.001", "bone.002"]
    item.use = True
    item.flag = True
    _build(tree)
    assert obj.pose.bones["bone.001"]["pole_vector"] is True
    assert obj.pose.bones["bone.002"]["pole_vector"] is True
    assert obj.pose.bones["bone.001"]["IK_FK"] == 0.0, "an unticked switch is left alone"


def test_switching_reaches_the_rigs_drivers():
    """Rigify switches work through drivers, which must see the new value."""
    obj, tree, node = _setup()
    con = obj.pose.bones["bone.003"].constraints[0]
    assert abs(con.influence) < 1e-6
    item = _item(node, "IK_FK")
    item.use = True
    item.value = 1.0
    _build(tree)
    assert abs(con.influence - 1.0) < 1e-6, f"the driver still reads {con.influence}"


def test_a_parent_switch_offers_rigifys_names():
    obj, tree, node = _setup()
    item = _item(node, "IK_parent")
    assert item.kind == "INT"
    assert '"Torso"' in item.options
    item.use = True
    item.choice = "2"
    _build(tree)
    value = obj.pose.bones["bone.001"]["IK_parent"]
    assert value == 2 and type(value) is int


def test_a_parent_switchs_choices_follow_the_bone():
    _obj, _tree, node = _setup(bone="")
    assert '"Torso"' in _item(node, "IK_parent").options
    node.bone = "bone.002"
    options = _item(node, "IK_parent").options
    assert '"Root"' in options and '"Torso"' not in options, options


def test_ticking_a_switch_starts_from_what_the_rig_has():
    obj, tree, node = _setup()
    obj.pose.bones["bone.001"]["IK_FK"] = 0.4
    item = _item(node, "IK_FK")
    item.use = True
    assert abs(item.value - 0.4) < 1e-6
    _build(tree)
    assert abs(obj.pose.bones["bone.001"]["IK_FK"] - 0.4) < 1e-6


def test_a_record_from_before_switches_is_never_written_to():
    """Nothing could put such a switch back, so the node sets none -- until
    Record Switches adds them."""
    from dataclasses import replace

    from armature_nodes.ops.bind import bind
    from armature_nodes.store import record as record_store

    obj = _rig()
    bind(obj)
    record = record_store.read(obj)
    legacy = {n: replace(b, pose=replace(b.pose, props=None)) for n, b in record.bones.items()}
    record_store.write(obj, record.with_bones(legacy))
    obj.pose.bones["bone.001"]["pole_vector"] = True  # set by hand before switches were recorded

    tree, node = _wire(obj, "bone.001")
    assert len(node.switches) == 0 and node.recorded()[2], "it says the record predates switches"

    assert bpy.ops.armature_nodes.record_switches(rig=obj.name) == {"FINISHED"}
    assert record_store.read(obj).bones["bone.001"].pose.props["pole_vector"] is True
    names = [item.name for item in node.switches]
    assert names == list(SWITCHES), names
    item = _item(node, "pole_vector")
    item.use = True
    item.flag = False
    _build(tree)
    assert obj.pose.bones["bone.001"]["pole_vector"] is False
    tree.nodes.remove(node)
    _build(tree)
    assert obj.pose.bones["bone.001"]["pole_vector"] is True, "back to what it had"


def test_a_change_in_rigifys_panel_is_taken_into_the_node():
    """Otherwise the next build would throw the change away."""
    obj, tree, node = _setup()
    item = _item(node, "IK_FK")
    item.use = True
    item.value = 1.0
    _build(tree)

    obj.pose.bones["bone.001"]["IK_FK"] = 0.0  # the animator, in Rigify's panel
    tree.is_dirty = True
    node.follow_live()
    assert item.value == 1.0, "while a build is due the rig has not caught up: keep the node's"
    tree.is_dirty = False
    node.follow_live()
    assert item.value == 0.0
    _build(tree)
    assert obj.pose.bones["bone.001"]["IK_FK"] == 0.0


def test_identical_builds_write_nothing():
    from armature_nodes import bridge
    from armature_nodes.apply import pipeline
    from armature_nodes.build import evaluate_tree
    from armature_nodes.store import record as record_store
    from armature_nodes.store import touched as touched_store

    obj, tree, node = _setup()
    item = _item(node, "pole_vector")
    item.use = True
    item.flag = True
    _build(tree)
    base = record_store.read(obj)
    _name, bone_defs = evaluate_tree(tree)
    target = bridge.overlay(base, bone_defs)
    result = pipeline.apply(obj, base, target, touched_store.read(obj))
    assert result.writes == 0, f"{result.writes} writes on a build that changes nothing"
