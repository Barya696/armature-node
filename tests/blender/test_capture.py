"""capture/ against a real armature, and the record through real JSON."""

import fixtures

from armature_nodes import capture
from armature_nodes.model.schema import from_json, to_json, validate
from armature_nodes.store import record as store_record
from armature_nodes.store import widgets_lib


def test_capture_sees_every_bone():
    obj = fixtures.make_rig(n_bones=5)
    rec = capture.capture_record(obj)
    assert len(rec.bones) == 5
    assert set(rec.bones) == {b.name for b in obj.data.bones}
    assert rec.source.object == obj.name
    assert rec.source.blender


def test_capture_is_parent_first():
    obj = fixtures.make_rig(n_bones=5)
    rec = capture.capture_record(obj)
    seen = set()
    for name in rec.names():
        parent = rec.bones[name].rest.parent
        assert parent is None or parent in seen, f"{name} precedes its parent"
        seen.add(name)


def test_capture_records_display_and_flags():
    obj = fixtures.make_rig(n_bones=4)
    rec = capture.capture_record(obj)
    for pbone in obj.pose.bones:
        bone = rec.bones[pbone.name]
        assert bone.rest.deform == pbone.bone.use_deform
        assert bone.pose.rotation_mode == pbone.rotation_mode
        shape = pbone.custom_shape
        assert bone.display.shape == (shape.name if shape else "")
        if shape:
            assert bone.display.scale == tuple(pbone.custom_shape_scale_xyz)
            assert bone.display.use_bone_size == pbone.use_custom_shape_bone_size
            assert bone.display.show_wire is True


def test_capture_records_collections():
    obj = fixtures.make_rig(n_bones=4)
    if not hasattr(obj.data, "collections"):
        return  # 3.6: no collections to record
    rec = capture.capture_record(obj)
    assert "Controls" in rec.armature.collections
    assert "Controls" in rec.bones["bone.001"].membership.collections
    assert rec.bones["root"].membership.collections == ()


def test_capture_constraints_are_generic():
    """Far more than the eight attributes v1 knew about."""
    obj = fixtures.make_rig(n_bones=4)
    rec = capture.capture_record(obj)
    cons = rec.bones["bone.003"].constraints
    assert len(cons) == 1
    con = cons[0]
    assert con.type == "COPY_TRANSFORMS"
    assert con.props["target"] == "target"
    assert con.props["subtarget"] == "root"
    assert abs(con.props["influence"] - 0.5) < 1e-6
    assert len(con.props) > 8, f"only {len(con.props)} props captured"
    # UI state must not be recorded: it would diff for no reason.
    for skipped in ("active", "show_expanded", "is_valid", "name", "type"):
        assert skipped not in con.props


def test_capture_widget_geometry():
    """The v1 bug: widgets stored by name only, and lost with the object."""
    obj = fixtures.make_rig(n_bones=4)
    lib = capture.capture_widgets(obj)
    assert set(lib.widgets) == {"WGT-rig_circle", "WGT-rig_box"}
    geo = lib.get("WGT-rig_circle")
    assert len(geo["verts"]) == 4
    assert len(geo["edges"]) == 4


def test_record_roundtrips_through_json():
    obj = fixtures.make_rig(n_bones=5)
    rec = capture.capture_record(obj)
    assert from_json(to_json(rec)) == rec
    assert validate(rec) == []


def test_record_roundtrips_through_the_object():
    obj = fixtures.make_rig(n_bones=5)
    rec, lib = capture.capture_all(obj)
    store_record.write(obj, rec)
    widgets_lib.write(obj, lib)
    assert store_record.read(obj) == rec
    assert widgets_lib.read(obj).widgets == lib.widgets


def test_capture_is_deterministic():
    """Two captures of an untouched rig are byte-identical.

    Test 10 (two builds write nothing) rests on this: if capture wobbled, the
    diff would never be empty.
    """
    obj = fixtures.make_rig(n_bones=5)
    a = to_json(capture.capture_record(obj))
    b = to_json(capture.capture_record(obj))
    assert a == b


def test_capture_refuses_during_a_build():
    from armature_nodes.store import lock

    obj = fixtures.make_rig(n_bones=2)
    lock.reset()
    raised = False
    with lock.held():
        try:
            capture.capture_record(obj)
        except RuntimeError:
            raised = True
    assert raised, "capture must be blocked while a build is in progress"
    lock.reset()


def test_capture_refuses_non_armature():
    fixtures.make_rig(n_bones=2)
    raised = False
    try:
        capture.capture_record(fixtures.make_widget("plain"))
    except capture.CaptureError:
        raised = True
    assert raised
