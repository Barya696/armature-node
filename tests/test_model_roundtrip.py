"""The record must survive JSON exactly -- it is the only copy of the rig."""

from conftest import FakeObject, make_record, make_widgets

from armature_nodes.model import diff as M_diff
from armature_nodes.model import ops
from armature_nodes.model.schema import (
    RECORD_VERSION,
    RecordError,
    from_json,
    peek_version,
    to_json,
    validate,
)
from armature_nodes.model.types import RigRecord, WidgetLib
from armature_nodes.store import lock
from armature_nodes.store import record as store_record
from armature_nodes.store import touched as store_touched
from armature_nodes.store import widgets_lib


def test_roundtrip_is_exact():
    r = make_record(5)
    assert from_json(to_json(r)) == r


def test_roundtrip_preserves_every_section():
    r = make_record(3)
    back = from_json(to_json(r))
    for name, bone in r.bones.items():
        other = back.bones[name]
        assert other.rest == bone.rest
        assert other.membership == bone.membership
        assert other.display == bone.display
        assert other.pose == bone.pose
        assert other.transform == bone.transform
        assert other.constraints == bone.constraints
    assert back.armature == r.armature
    assert back.source == r.source


def test_roundtrip_is_stable():
    """Same record -> same bytes. The no-op build test depends on this."""
    r = make_record(4)
    assert to_json(r) == to_json(from_json(to_json(r)))


def test_transform_survives_roundtrip():
    r = ops.set_pose(make_record(3), ["root"], location=(1.0, 2.0, 3.0))
    back = from_json(to_json(r))
    assert back.bones["root"].transform.location == (1.0, 2.0, 3.0)
    assert back.bones["root"].transform.rotation is None


def test_empty_record_roundtrips():
    r = RigRecord()
    assert from_json(to_json(r)) == r


def test_version_mismatch_is_refused():
    text = to_json(make_record(1)).replace('"version":2', '"version":99')
    try:
        from_json(text)
    except RecordError:
        return
    raise AssertionError("a newer record must not be silently read")


def test_bad_json_is_refused():
    for bad in ("", "{not json", "[]", '{"version":2}'):
        try:
            from_json(bad)
        except RecordError:
            continue
        raise AssertionError("bad input should not parse: " + repr(bad))


def test_peek_version():
    assert peek_version(to_json(make_record(1))) == RECORD_VERSION
    assert peek_version("{not json") is None
    assert peek_version("") is None


def test_validate_flags_dangling_parent():
    r = make_record(3)
    broken = M_diff.apply_changes(
        r, M_diff.ChangeSet(bones={"root": {"rest.parent": "nope"}})
    )
    assert any("nope" in p for p in validate(broken))
    assert validate(r) == []


# --- store ------------------------------------------------------------------


def test_store_roundtrip():
    obj = FakeObject()
    r = make_record(4)
    assert store_record.exists(obj) is False
    assert store_record.read(obj) is None
    store_record.write(obj, r)
    assert store_record.exists(obj) is True
    assert store_record.read(obj) == r
    assert store_record.stored_version(obj) == RECORD_VERSION


def test_store_forget():
    obj = FakeObject()
    store_record.write(obj, make_record(2))
    store_record.forget(obj)
    assert store_record.exists(obj) is False
    assert store_record.read(obj) is None


def test_store_read_never_writes():
    """Reading must not mutate the object -- that was the v1 capture bug."""
    obj = FakeObject()
    store_record.write(obj, make_record(3))
    before = dict(obj)
    for _ in range(3):
        store_record.read(obj)
    assert dict(obj) == before


def test_widgets_roundtrip():
    obj = FakeObject()
    lib = make_widgets()
    widgets_lib.write(obj, lib)
    back = widgets_lib.read(obj)
    assert back.widgets == lib.widgets
    assert "WGT-rig_bone.001" in back
    assert len(back) == 1


def test_widgets_compress_large():
    big = WidgetLib(
        widgets={
            "WGT-%d" % i: {
                "verts": [(float(j), 0.0, 0.0) for j in range(200)],
                "edges": [(j, j + 1) for j in range(199)],
                "faces": [],
            }
            for i in range(5)
        }
    )
    obj = FakeObject()
    widgets_lib.write(obj, big)
    stored = obj[widgets_lib.KEY]
    assert stored.startswith("z:"), "large libraries must be compressed"
    assert widgets_lib.read(obj).widgets == big.widgets
    count, nbytes = widgets_lib.stats(obj)
    assert count == 5 and nbytes == len(stored)


def test_widgets_corrupt_is_survivable():
    obj = FakeObject()
    obj[widgets_lib.KEY] = "z:not-base64!!"
    assert len(widgets_lib.read(obj)) == 0


def test_touched_roundtrip():
    obj = FakeObject()
    changes = M_diff.ChangeSet(
        bones={"root": {"display.shape": "X", "rest.deform": False}}
    )
    store_touched.write(obj, changes)
    assert store_touched.read(obj) == {"root": ("display.shape", "rest.deform")}
    assert store_touched.count(obj) == 2


def test_touched_empty_removes_property():
    obj = FakeObject()
    store_touched.write(obj, M_diff.ChangeSet(bones={"root": {"rest.deform": False}}))
    assert store_touched.exists(obj)
    store_touched.write(obj, M_diff.ChangeSet())
    assert not store_touched.exists(obj)
    assert store_touched.read(obj) == {}


def test_lock_blocks_capture():
    lock.reset()
    assert not lock.is_held()
    lock.guard("capture")  # no raise outside a build
    with lock.held():
        assert lock.is_held()
        raised = False
        try:
            lock.guard("capture")
        except RuntimeError:
            raised = True
        assert raised, "capture must be blocked during a build"
        with lock.held():  # re-entrant
            assert lock.is_held()
        assert lock.is_held(), "nested exit must not release the lock"
    assert not lock.is_held()
