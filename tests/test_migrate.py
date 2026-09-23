"""v1 -> v2 must be lossless and must never consult the live armature."""

import json

from conftest import FakeObject, v1_record

from armature_nodes.model import migrate
from armature_nodes.model.schema import RECORD_VERSION, RecordError, from_json, to_json
from armature_nodes.store import record as store_record


def test_needs_migration():
    assert migrate.needs_migration(1) is True
    assert migrate.needs_migration(RECORD_VERSION) is False
    assert migrate.needs_migration(RECORD_VERSION + 1) is False
    assert migrate.needs_migration(None) is False


def test_v1_fields_survive_exactly():
    data = v1_record(3)
    rec = migrate.migrate_dict(data)
    assert rec.version == RECORD_VERSION
    assert list(rec.bones) == [b["name"] for b in data["bones"]]
    for src in data["bones"]:
        bone = rec.bones[src["name"]]
        assert tuple(src["head"]) == bone.rest.head
        assert tuple(src["tail"]) == bone.rest.tail
        assert src["roll"] == bone.rest.roll
        assert src["parent"] == bone.rest.parent
        assert src["use_connect"] == bone.rest.connect
        assert src["use_deform"] == bone.rest.deform
        assert src["envelope_distance"] == bone.rest.envelope_distance
        assert src["envelope_weight"] == bone.rest.envelope_weight
        shape = src.get("shape")
        if shape:
            d = bone.display
            assert shape["widget"] == d.shape
            assert shape["preset"] == d.preset, "v1 preset must not be dropped"
            assert tuple(shape["scale"]) == d.scale
            assert tuple(shape["translation"]) == d.translation
            assert tuple(shape["rotation"]) == d.rotation
            assert shape["wire_width"] == d.wire_width
            assert shape["scale_to_bone_length"] == d.use_bone_size
            assert shape["show_wire"] == d.show_wire
        for i, c in enumerate(src.get("constraints", [])):
            got = bone.constraints[i]
            assert got.type == c["type"] and got.name == c["name"]
            assert got.props == c["params"], "v1 constraint params must survive"


def test_new_fields_take_defaults_not_live_data():
    bone = migrate.migrate_dict(v1_record(2)).bones["bone.001"]
    assert bone.membership.collections == ()
    assert bone.membership.layers is None
    assert bone.pose.rotation_mode == "QUATERNION"
    assert bone.pose.locks == {} and bone.pose.ik == {}
    assert bone.transform.location is None
    assert bone.display.color is None


def test_migrated_record_roundtrips():
    rec = migrate.migrate_dict(v1_record(3))
    assert from_json(to_json(rec)) == rec


def test_unknown_version_refused():
    for bad in ({"version": 99, "bones": []}, {"version": None, "bones": []}, {}):
        try:
            migrate.migrate_dict(bad)
        except RecordError:
            continue
        raise AssertionError("should not migrate: " + repr(bad))


def test_store_reads_v1_without_writing():
    """A v1 file opens fine and stays v1 until something explicitly upgrades."""
    obj = FakeObject()
    obj[migrate.V1_KEY] = json.dumps(v1_record(2))
    before = dict(obj)

    assert store_record.exists(obj)
    assert store_record.stored_version(obj) == 1
    assert store_record.needs_migration(obj)
    rec = store_record.read(obj)
    assert rec is not None and rec.version == RECORD_VERSION
    assert dict(obj) == before, "reading must not persist the migration"


def test_persist_migration_keeps_v1_until_v2_written():
    obj = FakeObject()
    obj[migrate.V1_KEY] = json.dumps(v1_record(2))
    expected = store_record.read(obj)

    rec = store_record.persist_migration(obj)
    assert rec == expected
    assert store_record.KEY in obj
    assert migrate.V1_KEY not in obj, "v1 is retired only after v2 is verified"
    assert store_record.read(obj) == expected
    assert store_record.needs_migration(obj) is False


def test_corrupt_v2_falls_back_to_v1():
    """A broken v2 must not hide a good legacy record underneath it."""
    obj = FakeObject()
    obj[migrate.V1_KEY] = json.dumps(v1_record(2))
    obj[store_record.KEY] = "{corrupt"
    rec = store_record.read(obj)
    assert rec is not None
    assert len(rec.bones) == 2
