"""diff / restore drive the whole build; they carry the convergence guarantee."""

from conftest import make_record

from armature_nodes.model import ops
from armature_nodes.model.diff import (
    ChangeSet,
    all_paths,
    apply_changes,
    diff,
    restore_paths,
    value_at,
    with_value,
)


def test_identical_records_diff_empty():
    r = make_record(4)
    changes = diff(r, r)
    assert not changes
    assert changes.bones == {}
    assert changes.touched_count() == 0


def test_single_field_change():
    base = make_record(3)
    target = ops.set_shape(base, ["bone.001"], shape="WGT-other")
    changes = diff(base, target)
    assert set(changes.bones) == {"bone.001"}
    assert changes.bones["bone.001"] == {"display.shape": "WGT-other"}
    assert changes.touched_count() == 1


def test_unchanged_bones_cost_nothing():
    base = make_record(20)
    # Flip the value rather than assuming one: setting a field to what it
    # already holds is correctly no change at all.
    flipped = not base.bones["bone.005"].rest.deform
    target = ops.set_deform(base, ["bone.005"], flipped)
    assert list(diff(base, target).bones) == ["bone.005"]


def test_setting_a_field_to_its_current_value_is_not_a_change():
    base = make_record(6)
    same = ops.set_deform(base, ["bone.005"], base.bones["bone.005"].rest.deform)
    assert not diff(base, same)


def test_added_and_removed_bones():
    base = make_record(3)
    fewer = base.with_bones({k: v for k, v in base.bones.items() if k != "bone.002"})
    changes = diff(base, fewer)
    assert changes.removed == ("bone.002",)
    assert changes.added == ()
    assert diff(fewer, base).added == ("bone.002",)


def test_apply_changes_reproduces_target():
    base = make_record(5)
    target = ops.set_pose(
        ops.set_shape(base, ["bone.001"], shape="W", wire_width=9.0),
        ["bone.002"],
        location=(1.0, 2.0, 3.0),
    )
    assert apply_changes(base, diff(base, target)) == target


def test_restore_paths_returns_stale_only():
    base = make_record(3)
    changes = diff(base, ops.set_shape(base, ["bone.001"], shape="W"))
    previously = {
        "bone.001": ("display.shape", "rest.deform"),
        "bone.002": ("display.scale",),
    }
    # display.shape is being written again, so it is not restored.
    assert restore_paths(previously, changes) == {
        "bone.001": ("rest.deform",),
        "bone.002": ("display.scale",),
    }


def test_restore_of_everything_when_graph_goes_empty():
    """Unplugging: nothing is changing, so everything touched is restored."""
    base = make_record(3)
    previously = {"bone.001": ("display.shape", "transform.location")}
    stale = restore_paths(previously, diff(base, base))
    assert stale == {"bone.001": ("display.shape", "transform.location")}


def test_convergence_round_trip():
    """record + graph, then graph removed, lands back on the record exactly."""
    base = make_record(6)
    target = ops.set_pose(
        ops.set_shape(base, ["bone.001", "bone.003"], shape="WGT-x", wire_width=4.0),
        ["bone.002"],
        location=(5.0, 0.0, 0.0),
    )
    changes = diff(base, target)
    live = apply_changes(base, changes)
    assert live == target

    # Now the nodes are gone: restore every touched path from the record.
    stale = restore_paths(changes.paths(), diff(base, base))
    restored = live
    for name, paths in stale.items():
        bone = restored.bones[name]
        for path in paths:
            bone = with_value(bone, path, value_at(base.bones[name], path))
        restored = restored.with_bones({**restored.bones, name: bone})
    assert restored == base


def test_value_at_and_with_value_cover_all_paths():
    bone = make_record(2).bones["bone.001"]
    for path in all_paths():
        original = value_at(bone, path)
        rebuilt = with_value(bone, path, original)
        assert value_at(rebuilt, path) == original
        assert rebuilt == bone


def test_transform_is_diffable():
    base = make_record(2)
    target = ops.set_pose(base, ["root"], location=(1.0, 0.0, 0.0))
    assert diff(base, target).bones["root"] == {"transform.location": (1.0, 0.0, 0.0)}
    cleared = ops.clear_pose(target, ["root"])
    assert diff(target, cleared).bones["root"] == {"transform.location": None}


def test_constraints_diff_as_a_whole():
    base = make_record(3)
    stripped = with_value(base.bones["bone.001"], "constraints", ())
    target = base.with_bones({**base.bones, "bone.001": stripped})
    assert diff(base, target).bones["bone.001"] == {"constraints": ()}


def test_changeset_paths_shape():
    changes = ChangeSet(bones={"a": {"rest.deform": False, "display.shape": "x"}})
    assert changes.paths() == {"a": ("display.shape", "rest.deform")}


# --- ops --------------------------------------------------------------------


def test_select_empty_pattern_is_everything():
    r = make_record(4)
    assert ops.select_names(r, "") == list(r.bones)
    assert ops.select_names(r, "   ") == list(r.bones)


def test_select_glob_and_alternatives():
    r = make_record(4)
    assert ops.select_names(r, "bone.001") == ["bone.001"]
    assert ops.select_names(r, "bone.*") == ["bone.001", "bone.002", "bone.003"]
    assert ops.select_names(r, "root;bone.002") == ["root", "bone.002"]
    assert ops.select_names(r, "nothing.*") == []


def test_set_shape_only_touches_given_fields():
    r = make_record(3)
    before = r.bones["bone.001"].display
    after = ops.set_shape(r, ["bone.001"], shape="W2").bones["bone.001"].display
    assert after.shape == "W2"
    assert after.scale == before.scale
    assert after.wire_width == before.wire_width


def test_ops_do_not_mutate_input():
    r = make_record(3)
    snapshot = r.bones["bone.001"]
    ops.set_shape(r, ["bone.001"], shape="ZZZ")
    ops.set_pose(r, ["bone.001"], location=(9.0, 9.0, 9.0))
    ops.set_deform(r, ["bone.001"], False)
    assert r.bones["bone.001"] == snapshot


def test_offset_pose_accumulates():
    r = make_record(2)
    once = ops.offset_pose(r, ["root"], location=(1.0, 0.0, 0.0))
    twice = ops.offset_pose(once, ["root"], location=(2.0, 0.0, 0.0))
    assert twice.bones["root"].transform.location == (3.0, 0.0, 0.0)


def test_mirror_names():
    assert ops.mirror_names("hand.L") == "hand.R"
    assert ops.mirror_names("hand.R") == "hand.L"
    assert ops.mirror_names("thigh_l") == "thigh_r"
    assert ops.mirror_names("spine") is None
