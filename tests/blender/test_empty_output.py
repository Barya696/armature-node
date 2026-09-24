"""Acceptance test 8: an Output with nothing connected equals the record.

Nothing deleted, nothing reset to defaults. This is the bug that started the
rewrite: unplugging the Input used to strip every custom shape, which made a
generated rig look exactly like its metarig.
"""

import fixtures


def test_empty_output_equals_record():
    obj = fixtures.make_rig()
    from armature_nodes.ops.bind import bind

    bind(obj)
    before = fixtures.snapshot(obj)

    # An Output with nothing wired into it at all.
    tree, _src, _out = fixtures.make_tree(obj, connected=False)
    fixtures.flush(tree)

    after = fixtures.snapshot(obj)
    assert after["shapes"] == before["shapes"], "shapes changed on an empty graph"
    assert after["deform"] == before["deform"], "deform flags changed"
    assert after["record"] == before["record"], "record changed"
    assert len(obj.data.bones) == len(before["deform"]), "bones were removed"


def test_unplug_replug_is_lossless():
    """The headline case: disconnect, flush, reconnect, flush -- five times."""
    obj = fixtures.make_rig()
    from armature_nodes.ops.bind import bind

    bind(obj)
    tree, src, out = fixtures.make_tree(obj)
    fixtures.flush(tree)
    snap = fixtures.snapshot(obj)

    for i in range(5):
        for link in list(tree.links):
            tree.links.remove(link)
        fixtures.flush(tree)
        assert fixtures.snapshot(obj)["shapes"] == snap["shapes"], \
            f"unplug {i} changed the rig"

        tree.links.new(src.outputs["Rig"], out.inputs["Rig"])
        fixtures.flush(tree)
        assert fixtures.snapshot(obj) == snap, f"replug {i} did not converge"


def test_record_bytes_unchanged_by_building():
    """A build must never rewrite the record -- it only reads it."""
    obj = fixtures.make_rig()
    from armature_nodes.ops.bind import bind
    from armature_nodes.store import record as record_store

    bind(obj)
    raw = record_store.raw(obj)
    tree, _src, _out = fixtures.make_tree(obj)
    for _ in range(3):
        fixtures.flush(tree)
    assert record_store.raw(obj) == raw, "the build rewrote the record"


def test_two_identical_builds_write_nothing():
    """Acceptance test 10, cheap to include here."""
    obj = fixtures.make_rig()
    from armature_nodes.ops.bind import bind
    from armature_nodes.apply import pipeline
    from armature_nodes.store import record as record_store
    from armature_nodes.store import touched as touched_store

    bind(obj)
    base = record_store.read(obj)
    first = pipeline.apply(obj, base, base, touched_store.read(obj))
    second = pipeline.apply(obj, base, base, touched_store.read(obj))
    assert second.writes == 0, f"second identical build wrote {second.writes} properties"
    assert first.errors == [] and second.errors == []
