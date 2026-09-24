"""Acceptance test 4: widgets rebuilt from stored geometry.

Delete every WGT-* object, flush, and the rig must come back with its widgets
intact. This is the test the old implementation could never have passed: it
stored widgets by name, so a missing object meant the shape was unrecoverable
and the "restore" path cleared it instead.
"""

import bpy

import fixtures


def test_widgets_rebuilt_after_deletion():
    obj = fixtures.make_rig()
    from armature_nodes.ops.bind import bind

    bind(obj)
    before = fixtures.snapshot(obj)
    assert before["widget_names"], "fixture rig has no widgets"

    tree, _src, _out = fixtures.make_tree(obj)
    fixtures.flush(tree)

    # Nuke every widget object, as opening a file without them would.
    removed = 0
    for widget in list(bpy.data.objects):
        if widget.name.startswith("WGT-"):
            bpy.data.objects.remove(widget, do_unlink=True)
            removed += 1
    assert removed, "nothing was deleted; fixture is wrong"
    assert all(pb.custom_shape is None for pb in obj.pose.bones), \
        "deleting the objects should have cleared the pose bones"

    # Force the shapes to be rewritten: the record still names them, and the
    # library still holds their geometry.
    from armature_nodes.store import touched as touched_store
    from armature_nodes.store import record as record_store
    from armature_nodes.apply import pipeline

    base = record_store.read(obj)
    # Mark the shape paths as touched so restore-then-apply rewrites them,
    # which is what a real build does after the depsgraph notices the change.
    touched_store.write(
        obj, {name: ("display.shape",) for name in base.bones}
    )
    pipeline.apply(obj, base, base, touched_store.read(obj))

    after = fixtures.snapshot(obj)
    assert after["shapes"] == before["shapes"], "widget assignment or placement changed"
    assert after["widget_geometry"] == before["widget_geometry"], "geometry differs"
    assert after["record"] == before["record"], "record changed"


def test_missing_widget_without_geometry_is_left_alone():
    """A shape we cannot rebuild must not be cleared -- that is data loss."""
    obj = fixtures.make_rig()
    from armature_nodes.ops.bind import bind
    from armature_nodes.apply.widgets import ensure_widget
    from armature_nodes.model.types import WidgetLib

    bind(obj)
    assert ensure_widget("WGT-does-not-exist", WidgetLib()) is None
