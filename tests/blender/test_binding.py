"""Binding is explicit, and the target follows the binding -- never selection.

Two rules the previous versions broke:

* Opening an editor on a rig must not record it. A rig whose widgets were
  already swapped would otherwise be captured as its own original, and saved
  that way for ever. Capture happens when the user clicks Bind, and nowhere
  else.
* In Modify mode the target is exclusively the Armature Input's source.
  Falling back to a typed name, or to whichever armature happens to be
  active, lets a graph modify a rig it was never pointed at.
"""

import bpy

import fixtures

from armature_nodes.model.migrate import V1_KEY
from armature_nodes.store import record as record_store


def _keys(obj):
    return {k for k in obj.keys() if k.startswith("an_")}


def test_opening_an_editor_does_not_capture():
    """Acceptance: no record appears until Bind is clicked."""
    obj = fixtures.make_rig()
    # A rig that has already been messed with, which is the dangerous case:
    # capturing now would record the damage as the original.
    obj.pose.bones["bone.001"].custom_shape = None

    assert _keys(obj) == set(), "fixture rig starts clean"

    fixtures.ensure_registered()
    from armature_nodes.decompile import decompile_armature_to_tree

    tree = bpy.data.node_groups.new("t", "ArmatureNodeTreeType")
    decompile_armature_to_tree(obj, tree)

    assert V1_KEY not in obj, "decompile wrote a legacy baseline"
    assert not record_store.exists(obj), "decompile captured a record"
    assert _keys(obj) == set(), f"decompile wrote {_keys(obj)}"


def test_unbound_rig_evaluates_to_nothing():
    """An unbound rig is inert, not silently bound on first evaluation."""
    obj = fixtures.make_rig()
    tree, src, _out = fixtures.make_tree(obj)
    before = fixtures.snapshot(obj)

    from armature_nodes.build import build_armature_from_tree

    build_armature_from_tree(tree)
    assert not record_store.exists(obj), "the build captured a record"
    assert fixtures.snapshot(obj) == before, "the build changed an unbound rig"
    assert src.eval_bones(None) == [], "an unbound Input must emit nothing"


def test_bind_is_what_captures():
    obj = fixtures.make_rig()
    from armature_nodes.ops.bind import bind

    assert not record_store.exists(obj)
    bind(obj)
    assert record_store.exists(obj)
    assert len(record_store.read(obj).bones) == len(obj.data.bones)


def test_bind_refuses_to_overwrite():
    obj = fixtures.make_rig()
    from armature_nodes.ops.bind import bind

    bind(obj)
    raw = record_store.raw(obj)
    raised = False
    try:
        bind(obj)
    except RuntimeError:
        raised = True
    assert raised, "a second bind must not silently re-record"
    assert record_store.raw(obj) == raw


def test_modify_target_is_the_input_source_only():
    """armature_name must not retarget a Modify graph."""
    obj = fixtures.make_rig(name="rig")
    other = fixtures.make_rig(name="decoy")
    from armature_nodes.ops.bind import bind

    bind(obj)
    bind(other)
    tree, _src, out = fixtures.make_tree(obj)
    assert out.mode == "MODIFY"

    out.armature_name = other.name  # a stale or mistyped name
    assert out.target_name() == obj.name, "armature_name retargeted a Modify graph"

    before = fixtures.snapshot(other)
    fixtures.flush(tree)
    assert fixtures.snapshot(other) == before, "the decoy rig was written to"


def test_full_rig_mode_still_honours_armature_name():
    obj = fixtures.make_rig()
    tree, _src, out = fixtures.make_tree(obj)
    out.mode = "FULL"
    out.armature_name = "generated"
    assert out.target_name() == "generated"


def test_activating_another_armature_leaves_the_record_alone():
    """Acceptance test 7: the editor follows binding, not selection."""
    obj = fixtures.make_rig(name="rig")
    metarig = fixtures.make_rig(name="metarig")
    from armature_nodes.ops.bind import bind

    bind(obj)
    raw = record_store.raw(obj)
    tree, src, _out = fixtures.make_tree(obj)

    # Make the other armature active, as selecting it in the viewport would.
    bpy.context.view_layer.objects.active = metarig

    from armature_nodes import sync

    sync.sync_editors_to_active()
    sync.sync_marker_handles()

    assert record_store.raw(obj) == raw, "the bound rig's record changed"
    assert not record_store.exists(metarig), "a record was created for the metarig"
    assert src.source is obj, "the tree was retargeted by selection"
