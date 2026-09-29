"""The Skeleton node's own buttons."""

import bpy

import fixtures


def test_clear_markers_runs():
    """Its poll read ``self`` in a classmethod: a NameError, so Blender
    greyed the button out for good."""
    fixtures.ensure_registered()
    tree = bpy.data.node_groups.new("Skeleton", "ArmatureNodeTreeType")
    skeleton = tree.nodes.new("ArmatureNodesSkeletonNode")
    assert len(skeleton.markers) == 33
    with bpy.context.temp_override(node=skeleton):
        assert bpy.ops.armature_nodes.skeleton_clear_markers.poll()
        assert bpy.ops.armature_nodes.skeleton_clear_markers() == {"FINISHED"}
    assert len(skeleton.markers) == 0
