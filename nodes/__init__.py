"""The Armature node tree's node types, one module per Shift+A category.

The graph is a modifier stack, the way Geometry Nodes is: **Armature Input**
hands the whole rig to **Armature Output**, and every node in between reads
the rig, changes the bones it selects, and passes the rest through untouched.
Decompiling a rig therefore produces two nodes, not one per bone -- the rig
is the input, not something the graph has to describe.

* ``armature_io`` -- ArmatureInputNode (nothing -> Rig), ArmatureOutputNode
  (Rig -> nothing).
* ``bone`` -- BoneNode poses one bone of the rig; ChainNode generates new ones.
* ``marker`` -- MarkerNode (one draggable handle, live with the bone its wire
  reaches), SkeletonNode (a bundle of markers, one output each).
* ``wrap`` -- WrapMarkersNode: fits the tree's markers onto a mesh, as the
  wrap add-on fits a template (pairs, Snap, Attract, Stick). Also Marker in
  Shift+A.
* ``transform`` -- PositionNode, RotationNode, TransformNode, SnapNode: pose
  modifiers. They never touch rest geometry, so they cannot deform a rig they
  are layered onto.
* ``shape`` -- CustomShapeNode: assigns a control widget.
* ``constraint`` -- IKConstraintNode, GenericConstraintNode.
* ``rigify`` -- RigifySwitchNode: sets the rig's switches (IK/FK, pole,
  parents, follow).

The group node is not here: it lives in ``groups.py``, with the rest of node
groups.

``base`` holds what every node builds on, ``marker_base`` what the two marker
nodes share. Every bone-producing node implements
eval_bones(ctx) -> list[BoneDef]; every constraint node implements
eval_constraints(ctx) -> list[ConstraintDef].

A new node goes in the module of its category, and into ``classes`` below.
"""

import logging

import bpy
from bpy.types import Node

from .armature_io import ArmatureInputNode, ArmatureOutputNode
from .bone import BoneNode, ChainNode
from .constraint import GenericConstraintNode, IKConstraintNode
from .marker import MarkerNode, SkeletonNode
from .marker_base import SkeletonMarker, deferred_marker_writes  # noqa: F401
from .rigify import ARMATURE_NODES_OT_rigify_switch_bone, RigifySwitch, RigifySwitchNode
from .shape import CustomShapeNode
from .transform import PositionNode, RotationNode, SnapNode, TransformNode
from .wrap import classes as wrap_classes

log = logging.getLogger(__name__)


classes = (
    SkeletonMarker,
    ArmatureInputNode,
    ArmatureOutputNode,
    BoneNode,
    ChainNode,
    MarkerNode,
    SkeletonNode,
    PositionNode,
    RotationNode,
    TransformNode,
    SnapNode,
    CustomShapeNode,
    IKConstraintNode,
    GenericConstraintNode,
    RigifySwitch,
    RigifySwitchNode,
    ARMATURE_NODES_OT_rigify_switch_bone,
    *wrap_classes,
)


def _on_node_prop_changed(self, context):
    """Any edited node value re-applies the tree to its armature (real time)."""
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


# UI-only toggles that must not trigger a rebuild.
_NO_REBUILD_PROPS = {
    "synced",
    "show_markers",
    "show_detail",
    "show_advanced",
    "show_handles",
    "markers",  # CollectionProperty: does not accept update=
    "lock_depth",
    "symmetric",
    "live_links",  # bookkeeping for the marker's live link
    "parent_link",  # bookkeeping for a marker's parent
    "parent_seen",
    "uid",  # what a marker's lines to other markers attach to
    "layout_version",  # the Transform node's one-time migration
    "wrap_stage",  # the Wrap Markers node's bookkeeping
    "wrap_original",
    "wrap_wrapped",
    "preview",  # moves the markers itself
}


def _inject_live_update(cls):
    """Add an ``update`` callback to every property of a node class.

    Blender only calls NodeTree.update() for link/node changes, not for value
    edits, so without this a scale tweak on a Custom Shape node would do
    nothing until the next structural change. CollectionProperty is skipped
    unconditionally -- Blender's RNA does not accept ``update`` for it at all
    (registration fails outright), and per-item updates on the PropertyGroup
    itself are the correct way to react to collection edits anyway.
    """
    for name, prop in list(cls.__annotations__.items()):
        keywords = getattr(prop, "keywords", None)
        if keywords is None or name in _NO_REBUILD_PROPS:
            continue
        function = getattr(prop, "function", None)
        if function is not None and getattr(function, "__name__", "") == "CollectionProperty":
            continue
        if "update" not in keywords:
            keywords["update"] = _on_node_prop_changed


def _check_reserved_names(cls):
    """Warn when a node property shadows one of Blender's own Node members.

    Registration succeeds either way, which is what makes this worth checking:
    a node that declares ``location`` overrides the node's position in the
    editor, so it registers cleanly and then Blender's own add-node operator
    dies setting ``node.location``, and the node can never be moved. Nothing
    else catches it -- the failure surfaces as a ValueError from Blender's
    code, with no hint that an addon property is the cause.
    """
    node_rna = getattr(Node, "bl_rna", None)
    if node_rna is None:  # not running inside Blender
        return
    reserved = set(node_rna.properties.keys()) - {"bl_idname"}
    clashes = sorted(set(getattr(cls, "__annotations__", {})) & reserved)
    if clashes:
        log.warning(
            "%s declares %s, which shadow bpy.types.Node properties. Rename them "
            "(e.g. 'location' -> 'bone_location') or the node will misbehave.",
            cls.__name__,
            ", ".join(clashes),
        )


def register():
    for cls in classes:
        if issubclass(cls, Node):
            _check_reserved_names(cls)
            _inject_live_update(cls)
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
