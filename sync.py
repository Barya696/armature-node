"""Shader-Editor style binding between armatures and node trees.

* Every armature object owns (at most) one Armature node tree, stored in
  ``Object.armature_nodes_tree`` -- like ``Object.active_material``.
* Whenever the active object changes, every open Armature node editor that is
  not pinned switches to that armature's tree. If the armature has no tree
  yet, one is created by decompiling the armature on the spot.
* Edits in the tree are applied to the armature immediately (see tree.py).
* The rig and the marker handles are read back into the nodes when they move
  (the live link, ``livelink``), from ``depsgraph_update_post``.

The handler only acts on updates that moved a bound rig or a marker handle.
Everything else Blender reports -- a node dragged in the editor, a value
typed, a material changed -- costs one pass over the update list. Reading the
nodes back on every update is what made the node editor lag.

Two things Blender does not report at all -- a Node Editor switched to
Armature Nodes, an armature changing mode -- are polled, twice a second, by a
timer that does nothing else.
"""

import logging

import bpy
from bpy.app.handlers import persistent
from bpy.props import PointerProperty

from .core import TREE_IDNAME

log = logging.getLogger(__name__)

# Nodes that own draggable marker handles.
_MARKER_NODES = ("ArmatureNodesSkeletonNode", "ArmatureNodesMarkerNode")
# Seconds between polls for what Blender does not report.
_POLL_INTERVAL = 0.5

# Owner token for msgbus subscriptions.
_MSGBUS_OWNER = object()
# Name of the armature the editors were last synced to.
_last_active = None
# Re-entrancy guard for the sync itself.
_syncing = False


# ---------------------------------------------------------------------------
# Tree <-> armature binding
# ---------------------------------------------------------------------------


def _poll_tree(self, tree):
    return tree.bl_idname == TREE_IDNAME


def is_armature(obj):
    return obj is not None and obj.type == "ARMATURE"


def tree_for_armature(obj, create=True):
    """Return the node tree bound to ``obj``, creating it by decompiling the
    armature when ``create`` is True and none exists yet."""
    if not is_armature(obj):
        return None

    tree = obj.armature_nodes_tree
    if tree is not None and tree.bl_idname == TREE_IDNAME:
        return tree

    # Adopt a tree that already outputs to this armature (older files).
    for candidate in bpy.data.node_groups:
        if candidate.bl_idname != TREE_IDNAME:
            continue
        for node in candidate.nodes:
            if node.bl_idname == "ArmatureNodesOutputNode" and node.target_name() == obj.name:
                obj.armature_nodes_tree = candidate
                return candidate

    if not create:
        return None
    return create_tree_for_armature(obj)


def create_tree_for_armature(obj, full=False):
    """Decompile ``obj`` into a new tree and bind it."""
    from .decompile import decompile_armature_to_tree
    from .tree import suspend_live_update

    tree = bpy.data.node_groups.new(f"{obj.name} Nodes", TREE_IDNAME)
    tree.live_update = True
    with suspend_live_update():
        decompile_armature_to_tree(obj, tree, full=full)
        tree.is_dirty = False
    obj.armature_nodes_tree = tree
    return tree


def resync_tree_from_armature(obj, full=False):
    """Re-read the armature into its existing tree (keeps editors on it)."""
    from .decompile import decompile_armature_to_tree
    from .tree import suspend_live_update

    tree = tree_for_armature(obj, create=False)
    if tree is None:
        return create_tree_for_armature(obj, full=full)
    with suspend_live_update():
        decompile_armature_to_tree(obj, tree, full=full)
        tree.is_dirty = False
    return tree


# ---------------------------------------------------------------------------
# Editors follow the active armature
# ---------------------------------------------------------------------------


def _armature_node_spaces():
    """Every node editor set to Armature Nodes, with its area.

    Only an editor explicitly set to Armature Nodes is ours: the addon never
    takes over a Shader, Geometry or Compositor editor.
    """
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "NODE_EDITOR":
                continue
            space = area.spaces.active
            if space is not None and space.tree_type == TREE_IDNAME:
                yield space, area


def show_tree_in_editors(tree):
    """Point every un-pinned Armature node editor at ``tree``."""
    for space, area in _armature_node_spaces():
        if getattr(space, "pin", False):
            continue
        if space.node_tree != tree:
            space.node_tree = tree
            area.tag_redraw()


def _active_object():
    view_layer = getattr(bpy.context, "view_layer", None)
    return view_layer.objects.active if view_layer is not None else None


def sync_editors_to_active(force=False):
    """Make the Armature node editors show the active armature's tree."""
    global _last_active, _syncing
    if _syncing:
        return
    obj = _active_object()
    if not is_armature(obj):
        return
    if not any(True for _ in _armature_node_spaces()):
        _last_active = obj.name
        return
    if not force and obj.name == _last_active:
        return

    _syncing = True
    try:
        tree = tree_for_armature(obj, create=True)
        if tree is not None:
            show_tree_in_editors(tree)
        _last_active = obj.name
    finally:
        _syncing = False


def _poll():
    """What Blender does not report: an editor switched to Armature Nodes,
    an armature changing mode, a group's interface edited."""
    try:
        from .groups import sync_all_group_nodes

        sync_editors_to_active()
        watch_armature_mode()
        sync_all_group_nodes()
    except Exception as exc:  # noqa: BLE001
        log.warning("Editor sync failed: %s", exc)
    return _POLL_INTERVAL


def request_sync():
    """Poll now, and from then on at ``_POLL_INTERVAL``."""
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)
    bpy.app.timers.register(_poll, first_interval=0.0)


def _on_active_object_changed(*_args):
    request_sync()


# ---------------------------------------------------------------------------
# Viewport -> node: the rig and the marker handles, read back
# ---------------------------------------------------------------------------


def armature_for_tree(tree):
    """The armature object a tree is bound to (Input source, else Output name)."""
    fallback = None
    for node in tree.nodes:
        if node.bl_idname == "ArmatureNodesInputNode":
            src = getattr(node, "source", None)
            if is_armature(src):
                return src
        elif node.bl_idname == "ArmatureNodesOutputNode" and fallback is None:
            fallback = bpy.data.objects.get(node.target_name())
    return fallback if is_armature(fallback) else None


def _trees_to_track():
    """Trees whose nodes should follow their rig: every tree shown in an
    Armature node editor plus the active armature's tree."""
    trees = {}
    for space, _area in _armature_node_spaces():
        tree = space.node_tree
        if tree is not None:
            trees[tree.name] = tree
    obj = _active_object()
    if is_armature(obj):
        tree = tree_for_armature(obj, create=False)
        if tree is not None:
            trees[tree.name] = tree
    return list(trees.values())


def _nodes_to_track(rigs=None, handle_trees=None):
    """Every node of the tracked trees and of the node groups they run,
    through groups inside groups -- each tree once, even when two rigs share
    a group. A marker or live node inside a group is as much the rig's as
    one outside it.

    Narrowed, when ``rigs`` or ``handle_trees`` is given, to the trees bound
    to one of those rigs or holding one of those trees' marker handles.
    """
    from .groups import trees_in_use

    seen = set()
    for root in _trees_to_track():
        trees = trees_in_use(root)
        if rigs is not None or handle_trees is not None:
            obj = armature_for_tree(root)
            wanted = (obj is not None and obj.name in (rigs or ())) or any(
                t.name in (handle_trees or ()) for t in trees
            )
            if not wanted:
                continue
        for tree in trees:
            if tree.name in seen:
                continue
            seen.add(tree.name)
            yield from tree.nodes


def refresh_marker_visibility():
    """Handles follow what is displayed, both ways.

    Created whenever one is missing -- a marker newly wired to an Output, a
    file load, an empty deleted by hand -- and removed once its node stops
    feeding a displaying Armature Output, so unwiring a marker clears it from
    the viewport. Run after the graph changes, not on every update.
    """
    from .primary_rig import (
        ensure_marker_empties,
        find_marker_empties,
        marker_empties_index,
        marker_node_visible,
        remove_marker_empties,
    )
    from .tree import is_updating

    if is_updating():
        return False
    changed, cache = False, {}
    index = marker_empties_index()
    for node in _nodes_to_track():
        if node.bl_idname not in _MARKER_NODES:
            continue
        try:
            visible = marker_node_visible(node, cache) and len(node.markers)
            shown = bool(find_marker_empties(node, index))
            if visible and not shown:
                ensure_marker_empties(node)
                changed = True
            elif shown and not visible:
                remove_marker_empties(node)
                changed = True
        except Exception as exc:  # noqa: BLE001
            log.warning("Marker handles failed on '%s': %s", node.name, exc)
    if changed:
        _redraw_editors()
    return changed


def _refresh_visibility_once():
    refresh_marker_visibility()
    return None


def request_visibility_refresh():
    """Refresh the handles as soon as Blender is idle (from any context)."""
    if not bpy.app.timers.is_registered(_refresh_visibility_once):
        bpy.app.timers.register(_refresh_visibility_once, first_interval=0.0)


def read_marker_drags(handle_trees=None):
    """Read dragged marker handles back into their nodes."""
    from .tree import is_updating

    if is_updating():
        return  # a rebuild is mid-flight; matrices are not trustworthy
    changed = False
    for node in _nodes_to_track(handle_trees=handle_trees):
        if node.bl_idname not in _MARKER_NODES:
            continue
        try:
            if node.sync_from_empties():
                changed = True
        except Exception as exc:  # noqa: BLE001
            log.warning("Marker sync failed on '%s': %s", node.name, exc)
    if changed:
        _redraw_editors()


def sync_marker_handles():
    """Handles made or removed as displayed, and drags read back: both."""
    refresh_marker_visibility()
    read_marker_drags()


# tree name -> the mode its armature was in on the last poll.
_armature_modes = {}


def watch_armature_mode():
    """Rebuild when a bound armature leaves Edit or Pose mode.

    Modify mode skips while the armature is in Edit mode (``armature.bones``
    is stale there); Full Rig defers in both Edit and Pose, because rebuilding
    edit bones would drag the user out of whatever they were doing. Nothing
    else would notice the mode change, so the graph would stay unapplied until
    the user happened to touch a node.
    """
    for tree in _trees_to_track():
        obj = armature_for_tree(tree)
        if obj is None:
            continue
        previous = _armature_modes.get(tree.name)
        _armature_modes[tree.name] = obj.mode
        if previous in ("EDIT", "POSE") and obj.mode != previous:
            tree.mark_dirty()


def sync_bone_nodes(rigs=None, handle_trees=None):
    """Pull the live rig into every live-linked node.

    What makes a Bone, Position, Rotation or Transform node show and hold the
    bone's real transform while the user poses it. The node tells its own
    writes apart from the user's with a snapshot -- see ``livelink``.
    """
    from .nodes import deferred_marker_writes
    from .store import lock
    from .tree import is_updating

    if is_updating() or lock.is_held():
        return  # mid-build: matrices are half-applied, and it is our own write

    changed = False
    # Marker writes land together at the end, parents first, so a child
    # marker carried by its parent is not also moved by its own bone's carry.
    with deferred_marker_writes():
        for node in _nodes_to_track(rigs, handle_trees):
            if not hasattr(node, "follow_live"):
                continue
            try:
                if node.follow_live():
                    changed = True
            except Exception as exc:  # noqa: BLE001
                log.warning("Live link failed on '%s': %s", node.name, exc)
    if changed:
        _redraw_editors()


def _redraw_editors():
    for _space, area in _armature_node_spaces():
        area.tag_redraw()


def _moved(depsgraph):
    """(armature names, trees owning marker handles) this update moved."""
    rigs, handle_trees = set(), set()
    for update in depsgraph.updates:
        obj = update.id
        if not isinstance(obj, bpy.types.Object):
            continue
        obj = obj.original
        if obj.type == "ARMATURE":
            rigs.add(obj.name)
        elif obj.type == "EMPTY":
            tree = obj.get("an_tree")
            if tree is not None:
                handle_trees.add(tree)
    return rigs, handle_trees


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


@persistent
def _on_depsgraph_update(scene, depsgraph=None):
    obj = _active_object()
    if obj is not None and is_armature(obj) and obj.name != _last_active:
        request_sync()  # msgbus misses some selection changes
    if depsgraph is None:
        return
    rigs, handle_trees = _moved(depsgraph)
    if not rigs and not handle_trees:
        return  # not about a rig or a handle: nothing to read back
    try:
        if handle_trees:
            read_marker_drags(handle_trees)
        sync_bone_nodes(rigs, handle_trees)
    except Exception as exc:  # noqa: BLE001
        log.warning("Node sync failed: %s", exc)


@persistent
def _on_frame_change(scene, depsgraph=None):
    # Playback and scrubbing animate the rig without a depsgraph update
    # handler call; the live nodes follow the animated bones.
    try:
        read_marker_drags()
        sync_bone_nodes()
    except Exception as exc:  # noqa: BLE001
        log.warning("Node sync failed: %s", exc)


@persistent
def _on_undo_redo(*_args):
    """Undo restores node values and pose together, but not the snapshots.

    Keeping them would make an undone grab look like a brand-new move, and it
    would be folded into the node a second time.
    """
    _forget_caches()
    request_visibility_refresh()


def _forget_caches():
    """Undo and file load free every node, link and object the caches hold."""
    from . import livelink
    from .core import forget_link_index
    from .primary_rig import forget_caches
    from .tree import graph_changed

    livelink.reset()
    forget_link_index()
    forget_caches()
    graph_changed()


@persistent
def _on_load_post(*_args):
    global _last_active
    _forget_caches()
    _armature_modes.clear()
    _last_active = None
    _subscribe_msgbus()  # msgbus subscriptions do not survive file load
    request_sync()
    request_visibility_refresh()


def _subscribe_msgbus():
    bpy.msgbus.clear_by_owner(_MSGBUS_OWNER)
    bpy.msgbus.subscribe_rna(
        key=(bpy.types.LayerObjects, "active"),
        owner=_MSGBUS_OWNER,
        args=(),
        notify=_on_active_object_changed,
        options={"PERSISTENT"},
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_HANDLERS = (
    ("depsgraph_update_post", _on_depsgraph_update),
    ("frame_change_post", _on_frame_change),
    ("load_post", _on_load_post),
    ("undo_post", _on_undo_redo),
    ("redo_post", _on_undo_redo),
)


def register():
    bpy.types.Object.armature_nodes_tree = PointerProperty(
        name="Armature Nodes",
        description="Node tree that defines and edits this armature",
        type=bpy.types.NodeTree,
        poll=_poll_tree,
    )
    _subscribe_msgbus()
    for name, handler in _HANDLERS:
        handlers = getattr(bpy.app.handlers, name)
        if handler not in handlers:
            handlers.append(handler)
    request_sync()


def unregister():
    global _last_active
    for timer in (_poll, _refresh_visibility_once):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
    for name, handler in _HANDLERS:
        handlers = getattr(bpy.app.handlers, name)
        if handler in handlers:
            handlers.remove(handler)
    bpy.msgbus.clear_by_owner(_MSGBUS_OWNER)
    del bpy.types.Object.armature_nodes_tree
    _armature_modes.clear()
    _last_active = None
