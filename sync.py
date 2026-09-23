"""Shader-Editor style binding between armatures and node trees.

* Every armature object owns (at most) one Armature node tree, stored in
  ``Object.armature_nodes_tree`` -- like ``Object.active_material``.
* Whenever the active object changes, every open Armature node editor that is
  not pinned switches to that armature's tree. If the armature has no tree
  yet, one is created by decompiling the armature on the spot.
* Edits in the tree are applied to the armature immediately (see tree.py).

Active-object changes are picked up through ``bpy.msgbus`` (instant) with a
``depsgraph_update_post`` handler as a fallback for cases msgbus misses.
"""

import bpy
from bpy.props import PointerProperty
from bpy.app.handlers import persistent

from .core import TREE_IDNAME

# Nodes that own draggable marker handles.
_MARKER_NODES = ("ArmatureNodesSkeletonNode", "ArmatureNodesMarkerNode")

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


def create_tree_for_armature(obj, shapes_only=False, full=False):
    """Decompile ``obj`` into a new tree and bind it."""
    from .decompile import decompile_armature_to_tree
    from .tree import suspend_live_update

    tree = bpy.data.node_groups.new(f"{obj.name} Nodes", TREE_IDNAME)
    tree.live_update = True
    with suspend_live_update():
        try:
            decompile_armature_to_tree(obj, tree, shapes_only=shapes_only, full=full)
        except RuntimeError as exc:
            print(f"[Armature Nodes] Could not decompile '{obj.name}': {exc}")
        tree.is_dirty = False
    obj.armature_nodes_tree = tree
    return tree


def resync_tree_from_armature(obj, shapes_only=False, full=False):
    """Re-read the armature into its existing tree (keeps editors on it)."""
    from .decompile import decompile_armature_to_tree
    from .tree import suspend_live_update

    tree = tree_for_armature(obj, create=False)
    if tree is None:
        return create_tree_for_armature(obj, shapes_only=shapes_only, full=full)
    with suspend_live_update():
        decompile_armature_to_tree(obj, tree, shapes_only=shapes_only, full=full)
        tree.is_dirty = False
    return tree


# ---------------------------------------------------------------------------
# Editors follow the active armature
# ---------------------------------------------------------------------------


def _armature_node_spaces():
    """Yield every unpinned node editor that can become an Armature Nodes
    editor. Do not require tree_type to already match: unlike the old code,
    this is what makes opening a normal Node Editor immediately follow the
    selected armature, just like Shader Editor follows a material."""
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
            # Only an editor explicitly set to Armature Nodes is ours. This
            # prevents the addon from hijacking Shader/Geometry/Compositor
            # Editors while still auto-populating the Armature editor.
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


def sync_editors_to_active(force=False):
    """Make the Armature node editors show the active armature's tree."""
    global _last_active, _syncing
    if _syncing:
        return
    ctx = bpy.context
    view_layer = getattr(ctx, "view_layer", None)
    obj = view_layer.objects.active if view_layer is not None else None
    if not is_armature(obj):
        return
    spaces = list(_armature_node_spaces())
    if not spaces:
        _last_active = obj.name
        return
    # Even if the active object name did not change, a newly opened generic
    # Node Editor still needs its tree_type switched to Armature Nodes.
    if not force and obj.name == _last_active and all(
        space.tree_type == TREE_IDNAME for space, _area in spaces
    ):
        return

    _syncing = True
    try:
        tree = tree_for_armature(obj, create=True)
        if tree is not None:
            show_tree_in_editors(tree)
        _last_active = obj.name
    finally:
        _syncing = False


# ---------------------------------------------------------------------------
# Topology watcher (backstop for node deletion)
# ---------------------------------------------------------------------------

# tree name -> (node count, link count, hash of node names)
_topology = {}


def _tree_signature(tree):
    return (
        len(tree.nodes),
        len(tree.links),
        hash(tuple(sorted(n.name for n in tree.nodes))),
    )


def watch_tree_topology():
    """Mark a tree dirty when nodes or links appear/disappear.

    ``NodeTree.update()`` covers most edits, but it is not guaranteed for
    removals -- nothing calls it for ``tree.nodes.remove()`` from Python, and
    a tree edited while its editor is closed reports nothing. Node.free()
    handles interactive deletes; this catches the rest. It compares three
    cheap numbers, so a tree that did not change costs almost nothing.
    """
    from .tree import is_updating

    if is_updating():
        return
    live = set()
    for tree in bpy.data.node_groups:
        if tree.bl_idname != TREE_IDNAME:
            continue
        live.add(tree.name)
        signature = _tree_signature(tree)
        previous = _topology.get(tree.name)
        _topology[tree.name] = signature
        if previous is not None and previous != signature:
            tree.mark_dirty()
    for gone in set(_topology) - live:
        del _topology[gone]


def _deferred_sync():
    """Polling fallback for editor creation/switching. Blender does not emit
    an RNA notification when a Node Editor area changes its editor subtype,
    so keep this lightweight watcher alive while the addon is enabled."""
    try:
        sync_editors_to_active()
        watch_tree_topology()  # catches node/link removals update() misses
        sync_marker_handles()  # catches drags that emit no depsgraph event
    except Exception as exc:  # noqa: BLE001
        print(f"[Armature Nodes] Editor sync failed: {exc}")
    return 0.25


def request_sync():
    if not bpy.app.timers.is_registered(_deferred_sync):
        bpy.app.timers.register(_deferred_sync, first_interval=0.0)


def _on_active_object_changed(*_args):
    request_sync()


# ---------------------------------------------------------------------------
# Viewport -> node: live world transform of controlled bones
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
    view_layer = getattr(bpy.context, "view_layer", None)
    obj = view_layer.objects.active if view_layer is not None else None
    if is_armature(obj):
        tree = tree_for_armature(obj, create=False)
        if tree is not None:
            trees[tree.name] = tree
    return list(trees.values())


def sync_marker_handles():
    """Read dragged marker handles back into their nodes.

    The viewport is the editor for a marker: you grab its empty and the node
    follows. Custom Shape nodes no longer mirror their bone's transform --
    in the modifier model the bone's position is an *output* of the graph, so
    reading it back would be a loop.
    """
    from .tree import is_updating

    if is_updating():
        return  # a rebuild is mid-flight; matrices are not trustworthy
    changed = False
    for tree in _trees_to_track():
        for node in tree.nodes:
            if node.bl_idname not in _MARKER_NODES:
                continue
            try:
                if node.sync_from_empties():
                    changed = True
            except Exception as exc:  # noqa: BLE001
                print(f"[Armature Nodes] Marker sync failed on '{node.name}': {exc}")
    if changed:
        for _space, area in _armature_node_spaces():
            area.tag_redraw()


@persistent
def _on_depsgraph_update(scene, depsgraph=None):
    # Fallback: selection clicks in the viewport always trigger a depsgraph
    # update, even when the msgbus notification is skipped.
    request_sync()
    # Bone/object transforms changed (or may have): refresh node values. The
    # nodes only write when a value actually differs, so this settles after
    # one pass and does not loop.
    try:
        sync_marker_handles()
    except Exception as exc:  # noqa: BLE001
        print(f"[Armature Nodes] Marker sync failed: {exc}")


@persistent
def _on_frame_change(scene, depsgraph=None):
    # Playback/scrubbing does not always emit depsgraph_update_post; keep the
    # node values following the animated bones.
    try:
        sync_marker_handles()
    except Exception as exc:  # noqa: BLE001
        print(f"[Armature Nodes] Marker sync failed: {exc}")


@persistent
def _on_load_post(*_args):
    global _last_active
    _topology.clear()  # signatures from the previous file mean nothing here
    _last_active = None
    _subscribe_msgbus()  # msgbus subscriptions do not survive file load
    request_sync()


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


def register():
    bpy.types.Object.armature_nodes_tree = PointerProperty(
        name="Armature Nodes",
        description="Node tree that defines and edits this armature",
        type=bpy.types.NodeTree,
        poll=_poll_tree,
    )
    _subscribe_msgbus()
    if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    if _on_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_on_frame_change)
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    request_sync()


def unregister():
    global _last_active
    if bpy.app.timers.is_registered(_deferred_sync):
        bpy.app.timers.unregister(_deferred_sync)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    if _on_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_on_frame_change)
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    bpy.msgbus.clear_by_owner(_MSGBUS_OWNER)
    del bpy.types.Object.armature_nodes_tree
    _topology.clear()
    _last_active = None
