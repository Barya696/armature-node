"""The Armature node tree data-block.

Registering a custom NodeTree gives the tree its own editor sub-type in the
Node Editor (own tree-type icon in the editor header, own Shift+A menu, own
datablock selector) -- the same mechanism Sverchok and Animation Nodes use.
The tree is a normal .blend data-block: saved with the file and undo-aware.

Like a material in the Shader Editor, an Armature node tree is bound to an
armature object and every edit is applied to that armature immediately: any
link/node change (Blender calls ``update``) or property change (our injected
property callbacks call ``mark_dirty``) schedules a deferred rebuild.
"""

import contextlib

import bpy
from bpy.types import NodeTree
from bpy.props import BoolProperty, FloatProperty, StringProperty

from .core import TREE_IDNAME

# Guard so the rebuild does not re-trigger itself through tree.update().
_updating = False
# Trees waiting for the debounce timer to fire.
_pending = set()
# Armature objects a deleted node owned, waiting to be removed. Node.free()
# runs in a restricted context where removing data-blocks is unsafe, so the
# removal is queued and performed by the same debounce timer as the rebuild.
_pending_removal = set()
# Delay used when something scheduled work without naming a tree.
_DEFAULT_DELAY = 0.08


def _schedule(delay=_DEFAULT_DELAY):
    """(Re)arm the debounce timer. Safe from any context."""
    if bpy.app.timers.is_registered(_flush_pending):
        bpy.app.timers.unregister(_flush_pending)
    bpy.app.timers.register(_flush_pending, first_interval=delay)


def queue_object_removal(name):
    """Remove an armature object once the current operation has finished.

    Called from ``Node.free()``: the node that owned the object is going away,
    so the object it generated must go with it.
    """
    if not name:
        return
    _pending_removal.add(name)
    _schedule()


def _process_removals():
    """Delete queued armature objects and their orphaned data."""
    while _pending_removal:
        name = _pending_removal.pop()
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        data = obj.data if obj.type == "ARMATURE" else None
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except (ReferenceError, RuntimeError) as exc:
            print(f"[Armature Nodes] Could not remove '{name}': {exc}")
            continue
        if data is not None and data.users == 0:
            try:
                bpy.data.armatures.remove(data)
            except (ReferenceError, RuntimeError):
                pass


def is_updating():
    """True while a rebuild (or bulk edit) is in progress."""
    return _updating


@contextlib.contextmanager
def suspend_live_update():
    """Bulk-edit a tree (e.g. decompile) without scheduling rebuilds for
    every property the code sets."""
    global _updating
    previous = _updating
    _updating = True
    try:
        yield
    finally:
        _updating = previous


def _override_context():
    """Build a full context override for use inside a ``bpy.app.timers`` callback.

    Timer callbacks run with no window/screen/area/region in ``bpy.context``,
    so ``bpy.ops.object.mode_set`` fails its poll and ``bpy.context.collection``
    is ``None``. Borrow the first window and a 3D Viewport (any area as a
    fallback) so the rebuild sees the same context an operator would.
    """
    wm = bpy.context.window_manager
    if wm is None or not wm.windows:
        return None
    window = wm.windows[0]
    screen = window.screen
    if screen is None:
        return None
    area = next((a for a in screen.areas if a.type == "VIEW_3D"), None)
    if area is None and screen.areas:
        area = screen.areas[0]
    ctx = {
        "window": window,
        "screen": screen,
        "scene": window.scene,
        "view_layer": window.view_layer,
    }
    if area is not None:
        ctx["area"] = area
        region = next((r for r in area.regions if r.type == "WINDOW"), None)
        if region is not None:
            ctx["region"] = region
    return ctx


def _redraw_all(ctx):
    screen = ctx.get("screen")
    if screen is None:
        return
    for area in screen.areas:
        area.tag_redraw()


def _flush_pending():
    """Timer: drop data owned by deleted nodes, then rebuild every dirty tree."""
    global _updating
    ctx = _override_context()
    if ctx is None:
        # No window yet (file load / background). Try again shortly.
        return 0.1

    _updating = True
    try:
        with bpy.context.temp_override(**ctx):
            _process_removals()
    except Exception as exc:  # noqa: BLE001
        print(f"[Armature Nodes] Cleanup failed: {exc}")
    finally:
        _updating = False

    names = list(_pending)
    _pending.clear()
    for tree_name in names:
        tree = bpy.data.node_groups.get(tree_name)
        if tree is None or not getattr(tree, "is_dirty", False):
            continue
        if not tree.live_update:
            continue
        _updating = True
        try:
            from .build import build_armature_from_tree

            with bpy.context.temp_override(**ctx):
                build_armature_from_tree(tree)
                ctx["view_layer"].update()
            tree.is_dirty = False
            tree.last_error = ""
        except Exception as exc:  # noqa: BLE001
            tree.last_error = str(exc)
            print(f"[Armature Nodes] Live update failed for '{tree.name}': {exc}")
        finally:
            _updating = False
    _redraw_all(ctx)
    return None


class ArmatureNodeTree(NodeTree):
    bl_idname = TREE_IDNAME
    bl_label = "Armature Nodes"
    bl_icon = "ARMATURE_DATA"

    live_update: BoolProperty(
        name="Live Update",
        description=(
            "Apply node changes to the armature immediately, like the Shader "
            "Editor applies material changes. Turn off to pause on very heavy rigs"
        ),
        default=True,
    )

    update_delay: FloatProperty(
        name="Delay",
        description="Seconds to wait after the last edit before applying (debounce)",
        default=0.08,
        min=0.0,
        max=2.0,
        precision=2,
    )

    is_dirty: BoolProperty(
        name="Dirty",
        description="Tree changed since it was last applied",
        default=False,
        options={"HIDDEN"},
    )

    last_error: StringProperty(
        name="Last Error",
        description="Why the last live update failed (empty when it succeeded)",
        default="",
        options={"HIDDEN"},
    )

    def mark_dirty(self):
        """Flag the tree and schedule a debounced apply (safe from any context)."""
        if _updating:
            return
        self.is_dirty = True
        if not self.live_update:
            return
        _pending.add(self.name)
        # update() / property callbacks / Node.free() run in a restricted
        # context where mode switches and operators are forbidden -> always
        # defer to a timer.
        _schedule(self.update_delay)

    def update(self):
        """Called by Blender when links or nodes in the tree change."""
        self.mark_dirty()


def register():
    bpy.utils.register_class(ArmatureNodeTree)


def unregister():
    if bpy.app.timers.is_registered(_flush_pending):
        bpy.app.timers.unregister(_flush_pending)
    _pending.clear()
    _pending_removal.clear()
    bpy.utils.unregister_class(ArmatureNodeTree)
