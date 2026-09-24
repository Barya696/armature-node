"""Procedural Armature Node System.

A custom, wireable node editor (same interaction model as the Shader Editor)
where a node graph compiles into a complete working armature, and an existing
armature decompiles back into the equivalent node graph.

The node tree is a real Blender data-block: it can be built by hand in the
GUI, or built/edited programmatically with zero UI involved.
"""

bl_info = {
    "name": "Armature Nodes",
    "author": "Armature Nodes",
    "version": (0, 2, 0),
    "blender": (3, 6, 0),
    "location": "Node Editor > Armature Nodes",
    "description": (
        "Procedural node system that compiles node graphs into armatures, "
        "decompiles armatures back into node graphs, places bones from "
        "markers, and drives Rigify rigs from a marker skeleton"
    ),
    "category": "Rigging",
}


# Import order is dependency order: nodes holds references to classes from
# sockets and core, so a stale sockets module would hand nodes the old socket
# classes.
_SUBMODULES = (
    "core",
    "compat",
    "model",
    "store",
    "capture",
    "apply",
    "bridge",
    "livelink",
    "legacy_adapter",
    "sockets",
    "tree",
    "primary_rig",
    "nodes",
    "build",
    "decompile",
    "operators",
    "ui",
    "sync",
)


def _reload_submodules():
    """Re-read edited source files before registering.

    Without this, editing the addon and toggling it off/on in Preferences (or
    hitting Reload Scripts) changes nothing: ``from . import nodes`` returns
    whatever is already in ``sys.modules``, so Blender keeps running the code
    it loaded the first time and the addon looks like it never changed.

    Only reloads modules already imported, so the first enable is untouched.
    """
    import importlib
    import sys

    for name in _SUBMODULES:
        module = sys.modules.get(f"{__name__}.{name}")
        if module is not None:
            importlib.reload(module)


def register():
    _reload_submodules()
    from . import ops, sockets, tree, nodes, operators, primary_rig, ui, sync

    ops.register()
    sockets.register()
    tree.register()
    nodes.register()
    operators.register()
    primary_rig.register()
    ui.register()
    sync.register()


def unregister():
    from . import ops, sockets, tree, nodes, operators, primary_rig, ui, sync

    sync.unregister()
    ui.unregister()
    primary_rig.unregister()
    operators.unregister()
    nodes.unregister()
    tree.unregister()
    sockets.unregister()
    ops.unregister()


if __name__ == "__main__":
    register()
