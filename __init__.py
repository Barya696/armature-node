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
        "decompiles armatures back into node graphs, and drives Rigify rigs "
        "from a MediaPipe marker skeleton"
    ),
    "category": "Rigging",
}


def register():
    from . import sockets, tree, nodes, operators, primary_rig, ui, sync

    sockets.register()
    tree.register()
    nodes.register()
    operators.register()
    primary_rig.register()
    ui.register()
    sync.register()


def unregister():
    from . import sockets, tree, nodes, operators, primary_rig, ui, sync

    sync.unregister()
    ui.unregister()
    primary_rig.unregister()
    operators.unregister()
    nodes.unregister()
    tree.unregister()
    sockets.unregister()


if __name__ == "__main__":
    register()
