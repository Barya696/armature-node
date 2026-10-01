"""Armature Nodes: rigging with a node graph.

A node tree is bound to an armature the way a material is to a mesh: Armature
Input hands the rig to Armature Output, and the nodes in between modify it --
pose bones, place markers, assign widgets, add constraints. Every edit is
applied live, and deleting a node undoes what it did.

README.md is the user guide; ``blender_manifest.toml`` describes the extension.
"""

# Read by Blender for a legacy add-on install; an extension install reads
# blender_manifest.toml instead.
bl_info = {
    "name": "Armature Nodes",
    "author": "baryaelimelec",
    "version": (0, 3, 0),
    "blender": (4, 2, 0),
    "location": "Node Editor > Armature Nodes",
    "description": "Rig with a node graph: modify an armature live, node by node",
    "category": "Rigging",
}

# The modules that register something, in registration order.
_MODULES = (
    "ops",
    "sockets",
    "tree",
    "nodes",
    "groups",
    "human_skeleton",
    "pose_fit",
    "operators",
    "primary_rig",
    "handles",
    "marker_links",
    "ui",
    "sync",
)
_LOG_FORMAT = "[Armature Nodes] %(levelname)s: %(message)s"


def _modules():
    import importlib

    return [importlib.import_module(f".{name}", __name__) for name in _MODULES]


def register():
    _add_log_handler()
    for module in _modules():
        module.register()


def unregister():
    for module in reversed(_modules()):
        module.unregister()
    _remove_log_handler()
    _forget_submodules()


def _forget_submodules():
    """Drop every submodule once unregistered, so that enabling the add-on
    again -- or Reload Scripts -- runs the code as it is on disk now,
    subpackages included, instead of what was imported the first time."""
    import sys

    prefix = __name__ + "."
    for name in [n for n in sys.modules if n.startswith(prefix)]:
        del sys.modules[name]


def _add_log_handler():
    """Warnings go to the system console, marked as this add-on's."""
    import logging

    log = logging.getLogger(__name__)
    if not any(getattr(h, "armature_nodes", False) for h in log.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        handler.armature_nodes = True
        log.addHandler(handler)
    log.propagate = False


def _remove_log_handler():
    import logging

    log = logging.getLogger(__name__)
    for handler in [h for h in log.handlers if getattr(h, "armature_nodes", False)]:
        log.removeHandler(handler)
