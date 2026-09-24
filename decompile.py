"""Reverse direction: an armature becomes a node graph.

In the modifier model there is nothing to reconstruct. The rig itself is the
input, so decompiling produces exactly two nodes -- **Armature Input** wired
straight into **Armature Output** -- the way opening Geometry Nodes on a mesh
gives you Group Input -> Group Output and not a node per vertex.

Everything you want to change about the rig goes *between* those two: a
Position node to move a control, a Custom Shape node to reshape one, a Bone
node to add one. Each selects the bones it affects by name and passes the rest
of the stream through untouched, so the graph stays the size of your edits
instead of the size of the rig.
"""

import bpy

NODE_X_SPACING = 260


def decompile_armature_to_tree(obj, tree, shapes_only=False, full=False):
    """Populate ``tree`` with the two-node stack that targets ``obj``.

    ``shapes_only`` and ``full`` are kept for call-site compatibility and
    only choose the Output node's mode:

    * default / ``shapes_only`` -> **Modify**: shapes and pose are written
      onto the existing rig; its bones, constraints and drivers are left
      alone. This is what you want on a generated rig.
    * ``full`` -> **Full Rig**: the graph owns the armature and rebuilds its
      bones, so nodes can add and remove them.
    """
    from .nodes import ArmatureInputNode, ArmatureOutputNode

    # Deliberately NO capture here. Opening an editor is not a promise that
    # the rig is unmodified -- it usually means the opposite, that the user is
    # about to change it, and a rig whose widgets were already swapped would
    # be recorded as its own original and saved that way for ever.
    #
    # An unbound rig gets a graph that reads nothing; the Input node shows
    # "Not bound" with a Bind button, and binding is the user's decision.
    tree.nodes.clear()

    source = tree.nodes.new(ArmatureInputNode.bl_idname)
    source.source = obj
    source.location = (0, 0)

    output = tree.nodes.new(ArmatureOutputNode.bl_idname)
    output.mode = "FULL" if full else "MODIFY"
    output.location = (NODE_X_SPACING, 0)

    tree.links.new(source.outputs["Rig"], output.inputs["Rig"])

    if hasattr(obj, "armature_nodes_tree"):
        obj.armature_nodes_tree = tree
    return tree
