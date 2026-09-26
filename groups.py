"""Node groups, the way Blender does them.

A node group is a node tree of its own -- a data-block in
``bpy.data.node_groups``, of this same tree type -- used from other trees
through a **group node**:

* The group's **interface** is its list of inputs and outputs.
* Inside the group, the **Group Input** node puts the inputs on wires and the
  **Group Output** node takes the outputs. Drag a wire into their empty
  socket to add an input or output.
* Outside, a **group node** points at the group and shows its interface as
  sockets. Every group node using a group shares it: edit the group once and
  every one of them changes.
* Evaluating a group node is a function call: what is wired into the group
  node arrives at Group Input, and what reaches Group Output comes back out
  (``core.EvalContext``).

Blender provides the pieces that are data -- the Group Input and Group Output
nodes, the interface, the group node's ``node_tree`` pointer -- to custom
trees like this one. It keeps the editing operators (Ctrl+G, Tab,
Ctrl+Alt+G) to its own tree types and leaves custom trees to define their own
on the same keys, which is what this module does. It also keeps a custom group
node's sockets in step with its group's interface, which Blender does not do
for custom trees.

Marker and Skeleton nodes go into groups like any other node. A marker
inside a group is shown -- and its handle grabbable -- when the group's output
reaches a rig's Armature Output through the group node using it. Like any
value in a Blender node group it is shared by every group node running the
group. When exactly one rig uses the group, the marker is live with its bone
as it is outside (``unique_rig``); a group used by several group nodes cannot
say which bone to follow, so there it drives but does not follow.
"""

import bpy
from bpy.props import BoolProperty
from bpy.types import NodeCustomGroup, Operator
from mathutils import Vector

from .core import GROUP_INPUT, GROUP_OUTPUT, TREE_IDNAME, socket_by_identifier

GROUP_NODE = "ArmatureNodesGroupNode"
# Nodes that stay in the main tree when a group is made from a selection:
# the rig's own ends, and a group's.
_NOT_GROUPED = {
    "ArmatureNodesInputNode",
    "ArmatureNodesOutputNode",
    GROUP_INPUT,
    GROUP_OUTPUT,
}
# Room between the grouped nodes and the Group Input / Output around them.
_MARGIN = 250.0


# ---------------------------------------------------------------------------
# What is a group, and who uses it
# ---------------------------------------------------------------------------


def armature_trees():
    return [t for t in bpy.data.node_groups if t.bl_idname == TREE_IDNAME]


def is_group_tree(tree):
    """A tree meant to be used as a group: it has a Group Input or Output."""
    return tree is not None and any(
        n.bl_idname in (GROUP_INPUT, GROUP_OUTPUT) for n in tree.nodes
    )


def group_nodes_using(group):
    """Every group node, in any tree, that runs ``group``."""
    out = []
    for tree in armature_trees():
        for node in tree.nodes:
            if node.bl_idname == GROUP_NODE and node.node_tree == group:
                out.append(node)
    return out


def group_users(group):
    """The trees that contain a group node running ``group``."""
    seen, out = set(), []
    for node in group_nodes_using(group):
        tree = node.id_data
        if tree.name not in seen:
            seen.add(tree.name)
            out.append(tree)
    return out


def uses_tree(tree, group, _seen=None):
    """True when ``tree`` runs ``group``, directly or through other groups."""
    if tree is None or group is None:
        return False
    if tree == group:
        return True
    seen = _seen if _seen is not None else set()
    if tree.name in seen:
        return False
    seen.add(tree.name)
    for node in tree.nodes:
        if node.bl_idname == GROUP_NODE and node.node_tree is not None:
            if uses_tree(node.node_tree, group, seen):
                return True
    return False


def group_trees():
    return [t for t in armature_trees() if is_group_tree(t)]


def trees_in_use(tree):
    """``tree`` and every group it runs, through groups inside groups --
    each once. What a rig's live link and handles have to look through."""
    out, seen, todo = [], set(), [tree]
    while todo:
        current = todo.pop(0)
        if current is None or current.name in seen:
            continue
        seen.add(current.name)
        out.append(current)
        for node in current.nodes:
            if node.bl_idname == GROUP_NODE and node.node_tree is not None:
                todo.append(node.node_tree)
    return out


def unique_rig(tree, _seen=None):
    """The armature ``tree`` -- a group -- is used on, when that is certain.

    Certain means one group node runs the group, in a rig's own tree or in a
    group that is itself certain. Used by several group nodes, or by none, a
    group has no one rig: None. Blender lets two groups use each other, so
    the walk up stops at a tree it has seen.
    """
    seen = set() if _seen is None else _seen
    if tree is None or tree.name in seen:
        return None
    seen.add(tree.name)
    users = group_nodes_using(tree)
    if len(users) != 1:
        return None
    parent = users[0].id_data
    for node in parent.nodes:
        if node.bl_idname == "ArmatureNodesInputNode":
            src = getattr(node, "source", None)
            if src is not None and getattr(src, "type", "") == "ARMATURE":
                return src
    return unique_rig(parent, seen)


# ---------------------------------------------------------------------------
# The group node
# ---------------------------------------------------------------------------


def sync_group_sockets(node):
    """Make a group node's sockets match its group's interface.

    Matched by identifier, which survives renaming: a socket is renamed in
    place and keeps its wires. One whose type changed is replaced. Returns
    True when anything changed.
    """
    group = node.node_tree
    items = []
    if group is not None:
        items = [i for i in group.interface.items_tree if i.item_type == "SOCKET"]
    changed = False
    for sockets, in_out in ((node.inputs, "INPUT"), (node.outputs, "OUTPUT")):
        wanted = [(i.identifier, i.name, i.socket_type) for i in items if i.in_out == in_out]
        types = {ident: kind for ident, _name, kind in wanted}
        for sock in list(sockets):
            if types.get(sock.identifier) != sock.bl_idname:
                sockets.remove(sock)
                changed = True
        for index, (ident, name, kind) in enumerate(wanted):
            sock = socket_by_identifier(sockets, ident)
            if sock is None:
                sock = sockets.new(kind, name, identifier=ident)
                changed = True
            elif sock.name != name:
                sock.name = name
                changed = True
            current = list(sockets).index(sock)
            if current != index:
                sockets.move(current, index)
                changed = True
    return changed


def sync_group_users(group):
    """The interface of ``group`` may have changed: bring its group nodes in line."""
    for node in group_nodes_using(group):
        sync_group_sockets(node)


def sync_all_group_nodes():
    """Backstop for edits nothing reported. Cheap when nothing changed."""
    for tree in armature_trees():
        for node in tree.nodes:
            if node.bl_idname == GROUP_NODE:
                sync_group_sockets(node)


class ArmatureNodesGroupNode(NodeCustomGroup):
    """A node group, used here: its inputs and outputs are the group's.

    Tab goes inside to edit the group, and every group node running the same
    group changes with it.
    """

    bl_idname = GROUP_NODE
    bl_label = "Group"
    bl_icon = "NODETREE"

    #: Recognised by the evaluator (``core``) and the value reader (``sockets``).
    is_armature_group = True

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == TREE_IDNAME

    def init(self, context):
        self.width = 180

    def draw_label(self):
        return self.node_tree.name if self.node_tree is not None else "Group"

    def draw_buttons(self, context, layout):
        layout.context_pointer_set("node", self)
        layout.template_ID(self, "node_tree", new="armature_nodes.group_new")

    def update(self):
        """Called when links change, and when the group is swapped for another."""
        sync_group_sockets(self)

    def free(self):
        tree = self.id_data
        if tree is not None and hasattr(tree, "mark_dirty"):
            tree.mark_dirty()


# ---------------------------------------------------------------------------
# Copying nodes between trees
# ---------------------------------------------------------------------------

# Node properties that are not the node's settings, or are set by the caller.
_SKIP = {
    "rna_type", "name", "location", "location_absolute", "dimensions", "inputs",
    "outputs", "internal_links", "parent", "select", "type", "is_active_output",
    "bl_idname", "bl_label", "bl_description", "bl_icon", "bl_static_type",
    "bl_width_default", "bl_width_min", "bl_width_max", "bl_height_default",
    "bl_height_min", "bl_height_max", "panel_states", "warning_propagation",
}


def _copy_props(src, dst):
    """Copy every writable property of ``src`` onto ``dst``, collections and
    nested groups included."""
    for prop in src.bl_rna.properties:
        ident = prop.identifier
        if ident in _SKIP:
            continue
        try:
            if prop.type == "COLLECTION":
                dst_items = getattr(dst, ident)
                if not hasattr(dst_items, "add"):
                    continue
                dst_items.clear()
                for item in getattr(src, ident):
                    _copy_props(item, dst_items.add())
            elif prop.type == "POINTER" and prop.is_readonly:
                inner_src, inner_dst = getattr(src, ident), getattr(dst, ident)
                if inner_src is not None and inner_dst is not None:
                    _copy_props(inner_src, inner_dst)
            elif not prop.is_readonly:
                setattr(dst, ident, getattr(src, ident))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            # A value the new node will not take (an enum whose items depend
            # on the scene, a pointer that fails its poll): keep its default.
            continue


def _matching(sockets, sock):
    return socket_by_identifier(sockets, sock.identifier) or sockets.get(sock.name)


def copy_node(src, tree):
    """A copy of node ``src`` in ``tree``: settings, socket values, look."""
    dst = tree.nodes.new(src.bl_idname)
    _copy_props(src, dst)
    if hasattr(dst, "sync_marker_sockets"):
        dst.sync_marker_sockets()
    if hasattr(dst, "ensure_inputs"):
        dst.ensure_inputs()
    if dst.bl_idname == GROUP_NODE:
        sync_group_sockets(dst)
    # Socket values last: setting a node's bone reads the bone into them.
    for sockets_src, sockets_dst in ((src.inputs, dst.inputs), (src.outputs, dst.outputs)):
        for sock in sockets_src:
            other = _matching(sockets_dst, sock)
            if other is None:
                continue
            other.hide = sock.hide
            if hasattr(sock, "default_value") and hasattr(other, "default_value"):
                try:
                    other.default_value = sock.default_value
                except (TypeError, ValueError, AttributeError):
                    pass
    dst.width = src.width
    if hasattr(dst, "live_links"):
        # A marker's record of what it was wired to names nodes of the old
        # tree; the copy starts afresh and adopts its new wires as they are.
        dst.live_links = ""
    return dst


def _where(node):
    return node.location_absolute.copy() if hasattr(node, "location_absolute") else node.location.copy()


def _place(node, location):
    if hasattr(node, "location_absolute"):
        node.location_absolute = location
    else:
        node.location = location


def _copy_nodes(nodes, tree, offset):
    """Copy ``nodes`` into ``tree``, shifted by ``offset``, frames kept.
    Returns {original name: copy}."""
    copies = {}
    # Frames first, so the nodes in them can be put back in.
    for node in sorted(nodes, key=lambda n: n.bl_idname != "NodeFrame"):
        copies[node.name] = copy_node(node, tree)
    for node in nodes:
        parent = node.parent
        if parent is not None and parent.name in copies:
            copies[node.name].parent = copies[parent.name]
    for node in nodes:
        _place(copies[node.name], _where(node) + offset)
    return copies


# ---------------------------------------------------------------------------
# Make Group and Ungroup
# ---------------------------------------------------------------------------


def _link_record(link):
    return (
        link.from_node.name,
        link.from_socket.identifier,
        link.to_node.name,
        link.to_socket.identifier,
        link.is_muted,
    )


def _out_socket(node, identifier):
    return socket_by_identifier(node.outputs, identifier)


def _in_socket(node, identifier):
    return socket_by_identifier(node.inputs, identifier)


def new_group_tree(name="NodeGroup", passthrough=True):
    """An empty group: Group Input and Output, with the rig passing through."""
    from .sockets import RigSocket

    group = bpy.data.node_groups.new(name, TREE_IDNAME)
    gin = group.nodes.new(GROUP_INPUT)
    gout = group.nodes.new(GROUP_OUTPUT)
    gin.location, gout.location = (-_MARGIN, 0.0), (_MARGIN, 0.0)
    if passthrough:
        group.interface.new_socket("Rig", in_out="INPUT", socket_type=RigSocket.bl_idname)
        group.interface.new_socket("Rig", in_out="OUTPUT", socket_type=RigSocket.bl_idname)
        group.links.new(gin.outputs[0], gout.inputs[0])
    return group


def groupable(node):
    return node.bl_idname not in _NOT_GROUPED


def make_group(tree, nodes, name="NodeGroup"):
    """Move ``nodes`` of ``tree`` into a new group; return its group node.

    As Blender's Ctrl+G does it: every wire crossing the edge of the selection
    becomes one of the group's inputs or outputs -- one per source, so a
    value feeding several nodes inside comes in once -- and the group node
    takes the nodes' place, wired as they were.
    """
    from .tree import suspend_live_update

    nodes = [n for n in nodes if groupable(n)]
    if not nodes:
        raise ValueError("Nothing to group: Input, Output and marker nodes stay outside")
    chosen = {n.name for n in nodes}
    links = [_link_record(l) for l in tree.links if l.is_valid]
    internal = [l for l in links if l[0] in chosen and l[2] in chosen]
    incoming = [l for l in links if l[0] not in chosen and l[2] in chosen]
    outgoing = [l for l in links if l[0] in chosen and l[2] not in chosen]

    spots = [_where(n) for n in nodes]
    left = min(p.x for p in spots)
    right = max(p.x + n.width for p, n in zip(spots, nodes))
    middle_y = sum(p.y for p in spots) / len(spots)
    center = Vector(((left + right) / 2.0, middle_y))

    with suspend_live_update():
        group = bpy.data.node_groups.new(name, TREE_IDNAME)
        copies = _copy_nodes(nodes, group, Vector((0.0, 0.0)))
        gin = group.nodes.new(GROUP_INPUT)
        gout = group.nodes.new(GROUP_OUTPUT)
        _place(gin, Vector((left - _MARGIN, middle_y)))
        _place(gout, Vector((right + _MARGIN * 0.4, middle_y)))

        for from_node, from_id, to_node, to_id, muted in internal:
            source = _out_socket(copies[from_node], from_id)
            target = _in_socket(copies[to_node], to_id)
            if source is not None and target is not None:
                group.links.new(source, target).is_muted = muted

        # One input per source outside, named after what it feeds inside.
        inputs = {}
        for from_node, from_id, to_node, to_id, _muted in incoming:
            target = _in_socket(copies[to_node], to_id)
            if target is None or target.bl_idname == "NodeSocketVirtual":
                continue
            key = (from_node, from_id)
            if key not in inputs:
                item = group.interface.new_socket(target.name, in_out="INPUT", socket_type=target.bl_idname)
                inputs[key] = item.identifier
            group.links.new(_out_socket(gin, inputs[key]), target)

        # One output per source inside.
        outputs = {}
        for from_node, from_id, _to_node, _to_id, _muted in outgoing:
            key = (from_node, from_id)
            if key in outputs:
                continue
            source = _out_socket(copies[from_node], from_id)
            if source is None:
                continue
            item = group.interface.new_socket(source.name, in_out="OUTPUT", socket_type=source.bl_idname)
            outputs[key] = item.identifier
            group.links.new(source, _in_socket(gout, item.identifier))

        group_node = tree.nodes.new(GROUP_NODE)
        group_node.node_tree = group
        sync_group_sockets(group_node)
        _place(group_node, center)
        if nodes[0].parent is not None and all(n.parent == nodes[0].parent for n in nodes):
            group_node.parent = nodes[0].parent

        for (from_node, from_id), ident in inputs.items():
            source = _out_socket(tree.nodes[from_node], from_id)
            if source is not None:
                tree.links.new(source, _in_socket(group_node, ident))
        for from_node, from_id, to_node, to_id, muted in outgoing:
            ident = outputs.get((from_node, from_id))
            target = _in_socket(tree.nodes[to_node], to_id)
            if ident is not None and target is not None:
                link = tree.links.new(_out_socket(group_node, ident), target)
                link.is_muted = muted

        for node in nodes:
            tree.nodes.remove(node)
        for node in tree.nodes:
            node.select = False
        group_node.select = True
        tree.nodes.active = group_node
    tree.mark_dirty()
    return group_node


def ungroup(tree, group_node):
    """Put a group node's nodes back in ``tree`` in its place; return them.

    The group itself is left alone -- other group nodes may run it.
    """
    from .tree import suspend_live_update

    group = group_node.node_tree
    if group is None:
        raise ValueError("The group node has no group")
    inner = [n for n in group.nodes if n.bl_idname not in (GROUP_INPUT, GROUP_OUTPUT)]
    links = [_link_record(l) for l in group.links if l.is_valid]
    outer_in = {}
    for sock in group_node.inputs:
        outer_in[sock.identifier] = [
            (l.from_node.name, l.from_socket.identifier) for l in sock.links if l.is_valid
        ]
    outer_out = {}
    for sock in group_node.outputs:
        outer_out[sock.identifier] = [
            (l.to_node.name, l.to_socket.identifier, l.is_muted) for l in sock.links if l.is_valid
        ]
    fields = {
        s.identifier: tuple(s.default_value) for s in group_node.inputs if hasattr(s, "default_value")
    }
    group_in = {n.name for n in group.nodes if n.bl_idname == GROUP_INPUT}
    group_out = {n.name for n in group.nodes if n.bl_idname == GROUP_OUTPUT}

    spots = [_where(n) for n in inner] or [_where(group_node)]
    mid = Vector((sum(p.x for p in spots) / len(spots), sum(p.y for p in spots) / len(spots)))
    offset = _where(group_node) - mid

    with suspend_live_update():
        copies = _copy_nodes(inner, tree, offset)

        def sources_of(from_node, from_id):
            """What feeds this inside socket, seen from ``tree``."""
            if from_node in group_in:
                return [(tree.nodes[n].outputs, i) for n, i in outer_in.get(from_id, [])]
            if from_node in copies:
                return [(copies[from_node].outputs, from_id)]
            return []

        def targets_of(to_node, to_id):
            if to_node in group_out:
                return [(tree.nodes[n].inputs, i, m) for n, i, m in outer_out.get(to_id, [])]
            if to_node in copies:
                return [(copies[to_node].inputs, to_id, False)]
            return []

        for from_node, from_id, to_node, to_id, muted in links:
            targets = targets_of(to_node, to_id)
            sources = sources_of(from_node, from_id)
            if from_node in group_in and not sources:
                # Nothing wired into the group node there: its field's value
                # goes into the socket it fed -- as if typed there, so a node
                # that took a wire to mean "set this" (Position, Transform)
                # still sets it.
                value = fields.get(from_id)
                for inputs, ident, _m in targets:
                    sock = socket_by_identifier(inputs, ident)
                    if value is None or not hasattr(sock, "default_value"):
                        continue
                    try:
                        sock.default_value = value
                    except (TypeError, ValueError):
                        continue
                    if hasattr(sock.node, "on_socket_edited"):
                        sock.node.on_socket_edited(sock)
                continue
            for outputs, out_id in sources:
                for inputs, in_id, outer_muted in targets:
                    source = socket_by_identifier(outputs, out_id)
                    target = socket_by_identifier(inputs, in_id)
                    if source is not None and target is not None:
                        link = tree.links.new(source, target)
                        link.is_muted = muted or outer_muted

        tree.nodes.remove(group_node)
        for node in tree.nodes:
            node.select = False
        for node in copies.values():
            node.select = True
    tree.mark_dirty()
    return list(copies.values())


# ---------------------------------------------------------------------------
# Operators: Ctrl+G, Ctrl+Alt+G, Tab
# ---------------------------------------------------------------------------


def _editor(context):
    space = context.space_data
    if space is None or space.type != "NODE_EDITOR" or space.tree_type != TREE_IDNAME:
        return None
    return space if space.edit_tree is not None else None


class ARMATURE_NODES_OT_group_make(Operator):
    """Put the selected nodes into a new node group, wired in where they were"""

    bl_idname = "armature_nodes.group_make"
    bl_label = "Make Group"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        space = _editor(context)
        return space is not None and any(
            n.select and groupable(n) for n in space.edit_tree.nodes
        )

    def execute(self, context):
        space = _editor(context)
        tree = space.edit_tree
        nodes = [n for n in tree.nodes if n.select and groupable(n)]
        left_out = [n.name for n in tree.nodes if n.select and not groupable(n)]
        group_node = make_group(tree, nodes)
        if left_out:
            self.report({"INFO"}, f"Left outside the group: {', '.join(left_out)}")
        # Like Blender: the new group opens, Tab goes back out.
        space.path.append(group_node.node_tree, node=group_node)
        return {"FINISHED"}


class ARMATURE_NODES_OT_group_ungroup(Operator):
    """Put the nodes of the selected groups back in this tree"""

    bl_idname = "armature_nodes.group_ungroup"
    bl_label = "Ungroup"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        space = _editor(context)
        return space is not None and any(
            n.select and n.bl_idname == GROUP_NODE for n in space.edit_tree.nodes
        )

    def execute(self, context):
        tree = _editor(context).edit_tree
        for node in [n for n in tree.nodes if n.select and n.bl_idname == GROUP_NODE]:
            if node.node_tree is not None:
                ungroup(tree, node)
        return {"FINISHED"}


class ARMATURE_NODES_OT_group_edit(Operator):
    """Tab: go into the selected group to edit it, or back out of the one you are in"""

    bl_idname = "armature_nodes.group_edit"
    bl_label = "Edit Group"

    exit: BoolProperty(name="Exit", description="Only go back out", default=False)

    @classmethod
    def poll(cls, context):
        return _editor(context) is not None

    def execute(self, context):
        space = _editor(context)
        node = space.edit_tree.nodes.active
        if not self.exit and node is not None and node.bl_idname == GROUP_NODE:
            if node.node_tree is not None:
                space.path.append(node.node_tree, node=node)
                return {"FINISHED"}
        if len(space.path) > 1:
            space.path.pop()
            return {"FINISHED"}
        return {"CANCELLED"}


class ARMATURE_NODES_OT_group_new(Operator):
    """A new, empty node group for this group node"""

    bl_idname = "armature_nodes.group_new"
    bl_label = "New Group"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        node = getattr(context, "node", None)
        group = new_group_tree()
        if node is not None and node.bl_idname == GROUP_NODE:
            node.node_tree = group
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Menus and keys
# ---------------------------------------------------------------------------


def add_menu_items(context):
    """Shift+A > Group: Group Input and Output inside a group, then every
    group that can be used here -- not one that would end up inside itself."""
    from nodeitems_utils import NodeItem

    space = context.space_data
    edit_tree = getattr(space, "edit_tree", None)
    items = []
    if is_group_tree(edit_tree):
        items += [NodeItem(GROUP_INPUT), NodeItem(GROUP_OUTPUT)]
    for group in sorted(group_trees(), key=lambda t: t.name.lower()):
        if edit_tree is not None and uses_tree(group, edit_tree):
            continue
        items.append(
            NodeItem(
                GROUP_NODE,
                label=group.name,
                settings={"node_tree": f"bpy.data.node_groups[{group.name!r}]"},
            )
        )
    return items


def _draw_group_menu(self, context):
    if _editor(context) is None:
        return
    layout = self.layout
    layout.separator()
    layout.operator(ARMATURE_NODES_OT_group_make.bl_idname, icon="NODETREE")
    layout.operator(ARMATURE_NODES_OT_group_ungroup.bl_idname)
    layout.operator(ARMATURE_NODES_OT_group_edit.bl_idname, text="Edit Group / Exit").exit = False


classes = (
    ArmatureNodesGroupNode,
    ARMATURE_NODES_OT_group_make,
    ARMATURE_NODES_OT_group_ungroup,
    ARMATURE_NODES_OT_group_edit,
    ARMATURE_NODES_OT_group_new,
)

# The same keys as Blender's own group operators, which do not run in custom
# trees: Blender passes the key on to these.
_KEYS = (
    (ARMATURE_NODES_OT_group_make.bl_idname, "G", {"ctrl": True}, {}),
    (ARMATURE_NODES_OT_group_ungroup.bl_idname, "G", {"ctrl": True, "alt": True}, {}),
    (ARMATURE_NODES_OT_group_edit.bl_idname, "TAB", {}, {"exit": False}),
    (ARMATURE_NODES_OT_group_edit.bl_idname, "TAB", {"ctrl": True}, {"exit": True}),
)
_keymap_items = []


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.NODE_MT_node.append(_draw_group_menu)
    bpy.types.NODE_MT_context_menu.append(_draw_group_menu)
    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig is not None:  # None in background mode
        keymap = keyconfig.keymaps.new(name="Node Editor", space_type="NODE_EDITOR")
        for idname, key, mods, props in _KEYS:
            item = keymap.keymap_items.new(idname, key, "PRESS", **mods)
            for prop, value in props.items():
                setattr(item.properties, prop, value)
            _keymap_items.append((keymap, item))


def unregister():
    for keymap, item in _keymap_items:
        try:
            keymap.keymap_items.remove(item)
        except (ReferenceError, RuntimeError):
            pass
    _keymap_items.clear()
    bpy.types.NODE_MT_context_menu.remove(_draw_group_menu)
    bpy.types.NODE_MT_node.remove(_draw_group_menu)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
