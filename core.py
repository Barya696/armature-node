"""Core data model for the Armature Nodes addon.

The node graph evaluates into these plain-Python intermediate structures,
which the Build operator then turns into real Blender data in two passes
(edit-mode bones, then pose-mode constraints).
"""

from dataclasses import dataclass, field
from typing import Optional

TREE_IDNAME = "ArmatureNodeTreeType"


@dataclass
class ConstraintDef:
    """A single pose-bone constraint to apply after the armature exists."""

    type: str  # Blender constraint type enum, e.g. 'IK', 'COPY_ROTATION'
    name: str = ""
    # params maps constraint attribute name -> value.
    # 'target' / 'pole_target' are stored as object NAMES and resolved
    # against bpy.data.objects at apply time.
    params: dict = field(default_factory=dict)


@dataclass
class ShapeDef:
    """Custom shape (control widget) assignment for a pose bone.

    Widgets follow the Rigify convention: mesh objects named ``WGT-rig_<x>``
    living in a ``WGTS_rig`` collection that is excluded from the view layer.
    ``widget`` is the widget OBJECT NAME; if it does not exist at build time
    and ``preset`` is set, the widget is generated into WGTS_rig.
    """

    widget: str = ""  # object name in bpy.data.objects
    preset: str = "NONE"  # generator preset when the widget is missing
    scale: tuple = (1.0, 1.0, 1.0)
    translation: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = (0.0, 0.0, 0.0)  # euler, radians
    wire_width: float = 1.0
    scale_to_bone_length: bool = True
    show_wire: bool = True
    # Captured widget mesh {'verts': [...], 'edges': [...]}. When the widget
    # object is missing at build time it is recreated from this, so a graph
    # decompiled from a rig can rebuild the rig's exact widgets on its own.
    geometry: Optional[dict] = None


@dataclass
class BoneDef:
    """A single bone definition, resolved from the node graph."""

    name: str
    head: tuple = (0.0, 0.0, 0.0)
    tail: tuple = (0.0, 0.0, 1.0)
    roll: float = 0.0
    parent: Optional[str] = None  # parent bone name, resolved at build time
    use_connect: bool = False
    use_deform: bool = True
    envelope_distance: float = 0.25
    envelope_weight: float = 1.0
    constraints: list = field(default_factory=list)  # list[ConstraintDef]
    shape: Optional[ShapeDef] = None  # control widget, applied in pose pass
    # Pose overrides written by the Transform-category nodes. ``None`` means
    # "leave whatever the rig has"; a tuple is a WORLD-space target applied in
    # the pose pass, after the bones exist. Rest geometry (head/tail/roll) is
    # never touched by those nodes -- that is what makes them non-destructive
    # modifiers on top of an existing rig rather than an edit of it.
    pose_location: Optional[tuple] = None
    pose_rotation: Optional[tuple] = None  # euler XYZ, radians
    pose_scale: Optional[tuple] = None
    # Deltas from Transform nodes, added on top of whatever pose the bone has
    # when no absolute target was set. They accumulate, so several Transform
    # nodes in a row stack instead of overwriting each other.
    pose_offset: tuple = (0.0, 0.0, 0.0)
    pose_rotation_offset: tuple = (0.0, 0.0, 0.0)
    # The same, along the bone's own axes -- its Location / Rotation channels.
    pose_local_offset: tuple = (0.0, 0.0, 0.0)
    pose_local_rotation: tuple = (0.0, 0.0, 0.0)
    # Which local channels a node SET outright ("location", "rotation"),
    # rather than added to. A zero offset normally means "no request" -- but
    # a channel set to zero means "at rest", and has to reach the rig.
    pose_local_set: tuple = ()
    # Straight from the rig's record and untouched. Bones are shared down the
    # stream until a node changes one (``editable``), and the build keeps the
    # record's own bone for every bone still pristine at the end.
    pristine: bool = field(default=False, compare=False, repr=False)


def editable(bones, chosen):
    """Copies of ``chosen``, put in their place in ``bones``, to change.

    Copy on write. A node hands the rest of the stream on as it came instead
    of copying the whole rig -- 700 bones per node on a Rigify rig, which was
    most of every build -- and copies only the bones it is about to change.
    """
    wanted = {id(b) for b in chosen}
    copies = {}
    for index, bone in enumerate(bones):
        key = id(bone)
        if key in wanted:
            if key not in copies:
                copies[key] = copy_bone(bone)
            bones[index] = copies[key]
    return [copies[id(b)] for b in chosen if id(b) in copies]


def unique_names(bones):
    """Ensure every BoneDef in the list has a unique name, renaming later
    duplicates with .001-style suffixes. Parent references are left as they
    are: they name the first bone of that name."""
    seen = {}
    for index, b in enumerate(bones):
        base = b.name or "Bone"
        if base not in seen:
            seen[base] = 0
            continue
        seen[base] += 1
        new_name = f"{base}.{seen[base]:03d}"
        while new_name in seen:
            seen[base] += 1
            new_name = f"{base}.{seen[base]:03d}"
        # A copy is renamed: the bone itself may be shared up the stream.
        bones[index] = renamed = copy_bone(b)
        renamed.name = new_name
        seen[new_name] = 0
    return bones


# ---------------------------------------------------------------------------
# Node groups: evaluation as a function call
# ---------------------------------------------------------------------------
#
# A group node runs its group's tree the way a function call runs a function:
# what is wired into the group node arrives inside at the Group Input node,
# and what reaches the Group Output node comes back out of the group node.
# While a group is running it sits on this stack. A Group Input node reads
# from the group node on top of it -- and does so *outside* it, in the frame
# the group node itself lives in, which is what makes groups inside groups
# work. One group tree used by several group nodes is evaluated once per
# group node, each with its own inputs.

GROUP_INPUT = "NodeGroupInput"
GROUP_OUTPUT = "NodeGroupOutput"
REROUTE = "NodeReroute"
# Deeper than this is a group that contains itself through another group.
MAX_GROUP_DEPTH = 32

_frames = []


class entering:
    """``with entering(group_node):`` -- evaluate inside that group node."""

    def __init__(self, group_node):
        self.group_node = group_node

    def __enter__(self):
        tree = self.group_node.node_tree
        if len(_frames) >= MAX_GROUP_DEPTH or any(g.node_tree == tree for g in _frames):
            raise RuntimeError(
                f"Node group '{tree.name if tree else '?'}' contains itself"
            )
        _frames.append(self.group_node)
        return self.group_node

    def __exit__(self, *exc):
        _frames.pop()
        return False


class outside:
    """``with outside():`` -- step out of the current group, into the frame
    of the group node that runs it. Where a Group Input node reads from."""

    def __enter__(self):
        self.group_node = _frames.pop() if _frames else None
        return self.group_node

    def __exit__(self, *exc):
        if self.group_node is not None:
            _frames.append(self.group_node)
        return False


def _frame_key():
    return tuple((g.id_data.name, g.name) for g in _frames)


def socket_by_identifier(sockets, identifier):
    """A node socket by its identifier -- names are labels and may repeat."""
    for sock in sockets:
        if sock.identifier == identifier:
            return sock
    return None


def group_output_node(tree):
    """The group's active Group Output node, or None."""
    if tree is None:
        return None
    outputs = [n for n in tree.nodes if n.bl_idname == GROUP_OUTPUT]
    for node in outputs:
        if getattr(node, "is_active_output", False):
            return node
    return outputs[0] if outputs else None


# tree pointer -> (graph version, {socket pointer: [link, ...]}).
_link_index = {}


def socket_links(sock):
    """The links on ``sock``, as ``sock.links`` gives them.

    ``NodeSocket.links`` walks every link of the tree on each call, and the
    evaluator and the live link ask it for socket after socket. This indexes
    a tree's links once, until the graph changes (``tree.graph_version``).
    """
    from .tree import graph_version

    tree = sock.id_data
    key, version = tree.as_pointer(), graph_version()
    entry = _link_index.get(key)
    if entry is None or entry[0] != version:
        index, into = {}, {}
        for link in tree.links:
            index.setdefault(link.from_socket.as_pointer(), []).append(link)
            into.setdefault(link.to_socket.as_pointer(), []).append(link)
        for links in into.values():
            # Several wires into one input: in their order on the socket,
            # the order ``NodeSocket.links`` uses.
            links.sort(key=lambda link: link.multi_input_sort_id, reverse=True)
        index.update(into)
        if len(_link_index) > 64:
            _link_index.clear()
        entry = _link_index[key] = (version, index)
    return entry[1].get(sock.as_pointer(), ())


def forget_link_index():
    """A file load or an undo freed every link the index holds."""
    _link_index.clear()


def _live_links(sock):
    return [l for l in socket_links(sock) if l.is_valid and not l.is_muted]


class EvalContext:
    """Per-build memoization so shared upstream nodes evaluate once.

    Keyed by where a node sits -- which group node, in which tree -- as well
    as by the node: a group's nodes run once for every group node using it,
    and two trees are free to use the same node names.

    **Several rigs into one input** (the Output's) are chained, in the order
    the wires come in: each branch runs again on top of what the wires before
    it made, from the point where it split off from them. So two Transform
    nodes side by side end up exactly as if one were plugged into the other,
    and a node the branches share still runs once. Each branch is a copy of
    the whole rig, so putting them side by side instead -- which is what this
    used to do -- kept one copy of every bone and threw the rest away, and
    with them every other branch's changes.

    That needs to know where each rig came from: every rig value is named by
    the node output it came out of (``_key``), and ``_made_from`` records the
    rigs each one was made from.
    """

    def __init__(self):
        self._bone_cache = {}
        self._constraint_cache = {}
        self._visiting = set()
        self._made_from = {}  # key of a rig -> keys of the rigs it was made from
        self._making = []  # keys of the rigs being made, innermost last
        self._order = {}  # key of a rig -> when it was first finished
        # A branch run again on top of other branches is a replay: a scope
        # of its own, where the rig it split off from is stood in for.
        self._scope = ()
        self._stand_ins = {}  # scope -> (key, rig standing in for it)
        self._replays = 0
        self._reads = 0

    def _key(self, node, identifier=""):
        return (_frame_key(), node.id_data.name, node.name, identifier)

    def _rig(self, key, make):
        """The rig ``key`` names: made once per scope, where it came from noted."""
        if self._making:
            self._made_from.setdefault(self._making[-1], set()).add(key)
        scope = self._scope
        for depth in range(len(scope), 0, -1):  # a replay inside a replay: both apply
            stand_in = self._stand_ins.get(scope[:depth])
            if stand_in is not None and stand_in[0] == key:
                return stand_in[1]
        cache_key = (scope, key)
        if cache_key in self._visiting:
            raise RuntimeError(f"Cycle detected at node '{key[2]}'")
        if cache_key in self._bone_cache:
            return self._bone_cache[cache_key]
        self._visiting.add(cache_key)
        self._making.append(key)
        try:
            result = make()
        finally:
            self._making.pop()
            self._visiting.discard(cache_key)
        self._bone_cache[cache_key] = result
        self._order.setdefault(key, len(self._order))
        return result

    def bones_from_node(self, node):
        return self._rig(
            self._key(node),
            lambda: node.eval_bones(self) if hasattr(node, "eval_bones") else [],
        )

    def constraints_from_node(self, node):
        key = self._key(node)
        if key in self._constraint_cache:
            return self._constraint_cache[key]
        result = (
            node.eval_constraints(self) if hasattr(node, "eval_constraints") else []
        )
        self._constraint_cache[key] = result
        return result

    def _from_socket(self, sock, kind):
        node = sock.node
        if node.bl_idname == REROUTE:
            return self._gather(node.inputs[0], kind)
        if node.bl_idname == GROUP_INPUT:
            # The value wired into the running group node, read where that
            # group node lives.
            with outside() as group_node:
                if group_node is None:
                    return []  # a group tree opened on its own: no caller
                outer = socket_by_identifier(group_node.inputs, sock.identifier)
                return self._gather(outer, kind) if outer is not None else []
        if getattr(node, "is_armature_group", False):
            return self._from_group(node, sock.identifier, kind)
        if kind == "bones":
            return self.bones_from_node(node)
        return self.constraints_from_node(node)

    def _from_group(self, group_node, identifier, kind):
        """Run the group: what reaches its Group Output, for this output."""
        key = self._key(group_node, identifier)
        output = group_output_node(group_node.node_tree)
        inner = socket_by_identifier(output.inputs, identifier) if output else None

        def run():
            if inner is None:
                return []
            with entering(group_node):
                return self._gather(inner, kind)

        if kind == "bones":
            return self._rig(key, run)
        if key not in self._constraint_cache:
            self._constraint_cache[key] = run()
        return self._constraint_cache[key]

    def _gather(self, sock, kind):
        links = _live_links(sock)
        if kind == "bones" and len(links) > 1:
            return self._chain(sock, links)
        out = []
        for link in links:
            out.extend(self._from_socket(link.from_socket, kind))
        return out

    # -- Several rigs into one input -------------------------------------------

    def _chain(self, sock, links):
        """The rigs on ``links``, chained in order (see the class docstring)."""
        node = sock.node
        key = ("chain", _frame_key(), node.id_data.name, node.name, sock.identifier)

        def make():
            merged, came_from = None, set()
            for link in links:
                rig, keys = self._read(link.from_socket)
                ancestry = self._ancestry(keys)
                if merged is None:
                    merged, came_from = list(rig), ancestry
                    continue
                shared = came_from & ancestry
                if shared:
                    split = max(shared, key=lambda k: self._order.get(k, -1))
                    merged = list(self._replay(link.from_socket, split, merged))
                else:
                    # Nothing in common -- two rigs built from scratch, say:
                    # side by side, as before.
                    merged = merged + list(rig)
                came_from |= ancestry
            return merged or []

        return self._rig(key, make)

    def _read(self, sock):
        """(the rig out of ``sock``, the keys of the rigs it is)."""
        self._reads += 1
        probe = ("read", self._reads)
        self._making.append(probe)
        try:
            rig = self._from_socket(sock, "bones")
        finally:
            self._making.pop()
        keys = self._made_from.pop(probe, set())
        if self._making:
            self._made_from.setdefault(self._making[-1], set()).update(keys)
        return rig, keys

    def _ancestry(self, keys):
        """``keys`` and every rig they were made from, all the way up."""
        out, todo = set(), list(keys)
        while todo:
            key = todo.pop()
            if key not in out:
                out.add(key)
                todo.extend(self._made_from.get(key, ()))
        return out

    def _replay(self, sock, split, base):
        """The rig out of ``sock`` made again with ``base`` in place of the
        rig ``split`` -- the branch's own nodes, run on top of ``base``."""
        self._replays += 1
        scope = self._scope + (self._replays,)
        self._stand_ins[scope] = (split, base)
        outer, self._scope = self._scope, scope
        try:
            return self._from_socket(sock, "bones")
        finally:
            self._scope = outer


def gather_input_bones(node, socket_name, ctx):
    """Collect BoneDefs from every link into the named input socket."""
    sock = node.inputs.get(socket_name)
    if sock is None:
        return []
    return ctx._gather(sock, "bones")


def gather_input_constraints(node, socket_name, ctx):
    """Collect ConstraintDefs from every link into the named input socket."""
    sock = node.inputs.get(socket_name)
    if sock is None:
        return []
    return ctx._gather(sock, "constraints")


def bone_roll(bone):
    """Roll of a (non-edit) Bone, recovered from its rest matrix.

    ``Bone`` has no ``roll``; only ``EditBone`` does. Reading it back out of
    the rest matrix is what lets the Armature Input describe a rig without
    dropping the whole armature into Edit mode first.
    """
    import bpy

    try:
        mat = bone.matrix_local.to_3x3()
        return float(bpy.types.Bone.AxisRollFromMatrix(mat)[1])
    except Exception:  # noqa: BLE001
        return 0.0


def select_bones(bones, pattern):
    """The bones a modifier node acts on.

    ``pattern`` is the node's Bone field: empty means every bone on the wire
    (the Geometry Nodes convention -- no selection is the whole stream), and
    several bones can be named at once, semicolon separated.
    """
    names = {n.strip() for n in pattern.split(";") if n.strip()}
    if not names:
        return list(bones)
    return [b for b in bones if b.name in names]


def copy_bone(b):
    return BoneDef(
        name=b.name,
        head=tuple(b.head),
        tail=tuple(b.tail),
        roll=b.roll,
        parent=b.parent,
        use_connect=b.use_connect,
        use_deform=b.use_deform,
        envelope_distance=b.envelope_distance,
        envelope_weight=b.envelope_weight,
        pose_location=b.pose_location,
        pose_rotation=b.pose_rotation,
        pose_scale=b.pose_scale,
        pose_offset=tuple(b.pose_offset),
        pose_rotation_offset=tuple(b.pose_rotation_offset),
        pose_local_offset=tuple(b.pose_local_offset),
        pose_local_rotation=tuple(b.pose_local_rotation),
        pose_local_set=tuple(b.pose_local_set),
        constraints=[
            ConstraintDef(type=c.type, name=c.name, params=dict(c.params))
            for c in b.constraints
        ],
        shape=(
            ShapeDef(
                widget=b.shape.widget,
                preset=b.shape.preset,
                scale=tuple(b.shape.scale),
                translation=tuple(b.shape.translation),
                rotation=tuple(b.shape.rotation),
                wire_width=b.shape.wire_width,
                scale_to_bone_length=b.shape.scale_to_bone_length,
                show_wire=b.shape.show_wire,
                geometry=b.shape.geometry,
            )
            if b.shape is not None
            else None
        ),
    )
