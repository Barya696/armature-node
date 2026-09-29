"""Rigify: the rig's own switches, set from the graph.

Rigify keeps its IK/FK, pole, parent and follow settings as custom properties
on a few bones -- a limb's on its ``*_parent`` bone (``upper_arm_parent.L``,
``thigh_parent.L``), the spine's on ``torso``. The Rigify Switch node sets
them the way the other nodes set a pose: over the rig's record, so deleting
the node puts back what the rig had when it was bound.

Any rig's number and boolean custom properties are switches to this node;
only the labels know about Rigify.
"""

import json

from bpy.types import Node, Operator, PropertyGroup
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)

from ..core import editable
from .base import ModifierNodeBase, bone_select_prop, bound_rig

#: Rigify property names that do not read well as words.
_LABELS = {"IK_FK": "IK / FK", "pole_vector": "Pole"}

#: Rigify's own panel order for the switches it has; any other comes after.
_ORDER = (
    "IK_FK", "pole_vector", "IK_Stretch", "IK_parent", "pole_parent", "FK_limb_follow",
    "neck_follow", "head_follow", "torso_parent",
)


def switch_label(name):
    if name in _LABELS:
        return _LABELS[name]
    return " ".join(w if w.isupper() else w.capitalize() for w in name.split("_"))


def _like(recorded, value):
    """``value`` as the type the rig stores the switch as."""
    if isinstance(recorded, bool):
        return bool(value)
    if isinstance(recorded, int):
        return int(round(value))
    return float(value)


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= 1e-6
    return a == b


# ---------------------------------------------------------------------------
# One switch on the node
# ---------------------------------------------------------------------------


def _rebuild(self, context):
    tree = self.id_data
    if tree is not None and hasattr(tree, "mark_dirty"):
        tree.mark_dirty()


def _on_use_changed(self, context):
    """Ticked: start from what the rig has now, so nothing jumps."""
    if self.use:
        live = self.live_value()
        if live is not None:
            self.put(live)
    _rebuild(self, context)


# A parent switch's choices, by its options text. Blender keeps only a pointer
# to the strings an enum callback returns, so they have to outlive the call.
_choices = {}


def _choice_items(self, context):
    items = _choices.get(self.options)
    if items is None:
        pairs = json.loads(self.options) if self.options else []
        items = _choices[self.options] = [(str(v), label, "", v) for v, label in pairs]
    return items


class RigifySwitch(PropertyGroup):
    """One switch on a Rigify Switch node: which property, and what to set."""

    # ``name`` is the custom property's name.
    kind: EnumProperty(
        items=(("FLOAT", "Number", ""), ("INT", "Whole Number", ""), ("BOOL", "Toggle", "")),
        options={"HIDDEN"},
    )
    use: BoolProperty(
        name="Set",
        description="Set this switch. Off, the rig keeps its own value",
        update=_on_use_changed,
    )
    value: FloatProperty(name="Value", soft_min=0.0, soft_max=1.0, update=_rebuild)
    number: IntProperty(name="Value", update=_rebuild)
    flag: BoolProperty(name="Value", update=_rebuild)
    choice: EnumProperty(
        name="Value",
        items=_choice_items,
        get=lambda self: self.number,
        set=lambda self, value: setattr(self, "number", value),
    )
    #: A parent switch's choices as Rigify names them, JSON [[value, label]].
    options: StringProperty(options={"HIDDEN"})
    #: The chosen bones that have this switch, ";"-separated.
    targets: StringProperty(options={"HIDDEN"})

    def current(self):
        return {"FLOAT": self.value, "INT": self.number, "BOOL": self.flag}[self.kind]

    def put(self, value):
        if self.kind == "BOOL":
            self.flag = bool(value)
        elif self.kind == "INT":
            self.number = int(value)
        else:
            self.value = float(value)

    def pose_bones(self):
        obj = bound_rig(self.id_data)
        if obj is None or obj.pose is None:
            return []
        found = (obj.pose.bones.get(n) for n in self.targets.split(";") if n)
        return [pb for pb in found if pb is not None]

    def live_value(self):
        """What the rig has now, on the first bone with the switch."""
        for pb in self.pose_bones():
            value = pb.get(self.name)
            if value is not None:
                return value
        return None

    def readout(self):
        value = self.live_value()
        if value is None:
            return "-"
        if self.kind == "BOOL":
            return "On" if value else "Off"
        if self.kind == "INT":
            labels = {v: label for v, label in json.loads(self.options or "[]")}
            return labels.get(value, str(value))
        return f"{value:.2f}"


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------


def _on_bone_changed(self, context):
    self.sync_switches()
    self.schedule_rebuild()


# (id(record), Bone field) -> (record, what those bones have to switch).
_recorded = {}


def _build_due(tree, _seen=None):
    """A build of ``tree`` is waiting -- or, for a group, of a tree using it:
    a group is never dirty itself, the trees running it are."""
    from ..groups import group_users

    seen = set() if _seen is None else _seen
    if tree is None or tree.name in seen:
        return False
    seen.add(tree.name)
    return bool(getattr(tree, "is_dirty", False)) or any(_build_due(u, seen) for u in group_users(tree))


class RigifySwitchNode(ModifierNodeBase, Node):
    """Rigify's switches -- IK/FK, pole, parents, follow -- set from the graph.

    Pick the bone that holds them, or leave Bone empty for every bone that
    has the switch: one node turns the poles on for all four limbs. A ticked
    switch is set; an unticked one shows what the rig has. Deleting the node
    puts back the values the rig was bound with.
    """

    bl_idname = "ArmatureNodesRigifySwitchNode"
    bl_label = "Rigify Switch"
    bl_icon = "PROPERTIES"

    bone: bone_select_prop(update=_on_bone_changed)
    switches: CollectionProperty(type=RigifySwitch)

    def init(self, context):
        super().init(context)
        self.width = 240
        self.sync_switches()

    def recorded(self):
        """``({switch: [bones]}, {switch: recorded value}, legacy)`` over the
        chosen bones; ``legacy`` when the record predates switches. None when
        there is no bound rig.

        Cached per record and Bone field: it runs on every live pass, and an
        empty field means every bone of a 700-bone rig.
        """
        from ..store import record as record_store

        obj = self.resolve_armature()
        record = record_store.read(obj) if obj is not None else None
        if record is None:
            return None
        key = (id(record), self.bone)
        hit = _recorded.get(key)
        if hit is None or hit[0] is not record:
            if len(_recorded) > 32:
                _recorded.clear()
            hit = _recorded[key] = (record, self._scan(record))
        return hit[1]

    def _scan(self, record):
        names = {n.strip() for n in self.bone.split(";") if n.strip()}
        holders, values, legacy = {}, {}, False
        for name, bone in record.bones.items():
            if names and name not in names:
                continue
            if bone.pose.props is None:
                legacy = True
                continue
            for key, value in bone.pose.props.items():
                holders.setdefault(key, []).append(name)
                values.setdefault(key, value)
        return holders, values, legacy

    def sync_switches(self):
        """One item per switch the chosen bones have, in Rigify's panel order,
        keeping what was set.

        Writes only on a real change: this runs on every live pass.
        """
        from ..tree import suspend_live_update

        found = self.recorded()
        if found is None:
            return
        holders, values, _legacy = found
        for index in reversed(range(len(self.switches))):
            if self.switches[index].name not in holders:
                self.switches.remove(index)
        order = sorted(holders, key=lambda key: _ORDER.index(key) if key in _ORDER else len(_ORDER))
        with suspend_live_update():
            for index, key in enumerate(order):
                targets = ";".join(holders[key])
                position = self.switches.find(key)
                if position < 0:
                    item = self.switches.add()
                    item.name = key
                    value = values[key]
                    item.kind = "BOOL" if isinstance(value, bool) else "INT" if isinstance(value, int) else "FLOAT"
                    item.targets = targets
                    item.options = self._options(item)
                    item.put(value)
                    position = len(self.switches) - 1
                elif self.switches[position].targets != targets:
                    item = self.switches[position]
                    item.targets = targets
                    item.options = self._options(item)  # the choices are the new bone's
                if position != index:
                    self.switches.move(position, index)

    @staticmethod
    def _options(item):
        """A parent switch's choices, from the UI data Rigify gives it."""
        for pb in item.pose_bones():
            if pb.get(item.name) is None:
                continue
            try:
                items = pb.id_properties_ui(item.name).as_dict().get("items")
            except TypeError:
                items = None
            return json.dumps([[entry[-1], entry[1]] for entry in items]) if items else ""
        return ""

    # -- Live, the other way -----------------------------------------------------

    def follow_live(self):
        """A switch changed in Rigify's own panel is taken into the node, or
        the next build would throw the change away."""
        from ..tree import suspend_live_update

        self.sync_switches()
        if _build_due(self.id_data):
            return False  # the rig has not caught up with the node yet
        changed = False
        for item in self.switches:
            if not item.use:
                continue
            live = {pb[item.name] for pb in item.pose_bones() if pb.get(item.name) is not None}
            if len(live) != 1:
                continue  # the bones disagree: there is no one value to take
            value = live.pop()
            if not _same(value, item.current()):
                with suspend_live_update():
                    item.put(value)
                changed = True
        return changed

    # -- UI and evaluation -------------------------------------------------------

    def draw_buttons(self, context, layout):
        layout.context_pointer_set("node", self)
        obj = self.rig_for_ui()
        row = layout.row(align=True)
        if obj is not None:
            row.prop_search(self, "bone", obj.data, "bones", text="", icon="BONE_DATA")
        else:
            row.prop(self, "bone", text="", icon="BONE_DATA")
        row.operator_menu_enum("armature_nodes.rigify_switch_bone", "bone", text="", icon="DOWNARROW_HLT")
        if not self.bone:
            layout.label(text="Every bone that has the switch", icon="INFO")

        if not self.switches:
            # Only then is the record worth reading: to say why.
            found = self.recorded()
            if found is None:
                layout.label(text="The rig is not bound", icon="ERROR")
            elif found[2]:
                col = layout.column(align=True)
                col.label(text="Bound before switches were recorded", icon="ERROR")
                col.operator("armature_nodes.record_switches", icon="FILE_TICK").rig = obj.name
            else:
                layout.label(text="No switches on this bone", icon="INFO")
            return
        for item in self.switches:
            row = layout.row(align=True)
            row.prop(item, "use", text="")
            label = switch_label(item.name)
            if not item.use:
                sub = row.row(align=True)
                sub.enabled = False
                sub.label(text=f"{label}:  {item.readout()}")
            elif item.kind == "BOOL":
                row.prop(item, "flag", text=label, toggle=True)
            elif item.kind == "INT" and item.options:
                row.label(text=label)
                row.prop(item, "choice", text="")
            elif item.kind == "INT":
                row.prop(item, "number", text=label)
            else:
                row.prop(item, "value", text=label, slider=True)

    def eval_bones(self, ctx):
        bones = self.stream(ctx)
        wanted = {item.name: item.current() for item in self.switches if item.use}
        if not wanted:
            return bones
        chosen = [b for b in self.selected(bones) if b.props and not wanted.keys().isdisjoint(b.props)]
        for b in editable(bones, chosen):
            b.props = {
                key: _like(value, wanted[key]) if key in wanted else value
                for key, value in b.props.items()
            }
        return bones


# ---------------------------------------------------------------------------
# Picking the bone
# ---------------------------------------------------------------------------

_bone_items = []


def _switch_bone_items(self, context):
    """The bones that have switches, busiest first: a limb's ``*_parent``
    bones, then ``torso``, then the one-switch bones."""
    from ..store import record as record_store

    node = getattr(context, "node", None)
    obj = node.resolve_armature() if node is not None else None
    record = record_store.read(obj) if obj is not None else None
    holders = [(n, b.pose.props) for n, b in (record.bones.items() if record else ()) if b.pose.props]
    holders.sort(key=lambda h: (-len(h[1]), h[0]))
    _bone_items[:] = [("*", "All Bones", "Every bone that has the switch")] + [
        (name, name, ", ".join(switch_label(k) for k in props)) for name, props in holders
    ]
    return _bone_items


class ARMATURE_NODES_OT_rigify_switch_bone(Operator):
    """Pick a bone that has switches"""

    bl_idname = "armature_nodes.rigify_switch_bone"
    bl_label = "Bone with Switches"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    bone: EnumProperty(name="Bone", items=_switch_bone_items)

    def execute(self, context):
        node = getattr(context, "node", None)
        if node is None:
            return {"CANCELLED"}
        node.bone = "" if self.bone == "*" else self.bone
        return {"FINISHED"}
