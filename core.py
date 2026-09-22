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


def unique_names(bones):
    """Ensure every BoneDef in the list has a unique name.

    Renames duplicates with .001-style suffixes and fixes up parent
    references that pointed at the renamed bone *within the same list order*.
    """
    seen = {}
    rename_map = {}
    for b in bones:
        base = b.name or "Bone"
        if base not in seen:
            seen[base] = 0
            continue
        seen[base] += 1
        new_name = f"{base}.{seen[base]:03d}"
        while new_name in seen:
            seen[base] += 1
            new_name = f"{base}.{seen[base]:03d}"
        rename_map[id(b)] = (b.name, new_name)
        b.name = new_name
        seen[new_name] = 0
    return bones


class EvalContext:
    """Per-build memoization so shared upstream nodes evaluate once."""

    def __init__(self):
        self._bone_cache = {}
        self._constraint_cache = {}
        self._visiting = set()

    def bones_from_node(self, node):
        key = node.name
        if key in self._visiting:
            raise RuntimeError(f"Cycle detected at node '{node.name}'")
        if key in self._bone_cache:
            return self._bone_cache[key]
        self._visiting.add(key)
        try:
            result = node.eval_bones(self) if hasattr(node, "eval_bones") else []
        finally:
            self._visiting.discard(key)
        self._bone_cache[key] = result
        return result

    def constraints_from_node(self, node):
        key = node.name
        if key in self._constraint_cache:
            return self._constraint_cache[key]
        result = (
            node.eval_constraints(self) if hasattr(node, "eval_constraints") else []
        )
        self._constraint_cache[key] = result
        return result


def gather_input_bones(node, socket_name, ctx):
    """Collect BoneDefs from every link into the named input socket."""
    sock = node.inputs.get(socket_name)
    if sock is None:
        return []
    bones = []
    for link in sock.links:
        if not link.is_valid:
            continue
        bones.extend(ctx.bones_from_node(link.from_node))
    return bones


def gather_input_constraints(node, socket_name, ctx):
    """Collect ConstraintDefs from every link into the named input socket."""
    sock = node.inputs.get(socket_name)
    if sock is None:
        return []
    out = []
    for link in sock.links:
        if not link.is_valid:
            continue
        out.extend(ctx.constraints_from_node(link.from_node))
    return out


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
