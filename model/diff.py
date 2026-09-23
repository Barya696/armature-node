"""What changed between two records.

The build pipeline is restore-then-apply, and both halves are driven from
here. A :class:`ChangeSet` names the leaf fields that differ, keyed by bone,
which gives three things the old rewrite-everything build could not:

* **Cost.** Only changed fields are written. Two identical builds write zero
  properties.
* **Restoration.** The set of fields the last build wrote is remembered; the
  next build restores any of them it is no longer changing back to the record.
  That is what makes unplugging a node genuinely undo it.
* **Diagnostics.** The Output can say which bones a graph actually affects.

Paths are dotted leaf names -- ``rest.head``, ``display.shape`` -- except
``constraints``, which is compared and written as a whole list. Constraints
are ordered and interdependent, so a per-property diff of them would produce
changes that cannot be applied in isolation.
"""

from dataclasses import dataclass, field, fields

from .types import BoneDef, DisplayDef, MembershipDef, PoseDef, RestDef, TransformDef

#: Sub-records compared field by field. ``constraints`` is handled separately.
_SECTIONS = (
    ("rest", RestDef),
    ("membership", MembershipDef),
    ("display", DisplayDef),
    ("pose", PoseDef),
    ("transform", TransformDef),
)

CONSTRAINTS_PATH = "constraints"


@dataclass(frozen=True)
class ChangeSet:
    """Per-bone, per-path changes.

    ``bones`` maps a bone name to ``{path: new_value}``. ``added`` and
    ``removed`` name bones that exist in only one of the two records; a build
    in Modify mode reports those rather than acting on them, because adding or
    deleting bones is a Full Rig operation.
    """

    bones: dict = field(default_factory=dict)
    added: tuple = ()
    removed: tuple = ()

    def __bool__(self):
        return bool(self.bones or self.added or self.removed)

    def paths(self):
        """``{bone: (path, ...)}`` -- the shape stored in ``an_rig_touched``."""
        return {name: tuple(sorted(changes)) for name, changes in self.bones.items()}

    def touched_count(self):
        return sum(len(c) for c in self.bones.values())

    def for_bone(self, name):
        return self.bones.get(name, {})


def _section_changes(prefix, old, new):
    out = {}
    for f in fields(new):
        a = getattr(old, f.name)
        b = getattr(new, f.name)
        if a != b:
            out[f"{prefix}.{f.name}"] = b
    return out


def diff_bone(old, new):
    """``{path: new_value}`` for one bone. Empty when they are equal."""
    changes = {}
    for prefix, _cls in _SECTIONS:
        changes.update(_section_changes(prefix, getattr(old, prefix), getattr(new, prefix)))
    if old.constraints != new.constraints:
        changes[CONSTRAINTS_PATH] = new.constraints
    return changes


def diff(base, target):
    """Changes needed to turn ``base`` into ``target``.

    Bones only in ``target`` are ``added``, bones only in ``base`` are
    ``removed``; neither contributes field changes, because there is nothing
    to compare against.
    """
    bones = {}
    base_names = set(base.bones)
    target_names = set(target.bones)
    for name in target_names & base_names:
        changes = diff_bone(base.bones[name], target.bones[name])
        if changes:
            bones[name] = changes
    return ChangeSet(
        bones=bones,
        added=tuple(sorted(target_names - base_names)),
        removed=tuple(sorted(base_names - target_names)),
    )


def restore_paths(previously_touched, changes):
    """Paths to put back to their recorded values before applying ``changes``.

    Everything the last build wrote that this build is *not* writing. Without
    this step a deleted node would leave its effect baked onto the rig, which
    is precisely the bug this architecture exists to remove.
    """
    out = {}
    for name, paths in (previously_touched or {}).items():
        now = set(changes.for_bone(name))
        stale = tuple(sorted(p for p in paths if p not in now))
        if stale:
            out[name] = stale
    return out


def value_at(bone, path):
    """Read a dotted path off a :class:`BoneDef`."""
    if path == CONSTRAINTS_PATH:
        return bone.constraints
    section, _, leaf = path.partition(".")
    if not leaf:
        raise KeyError(path)
    return getattr(getattr(bone, section), leaf)


def with_value(bone, path, value):
    """A copy of ``bone`` with ``path`` set to ``value``.

    Used by restoration and by ``model.ops``; a record is never mutated, so
    every write rebuilds the small frozen dataclass that owns the field.
    """
    from dataclasses import replace

    if path == CONSTRAINTS_PATH:
        return replace(bone, constraints=tuple(value))
    section, _, leaf = path.partition(".")
    if not leaf:
        raise KeyError(path)
    sub = replace(getattr(bone, section), **{leaf: value})
    return replace(bone, **{section: sub})


def apply_changes(record, changes):
    """A new record with ``changes`` folded in. Pure -- for tests and preview."""
    from dataclasses import replace

    bones = dict(record.bones)
    for name, paths in changes.bones.items():
        bone = bones.get(name)
        if bone is None:
            continue
        for path, value in paths.items():
            bone = with_value(bone, path, value)
        bones[name] = bone
    return replace(record, bones=bones)


def all_paths():
    """Every diffable path. Used by the import-boundary and coverage tests."""
    out = []
    for prefix, cls in _SECTIONS:
        out.extend(f"{prefix}.{f.name}" for f in fields(cls))
    out.append(CONSTRAINTS_PATH)
    return tuple(out)


__all__ = [
    "ChangeSet",
    "CONSTRAINTS_PATH",
    "diff",
    "diff_bone",
    "restore_paths",
    "value_at",
    "with_value",
    "apply_changes",
    "all_paths",
    "BoneDef",
]
