"""The write counter.

Every property written to the armature goes through a :class:`Writer`, which
gives two things the old build could not:

* **A no-op build really is one.** ``set`` compares before assigning, so
  applying the same record twice writes nothing and the counter proves it.
  That is acceptance test 10, and it is only meaningful because the count is
  taken at the point of assignment rather than estimated.
* **Errors are collected, not raised.** A rig with one unknown constraint type
  should still get the other 400 bones applied; the problems come back on the
  result for the Output node to show.
"""

__all__ = ["Writer"]

#: Floats compare equal within this. Blender stores single precision, so a
#: value that round-tripped through JSON is rarely bit-identical and would
#: otherwise be rewritten on every build for ever.
EPS = 1e-6


def _same(a, b):
    if isinstance(a, float) and isinstance(b, (int, float)):
        return abs(a - float(b)) <= EPS
    if hasattr(a, "__len__") and hasattr(b, "__len__") and not isinstance(a, str):
        if len(a) != len(b):
            return False
        return all(_same(x, y) for x, y in zip(a, b))
    return a == b


class Writer:
    """Counts property writes and collects problems."""

    def __init__(self):
        self.writes = 0
        self.errors = []
        self.bones = set()
        # (pose_bone, intended matrix) pairs, checked after the view layer is
        # updated. pbone.matrix is COMPUTED from matrix_basis plus the parent
        # chain and constraints, so reading it back before the depsgraph
        # re-evaluates returns the old value -- verifying inline would report
        # every successful write as blocked.
        self.pending_poses = []

    def count(self, changed=True):
        if changed:
            self.writes += 1
        return changed

    def note(self, message):
        self.errors.append(message)

    def touch(self, bone_name):
        self.bones.add(bone_name)

    def set(self, owner, attr, value):
        """Assign ``owner.attr = value`` when it differs. Returns True if written."""
        try:
            current = getattr(owner, attr)
        except AttributeError:
            # A property this Blender does not have is not an error: the
            # record may come from a newer build.
            return False
        try:
            if _same(current, value):
                return False
        except (TypeError, ValueError):
            pass  # not comparable; fall through and write
        try:
            setattr(owner, attr, value)
        except (AttributeError, TypeError, ValueError) as exc:
            self.note(f"{attr}: {exc}")
            return False
        self.writes += 1
        return True
