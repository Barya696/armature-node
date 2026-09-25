"""Two-way link between a node and the one bone it poses.

A node writes its value to the bone; the user grabs the bone in Pose mode;
both have to show up on the node, live. The whole difficulty is telling them
apart. Read back a value the node itself just wrote and it chases its own
output -- on a constrained bone the evaluated result differs from what was
written, so the two would oscillate for ever.

The rule that makes it safe:

* After every build, **snapshot** each linked bone. Whatever it looks like
  then is the graph's own doing.
* On every tick, anything that differs from the snapshot was done by
  someone else -- a grab, a keyframe, a parent being moved. Only that
  **delta** is folded into the node, and the snapshot moves on with it.

The snapshot never enters the undo stack, so undo, redo and file load reset
it: otherwise undoing a grab would look like a fresh move and be folded in
a second time.

This module reads the armature and writes node fields and marker handles --
never the armature itself. That stays ``apply/``'s job.
"""

import math

from mathutils import Matrix, Vector

__all__ = [
    "EPS",
    "BoneState",
    "Delta",
    "driven_world",
    "delta_since",
    "remember",
    "forget",
    "reset",
]

EPS = 1e-5

# node pointer -> (bone name, BoneState)
_snapshots = {}


class BoneState:
    """A bone's pose, in the three frames the nodes express values in."""

    __slots__ = ("world", "rest", "basis")

    def __init__(self, world, rest, basis):
        self.world = world
        self.rest = rest
        self.basis = basis

    @classmethod
    def read(cls, obj, pbone):
        rest_armature = obj.convert_space(
            pose_bone=pbone,
            matrix=Matrix.Identity(4),
            from_space="LOCAL",
            to_space="POSE",
        )
        return cls(
            world=obj.matrix_world @ pbone.matrix,
            # Rest follows the parent's current pose. Relative values are
            # measured from here, so moving a parent -- or the whole object --
            # is not mistaken for a change to the child's own offset.
            rest=obj.matrix_world @ rest_armature,
            basis=pbone.matrix_basis.copy(),
        )


def driven_world(obj, pbone):
    """The bone's world matrix before its own constraints.

    This is the value to seed a node from so that driving the bone changes
    nothing. The pose is written before constraints run, so on a constrained
    bone ``pbone.matrix`` -- the evaluated result -- is the wrong thing to
    write back: the constraint would be applied on top of it again, and the
    bone would move the moment the node took it over.
    """
    armature = obj.convert_space(
        pose_bone=pbone,
        matrix=pbone.matrix_basis,
        from_space="LOCAL",
        to_space="POSE",
    )
    return obj.matrix_world @ armature


def _angle(q):
    """Rotation angle of a quaternion, immune to the q / -q double cover."""
    return 2.0 * math.acos(min(1.0, abs(q.w)))


class Delta:
    """How the bone moved since the node last saw it."""

    def __init__(self, before, now):
        wl0, wr0, ws0 = before.world.decompose()
        wl1, wr1, ws1 = now.world.decompose()
        rl0, rr0, _ = before.rest.decompose()
        rl1, rr1, _ = now.rest.decompose()
        bl0, br0, _ = before.basis.decompose()
        bl1, br1, _ = now.basis.decompose()

        self.now = now
        # Absolute, world axes.
        self.world_loc = wl1 - wl0
        self.world_rot = wr1 @ wr0.inverted()
        self.scale_ratio = Vector(
            tuple(b / a if abs(a) > 1e-9 else 1.0 for a, b in zip(ws0, ws1))
        )
        # Relative to rest, world axes: parent motion cancels out.
        self.rel_loc = (wl1 - rl1) - (wl0 - rl0)
        self.rel_rot = (wr1 @ rr1.inverted()) @ (wr0 @ rr0.inverted()).inverted()
        # The bone's own channels.
        self.local_loc = bl1 - bl0
        self.local_rot = br1 @ br0.inverted()

    def moved(self):
        return (
            self.world_loc.length > EPS
            or _angle(self.world_rot) > EPS
            or (self.scale_ratio - Vector((1.0, 1.0, 1.0))).length > EPS
            or self.local_loc.length > EPS
            or _angle(self.local_rot) > EPS
        )


def delta_since(node, obj, pbone):
    """The user's move since the last snapshot, or None on first sight.

    Always moves the snapshot on, so each call reports only what is new.
    """
    now = BoneState.read(obj, pbone)
    key = node.as_pointer()
    previous = _snapshots.get(key)
    _snapshots[key] = (pbone.name, now)
    if previous is None or previous[0] != pbone.name:
        return None  # first look at this bone: nothing to compare against
    return Delta(previous[1], now)


def remember(node, obj, pbone):
    """Record the bone as it is now -- called after the graph wrote it."""
    _snapshots[node.as_pointer()] = (pbone.name, BoneState.read(obj, pbone))


def forget(node):
    _snapshots.pop(node.as_pointer(), None)


def reset():
    """Drop every snapshot. Undo, redo and file load all invalidate them."""
    _snapshots.clear()


def compose(rotation, euler, compat=None):
    """``rotation`` (a quaternion) applied on top of ``euler``, as an Euler.

    ``compat`` keeps the result continuous with the previous value, so a
    slider does not flip 180 degrees the moment the angle wraps.
    """
    from mathutils import Euler

    q = rotation @ Euler(tuple(euler), "XYZ").to_quaternion()
    e = q.to_euler("XYZ", Euler(tuple(compat or euler), "XYZ"))
    return (e.x, e.y, e.z)
