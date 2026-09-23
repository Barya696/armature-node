"""Re-entrancy guard: "a build is in progress".

Applying a record writes to the armature, and writing to an armature makes
Blender fire depsgraph handlers, which are exactly the things that mark the
tree dirty and schedule another build. Without a guard a single build
re-enters itself and the addon live-locks.

It also blocks capture. A capture taken mid-build would read half-applied
state and store it as the original -- the failure mode this architecture
exists to prevent, arriving through the back door.
"""

import contextlib

_depth = 0

__all__ = ["held", "is_held", "guard", "reset"]


def is_held():
    """True while a build is running."""
    return _depth > 0


@contextlib.contextmanager
def held(name="build"):
    """Mark a build in progress for the duration of the block.

    Re-entrant by depth so nested applies (a pipeline calling a sub-step that
    also guards) do not release the lock early.
    """
    global _depth
    _depth += 1
    try:
        yield
    finally:
        _depth -= 1
        if _depth < 0:  # pragma: no cover - defensive
            _depth = 0


def guard(what="operation"):
    """Raise when called during a build. Use on captures and other writers."""
    if is_held():
        raise RuntimeError(f"{what} is not allowed while a build is in progress")


def reset():
    """Force the lock open. Only for error recovery and tests."""
    global _depth
    _depth = 0
