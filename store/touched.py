"""``an_rig_touched`` -- what the last build wrote.

The pivot of the whole restore-then-apply design. Each build records the field
paths it wrote; the next build restores any of them it is *not* writing back
to the record before applying its own changes.

That is what makes the viewport converge on ``record + graph`` no matter what
the user did: deleting a node, unplugging the Input, or emptying the tree all
leave paths in this set that the next build then restores. There is no
special "do nothing when the stream is empty" branch, because there does not
need to be one.

Only paths are stored, never values -- the values to restore to come from the
record, which is the single source of truth.
"""

import json

KEY = "an_rig_touched"

__all__ = ["KEY", "read", "write", "clear", "exists", "count"]


def read(obj):
    """``{bone_name: (path, ...)}`` for the previous build, or ``{}``."""
    if obj is None:
        return {}
    try:
        value = obj.get(KEY)
    except (AttributeError, TypeError):
        return {}
    if not isinstance(value, str) or not value:
        return {}
    try:
        data = json.loads(value)
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for name, paths in data.items():
        if isinstance(paths, list):
            out[name] = tuple(p for p in paths if isinstance(p, str))
    return out


def write(obj, touched):
    """Store the touched set. Accepts a ChangeSet or a plain ``{name: paths}``.

    An empty set removes the property rather than storing ``{}``: a rig the
    graph no longer touches should look untouched, not carry an empty marker.
    """
    if obj is None:
        return 0
    paths = touched.paths() if hasattr(touched, "paths") else touched
    paths = {k: sorted(v) for k, v in (paths or {}).items() if v}
    if not paths:
        clear(obj)
        return 0
    obj[KEY] = json.dumps(paths, separators=(",", ":"), sort_keys=True)
    return sum(len(v) for v in paths.values())


def clear(obj):
    try:
        del obj[KEY]
    except (KeyError, TypeError):
        pass


def exists(obj):
    if obj is None:
        return False
    try:
        return bool(obj.get(KEY))
    except (AttributeError, TypeError):
        return False


def count(obj):
    """How many fields the last build wrote."""
    return sum(len(v) for v in read(obj).values())
