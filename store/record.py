"""``an_rig_record`` — the rig's own copy of what it was.

The armature object is the database. This module is the only one that reads or
writes that property, so there is exactly one answer to "where does the
original live" and exactly one place to look when it is wrong.

Everything here takes the object as a plain mapping (``obj[key]``), which both
a ``bpy.types.Object`` and a ``dict`` satisfy — so the store is testable
without Blender, and nothing in here imports ``bpy``.

**Reads never write.** A v1 record is migrated in memory on every read and
persisted only by an explicit operator. Capturing or rewriting during
evaluation is what let the old implementation record an already-modified rig
as the original.
"""

from ..model import migrate as _migrate
from ..model.schema import RecordError, from_json, peek_version, to_json

KEY = "an_rig_record"

__all__ = ["KEY", "exists", "read", "write", "forget", "raw", "stored_version",
           "needs_migration", "persist_migration"]


def raw(obj):
    """The stored JSON text, or ``""``."""
    if obj is None:
        return ""
    try:
        value = obj.get(KEY)
    except (AttributeError, TypeError):
        return ""
    return value if isinstance(value, str) else ""


def exists(obj):
    """True when this object carries a record of any readable version."""
    return bool(raw(obj)) or bool(_v1_raw(obj))


def stored_version(obj):
    """Version of the stored record: 2, 1 for a legacy one, or ``None``."""
    version = peek_version(raw(obj))
    if version is not None:
        return version
    return 1 if _v1_raw(obj) else None


def _v1_raw(obj):
    if obj is None:
        return ""
    try:
        value = obj.get(_migrate.V1_KEY)
    except (AttributeError, TypeError):
        return ""
    return value if isinstance(value, str) else ""


def read(obj):
    """The record for ``obj``, or ``None`` when it is not bound.

    A legacy v1 record is upgraded in memory. Nothing is written back — the
    caller decides when to persist, and until then the v1 data stays untouched
    so a failed upgrade cannot lose it.
    """
    text = raw(obj)
    if text:
        try:
            return from_json(text)
        except RecordError:
            # Fall through to v1 rather than raising: a corrupt v2 record
            # should not hide a perfectly good legacy one underneath it.
            pass
    legacy = _v1_raw(obj)
    if legacy:
        import json

        try:
            data = json.loads(legacy)
        except (ValueError, TypeError):
            return None
        try:
            return _migrate.migrate_dict(data)
        except RecordError:
            return None
    return None


def write(obj, record):
    """Store ``record`` on ``obj``. Returns the number of bones stored."""
    if obj is None:
        return 0
    obj[KEY] = to_json(record)
    return len(record.bones)


def needs_migration(obj):
    """True when this object holds a legacy record and no current one."""
    return not raw(obj) and bool(_v1_raw(obj))


def persist_migration(obj):
    """Write the upgraded record, then retire the legacy property.

    Ordered so the legacy data survives a failure: v2 is written and read back
    before v1 is dropped. A migration that half-ran would otherwise leave the
    rig with no record at all.
    """
    if not needs_migration(obj):
        return None
    record = read(obj)
    if record is None:
        return None
    write(obj, record)
    if from_json(raw(obj)) != record:  # pragma: no cover - paranoia
        raise RecordError("migrated record did not read back identically")
    try:
        del obj[_migrate.V1_KEY]
    except (KeyError, TypeError):
        pass
    return record


def forget(obj):
    """Remove the record. The rig is then unbound, not modified."""
    for key in (KEY, _migrate.V1_KEY):
        try:
            del obj[key]
        except (KeyError, TypeError):
            pass
