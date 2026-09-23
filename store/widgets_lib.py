"""``an_rig_widgets`` -- the widget mesh library.

Widgets are stored as geometry, not as object names. The v1 implementation
stored only the name, so when a ``WGT-*`` object went missing the rig could
not be restored and the "baseline" itself stripped every custom shape. A name
is a pointer into a scene that may not have the object any more; geometry is
the thing itself.

Kept out of the record because it is bulky and changes on a different cadence:
the record is read on every evaluation, the library only when a widget object
has to be rebuilt. It is zlib-compressed and base64-armoured, because a
Rigify rig's widget set is megabytes of float lists as plain JSON.
"""

import base64
import json
import zlib

from ..model.schema import widgets_from_dict, widgets_to_dict
from ..model.types import WidgetLib

KEY = "an_rig_widgets"

#: Below this, compression costs more than it saves and hurts readability
#: when debugging a stored property by eye.
_COMPRESS_OVER = 512
_MAGIC = "z:"

__all__ = ["KEY", "read", "write", "exists", "forget", "encode", "decode", "stats"]


def encode(lib):
    """Serialise a :class:`WidgetLib` to a storable string."""
    text = json.dumps(widgets_to_dict(lib), separators=(",", ":"), sort_keys=True)
    if len(text) < _COMPRESS_OVER:
        return text
    packed = zlib.compress(text.encode("utf-8"), 6)
    return _MAGIC + base64.b64encode(packed).decode("ascii")


def decode(text):
    """Parse a stored string back into a :class:`WidgetLib`."""
    if not text:
        return WidgetLib()
    try:
        if text.startswith(_MAGIC):
            raw = zlib.decompress(base64.b64decode(text[len(_MAGIC):])).decode("utf-8")
        else:
            raw = text
        return widgets_from_dict(json.loads(raw))
    except (ValueError, TypeError, zlib.error, base64.binascii.Error):
        # A corrupt library costs the widgets, not the rig: the record is
        # stored separately and is still perfectly readable.
        return WidgetLib()


def read(obj):
    if obj is None:
        return WidgetLib()
    try:
        value = obj.get(KEY)
    except (AttributeError, TypeError):
        return WidgetLib()
    return decode(value if isinstance(value, str) else "")


def write(obj, lib):
    if obj is None:
        return 0
    obj[KEY] = encode(lib)
    return len(lib)


def exists(obj):
    if obj is None:
        return False
    try:
        return bool(obj.get(KEY))
    except (AttributeError, TypeError):
        return False


def forget(obj):
    try:
        del obj[KEY]
    except (KeyError, TypeError):
        pass


def stats(obj):
    """(widget count, stored bytes) -- for the N-panel and diagnostics."""
    try:
        value = obj.get(KEY) or ""
    except (AttributeError, TypeError):
        value = ""
    return len(decode(value)), len(value)
