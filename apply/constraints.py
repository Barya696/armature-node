"""Write a bone's constraint stack.

Constraints are ordered and interdependent, so they are diffed and written as
a whole list: the stack is cleared and rebuilt rather than patched. That is
the only way to get the order right, and a bone's stack is short enough that
rebuilding it costs nothing measurable.

Properties are written generically from the captured dict, which is what lets
a constraint type this addon has never heard of round-trip intact.
"""

import bpy

__all__ = ["apply_constraints"]

#: Properties that name a datablock; the record stores them by name.
_ID_PROPS = ("target", "pole_target", "subtarget", "action", "camera", "object")


def _resolve(con, key, value, writer):
    prop = con.bl_rna.properties.get(key)
    if prop is None or prop.is_readonly:
        return
    if prop.type == "POINTER":
        # Stored by name; None means the slot was empty when captured, and
        # objects.get(None) raises rather than returning None.
        if not value:
            return
        # Look it up, and leave the slot alone rather than clearing it when
        # the object is gone -- a missing target is a broken rig, but clearing
        # it silently is worse.
        target = bpy.data.objects.get(value)
        if target is not None and getattr(con, key, None) is not target:
            setattr(con, key, target)
            writer.count()
        return
    writer.set(con, key, value)


def apply_constraints(pbone, defs, writer):
    """Replace ``pbone``'s stack with ``defs`` (a tuple of ConstraintDef)."""
    if pbone is None:
        return
    while pbone.constraints:
        pbone.constraints.remove(pbone.constraints[0])
        writer.count()
    for cdef in defs or ():
        try:
            con = pbone.constraints.new(cdef.type)
        except (RuntimeError, TypeError):
            writer.errors.append(f"unknown constraint type {cdef.type!r}")
            continue
        writer.count()
        if cdef.name:
            con.name = cdef.name
        # subtarget must be written after target, or Blender discards it.
        props = dict(cdef.props)
        ordered = [k for k in _ID_PROPS if k in props] + [
            k for k in props if k not in _ID_PROPS
        ]
        for key in ordered:
            _resolve(con, key, props[key], writer)
