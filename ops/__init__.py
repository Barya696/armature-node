"""Operators: thin wrappers with no logic of their own."""

from . import bind, record  # noqa: F401


def register():
    bind.register()
    record.register()


def unregister():
    record.unregister()
    bind.unregister()
