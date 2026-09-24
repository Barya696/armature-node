"""Run the Blender-side tests.

    blender -b --factory-startup -P tests/blender/run.py
    blender -b --factory-startup -P tests/blender/run.py -- test_capture

Discovers ``test_*.py`` beside this file and runs every ``test_*`` function in
each. Plain functions and bare asserts, like the pure suite -- no pytest,
because Blender does not ship one.

Each test gets a clean scene, so a failure cannot cascade into the next.
"""

import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON = os.path.dirname(os.path.dirname(HERE))
PARENT = os.path.dirname(ADDON)
for path in (HERE, PARENT):
    if path not in sys.path:
        sys.path.insert(0, path)


def reset_scene():
    """A clean file between tests, without relying on operator context."""
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)


def discover(only=None):
    names = sorted(
        f[:-3]
        for f in os.listdir(HERE)
        if f.startswith("test_") and f.endswith(".py")
    )
    if only:
        names = [n for n in names if n in only or n.replace("test_", "") in only]
    return names


def run(only=None):
    passed = failed = 0
    failures = []
    for mod_name in discover(only):
        module = importlib.import_module(mod_name)
        tests = sorted(n for n in dir(module) if n.startswith("test_"))
        print(f"\n{mod_name}  ({len(tests)} tests)")
        for name in tests:
            fn = getattr(module, name)
            if not callable(fn):
                continue
            try:
                reset_scene()
                fn()
            except Exception:
                failed += 1
                failures.append((mod_name, name, traceback.format_exc()))
                print(f"  FAIL  {name}")
            else:
                passed += 1
                print(f"  ok    {name}")

    print("\n" + "=" * 64)
    for mod_name, name, tb in failures:
        print(f"\n--- {mod_name}.{name} ---\n{tb}")
    print(f"BLENDER TESTS: {passed} passed, {failed} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    code = run(only=argv or None)
    # -b still exits 0 on an uncaught error, so the status has to be explicit.
    sys.exit(code)
