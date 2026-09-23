"""Run the pure-Python tests without pytest.

The test modules are ordinary pytest files -- ``def test_*`` with bare
``assert`` -- so ``pytest`` runs them unchanged. This runner exists because
Blender does not ship pytest, and the pure suite should be runnable anywhere
the addon is.

    python tests/run_pure.py
"""

import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON = os.path.dirname(HERE)
PARENT = os.path.dirname(ADDON)
for path in (HERE, PARENT):
    if path not in sys.path:
        sys.path.insert(0, path)

MODULES = ("test_model_roundtrip", "test_diff", "test_migrate")


def run():
    passed = failed = 0
    failures = []
    for mod_name in MODULES:
        module = importlib.import_module(mod_name)
        names = sorted(n for n in dir(module) if n.startswith("test_"))
        print(f"\n{mod_name}  ({len(names)} tests)")
        for name in names:
            fn = getattr(module, name)
            if not callable(fn):
                continue
            try:
                fn()
            except Exception:
                failed += 1
                failures.append((mod_name, name, traceback.format_exc()))
                print(f"  FAIL  {name}")
            else:
                passed += 1
                print(f"  ok    {name}")

    print("\n" + "=" * 62)
    for mod_name, name, tb in failures:
        print(f"\n--- {mod_name}.{name} ---\n{tb}")
    print(f"{passed} passed, {failed} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(run())
