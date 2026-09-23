"""Shared fixtures for the pure-Python tests.

These run under plain ``pytest`` with no Blender: everything under ``model/``
and ``store/`` is importable on its own, which is the point of the dependency
rule. ``tests/blender/`` is the other half and needs ``blender -b -P``.
"""

import os
import sys

# The addon is imported as a package, so its parent has to be importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ADDON = os.path.dirname(_HERE)
_PARENT = os.path.dirname(_ADDON)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

PACKAGE = os.path.basename(_ADDON)

from armature_nodes.model.types import (  # noqa: E402
    ArmatureDef,
    BoneDef,
    ConstraintDef,
    DisplayDef,
    MembershipDef,
    PoseDef,
    RestDef,
    RigRecord,
    SourceDef,
    TransformDef,
    WidgetLib,
)


class FakeObject(dict):
    """Stands in for a ``bpy.types.Object``.

    The store only ever uses mapping access on the object, which is exactly
    how Blender custom properties behave -- so a dict is a faithful double and
    the store needs no Blender to test.
    """

    def __init__(self, name="rig"):
        super().__init__()
        self.name = name
        self.type = "ARMATURE"


def make_bone(name, parent=None, shape="", deform=True, collections=(), constraints=()):
    return BoneDef(
        name=name,
        rest=RestDef(
            head=(0.0, 0.0, 1.0),
            tail=(0.0, 0.0, 1.3),
            roll=0.25,
            parent=parent,
            connect=bool(parent),
            deform=deform,
            inherit_scale="ALIGNED",
            envelope_distance=0.3,
        ),
        membership=MembershipDef(collections=tuple(collections)),
        display=DisplayDef(
            shape=shape,
            preset="CIRCLE" if shape else "NONE",
            scale=(2.0, 2.0, 2.0),
            translation=(0.0, 0.1, 0.0),
            rotation=(0.0, 0.0, 0.5),
            wire_width=3.0,
            use_bone_size=False,
            show_wire=bool(shape),
            color={"palette": "THEME04"} if shape else None,
        ),
        pose=PoseDef(
            rotation_mode="XYZ",
            locks={"location": [True, False, True]},
            ik={"lock_ik_x": True, "ik_stiffness_x": 0.5},
        ),
        constraints=tuple(constraints),
    )


def make_record(n_bones=3):
    """A small but fully populated record, every section non-default."""
    bones = {}
    parent = None
    for i in range(n_bones):
        name = f"bone.{i:03d}" if i else "root"
        bones[name] = make_bone(
            name,
            parent=parent,
            shape=f"WGT-rig_{name}" if i else "",
            deform=(i % 2 == 0),
            collections=("Arm.L (IK)", "Main") if i else (),
            constraints=(
                ConstraintDef(
                    type="COPY_TRANSFORMS",
                    name="Copy",
                    props={"target": "rig", "subtarget": "root", "influence": 0.75,
                           "mix_mode": "REPLACE"},
                ),
            )
            if i
            else (),
        )
        parent = name
    return RigRecord(
        version=2,
        source=SourceDef(object="rig", armature="rig-data",
                         captured_at="2026-09-23T12:00:00", blender="4.2.0"),
        armature=ArmatureDef(display_type="WIRE", show_in_front=True,
                             pose_position="POSE",
                             collections=("Main", "Arm.L (IK)")),
        bones=bones,
    )


def make_widgets():
    return WidgetLib(
        widgets={
            "WGT-rig_bone.001": {
                "verts": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
                "edges": [(0, 1), (1, 2), (2, 0)],
                "faces": [(0, 1, 2)],
            }
        }
    )


def v1_record(n_bones=2):
    """A record in the shape the previous implementation wrote."""
    bones = []
    parent = None
    for i in range(n_bones):
        name = f"bone.{i:03d}" if i else "root"
        entry = {
            "name": name,
            "head": [0.0, 0.0, 1.0],
            "tail": [0.0, 0.0, 1.3],
            "roll": 0.25,
            "parent": parent,
            "use_connect": bool(parent),
            "use_deform": (i % 2 == 0),
            "envelope_distance": 0.3,
            "envelope_weight": 1.0,
            "shape": None,
            "constraints": [],
        }
        if i:
            entry["shape"] = {
                "widget": f"WGT-rig_{name}",
                "preset": "CIRCLE",
                "scale": [2.0, 2.0, 2.0],
                "translation": [0.0, 0.1, 0.0],
                "rotation": [0.0, 0.0, 0.5],
                "wire_width": 3.0,
                "scale_to_bone_length": False,
                "show_wire": True,
            }
            entry["constraints"] = [
                {"type": "COPY_TRANSFORMS", "name": "Copy",
                 "params": {"target": "rig", "subtarget": "root", "influence": 0.75}}
            ]
        bones.append(entry)
        parent = name
    return {"version": 1, "bones": bones}


# pytest picks these up as fixtures when it is available; the bundled runner
# calls the helpers directly.
try:
    import pytest

    @pytest.fixture
    def record():
        return make_record()

    @pytest.fixture
    def widgets():
        return make_widgets()

    @pytest.fixture
    def obj():
        return FakeObject()

except ImportError:  # pragma: no cover - pytest is optional for the runner
    pass
