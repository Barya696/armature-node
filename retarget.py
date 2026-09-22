"""Match the marker skeleton onto a Rigify rig and drive it.

Which bone layer is targeted, and why
-------------------------------------
Generated Rigify rigs have four layers of bones:

* ``DEF-*``   the skinned bones -- every one carries a Copy Transforms /
  Stretch To constraint from ``ORG-``/``MCH-``, so anything written into their
  pose is overwritten on the next depsgraph evaluation. They cannot drive.
* ``ORG-*``   the original metarig bones -- likewise constrained to the
  controls in a *generated* rig (they are only free on the metarig itself).
* ``MCH-*``   internal mechanism. Writing to it desynchronises the rig.
* controls (no prefix: ``torso``, ``chest``, ``hand_fk.L``, ...) -- the only
  unconstrained, animator-writable layer.

So the match targets **control bones**. ``ORG-``/``DEF-`` names are kept as
fallbacks only, because on a hand-made or non-Rigify rig they are sometimes
the only names present.

FK is the default drive mode: an FK control maps 1:1 onto a skeleton bone's
orientation. IK controls are position targets and would fight a driven FK
chain, so IK is opt-in; in that mode the hands/feet drive ``*_ik`` controls
and the elbow/knee positions drive the pole targets.
"""

import bpy
from mathutils import Matrix, Vector

from .primary_rig import skeleton_key_map, vec_roll_to_mat3

# ---------------------------------------------------------------------------
# Mapping table
# ---------------------------------------------------------------------------
#
# skeleton bone key -> ordered candidate control names. ``{S}`` is the side
# token (L/R). The first candidate that exists on the rig wins.

CENTER_MAP = {
    "hips": ("torso", "hips", "ORG-spine", "DEF-spine"),
    "spine": ("tweak_spine.001", "spine_fk.001", "ORG-spine.001", "DEF-spine.001"),
    "spine1": ("tweak_spine.002", "spine_fk.002", "ORG-spine.002", "DEF-spine.002"),
    "spine2": ("chest", "tweak_spine.003", "ORG-spine.003", "DEF-spine.003"),
    "neck": ("neck", "tweak_spine.004", "ORG-spine.004", "DEF-spine.004"),
    "head": ("head", "tweak_spine.005", "ORG-spine.005", "DEF-spine.005"),
}

SIDE_MAP = {
    "shoulder": ("shoulder.{S}", "ORG-shoulder.{S}", "DEF-shoulder.{S}"),
    "upper_arm": ("upper_arm_fk.{S}", "ORG-upper_arm.{S}", "DEF-upper_arm.{S}"),
    "forearm": ("forearm_fk.{S}", "ORG-forearm.{S}", "DEF-forearm.{S}"),
    "hand": ("hand_fk.{S}", "ORG-hand.{S}", "DEF-hand.{S}"),
    "thigh": ("thigh_fk.{S}", "ORG-thigh.{S}", "DEF-thigh.{S}"),
    "shin": ("shin_fk.{S}", "ORG-shin.{S}", "DEF-shin.{S}"),
    "foot": ("foot_fk.{S}", "ORG-foot.{S}", "DEF-foot.{S}"),
    # Rigify renamed this to toe_fk around 0.6.3; older rigs use bare toe.
    "toe": ("toe_fk.{S}", "toe.{S}", "ORG-toe.{S}", "DEF-toe.{S}"),
    "thumb": ("thumb.01.{S}", "ORG-thumb.01.{S}", "DEF-thumb.01.{S}"),
}

# IK mode replaces these entries. ``POLE`` entries are driven by position
# only (the pole target is placed at the marker joint), the rest by the full
# transform, since an IK control owns its own location.
IK_MAP = {
    "upper_arm": ("upper_arm_ik.{S}", "upper_arm_fk.{S}"),
    "forearm": ("upper_arm_ik_target.{S}", "forearm_fk.{S}"),
    "hand": ("hand_ik.{S}", "hand_fk.{S}"),
    "thigh": ("thigh_ik.{S}", "thigh_fk.{S}"),
    "shin": ("thigh_ik_target.{S}", "shin_fk.{S}"),
    "foot": ("foot_ik.{S}", "foot_fk.{S}"),
}
IK_POLES = {"forearm", "shin"}

# Bone keys whose control owns its world location as well as its rotation.
# Everything else is rotation-only: the rig decides where its bones sit.
LOCATION_KEYS = {"hips"}
IK_LOCATION_KEYS = {"hand", "foot", "forearm", "shin"}

# Limb property holders that carry Rigify's IK/FK slider.
IK_FK_HOLDERS = (
    "upper_arm_parent.{S}",
    "thigh_parent.{S}",
)
IK_FK_PROPS = ("IK_FK", "IK/FK")

# Bone keys deliberately left unmapped: three MediaPipe points per hand are
# not enough to solve per-phalanx rotations, so the finger controls stay with
# the animator. Add them through the node's overrides if you want them.
UNMAPPED_NOTE = (
    "Finger controls (f_index/f_middle/f_ring/f_pinky), palm, face controls "
    "and root are never driven"
)


def candidates(key, side, mode="FK"):
    """Ordered candidate control names for a skeleton bone key."""
    if side is None:
        return CENTER_MAP.get(key, ())
    names = []
    if mode == "IK" and key in IK_MAP:
        names.extend(IK_MAP[key])
    names.extend(SIDE_MAP.get(key, ()))
    return tuple(n.format(S=side) for n in names)


def drives_location(key, mode="FK"):
    if key in LOCATION_KEYS:
        return True
    return mode == "IK" and key in IK_LOCATION_KEYS


def is_pole(key, mode="FK"):
    return mode == "IK" and key in IK_POLES


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_bones(obj, bone_defs, mode="FK", overrides=None):
    """Resolve skeleton BoneDefs against the rig's pose bones.

    Returns ``(matches, unmatched)`` where ``matches`` is a list of
    ``(bone_def, pose_bone, key)`` and ``unmatched`` is the list of skeleton
    bone names with no control on this rig.
    """
    overrides = overrides or {}
    key_map = skeleton_key_map()
    pose_bones = obj.pose.bones
    matches, unmatched = [], []
    for bdef in bone_defs:
        entry = key_map.get(bdef.name)
        if entry is None:
            continue  # not a marker-skeleton bone (e.g. a wired-in parent)
        key, side = entry
        override = overrides.get(bdef.name)
        if override:
            # A manual override is a decision, not a hint: if the named bone
            # is not on this rig, say so instead of silently auto-matching
            # something else.
            pbone = obj.pose.bones.get(override)
            if pbone is None:
                unmatched.append(f"{bdef.name} (override '{override}' missing)")
                continue
            matches.append((bdef, pbone, key))
            continue
        names = list(candidates(key, side, mode))
        pbone = next((pose_bones.get(n) for n in names if pose_bones.get(n)), None)
        if pbone is None:
            unmatched.append(bdef.name)
            continue
        matches.append((bdef, pbone, key))
    return matches, unmatched


def report_lines(obj, bone_defs, mode="FK", overrides=None):
    """Human-readable match report for the node UI."""
    matches, unmatched = match_bones(obj, bone_defs, mode, overrides)
    lines = [f"{m[0].name}  ->  {m[1].name}" for m in matches]
    return lines, unmatched


# ---------------------------------------------------------------------------
# Pose application
# ---------------------------------------------------------------------------


def _bone_matrix(bdef):
    """World matrix of a skeleton BoneDef: +Y along the bone, +Z from roll."""
    head = Vector(bdef.head)
    tail = Vector(bdef.tail)
    mat = vec_roll_to_mat3(tail - head, bdef.roll).to_4x4()
    mat.translation = head
    return mat


def _depth(pbone):
    d = 0
    parent = pbone.parent
    while parent is not None:
        d += 1
        parent = parent.parent
    return d


def _set_ik_fk(obj, mode):
    """Flip Rigify's IK/FK slider so the driven chain is the visible one."""
    value = 1.0 if mode == "FK" else 0.0
    for pattern in IK_FK_HOLDERS:
        for side in ("L", "R"):
            pbone = obj.pose.bones.get(pattern.format(S=side))
            if pbone is None:
                continue
            for prop in IK_FK_PROPS:
                try:
                    if prop in pbone:
                        pbone[prop] = value
                except (TypeError, KeyError):  # noqa: PERF203
                    continue


def apply_pose(obj, bone_defs, mode="FK", overrides=None, set_switches=True):
    """Drive ``obj``'s controls from the marker skeleton.

    Pose matrices are written parent-before-child with a view-layer update per
    hierarchy level, which Blender requires: a child's ``matrix`` is resolved
    against its parent's *evaluated* transform, so without the flush every
    child would inherit a stale parent.

    Returns ``(applied, unmatched)``.
    """
    matches, unmatched = match_bones(obj, bone_defs, mode, overrides)
    if not matches:
        return 0, unmatched
    if set_switches:
        _set_ik_fk(obj, mode)

    world_inv = obj.matrix_world.inverted()
    matches.sort(key=lambda m: _depth(m[1]))
    applied = 0
    depth = None
    view_layer = bpy.context.view_layer
    for bdef, pbone, key in matches:
        d = _depth(pbone)
        if depth is not None and d != depth and view_layer is not None:
            view_layer.update()
        depth = d

        target = _bone_matrix(bdef)
        if is_pole(key, mode):
            # A pole target is a position hint only; its orientation is
            # meaningless to the solver.
            current = obj.matrix_world @ pbone.matrix
            current.translation = target.translation
            target = current
        elif not drives_location(key, mode):
            # Rotation only: the rig keeps its own rest placement.
            current = obj.matrix_world @ pbone.matrix
            target.translation = current.translation

        # Preserve the control's scale; only location/rotation are retargeted.
        loc, rot, _scale = target.decompose()
        _l, _r, scale = (obj.matrix_world @ pbone.matrix).decompose()
        target = (
            Matrix.Translation(loc)
            @ rot.to_matrix().to_4x4()
            @ Matrix.Diagonal(scale).to_4x4()
        )
        try:
            pbone.matrix = world_inv @ target
        except (AttributeError, ValueError) as exc:
            print(f"[Armature Nodes] Could not pose '{pbone.name}': {exc}")
            continue
        applied += 1
    if view_layer is not None:
        view_layer.update()
    return applied, unmatched
