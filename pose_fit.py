"""Pose: a mesh posing the rig -- the other way round from the rig posing
the mesh.

A character rigged with Rigify, and a mesh of the same character in another
pose -- one generated from a picture, say, with a topology of its own. First
the Human Skeleton's markers go onto that mesh's joints (Fit to Mesh, Pick
Pairs, or by hand), and the rig follows them: its IK limbs reach for them,
joint for joint. What the markers cannot say is how each part is turned --
the lean of the chest, the head's turn, the hands' and the feet's -- and
that is what **Pose** finds, the way the wrap add-on wraps a template onto a
scan:

* each pass takes points spread over the rigged mesh and over the target,
  pairs each with the closest point of the other, and turns the parts to
  bring the pairs together (``pose_solver``): Snap on the markers alone
  first, then Attract with its gates annealed, then Stick;
* a part's rest shape also says where the markers beyond it belong -- the
  chest's, where the neck and the shoulders are -- so the markers steer its
  lean as landmarks;
* a turn the surface cannot see is eased back to where it began.

The markers stay where they were put. The turns go to them -- pelvis, chest,
neck, head, hands and feet -- and the rig poses itself from them, IK and all;
the next pass reads what it did. The result is an ordinary pose of the rig,
held by the Human Skeleton's markers.
"""

import re

import bpy
import numpy as np
from bpy.props import StringProperty
from bpy.types import Operator
from mathutils import Euler, Matrix, Vector

from .pose_solver import FIXED, ROOT, TURN, Figure, schedule

# The parts of the rigged mesh: (part, the Human Skeleton marker it turns
# about -- None: it only follows the markers around it -- and the deform
# bones it holds, "DEF-" left off). A clavicle turns with the chest.
_PARTS = (
    ("pelvis", "Pelvis", ("spine", "spine.001", "pelvis.L", "pelvis.R")),
    ("chest", "Chest", ("spine.002", "spine.003", "breast.L", "breast.R", "shoulder.L", "shoulder.R")),
    ("neck", "Neck", ("spine.004", "spine.005")),
    ("head", "Head", ("spine.006",)),
    ("arm.L", None, ("upper_arm.L", "forearm.L")),
    ("hand.L", "Hand.L", ("hand.L",)),
    ("leg.L", None, ("thigh.L", "shin.L")),
    ("foot.L", "Foot.L", ("foot.L", "toe.L")),
    ("arm.R", None, ("upper_arm.R", "forearm.R")),
    ("hand.R", "Hand.R", ("hand.R",)),
    ("leg.R", None, ("thigh.R", "shin.R")),
    ("foot.R", "Foot.R", ("foot.R", "toe.R")),
)
# Controls a deform bone's parents may pass through on the way up.
_CONTROLS = {"torso": "pelvis", "hips": "pelvis", "chest": "chest", "neck": "neck", "head": "head"}
# Markers a part's rest shape says where they belong: the lean of the pelvis
# puts the chest and the hips, the chest's the neck and the shoulders, the
# neck's the head.
_REACHES = {
    "Chest": "pelvis", "Hip.L": "pelvis", "Hip.R": "pelvis",
    "Neck": "chest", "Shoulder.L": "chest", "Shoulder.R": "chest", "Head": "neck",
}
# Where a marker that moves no control stands on the rig: a hip, on the thigh's joint.
_JOINTS = {"Hip.L": "DEF-thigh.L", "Hip.R": "DEF-thigh.R"}
# Markers moving a control through a Position node whose offset turns with
# a part: the head's, and each clavicle's with the chest.
_OFFSETS = {"Head": "head", "Shoulder.L": "chest", "Shoulder.R": "chest"}
# Poles, set from where the markers bend each limb: (root joint's bone, the
# marker on the joint, the marker at the end).
_POLES = {
    "Elbow.L": ("DEF-upper_arm.L", "Hand.L"), "Elbow.R": ("DEF-upper_arm.R", "Hand.R"),
    "Knee.L": ("DEF-thigh.L", "Foot.L"), "Knee.R": ("DEF-thigh.R", "Foot.R"),
}
_TRAIL = re.compile(r"\.\d{3}$")
_SAMPLES = 4000
# Passes: Snap on the markers alone, Attract with its gates annealed, Stick.
_SNAP, _ATTRACT, _STICK = 2, 16, 6
_LANDMARKS = 2.0  # the markers, all told, against the whole surface's 1 -- each way
_HOLD = 0.02  # a turn held back as if a fiftieth of its part's surface stood against it
_ANGLE = 65.0
# A marker is a strong landmark, not a fixed one: the head, the hands and the
# feet may slide to where the surface says they are, held where they were put
# as firmly as a third of their part's surface at first, a fifth of that by
# Stick -- and at most this share of the height a pass. (The pelvis, chest
# and neck turn only: the markers beyond them reach from where they stand.)
_PLACE = 0.3
_SLIDE = 0.01


def rigged_mesh(rig, exclude=None):
    """The mesh ``rig`` deforms -- the biggest, if it deforms several -- or None."""
    meshes = [
        o for o in bpy.context.scene.objects
        if o.type == "MESH" and o != exclude and any(m.type == "ARMATURE" and m.object == rig for m in o.modifiers)
    ]
    return max(meshes, key=lambda o: len(o.data.vertices), default=None)


def _part_of(bone, lookup):
    """The part a deform bone moves with: by its name, or -- a finger, the
    jaw -- by its nearest ancestor's. The pelvis, failing all."""
    while bone is not None:
        name = bone.name[4:] if bone.name[:4] in ("DEF-", "ORG-", "MCH-") else bone.name
        for key in (name, _TRAIL.sub("", name)):
            if key in lookup:
                return lookup[key]
        bone = bone.parent
    return 0


def _sample(obj, rig, count):
    """(vertex indices, weights (n, parts)): vertices spread over ``obj`` by
    area, and how much of each part's motion each takes."""
    mesh = obj.data
    mesh.calc_loop_triangles()
    n = len(mesh.vertices)
    co = np.empty(n * 3)
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.vertices.foreach_get("co", co)
    mesh.loop_triangles.foreach_get("vertices", tris)
    co, tris = co.reshape(-1, 3), tris.reshape(-1, 3)
    area = np.linalg.norm(np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]]), axis=1)
    share = np.zeros(n)
    for k in range(3):
        np.add.at(share, tris[:, k], area)
    names = [name for name, _m, _b in _PARTS]
    lookup = {bone: i for i, (_n, _m, bones) in enumerate(_PARTS) for bone in bones}
    lookup.update({control: names.index(part) for control, part in _CONTROLS.items()})
    parts = {}
    for group in obj.vertex_groups:
        bone = rig.data.bones.get(group.name)
        if bone is not None and bone.use_deform:
            parts[group.index] = _part_of(bone, lookup)
    rng = np.random.default_rng(0)
    usable = np.flatnonzero(share > 0.0)
    picked = rng.choice(usable, size=min(count, len(usable)), replace=False, p=share[usable] / share[usable].sum())
    weights = np.zeros((len(picked), len(_PARTS)))
    for row, i in enumerate(picked):
        for g in mesh.vertices[i].groups:
            part = parts.get(g.group)
            if part is not None:
                weights[row, part] += g.weight
    total = weights.sum(axis=1)
    keep = total > 1e-6
    return picked[keep], weights[keep] / total[keep, None]


def _pole(root, joint, end, length):
    """Where a pole target stands for the limb root-joint-end to bend as it
    does: on the side the joint bends to, ``length`` out. None when the limb
    is straight."""
    axis = end - root
    foot = root + axis * ((joint - root).dot(axis) / max(axis.dot(axis), 1e-12))
    out = joint - foot
    if out.length < 1e-3 * axis.length:
        return None
    return joint + out.normalized() * length


class PoseRun:
    """Pose in flight: its passes, fed to the rig one at a time -- by the
    modal operator, one per tick, so the rig settles into the pose in view;
    or all at once from a script (``finish``)."""

    def __init__(self, wrap, depsgraph):
        from .groups import unique_rig
        from .human_skeleton import spec
        from .nodes.base import bound_rig
        from .nodes.wrap import MeshTarget, Refused, gather, tree_markers

        if wrap.target is None:
            raise Refused("Fit the skeleton to the posed character first")
        rig = bound_rig(wrap.id_data) or unique_rig(wrap.id_data)
        if rig is None:
            raise Refused("This skeleton poses no rig")
        body = rigged_mesh(rig, exclude=wrap.target)
        if body is None:
            raise Refused(f"No mesh is rigged to '{rig.name}': give the character an Armature modifier on it")
        missing = [f"DEF-{bones[0]}" for _n, _m, bones in _PARTS if f"DEF-{bones[0]}" not in rig.pose.bones]
        if missing:
            raise Refused(f"'{rig.name}' is not a Rigify human: it has no {missing[0]}")
        if len(body.evaluated_get(depsgraph).data.vertices) != len(body.data.vertices):
            raise Refused(f"'{body.name}' has modifiers that change its vertices: turn them off to pose")
        self.rig_name, self.body_name, self.target_name = rig.name, body.name, wrap.target.name
        self.tree_name = wrap.id_data.name
        self.mesh = MeshTarget(wrap.target, depsgraph)
        self.samples, self.weights = _sample(body, rig, _SAMPLES)
        # And points spread over the target, for the other way round: each
        # pulls the nearest of ours -- a raised arm, a turned nose.
        rng = np.random.default_rng(1)
        back = rng.choice(len(self.mesh.points), size=min(_SAMPLES, len(self.mesh.points)), replace=False)
        self.back, self.back_normals = self.mesh.points[back], self.mesh.normals[back]

        self.nodes = {}  # marker name: (node name, key)
        for node, marker in tree_markers(wrap.id_data):
            self.nodes.setdefault(marker.name, (node.name, marker.key))
        where = {name: (control, seed) for name, control, seed, _p, _w, _s in spec() if control}
        # The parts that turn: (part index, marker, its control).
        self.turning = [
            (i, marker, where[marker][0]) for i, (_n, marker, _b) in enumerate(_PARTS)
            if marker in self.nodes and marker in where
        ]
        if not self.turning:
            raise Refused("There are no Human Skeleton markers here to pose from")
        turned = {i for i, _m, _c in self.turning}
        reaching = set(_REACHES.values())
        self.kinds = [
            (TURN if name in reaching else ROOT) if i in turned else FIXED for i, (name, _m, _b) in enumerate(_PARTS)
        ]
        part = {name: i for i, (name, _m, _b) in enumerate(_PARTS)}
        controls = {name: where[marker][0] for name, marker, _b in _PARTS if marker in where}
        # The landmarks: (part, its control, the marker's rig point at rest).
        self.reaches = []
        for marker, reaching in _REACHES.items():
            if marker in self.nodes and (marker in where or marker in _JOINTS) and reaching in controls:
                control, seed = where.get(marker, (_JOINTS.get(marker), None))
                bone, along = seed or (control, 0.0)
                rest = rig.data.bones.get(bone)
                if rest is not None:
                    self.reaches.append((part[reaching], controls[reaching], marker,
                                         rest.head_local.lerp(rest.tail_local, along)))
        # The Position nodes whose offsets move: (node name, marker, part or pole).
        self.offsets = []
        for marker, (node_name, _key) in self.nodes.items():
            if marker not in _OFFSETS and marker not in _POLES:
                continue
            for link in wrap.id_data.nodes[node_name].outputs[0].links:
                if link.to_node.bl_idname == "ArmatureNodesPositionNode":
                    self.offsets.append((link.to_node.name, marker, part.get(_OFFSETS.get(marker))))
        # What Original shows, if nothing is kept yet -- or it shows now.
        wrap.remember(gather(wrap.id_data)[0], replace=wrap.preview == "ORIGINAL")
        self.wrap_name = wrap.name
        self.start = self._snapshot()
        self.places = dict(self.start[2])  # where the markers are now: the slid ones move on
        self.height = max(float(np.ptp(self.mesh.points[:, 2])), 1e-6)
        self.distance = wrap.stick_distance * 0.01 * self.height
        self.gap = None
        self.passes = self._passes()

    # -- The rig and the nodes --------------------------------------------

    def tree(self):
        return bpy.data.node_groups.get(self.tree_name)

    def _marker(self, name):
        node_name, key = self.nodes[name]
        node = self.tree().nodes.get(node_name)
        return node, (node.marker_by_key(key) if node is not None else None)

    def _at(self, name):
        node, marker = self._marker(name)
        return Vector(node.marker_value(marker, "position"))

    def _snapshot(self):
        """What a stopped pose puts back: the turns, the offsets."""
        turns = []
        for _i, name, _c in self.turning:
            node, marker = self._marker(name)
            turns.append((name, node.marker_value(marker, "rotation")))
        tree = self.tree()
        places = [(name, tuple(self._at(name))) for name in self.nodes]
        return turns, [(name, tuple(tree.nodes[name].socket_value("Offset"))) for name, _m, _p in self.offsets], places

    def _settle(self):
        """Build the rig's graph now: the next pass reads what the last one did."""
        from .build import build_armature_from_tree

        rig = bpy.data.objects[self.rig_name]
        build_armature_from_tree(rig.armature_nodes_tree)
        bpy.context.view_layer.update()
        return rig

    def _read(self):
        """(figure, points, normals, landmarks) for the rig as it stands."""
        rig = self._settle()
        mw = rig.matrix_world
        pivots = np.zeros((len(_PARTS), 3))
        for i, name, _control in self.turning:
            pivots[i] = self._at(name)
        figure = Figure(pivots, [-1] * len(_PARTS), self.kinds)  # each part turns about its marker
        body = bpy.data.objects[self.body_name]
        evaluated = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            n = len(mesh.vertices)
            co, normals = np.empty(n * 3), np.empty(n * 3)
            mesh.vertices.foreach_get("co", co)
            mesh.vertex_normals.foreach_get("vector", normals)
        finally:
            evaluated.to_mesh_clear()
        m = np.array(body.matrix_world)
        points = co.reshape(-1, 3)[self.samples] @ m[:3, :3].T + m[:3, 3]
        normals = normals.reshape(-1, 3)[self.samples] @ np.linalg.inv(m[:3, :3])
        normals /= np.maximum(np.linalg.norm(normals, axis=1), 1e-12)[:, None]
        # Where each part's rest shape puts the markers beyond it, and where they are.
        at, rides, goals = [], [], []
        for part, control, marker, rest in self.reaches:
            pb = rig.pose.bones.get(control)
            if pb is not None:
                at.append(mw @ (pb.matrix @ (pb.bone.matrix_local.inverted() @ rest)))
                rides.append(part)
                goals.append(self._at(marker))
        strength = np.full(len(at), _LANDMARKS / max(len(at), 1))
        landmarks = (np.array(at).reshape(-1, 3), rides, np.array(goals).reshape(-1, 3), strength)
        return figure, points, normals, landmarks

    def _closest(self, points, normals, radius, gate):
        """(targets, their normals, pull, gaps): each point's closest point on
        the target, if within ``radius`` and facing its way within ``gate``;
        the pull falls off with distance and sums to one."""
        bvh = self.mesh.bvh
        targets, facing, pull = np.zeros_like(points), np.zeros_like(points), np.zeros(len(points))
        gaps = []
        for i, (p, n) in enumerate(zip(points, normals)):
            co, normal, _index, dist = bvh.find_nearest(Vector(p))
            if co is None:
                continue
            gaps.append(dist)
            if dist < radius and Vector(n).dot(normal) >= gate:
                targets[i], facing[i] = co, normal
                pull[i] = 0.3 + 0.7 * (1.0 - dist / max(radius, 1e-12))
        if pull.sum() > 0.0:
            pull /= pull.sum()
        self.gap = float(np.mean(gaps)) if gaps else None
        return targets, facing, pull, gaps

    def _reverse(self, points, normals, radius, gate):
        """(which of ``points``, targets, their normals, pull): the other way
        round -- each point spread over the target pulls the nearest of
        ours, under the same gates; the pull sums to one."""
        from mathutils.kdtree import KDTree

        tree = KDTree(len(points))
        for i, p in enumerate(points):
            tree.insert(p, i)
        tree.balance()
        which, targets, facing, pull = [], [], [], []
        for t, n in zip(self.back, self.back_normals):
            _co, i, dist = tree.find(t)
            if i is not None and dist < radius and float(normals[i] @ n) >= gate:
                which.append(i)
                targets.append(t)
                facing.append(n)
                pull.append(0.3 + 0.7 * (1.0 - dist / max(radius, 1e-12)))
        pull = np.array(pull)
        if pull.sum() > 0.0:
            pull /= pull.sum()
        return np.array(which, dtype=int), np.array(targets).reshape(-1, 3), np.array(facing).reshape(-1, 3), pull

    def _holds(self, figure, points):
        """(parts, turned since the start, strength): each turn eased back
        unless the surface asks for it, as firmly as a fiftieth of its part's
        points turning with it."""
        starts = dict(self.start[0])
        which, turned, strength = [], [], []
        for i, name, _control in self.turning:
            node, marker = self._marker(name)
            now = Euler(node.marker_value(marker, "rotation"), "XYZ").to_quaternion()
            axis, angle = (now @ Euler(starts[name], "XYZ").to_quaternion().inverted()).to_axis_angle()
            if angle > np.pi:
                angle -= 2.0 * np.pi
            lever = np.sum((points - figure.pivots[i]) ** 2, axis=1)
            which.append(i)
            turned.append(np.array(axis) * angle)
            strength.append(_HOLD * float(np.mean(self.weights[:, i] * lever)))
        return which, np.array(turned).reshape(-1, 3), np.array(strength)

    def _pins(self, figure, s):
        """(points, parts, goals, strength): each sliding part's marker held
        where it was put, ``s`` of the way from as firmly as its part's
        surface (1) to a fifth of that (0)."""
        starts = dict(self.start[2])
        at, rides, goals, strength = [], [], [], []
        for i, name, _control in self.turning:
            if self.kinds[i] == ROOT:
                at.append(figure.pivots[i])
                rides.append(i)
                goals.append(starts[name])
                strength.append(_PLACE * float(np.mean(self.weights[:, i])) * (0.2 + 0.8 * s))
        return np.array(at).reshape(-1, 3), rides, np.array(goals).reshape(-1, 3), np.array(strength)

    def _poles(self):
        """Each pole target where its limb's markers bend it."""
        rig = bpy.data.objects[self.rig_name]
        tree = self.tree()
        for name, marker, part in self.offsets:
            if part is not None or marker not in _POLES:
                continue
            root_bone, end = _POLES[marker]
            pb = rig.pose.bones.get(root_bone)
            if pb is None or end not in self.nodes:
                continue
            move = tree.nodes[name]
            joint = self._at(marker)
            pole = _pole(rig.matrix_world @ pb.head, joint, self._at(end), Vector(move.socket_value("Offset")).length)
            if pole is not None:
                move.write_socket("Offset", tuple(pole - joint))
        tree.mark_dirty()

    def _write(self, moves):
        """The step's turns, to the markers and the offsets that turn with them."""
        from .nodes.marker_base import deferred_marker_writes

        turns = [Matrix(m[:3, :3].tolist()) for m in moves]
        limit = _SLIDE * self.height
        for i, name, _control in self.turning:  # a sliding part takes its marker along
            if self.kinds[i] == ROOT:
                was = np.array(self.places[name])
                slide = (moves[i] @ np.append(was, 1.0))[:3] - was
                far = float(np.linalg.norm(slide))
                self.places[name] = tuple(was + slide * (min(far, limit) / far if far > 0.0 else 0.0))
        with deferred_marker_writes():
            # The others stay where they were put: a turned marker would swing
            # its children, and a pole target carried by a turn drags its own.
            for name, at in self.places.items():
                node, marker = self._marker(name)
                node.write_marker(marker, "position", at)
            for i, name, _control in self.turning:
                node, marker = self._marker(name)
                was = Euler(node.marker_value(marker, "rotation"), "XYZ")
                node.write_marker(marker, "rotation", tuple((turns[i] @ was.to_matrix()).to_euler("XYZ", was)))
        tree = self.tree()
        for name, _marker, part in self.offsets:
            if part is not None:
                move = tree.nodes[name]
                move.write_socket("Offset", tuple(turns[part] @ Vector(move.socket_value("Offset"))))
        tree.mark_dirty()

    # -- The passes ---------------------------------------------------------

    def _passes(self):
        reach = None
        for k in range(_SNAP + _ATTRACT + _STICK):
            self._poles()  # the limbs bend as their markers stand now
            figure, points, normals, landmarks = self._read()
            holds = self._holds(figure, points)
            s = 1.0 if k < _SNAP else (1.0 - min(1.0, (k - _SNAP + 1) / _ATTRACT)) ** 2
            at, rides, goals, strength = landmarks
            pat, prides, pgoals, pstrength = self._pins(figure, s)
            landmarks = (np.vstack([at, pat]), list(rides) + prides, np.vstack([goals, pgoals]), np.concatenate([strength, pstrength]))
            if k < _SNAP:  # the markers alone
                moves = figure.step(points[:0], self.weights[:0], points[:0], np.zeros(0),
                                    landmarks=landmarks, holds=holds, damping=0.5, max_turn=0.15)
            else:
                if reach is None:
                    _t, _f, _p, gaps = self._closest(points, normals, np.inf, -1.0)
                    reach = max(self.distance, float(np.percentile(gaps, 95)) * 1.25)
                t = min(1.0, (k - _SNAP + 1) / _ATTRACT)
                radius, gate, damping, slide = schedule(t, self.distance, reach, _ANGLE, loose=0.0)
                targets, facing, pull, _gaps = self._closest(points, normals, radius, gate)
                which, back, back_facing, back_pull = self._reverse(points, normals, radius, gate)
                moves = figure.step(
                    np.vstack([points, points[which]]), np.vstack([self.weights, self.weights[which]]),
                    np.vstack([targets, back]), np.concatenate([pull, back_pull]),
                    normals=np.vstack([facing, back_facing]) if slide else None,
                    landmarks=landmarks, holds=holds, damping=damping, max_turn=0.15,
                )
            self._write(moves)
            yield k

    def advance(self):
        """One pass; False once there are none left."""
        return next(self.passes, None) is not None

    def finish(self):
        while self.advance():
            pass
        self._settle()

    def cancel(self):
        """Put the turns and the offsets back where the pose found them."""
        from .nodes.marker_base import deferred_marker_writes

        turns, offsets, places = self.start
        with deferred_marker_writes():
            for name, at in places:
                node, marker = self._marker(name)
                if marker is not None:
                    node.write_marker(marker, "position", at)
            for name, rotation in turns:
                node, marker = self._marker(name)
                if marker is not None:
                    node.write_marker(marker, "rotation", rotation)
        tree = self.tree()
        for name, offset in offsets:
            if name in tree.nodes:
                tree.nodes[name].write_socket("Offset", offset)
        tree.mark_dirty()
        self._settle()


class ARMATURE_NODES_OT_wrap_pose(Operator):
    """Pose the rig from the character the skeleton is fitted to: its own
    mesh wraps onto that one, each part turning to match -- the pelvis,
    chest, neck, head, hands and feet -- the markers staying where they are"""

    bl_idname = "armature_nodes.wrap_pose"
    bl_label = "Pose"
    bl_options = {"UNDO"}  # one pose, one undo

    tree: StringProperty(name="Tree", options={"HIDDEN", "SKIP_SAVE"})
    node: StringProperty(name="Node", options={"HIDDEN", "SKIP_SAVE"})

    def _begin(self, context):
        from .nodes.wrap import WRAP_NODE, Refused

        tree = bpy.data.node_groups.get(self.tree) if self.tree else None
        wrap = tree.nodes.get(self.node) if tree is not None else getattr(context, "node", None)
        if getattr(wrap, "bl_idname", "") != WRAP_NODE:
            return None
        try:
            return PoseRun(wrap, context.evaluated_depsgraph_get())
        except (Refused, ValueError) as exc:
            self.report({"WARNING"}, str(exc))
            return None

    def execute(self, context):
        run = self._begin(context)
        if run is None:
            return {"CANCELLED"}
        run.finish()
        return self._done(run)

    def invoke(self, context, event):
        self._run = self._begin(context)
        if self._run is None:
            return {"CANCELLED"}
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        context.workspace.status_text_set(f"Pose: {self._run.body_name} onto {self._run.target_name}...   Esc: stop")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        from .nodes.wrap import _NAVIGATION

        if event.type == "ESC" and event.value == "PRESS":
            self._stop(context)
            self._run.cancel()
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"} if event.type in _NAVIGATION else {"RUNNING_MODAL"}
        try:
            more = self._run.advance()
        except Exception as exc:  # noqa: BLE001 -- a marker deleted under the pose, say
            self._stop(context)
            self.report({"WARNING"}, f"Pose stopped: {exc}")
            return {"CANCELLED"}
        if more:
            return {"RUNNING_MODAL"}
        self._stop(context)
        self._run.finish()
        return self._done(self._run)

    def _stop(self, context):
        context.window_manager.event_timer_remove(self._timer)
        context.workspace.status_text_set(None)

    def _done(self, run):
        from .nodes.wrap import _redraw

        wrap = run.tree().nodes.get(run.wrap_name)
        if wrap is not None:
            wrap["wrap_preview"] = 1  # what shows now: Wrapped, the pose and all
        _redraw()
        gap = f", {run.gap * 1000.0:.1f} mm from it on average" if run.gap is not None else ""
        self.report({"INFO"}, f"Posed: {run.body_name} on {run.target_name}{gap}")
        return {"FINISHED"}


classes = (ARMATURE_NODES_OT_wrap_pose,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
