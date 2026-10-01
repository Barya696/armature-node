"""Posing a skinned mesh onto a target by turning its bones: a three-segment
figure -- a body, an upper and a lower limb -- posed by a known motion, and
found again from where its points went. Needs numpy (Blender ships it)."""

import conftest  # noqa: F401  (puts the add-on on the path)
import numpy as np

from armature_nodes.pose_solver import ROOT, TURN, Figure, carry, rotation, schedule

PIVOTS = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0)])
PARENTS = [-1, 0, 1]
KINDS = [ROOT, TURN, TURN]


def _limb(rings=30, around=12):
    """Points on a capsule from z = -0.5 to 3 -- the body below 1, the upper
    limb to 2, the lower beyond -- blended over 0.2 at each joint."""
    points, weights = [], []
    for z in np.linspace(-0.5, 3.0, rings):
        for a in np.linspace(0.0, 2.0 * np.pi, around, endpoint=False):
            points.append((0.2 * np.cos(a), 0.2 * np.sin(a), z))
            w = np.zeros(3)
            share = np.clip((z - np.array([-np.inf, 1.0, 2.0]) + 0.1) / 0.2, 0.0, 1.0)
            w[0], w[1], w[2] = 1.0 - share[1], share[1] - share[2], share[2]
            weights.append(w)
    return np.array(points), np.array(weights)


def _truth():
    """The motion to find: the limb raised 35 degrees, bent 50 at the elbow,
    the body moved and turned a little."""
    moves = np.zeros(3)
    d = np.zeros(18)
    d[0:3] = (0.0, 0.0, 0.2)  # the body turns about z...
    d[3:6] = (0.1, -0.05, 0.0)  # ...and moves
    d[6:9] = (np.radians(35.0), 0.0, 0.0)
    d[9:12] = (0.0, np.radians(50.0), 0.0)
    del moves
    return Figure(PIVOTS, PARENTS, KINDS).moves(d, max_turn=10.0)


def _follow(pivots, moves):
    """Each segment's pivot, carried by its parent (the root's by itself)."""
    out = pivots.copy()
    for a, parent in enumerate(PARENTS):
        m = moves[a] if parent < 0 else moves[parent]
        out[a] = m[:3, :3] @ pivots[a] + m[:3, 3]
    return out


def _fit(rest, weights, find, passes, landmarks=None, normals=False):
    """The loop a rig runs: skin the points from rest with the segments'
    transforms, step, compose the step onto the transforms -- the solver
    never sees its own prediction, only the skinned result."""
    posed = np.repeat(np.eye(4)[None], len(PARENTS), axis=0)
    for _ in range(passes):
        points, pivots = carry(posed, weights, rest), _follow(PIVOTS, posed)
        targets, dirs = find(points)
        lm = None
        if landmarks is not None:
            lm = (pivots[[1, 2]], [0, 1], landmarks, np.full(2, 5.0))
        moves = Figure(pivots, PARENTS, KINDS).step(
            points, weights, targets, np.ones(len(points)) / len(points),
            normals=dirs if normals else None, landmarks=lm, damping=0.05,
        )
        posed = moves @ posed
    return carry(posed, weights, rest), _follow(PIVOTS, posed)


def test_rotation_is_a_proper_turn():
    R = rotation((0.0, 0.0, np.pi / 2))
    assert np.allclose(R @ (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)


def test_moves_carry_a_child_with_its_parent():
    d = np.zeros(18)
    d[6:9] = (np.pi / 2, 0.0, 0.0)  # the upper limb turns a quarter about x, at z = 1
    moves = Figure(PIVOTS, PARENTS, KINDS).moves(d, max_turn=10.0)
    tip = moves[2] @ (0.0, 0.0, 3.0, 1.0)
    assert np.allclose(tip[:3], (0.0, -2.0, 1.0)), tip


def test_known_places_bring_the_pose_back():
    """Each point told where it belongs: the steps land on the pose."""
    points, weights = _limb()
    goal = carry(_truth(), weights, points)
    got, _pivots = _fit(points, weights, lambda p: (goal, None), passes=20)
    assert np.abs(got - goal).max() < 2e-4, np.abs(got - goal).max()


def test_closest_points_and_the_joints_bring_the_pose_back():
    """What a real target gives: no pairs, only its surface -- the closest
    of its points to each of ours, along its normal once close -- and the
    markers on its joints. A round limb's turn about itself shows on no
    surface, so what is checked is the fit: on the surface, joints home."""
    points, weights = _limb()
    truth = _truth()
    dense, dense_weights = _limb(rings=120, around=48)
    surface = carry(truth, dense_weights, dense)
    joints = _follow(PIVOTS, truth)[[1, 2]]

    def closest(p):
        near = np.argmin(((p[:, None, :] - surface[None, :, :]) ** 2).sum(axis=2), axis=1)
        return surface[near], None

    def gap(p):
        return np.sqrt(((p[:, None, :] - surface[None, :, :]) ** 2).sum(axis=2).min(axis=1))

    got, pivots = _fit(points, weights, closest, passes=30, landmarks=joints)
    assert np.allclose(pivots[[1, 2]], joints, atol=0.01), pivots
    floor = gap(carry(truth, weights, points))  # the true pose, against the sampled surface
    assert gap(got).max() < floor.max() + 0.01 and gap(got).mean() < floor.mean() + 0.005, (gap(got).max(), floor.max())


def test_the_schedule_starts_wide_and_ends_on_the_users_gates():
    first = schedule(0.05, distance=0.03, reach=0.4, angle=65.0)
    last = schedule(1.0, distance=0.03, reach=0.4, angle=65.0)
    assert first[0] > 0.3 and first[1] < -0.8 and not first[3]
    assert np.isclose(last[0], 0.03) and np.isclose(last[1], np.cos(np.radians(65.0))) and last[3]
    assert last[2] < first[2]


def test_a_swing_never_twists():
    """A limb IK poses: it swings, but its twist is not its own."""
    from armature_nodes.pose_solver import SWING

    points = np.array([(0.2, 0.0, 1.0), (-0.2, 0.0, 1.0), (0.0, 0.2, 1.5)])
    twisted = carry(Figure([(0.0, 0.0, 0.0)], [-1], [TURN]).moves(np.array([0.0, 0.0, 0.4])), np.ones((3, 1)), points)
    figure = Figure([(0.0, 0.0, 0.0)], [-1], [SWING], along=[(0.0, 0.0, 1.0)])
    R = figure.step(points, np.ones((3, 1)), twisted, np.ones(3), damping=0.0)[0][:3, :3]
    assert abs(R[1, 0] - R[0, 1]) < 1e-9, "a turn about its own length"


def test_a_hold_turns_back_what_nothing_measures():
    figure = Figure([(0.0, 0.0, 0.0)], [-1], [TURN])
    empty = np.zeros((0, 3))
    moves = figure.step(empty, np.zeros((0, 1)), empty, np.zeros(0), holds=([0], np.array([(0.0, 0.0, 0.2)]), np.ones(1)),
                        damping=0.0)
    assert np.allclose(moves[0][:3, :3], rotation((0.0, 0.0, -0.2)), atol=1e-6)


def test_a_fixed_part_only_follows():
    from armature_nodes.pose_solver import FIXED

    figure = Figure([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)], [-1, -1], [TURN, FIXED])
    assert figure.size == 3
    points = np.array([(0.0, 0.2, 1.5), (0.2, 0.0, 1.5)])
    moves = figure.step(points, np.array([(0.0, 1.0), (0.0, 1.0)]), points + (0.1, 0.0, 0.0), np.ones(2))
    assert np.allclose(moves, np.eye(4)), "nothing turns what only follows"
