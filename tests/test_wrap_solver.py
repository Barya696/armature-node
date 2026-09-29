"""The marker wrap's maths: Snap, Attract, Stick and symmetry, on shapes
simple enough to know the answer. Needs numpy (Blender ships it)."""

import conftest  # noqa: F401  (puts the add-on on the path)
import numpy as np

from armature_nodes.wrap_solver import FIXED, FREE, INSIDE, SURFACE, Skeleton

CHAIN = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 3.0)]  # bones of 1 and 2
BONES = [(0, 1), (1, 2)]


def _last(passes):
    out = None
    for out in passes:
        pass
    return out


def _close(a, b, eps=1e-6):
    assert np.allclose(a, b, atol=eps), f"got {np.round(a, 4).tolist()}, want {np.round(b, 4).tolist()}"


def test_one_pair_carries_its_whole_skeleton():
    sk = Skeleton(CHAIN, BONES, pairs={0: (1.0, 0.0, 0.0)})
    _close(_last(sk.snap(CHAIN)), np.array(CHAIN) + (1.0, 0.0, 0.0))


def test_the_snap_glides_there():
    sk = Skeleton(CHAIN, BONES, pairs={0: (1.0, 0.0, 0.0)})
    frames = list(sk.snap(CHAIN))
    assert len(frames) > 2
    xs = [f[0][0] for f in frames]
    assert xs == sorted(xs) and 0.0 < xs[0] < 1.0


def test_between_two_pairs_a_marker_moves_by_how_far_along_the_bones_it_is():
    # Marker 1 is a third of the way along the chain: it takes a third of
    # the tip's move, not half (not by index) and not the nearest pair's.
    sk = Skeleton(CHAIN, BONES, pairs={0: (0.0, 0.0, 0.0), 2: (0.9, 0.0, 3.0)})
    x = _last(sk.snap(CHAIN))
    _close(x[1], (0.3, 0.0, 1.0))
    _close(x[2], (0.9, 0.0, 3.0))


def test_the_skeleton_is_scaled_to_the_pairs():
    sk = Skeleton(CHAIN, BONES, pairs={0: (0.0, 0.0, 0.0), 2: (0.0, 0.0, 1.5)})
    _close(_last(sk.snap(CHAIN)), np.array(CHAIN) * 0.5)


def test_snap_starts_from_the_rest_shape_whatever_the_markers_did_since():
    sk = Skeleton(CHAIN, BONES, pairs={0: (1.0, 0.0, 0.0)})
    moved = np.array(CHAIN) + (0.0, 5.0, 0.0)
    _close(_last(sk.snap(moved)), np.array(CHAIN) + (1.0, 0.0, 0.0))


def test_a_fixed_marker_stays_and_a_free_one_is_carried():
    sk = Skeleton(CHAIN, BONES, roles=[INSIDE, FREE, FIXED], pairs={0: (0.0, 0.0, -1.0)})
    x = _last(sk.snap(CHAIN))
    _close(x[2], CHAIN[2])
    _close(x[0], (0.0, 0.0, -1.0))
    assert -1.0 < x[1][2] < 1.0, "the free marker was not carried between them"


def test_a_marker_on_its_own_moves_by_the_pairs_average():
    points = CHAIN + [(4.0, 0.0, 0.0)]  # no bone to it
    sk = Skeleton(points, BONES, pairs={0: (0.0, 0.0, 0.0), 2: (0.0, 0.0, 6.0)})
    _close(_last(sk.snap(points))[3], (4.0, 0.0, 1.5))


def test_hands_paired_lower_swing_the_arms_down_and_leave_the_body():
    """A T-pose skeleton paired onto an A-pose character: the arms turn down
    at the shoulders. The shoulders and chest stay put and every bone keeps
    its length -- a straight-line solve dragged the shoulders 17 cm down
    into the chest with the hands."""
    rest = [
        (0.0, 0.0, 1.0), (0.0, 0.0, 1.3), (0.0, 0.0, 1.6),  # pelvis, chest, head
        (0.2, 0.0, 1.45), (0.5, 0.0, 1.45), (0.8, 0.0, 1.45),  # shoulder, elbow, hand .L
        (-0.2, 0.0, 1.45), (-0.5, 0.0, 1.45), (-0.8, 0.0, 1.45),  # .R
    ]
    bones = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (1, 6), (6, 7), (7, 8)]
    down = 0.6 * np.array([np.cos(np.radians(60.0)), 0.0, -np.sin(np.radians(60.0))])
    pairs = {0: rest[0], 2: rest[2], 5: np.array(rest[3]) + down, 8: (np.array(rest[6]) + down * (-1, 1, 1))}
    x = _last(Skeleton(rest, bones, pairs=pairs, mirror=[0, 1, 2, 6, 7, 8, 3, 4, 5]).snap(rest))
    for i, what in ((1, "chest"), (3, "shoulder.L"), (6, "shoulder.R")):
        moved = np.linalg.norm(x[i] - rest[i])
        assert moved < 0.02, f"the {what} was dragged {moved * 100:.1f} cm"
    lengths = [np.linalg.norm(x[i] - x[j]) / np.linalg.norm(np.subtract(rest[i], rest[j])) for i, j in bones]
    assert max(abs(r - 1.0) for r in lengths) < 0.02, f"bone lengths changed: {np.round(lengths, 3).tolist()}"


def test_no_marker_moves_further_than_the_pairs_do():
    """Two hands paired further apart and lower, as when the skeleton's arms
    are bent and the mesh's are out: the body must not be scaled up with
    them. (A whole-skeleton scale fitted to the hands threw the feet 1.8 m.)"""
    rest = [
        (0.0, 0.0, 1.5), (0.2, 0.0, 1.5), (-0.2, 0.0, 1.5),  # chest, shoulders
        (0.3, 0.0, 1.5), (-0.3, 0.0, 1.5),  # hands, bent in
        (0.0, 0.0, 1.0), (0.1, 0.0, 0.1), (-0.1, 0.0, 0.1),  # pelvis, feet
    ]
    bones = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 6), (5, 7)]
    pairs = {3: (0.66, 0.0, 1.3), 4: (-0.66, 0.0, 1.3)}
    x = _last(Skeleton(rest, bones, pairs=pairs).snap(rest))
    reach = max(np.linalg.norm(np.array(p) - rest[i]) for i, p in pairs.items())
    moves = np.linalg.norm(x - np.array(rest), axis=1)
    assert moves.max() <= reach + 1e-6, f"moves {np.round(moves, 3).tolist()}, pairs moved {reach:.3f}"


def _axis(points, roles):
    """A limb standing on the Z axis: its middle is the axis, its skin a
    cylinder of radius 0.1 around it."""
    out = np.full((len(points), 3), np.nan)
    for i, (p, role) in enumerate(zip(points, roles)):
        if role == INSIDE:
            out[i] = (0.0, 0.0, p[2])
        elif role == SURFACE:
            radial = np.array((p[0], p[1], 0.0))
            out[i] = (0.0, 0.0, p[2]) + 0.1 * radial / max(np.linalg.norm(radial), 1e-9)
    return out


def test_attract_draws_markers_toward_the_middle_and_stick_lands_them():
    start = np.array([(0.04, 0.02, 0.0), (0.05, 0.0, 1.0), (-0.03, 0.01, 2.0)])
    sk = Skeleton(start, [(0, 1), (1, 2)])
    x = _last(sk.attract(start, _axis, 0.1))
    before = np.linalg.norm(start[:, :2], axis=1)
    after = np.linalg.norm(x[:, :2], axis=1)
    assert (after < 0.5 * before).all(), f"attract left them at {after}"
    x = _last(sk.stick(x, _axis, 0.1))
    _close(x[:, :2], np.zeros((3, 2)))


def test_a_surface_marker_sticks_to_the_skin():
    start = np.array([(0.0, 0.0, 0.0), (0.03, 0.0, 1.0)])
    sk = Skeleton(start, [(0, 1)], roles=[INSIDE, SURFACE])
    x = _last(sk.stick(start, _axis, 0.2))
    _close(x[1], (0.1, 0.0, 1.0))
    _close(x[0], (0.0, 0.0, 0.0))


def test_a_pair_holds_through_attract():
    start = np.array([(0.05, 0.0, 0.0), (0.05, 0.0, 1.0)])
    sk = Skeleton(start, [(0, 1)], pairs={0: (0.05, 0.0, 0.0)})
    x = _last(sk.stick(start, _axis, 0.2))
    _close(x[0], (0.05, 0.0, 0.0))
    _close(x[1][:2], (0.0, 0.0))


def test_symmetry_mirrors_across_the_frame_not_the_world():
    frame = np.eye(4)
    frame[0, 3] = 2.0  # the mesh stands at x = 2
    points = [(2.3, 0.0, 1.0), (1.5, 0.2, 1.0), (2.1, 0.0, 2.0)]  # L, R, middle
    sk = Skeleton(points, [], mirror=[1, 0, 2], frame=frame)
    x = sk.symmetrize(np.array(points))
    _close(x[0], (2.4, 0.1, 1.0))
    _close(x[1], (1.6, 0.1, 1.0))
    _close(x[2], (2.0, 0.0, 2.0))


def test_symmetric_pairs_give_a_symmetric_snap():
    rest = [(0.0, 0.0, 1.0), (0.3, 0.0, 1.0), (-0.3, 0.0, 1.0), (0.6, 0.0, 1.0), (-0.6, 0.0, 1.0)]
    bones = [(0, 1), (0, 2), (1, 3), (2, 4)]
    sk = Skeleton(rest, bones, pairs={3: (0.5, 0.1, 0.8), 4: (-0.5, 0.1, 0.8)}, mirror=[0, 2, 1, 4, 3])
    x = _last(sk.snap(rest))
    _close(x[1] * (-1.0, 1.0, 1.0), x[2])
    _close(x[0][0], 0.0)


def test_a_pair_wins_over_the_mirror():
    points = [(0.3, 0.0, 1.0), (-0.3, 0.0, 1.0)]
    sk = Skeleton(points, [(0, 1)], pairs={0: (0.5, 0.0, 1.0)}, mirror=[1, 0])
    _close(_last(sk.snap(points))[0], (0.5, 0.0, 1.0))
