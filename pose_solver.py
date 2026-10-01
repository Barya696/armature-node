"""Pose a skinned mesh onto a target surface: the wrap add-on's Snap,
Attract and Stick, with a skeleton for the elastic model.

The wrap moves every vertex, held in shape by a Laplacian. Here a vertex can
only move the way its bones move it -- carried by each bone it is weighted
to, in proportion (linear blend skinning) -- and the bones are segments of a
tree: each turns about its joint, carrying its children; the root also
moves. A segment may slide instead (a Rigify clavicle, which its marker
moves), or swing without twisting (an IK limb, whose twist is its pole's and
its hand's business). The mesh bends only where, and how, the rig bends.

``Figure.step`` is one Gauss-Newton pass, linearised about the pose the rig
is in now::

    minimize  sum_i a_i |u_i . (c_i - x_i(d))|^2   the surface: each point to
                                                  its closest point on the
                                                  target -- along the
                                                  target's normal once close,
                                                  so it may slide over it
            + sum_k b_k |m_k - q_k(d)|^2           landmarks: the markers
            + sum_j h_j |r_j + t_j(d)|^2           holds: a segment turned r_j
                                                  from where it started is
                                                  eased back, so a turn the
                                                  surface cannot see stays put
            + damping * d^T diag(A) d              Levenberg-Marquardt

``d`` holds each segment's small turn (along its axes, times the angle) and
move. The caller turns the step into the rig's own controls, lets the rig
pose itself and reads the result back for the next pass: the rig, not this
linear model, has the last word, so its IK and its spine's in-betweens stand.

``schedule`` anneals the passes as the wrap's Attract does: the search radius
and the normal gate wide at first, the user's at the end; the damping falls
as the pull takes over; points slide over the surface once close.
"""

import numpy as np

TURN, SLIDE, ROOT, SWING, FIXED = "TURN", "SLIDE", "ROOT", "SWING", "FIXED"
_AXES = np.eye(3)


def rotation(axis_angle):
    """3x3 rotation for a turn given as its axis times its angle (Rodrigues)."""
    v = np.asarray(axis_angle, dtype=float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _across(direction):
    """Two unit axes square to ``direction`` and to each other."""
    d = np.asarray(direction, dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    other = _AXES[int(np.argmin(np.abs(d)))]
    e1 = np.cross(d, other)
    e1 /= np.linalg.norm(e1)
    return np.array([e1, np.cross(d, e1)])


class Figure:
    """A tree of segments: where each turns (its pivot, in the world), its
    parent (-1 for the root) and how it moves -- TURN, SLIDE, ROOT (both),
    SWING about its pivot but never about ``along[a]``, its own length -- or
    FIXED: it only follows."""

    def __init__(self, pivots, parents, kinds, along=None):
        self.pivots = np.asarray(pivots, dtype=float).reshape(-1, 3)
        self.parents = list(parents)
        self.kinds = list(kinds)
        n = len(self.parents)
        # carried[b, a]: segment b moves with segment a -- a is b, or above it.
        self.carried = np.zeros((n, n))
        for b in range(n):
            a = b
            while a >= 0:
                self.carried[b, a] = 1.0
                a = self.parents[a]
        self.order = sorted(range(n), key=lambda b: self.carried[b].sum())  # parents first
        # One column per unknown: (segment, turns or slides, along which axis).
        self.columns = []
        for a, kind in enumerate(self.kinds):
            turns = _across(along[a]) if kind == SWING else _AXES if kind in (TURN, ROOT) else ()
            self.columns += [(a, True, axis) for axis in turns]
            if kind in (SLIDE, ROOT):
                self.columns += [(a, False, axis) for axis in _AXES]
        self.size = len(self.columns)

    def _rows(self, points, dirs, carried):
        """(m, size): how far each point moves along its direction per unit
        of each unknown, taking ``carried`` (m, segments) of each segment's
        motion. A turn moves a point by turn x (point - pivot)."""
        J = np.zeros((len(points), self.size))
        levers = {}
        for c, (a, turns, axis) in enumerate(self.columns):
            if turns:
                if a not in levers:
                    levers[a] = np.cross(points - self.pivots[a], dirs)
                J[:, c] = carried[:, a] * (levers[a] @ axis)
            else:
                J[:, c] = carried[:, a] * (dirs @ axis)
        return J

    def step(self, points, weights, targets, pull, normals=None, landmarks=None, holds=None, damping=0.1,
             max_turn=0.3):
        """One pass: each segment's move, (segments, 4, 4) world matrices,
        bringing ``points`` (n, 3) -- carried by the segments as ``weights``
        (n, segments) say, rows summing to one -- toward ``targets`` (n, 3)
        with strength ``pull`` (n,): along ``normals`` (n, 3) when given, all
        the way otherwise. ``landmarks``: (points (k, 3), the segment each
        rides with (k,), where it belongs (k, 3), strength (k,)). ``holds``:
        (segments (j,), how far each has turned since the start (j, 3), its
        axis times its angle, in the world; strength (j,)). A step turning
        any segment more than ``max_turn`` radians is scaled down."""
        rows, gaps, strengths = [], [], []
        if len(points):
            carried = weights @ self.carried
            if normals is not None:
                rows.append(self._rows(points, normals, carried))
                gaps.append(np.einsum("ij,ij->i", targets - points, normals))
                strengths.append(pull)
            else:
                for axis in _AXES:
                    rows.append(self._rows(points, np.broadcast_to(axis, points.shape), carried))
                    gaps.append((targets - points) @ axis)
                    strengths.append(pull)
        if landmarks is not None and len(landmarks[0]):
            at, rides, goal, strength = landmarks
            carried = self.carried[np.asarray(rides, dtype=int)]
            for axis in _AXES:
                rows.append(self._rows(at, np.broadcast_to(axis, at.shape), carried))
                gaps.append((goal - at) @ axis)
                strengths.append(strength)
        if holds is not None and len(holds[0]):
            which, turned, strength = holds
            carried = self.carried[np.asarray(which, dtype=int)]
            for axis in _AXES:
                # A segment's turn in the world: its own and all above it.
                row = np.zeros((len(which), self.size))
                for c, (a, turns, along) in enumerate(self.columns):
                    if turns:
                        row[:, c] = carried[:, a] * (along @ axis)
                rows.append(row)
                gaps.append(-(np.asarray(turned) @ axis))
                strengths.append(strength)
        if not rows:
            return np.repeat(np.eye(4)[None], len(self.parents), axis=0)
        J, r, w = np.vstack(rows), np.concatenate(gaps), np.concatenate(strengths)
        A = J.T @ (w[:, None] * J)
        g = J.T @ (w * r)
        diag = np.diag(A).copy()
        floor = 1e-9 * max(float(diag.max()), 1e-12)  # an unknown nothing measures stays put
        d = np.linalg.solve(A + np.diag(damping * diag + floor), g)
        return self.moves(d, max_turn)

    def moves(self, d, max_turn=0.3):
        """(segments, 4, 4): the unknowns ``d`` as each segment's world move,
        a child's composed with its parent's."""
        n = len(self.parents)
        turns, slides = np.zeros((n, 3)), np.zeros((n, 3))
        for (a, turn, axis), value in zip(self.columns, d):
            (turns if turn else slides)[a] += value * axis
        biggest = float(np.linalg.norm(turns, axis=1).max()) if n else 0.0
        if biggest > max_turn:
            turns, slides = turns * (max_turn / biggest), slides * (max_turn / biggest)
        out = np.zeros((n, 4, 4))
        for a in self.order:
            own = np.eye(4)
            R = rotation(turns[a])
            own[:3, :3] = R
            own[:3, 3] = self.pivots[a] - R @ self.pivots[a] + slides[a]
            parent = self.parents[a]
            out[a] = (out[parent] if parent >= 0 else np.eye(4)) @ own
        return out


def carry(moves, weights, points):
    """Where ``points`` (n, 3) go when the segments make ``moves``, each point
    carried by them as ``weights`` (n, segments) say."""
    homogeneous = np.c_[points, np.ones(len(points))]
    moved = np.einsum("sij,nj->nsi", moves[:, :3, :], homogeneous)  # (n, segments, 3)
    return np.einsum("ns,nsi->ni", weights, moved)


def schedule(t, distance, reach, angle, loose=-1.0):
    """(radius, cosine gate, damping, slide) for a pass at progress ``t`` in
    (0, 1]: the search from ``reach`` down to ``distance``, the normal gate
    from ``loose`` (a cosine; -1 takes anything) to ``angle`` degrees, the
    damping from stiff to light -- the wrap's Attract, on a skeleton."""
    s = (1.0 - t) ** 2
    radius = distance + (reach - distance) * s
    gate = np.cos(np.radians(angle)) * (1.0 - s) + loose * s
    return radius, gate, 1.0 * s + 0.05 * (1.0 - s), t > 0.5
