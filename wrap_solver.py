"""Wrap a marker skeleton onto a mesh: the maths, and nothing else.

The wrap add-on fits a template mesh onto a scan in three steps -- Snap,
Attract, Stick -- driven by point pairs. This is the same with the markers
as the template: a skeleton whose bones are the lines the viewport draws
between them (a marker and its parent, joined markers, MediaPipe's bones).

* **Snap**: the pairs pull their markers to where they belong and the rest
  of the skeleton follows, as a skeleton does: as rigidly as it can. Each
  bone keeps its length and turns at its joints, so hands paired lower
  swing the arms down at the shoulders instead of dragging the chest after
  them; a chain the pairs stretch or squash (a smaller character) shares
  that out along its bones; one pair carries its whole skeleton. The pairs
  land exactly. A part holding no pair at all (a stray marker) moves by the
  pairs' average. Snap always starts from the skeleton as it was before any
  step, so it depends on the pairs alone.
* **Attract**: every marker with a place on the mesh -- the middle of the
  limb it is in, or the skin -- is drawn toward it, softly, over passes in
  which the reach shrinks from the whole gap down to the chosen distance.
  The bones keep the skeleton in one piece; a Free marker is only carried.
* **Stick**: the same, harder, and on the last pass every marker within the
  distance lands exactly on its place.

Symmetry is the wrap's too: a marker with a partner (Hand.L, Hand.R) is
averaged with its partner's mirror image after every pass, across the YZ
plane of a frame -- the mesh's own, so a character standing away from the
world origin still mirrors about its middle.

Pure: numpy, no bpy. Where a marker's place is comes from a callable built
from the mesh by ``nodes/wrap.py``, so a test can hand in any shape.
"""

import numpy as np

INSIDE, SURFACE, FREE, FIXED = "INSIDE", "SURFACE", "FREE", "FIXED"

_PIN = 1e8  # the weight of a pair or a Fixed marker: it does not give
_SNAP_FRAMES = 12  # the Snap glide, streamed in this many steps
_RIGID_ROUNDS = 40  # of the as-rigid-as-possible Snap
_ATTRACT_PASSES = 12
_STICK_PASSES = 6


def _bones(points, bones):
    """([(i, j, weight)], graph Laplacian): each bone weighted by 1 / its
    length, so a field harmonic over the bones moves a marker by how far
    along them it is -- the way a chain bends, not by how near it is in
    space: a hand hanging beside the thigh is not moved by the knee."""
    n = len(points)
    edges = sorted({(min(i, j), max(i, j)) for i, j in bones if i != j and 0 <= min(i, j) and max(i, j) < n})
    lengths = [float(np.linalg.norm(points[i] - points[j])) for i, j in edges]
    mean = float(np.mean(lengths)) if lengths else 1.0
    weighted = [(i, j, mean / max(length, 1e-3 * mean)) for (i, j), length in zip(edges, lengths)]
    lap = np.zeros((n, n))
    for i, j, w in weighted:
        lap[i, i] += w
        lap[j, j] += w
        lap[i, j] -= w
        lap[j, i] -= w
    return weighted, lap


class Skeleton:
    """Markers joined by bones, and what the wrap knows about each.

    ``rest``    (n, 3) the shape to keep: where the markers were before the
                first step
    ``bones``   [(i, j)] marker indices
    ``roles``   per marker: INSIDE or SURFACE (drawn to that place on the
                mesh), FREE (carried by the bones only), FIXED (never moves)
    ``pairs``   {i: where marker i goes}
    ``mirror``  per marker its partner's index, its own on the middle line,
                -1 without one; None: no symmetry
    ``frame``   4x4 matrix whose YZ plane is the mirror
    """

    def __init__(self, rest, bones, roles=None, pairs=None, mirror=None, frame=None):
        self.rest = np.asarray(rest, dtype=float).reshape(-1, 3)
        n = len(self.rest)
        self.roles = list(roles) if roles is not None else [INSIDE] * n
        self.pairs = {int(i): np.asarray(p, dtype=float) for i, p in (pairs or {}).items()}
        self.mirror = None if mirror is None else np.asarray(mirror, dtype=int)
        self.frame = np.eye(4) if frame is None else np.asarray(frame, dtype=float)
        self.bones, self.laplacian = _bones(self.rest, bones)

    # -- The steps ---------------------------------------------------------------

    def snap(self, start):
        """Yield the markers gliding from ``start`` to their snapped places."""
        start = np.asarray(start, dtype=float)
        goals, weights = self._pins(start)
        paired = [i for i in self.pairs if self.roles[i] != FIXED]
        # Barely damped, toward the pairs' average move: where a pair reaches,
        # the bones decide; where none does, the average is what is left.
        average = np.mean([self.pairs[i] - self.rest[i] for i in paired], axis=0) if paired else np.zeros(3)
        x = self._rigid(goals, weights, self.rest + average)
        x = self._settle(x, goals, weights)
        for k in range(1, _SNAP_FRAMES + 1):
            t = k / _SNAP_FRAMES
            yield start + t * t * (3.0 - 2.0 * t) * (x - start)

    def attract(self, start, place, distance):
        """Yield each Attract pass. ``place(points, roles)`` gives every
        marker's place on the mesh, NaN where it has none; ``distance`` is
        the reach of the last pass."""
        return self._draw_in(start, place, distance, _ATTRACT_PASSES, False)

    def stick(self, start, place, distance):
        """Yield each Stick pass; the last leaves every marker within
        ``distance`` of its place exactly on it."""
        return self._draw_in(start, place, distance, _STICK_PASSES, True)

    def _draw_in(self, start, place, distance, passes, stick):
        """Attract and Stick, as the wrap anneals them: the reach starts wide
        enough to cover the gap and shrinks to ``distance``, the pull grows,
        and each pass keeps the skeleton's shape (the bones) without going
        far from the last one (the damping)."""
        base = np.asarray(start, dtype=float)
        x = base.copy()
        goals0, weights0 = self._pins(base)
        drawn = np.array([r in (INSIDE, SURFACE) for r in self.roles]) & (weights0 == 0.0)
        gap = np.linalg.norm(place(x, self.roles) - x, axis=1)[drawn]
        gap = gap[np.isfinite(gap)]
        reach0 = max(distance, 1.25 * float(np.percentile(gap, 95))) if len(gap) else distance
        for k in range(passes):
            t = (k + 1) / passes
            reach = distance + (reach0 - distance) * (1.0 - t) ** 2
            spots = place(x, self.roles)
            gap = np.linalg.norm(spots - x, axis=1)
            near = drawn & np.isfinite(gap) & (gap <= reach)
            goals, weights = goals0.copy(), weights0.copy()
            goals[near] = spots[near]
            pull = (50.0 if stick else 5.0) * (0.1 + 0.9 * t)
            weights[near] = pull * (0.3 + 0.7 * (1.0 - gap[near] / max(reach, 1e-12)))
            x = self._solve(base, goals, weights, 1.0, x - base)
            if stick:
                # Onto the place itself, more of the way each pass: all of it
                # on the last, for every marker within the chosen distance.
                spots = place(x, self.roles)
                gap = np.linalg.norm(spots - x, axis=1)
                near = drawn & np.isfinite(gap) & (gap <= max(reach, distance))
                x[near] += t * t * (spots[near] - x[near])
            x = self._settle(x, goals, weights)
            yield x.copy()

    # -- Symmetry ----------------------------------------------------------------

    def symmetrize(self, x):
        """Each marker averaged with its partner's mirror image, in the
        frame's own space; one on the middle line goes onto the plane."""
        if self.mirror is None or not (self.mirror >= 0).any():
            return x
        has = self.mirror >= 0
        inv = np.linalg.inv(self.frame)
        local = x @ inv[:3, :3].T + inv[:3, 3]
        flipped = local[self.mirror[has]] * np.array([-1.0, 1.0, 1.0])
        local[has] = 0.5 * (local[has] + flipped)
        return local @ self.frame[:3, :3].T + self.frame[:3, 3]

    # -- Parts ---------------------------------------------------------------------

    def _pins(self, start):
        """(goals, weights) holding the pairs on their places and the Fixed
        markers where they start. NaN goals are markers nothing holds."""
        n = len(start)
        goals, weights = np.full((n, 3), np.nan), np.zeros(n)
        for i, role in enumerate(self.roles):
            if role == FIXED:
                goals[i], weights[i] = start[i], _PIN
        for i, spot in self.pairs.items():
            if self.roles[i] != FIXED:
                goals[i], weights[i] = spot, _PIN
        return goals, weights

    def _rigid(self, goals, weights, toward, damp=1e-6):
        """The rest skeleton held by the pins, bent the way a skeleton bends.

        Every bone keeps its length. A marker where three or more bones meet
        -- a chest with its neck and shoulders, a pelvis with its legs -- is
        a solid piece: its bones turn together, as one rotation. One with one
        or two bones is a joint, free to bend: an arm turns at the shoulder
        and the elbow without moving the chest. Rounds of: fit the rotations
        to where the markers are, then place the markers to fit the turned
        bones (Sorkine & Alexa's as-rigid-as-possible, with free joints).

        Starts from the straight-line answer, where nothing turns: that one
        spreads a hand's move down the arm into the chest.
        """
        n = len(self.rest)
        a = self.laplacian + np.diag(weights + damp)
        held = np.where(weights[:, None] > 0.0, goals, 0.0) * weights[:, None] + damp * toward
        x = np.linalg.solve(a, self.laplacian @ self.rest + held)
        if not self.bones:
            return x
        solid = np.bincount([k for i, j, _w in self.bones for k in (i, j)], minlength=n) >= 3
        for _ in range(_RIGID_ROUNDS):
            cov = np.zeros((n, 3, 3))
            for i, j, w in self.bones:
                outer = w * np.outer(self.rest[i] - self.rest[j], x[i] - x[j])
                cov[i] += outer
                cov[j] += outer
            u, _s, vt = np.linalg.svd(cov)
            u[np.linalg.det(u @ vt) < 0.0, :, 2] *= -1.0  # a turn, never a mirror
            turn = np.transpose(vt, (0, 2, 1)) @ np.transpose(u, (0, 2, 1))
            rhs = held.copy()
            for i, j, w in self.bones:
                rest, now = self.rest[i] - self.rest[j], x[i] - x[j]
                # At a joint the bone points where it points, at its length.
                free = np.linalg.norm(rest) * now / max(float(np.linalg.norm(now)), 1e-12)
                ends = [turn[k] @ rest if solid[k] else free for k in (i, j)]
                bone = 0.5 * w * (ends[0] + ends[1])
                rhs[i] += bone
                rhs[j] -= bone
            x = np.linalg.solve(a, rhs)
        return x

    def _solve(self, base, goals, weights, damp, toward):
        """Positions shaped like ``base`` with marker i drawn to ``goals[i]``
        by ``weights[i]``. Minimises, over the moves d = x - base,

            sum over bones  w |d_i - d_j|^2                 keep the shape
          + weights_i |d_i - (goals_i - base_i)|^2           go to the goals
          + damp |d_i - toward_i|^2                          not all at once

        One small dense solve: a skeleton is tens of markers, not a mesh.
        """
        want = np.where(weights[:, None] > 0.0, goals - base, 0.0)
        rhs = weights[:, None] * want + damp * toward
        return base + np.linalg.solve(self.laplacian + np.diag(weights + damp), rhs)

    def _settle(self, x, goals, weights):
        """Pins exactly on, symmetric -- and the pins win over the mirror."""
        pinned = weights >= _PIN
        x[pinned] = goals[pinned]
        x = self.symmetrize(x)
        x[pinned] = goals[pinned]
        return x
