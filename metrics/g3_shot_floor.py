# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-09-01  Created. G3, the shot-floor margin, as a reading of the G1
#               grid. Replaces the old single-point G3 of metrics_part2.py
#               (one n = 2, d = 1 number, scored 0-100) with the ratio surface
#               its own audit asked for. No new sweep: G1 already measures the
#               variance at every grid point.

"""G3 - shot-floor margin: how far above finite sampling the gradient sits.

Why this is a reading, not a sweep
----------------------------------
The old scorecard ran G3 as its own experiment at one point (n = 2, d = 1) and
scored it 0-100. Its own results slide called that out: one point describes
one point, and the fix on record was "repeat at every grid point and report
the ratio as a surface". The G1 grid *is* that repeat - it holds the global
and local gradient variance at every (family, n, d) - so G3 here is a pure
function of finished G1 records, exactly like the onset slope (G1) and the
depth retention (G2). One measured grid, four questions; none of them can
disagree with the others.

The quantity
------------
Larocca et al. 2025, Box 3: an N-shot estimate of a loss carries sample
variance ``Var_s = Var[O] / N``, so statistical fluctuations are of order
``1/sqrt(N)``. A gradient whose variance over random angles sits at or below
that floor cannot be resolved from shot noise on hardware sampled at N shots.
The margin

    margin(n, d) = Var[dC](n, d) / (1 / N_shots)

says how many times over the floor the family still is at that grid point,
and ``crossing_width`` reports where the margin falls under 1 along the width
axis - the width at which training with N-shot gradients stops being possible.

Conventions to state next to any number
---------------------------------------
* ``N_shots = 1024`` (the floor 1/1024) is a convention carried from the old
  scorecard, not a measured or analytic value. The bound assumes an observable
  of unit spectral norm, which both ``Z...Z`` and ``Z_0`` satisfy.
* The floor is conservative by up to a factor 2: the shift-rule estimate of a
  derivative combines two N-shot estimates, so its variance is bounded by
  ``1/(2N)``..``1/N`` depending on how shots are split. The old deck flagged
  this ("silence is not an option"); it is now stated here, once, in code.
* Every simulation in this package is analytic. G3 compares a measured
  variance against a computed bound; no shots are ever sampled.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .g1_gradient import SHOTS_FLOOR


def shot_floor_margin(variance: float, floor: float = SHOTS_FLOOR) -> float:
    """Return how many times over the shot floor one measured variance sits.

    Args:
        variance: Gradient variance at one grid point (global or local cost).
        floor: The finite-shot floor. Defaults to :data:`..g1_gradient.SHOTS_FLOOR`.

    Returns:
        The dimensionless ratio ``variance / floor``. Below 1 means the
        gradient is not resolvable at that shot budget.
    """
    return float(variance) / float(floor)


def crossing_width(
    widths: Sequence[int],
    variances: Sequence[float],
    floor: float = SHOTS_FLOOR,
) -> float | None:
    """Return the interpolated width where the variance crosses the floor.

    A log-linear interpolation between the last grid width above the floor and
    the first below it, so the answer is not quantised to the swept widths
    (the same complaint the old G5 audit made about its p_crit).

    Args:
        widths: Qubit counts, in any order.
        variances: Matching gradient variances.
        floor: The finite-shot floor.

    Returns:
        The crossing width as a float, or ``None`` when the variance never
        falls under the floor inside the swept range.
    """
    order = np.argsort(widths)
    ns = np.asarray(widths, dtype=float)[order]
    vs = np.asarray(variances, dtype=float)[order]
    below = vs < floor
    if not below.any():
        return None
    first = int(np.argmax(below))
    if first == 0:
        return float(ns[0])
    n_hi, n_lo = ns[first - 1], ns[first]
    v_hi, v_lo = np.log(vs[first - 1] + 1e-300), np.log(vs[first] + 1e-300)
    f = np.log(floor)
    return float(n_hi + (n_lo - n_hi) * (v_hi - f) / (v_hi - v_lo))
