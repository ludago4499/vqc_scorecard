# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-18  Created. Gradient-variance grid for the trainability series, on
#               all four families. G1 (barren-plateau onset slope) and G2
#               (gradient retention vs depth) are read from this one grid by the
#               analysis functions below. UNVERIFIED - not yet checked against a
#               published curve; treat the numbers as indicative.

"""G-series - gradient variance, and the G1 / G2 readings of it.

One measurement, two questions
------------------------------
At every grid point this metric measures the variance of a single cost
derivative over random angles, for both a global and a local cost (see
:mod:`..probes.gradient`). That one grid answers both G-series questions the deck
asks:

* **G1, barren-plateau onset.** How fast the global gradient variance decays with
  the qubit count: ``Var[dC] ~ exp(-alpha * n)``. For an exact 2-design
  ``alpha -> ln 2 = 0.693``. :func:`bp_onset_slope` fits ``alpha`` per family.
* **G2, gradient retention vs depth.** How the variance moves as depth grows at a
  fixed width, and the depth at which it falls under the finite-shot floor.
  :func:`depth_retention` reports the curve and that crossing.

Because both are views of the same measured grid, they cannot disagree with each
other, and neither needs its own sweep.

Protocol and its limits
------------------------
* The derivative is a central finite difference of the cost (see
  :mod:`..probes.gradient`). It is correct for every gate, including the
  controlled rotations of A4, and is validated against autodiff in the tests.
  The variance is averaged over a spread of angles across the circuit, so no
  single degenerate angle can dominate.
* The mean derivative is zero by symmetry over random angles, so the variance is
  the whole signal.
* The finite-shot floor is the proxy ``1/1024`` of the old scorecard: a variance
  below it is not resolvable from shot noise at 1024 shots. It is a convention,
  not a measured or analytic quantity.
* **Unverified.** These numbers have not been checked against a published
  gradient-variance curve. Read them as indicative.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..ansatze.base import Ansatz
from ..probes.gradient import FD_STEP, diagonal_observable, gradient_variance
from ..probes.statevector import build_qnode
from .base import Metric

SHOTS_FLOOR: float = 1.0 / 1024
"""Finite-shot gradient-variance floor. A variance below it is treated as
indistinguishable from shot noise at 1024 shots. A convention, not a measurement."""

N_ANGLES: int = 6
"""How many angles the variance is averaged over. A single angle can be
degenerate at shallow depth (some gradients are exactly zero there), which makes
one number unrepresentative. Averaging over a spread of angles gives the typical
gradient scale, as in McClean 2018, and removes those zeros."""


def _angle_indices(p: int) -> list[int]:
    """Return up to :data:`N_ANGLES` angle indices spread across the circuit.

    Evenly spaced from the first to the last angle, so the average samples early,
    middle and late gates rather than one arbitrary position.
    """
    if p <= N_ANGLES:
        return list(range(p))
    return sorted({int(round(i)) for i in np.linspace(0, p - 1, N_ANGLES)})


def measure(
    ansatz: Ansatz,
    n: int,
    d: int,
    samples: int,
    rng: np.random.Generator,
    encoder_input=None,
) -> Mapping[str, float]:
    """Measure the cost-gradient variance at one grid point, from one seed.

    The two shifted circuits are evaluated once and reused for both the global
    and the local cost, so the two variances share the same random angles.

    Args:
        ansatz: The family to measure.
        n: Number of qubits.
        d: Depth, in the unit of the family.
        samples: Number of random angle draws behind each variance.
        rng: Generator from :mod:`..core.seeding`.
        encoder_input: Fixed input vector, or None.

    Returns:
        A mapping with the variance and the mean absolute gradient of the global
        and the local cost, each averaged over the sampled angles: gradvar_global,
        gradvar_local, gradabs_global, gradabs_local.
    """
    p = ansatz.n_params(n, d)
    circuit = build_qnode(ansatz, n, d, encoder_input)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(samples, p))  # one base, shared by all angles
    diag_global = diagonal_observable(n, "global")
    diag_local = diagonal_observable(n, "local")

    var_g, var_l, abs_g, abs_l = [], [], [], []
    for k in _angle_indices(p):
        plus, minus = theta.copy(), theta.copy()
        plus[:, k] += FD_STEP
        minus[:, k] -= FD_STEP
        states = np.asarray(circuit(np.concatenate([plus, minus], axis=0))).reshape(2 * samples, -1)
        upper, lower = states[:samples], states[samples:]
        vg, ag = gradient_variance(upper, lower, diag_global)
        vl, al = gradient_variance(upper, lower, diag_local)
        var_g.append(vg); abs_g.append(ag); var_l.append(vl); abs_l.append(al)

    return {
        "gradvar_global": float(np.mean(var_g)),
        "gradvar_local": float(np.mean(var_l)),
        "gradabs_global": float(np.mean(abs_g)),
        "gradabs_local": float(np.mean(abs_l)),
    }


def reference(n: int, samples: int) -> Mapping[str, float]:
    """Return the finite-shot floor proxy, the same at every grid point.

    Args:
        n: Number of qubits. Unused; the floor is width-independent here.
        samples: Unused. Accepted for the common signature.

    Returns:
        A mapping with shot_floor.
    """
    del n, samples
    return {"shot_floor": SHOTS_FLOOR}


# --------------------------------------------------------------------------- #
# G1 and G2 are read from the measured grid by these functions. They take the
# finished records (or a simple dict) so they can be run at deck-build time
# without touching the simulator again.
# --------------------------------------------------------------------------- #

def bp_onset_slope(widths: Sequence[int], variances: Sequence[float]) -> float:
    """Return the barren-plateau onset slope alpha of ``Var ~ exp(-alpha n)``.

    Fits a straight line to ``log(Var)`` against the qubit count and returns the
    negative of its slope. ``alpha`` near ``ln 2 = 0.693`` means the family
    decays like an exact 2-design; larger means faster onset.

    Args:
        widths: Qubit counts, in any order.
        variances: The matching global gradient variances.

    Returns:
        The fitted alpha. Positive when the variance decays with width.
    """
    order = np.argsort(widths)
    x = np.asarray(widths, dtype=float)[order]
    y = np.log(np.asarray(variances, dtype=float)[order] + 1e-18)
    slope = np.polyfit(x, y, 1)[0]
    return float(-slope)


def depth_retention(
    depths: Sequence[int], variances: Sequence[float], floor: float = SHOTS_FLOOR
) -> dict[str, object]:
    """Return the depth curve of the variance and the first sub-floor depth.

    Args:
        depths: Depths, in any order.
        variances: The matching global gradient variances.
        floor: The shot floor to cross. Defaults to :data:`SHOTS_FLOOR`.

    Returns:
        A dict with the sorted depths, the variances in that order, and
        ``crossing_depth``, the first depth whose variance is below the floor, or
        ``None`` when the variance never falls under it in the swept range.
    """
    order = np.argsort(depths)
    ds = [int(depths[i]) for i in order]
    vs = [float(variances[i]) for i in order]
    crossing = next((ds[i] for i, v in enumerate(vs) if v < floor), None)
    return {"depths": ds, "variances": vs, "crossing_depth": crossing}


G1 = Metric(
    tag="G1",
    name="Gradient variance (barren-plateau onset G1, depth retention G2)",
    unit="variance of one finite-difference cost derivative over random angles",
    source="McClean et al., Nat. Commun. 9, 4812 (2018); Cerezo et al., Nat. "
           "Commun. 12, 1791 (2021); definitions per Larocca et al. 2025",
    measure=measure,
    reference=reference,
    reference_kind="none",
    deviations=(
        "PARTIALLY VERIFIED. The gradient is checked against PennyLane autodiff on "
        "all four families (the finite difference agrees to its O(step**2) error), "
        "and A1's onset slope reproduces the old HEA value (alpha ~ 0.71). The "
        "cross-family onset slopes are not yet checked against a published curve.",
        "The derivative is a central finite difference, not a two-term parameter "
        "shift: the two-term shift is exact only for Pauli rotations and is wrong "
        "for A4's controlled CRZ gates (checked: off by ~7%). See probes.gradient.",
        "The variance is averaged over N_ANGLES angles spread across the circuit, "
        "not one fixed angle. A single angle can be exactly degenerate at shallow "
        "depth (zero gradient), which the average removes.",
        "Depth and width are swept independently, unlike the old code which tied "
        "d = n for G1. The onset slope is therefore fitted at a fixed depth; the "
        "d = n diagonal is available in the grid if that comparison is wanted.",
        "The shot floor 1/1024 is a convention carried from the old scorecard, "
        "not a measured or analytic value.",
    ),
)
