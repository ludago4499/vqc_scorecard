# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-09-01  Created. Noise-induced barren plateau, measured. Replaces the
#               old G5 of metrics_part2.py, which swept 5 noise values at one
#               (n, d) point, never swept depth (the axis Wang's bound is
#               exponential in), quantised p_crit to the swept values, and
#               silently scored an uncrossed floor as perfect robustness.
#               Here depth is a grid axis, p_crit is interpolated, and there
#               is no score. Noise model is PER-GATE depolarizing - a declared
#               deviation from the per-layer source model, see below.

"""G5 - noise-induced barren plateau: the gradient under depolarizing noise.

The claim being tested
----------------------
Wang, Fontana, Cerezo, Sharma, Sone, Cincio & Coles, Nat. Commun. 12, 6961
(2021), Theorem 1: under local Pauli noise of strength q per layer, every cost
derivative is bounded as ``|dC| <= F(n) * q^(L+1)`` - exponential decay in the
DEPTH L. Larocca et al. 2025, Box 8 give the closed form for global
depolarizing noise:

    loss = (1-p)^L * Tr[U rho U^dag O] + (1 - (1-p)^L) * Tr[O] / 2^n,

so the whole landscape contracts deterministically toward the maximally mixed
value. This is a different object from the randomness-induced plateau of G1:
it flattens every point, not just most of them, and error mitigation is not
expected to undo it. Depth is therefore the axis this metric must sweep - the
one axis the old implementation held fixed.

Noise model (declared deviation)
--------------------------------
A ``DepolarizingChannel(p)`` after **every gate** on the wires it touches, not
one channel per layer as in the source. The reason and both consequences are
documented in :mod:`..probes.noise`: a "layer" is not the same object across
the four families, while a gate is; per-gate noise makes gate-heavy circuits
noisier at equal (n, d), which couples G5 to the cost control P6 and is the
NISQ-faithful reading; and the depth decay stays exponential, only with an
effective rate set by gates-per-depth-unit rather than by p alone.

What is measured at one grid point
----------------------------------
The global-cost gradient variance (same estimator as G1: central finite
difference, pooled over a spread of angle indices) at every noise level in
:data:`NOISE_LEVELS`, including the noiseless control p = 0, which must match
the G1 statevector value - that agreement is a permanent test. The readings
``noise_crossing`` (interpolated p_crit at fixed (n, d)) and
``noise_depth_rate`` (fitted decay rate vs depth at fixed p) are pure
functions of finished records, like every other G reading.

Cost warning
------------
Density-matrix simulation, one call per sample. This is the expensive metric
of the package: keep the grid to small n and use the checkpoint - the sweep
resumes, so a stopped run loses nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..ansatze.base import Ansatz
from ..probes.noise import noisy_gradient_variance
from .base import Metric
from .g1_gradient import SHOTS_FLOOR, _angle_indices

NOISE_LEVELS: tuple[float, ...] = (0.0, 1e-3, 1e-2, 1e-1)
"""Per-gate depolarizing probabilities swept at every grid point. p = 0 is the
noiseless control that ties G5 to G1. The three finite values are one per
decade, bracketing reported NISQ two-qubit error rates (~1e-3..1e-2)."""

N_ANGLES_G5: int = 3
"""Angle indices pooled per variance. Fewer than G1's 6 because every noisy
evaluation is a full density-matrix simulation; 3 spread indices still remove
the single-angle degeneracies that motivated averaging in G1."""


def _g5_angle_indices(p: int) -> list[int]:
    """Return up to :data:`N_ANGLES_G5` indices spread across the circuit."""
    full = _angle_indices(p)
    if len(full) <= N_ANGLES_G5:
        return full
    picks = np.linspace(0, len(full) - 1, N_ANGLES_G5).round().astype(int)
    return [full[i] for i in dict.fromkeys(picks.tolist())]


def measure(
    ansatz: Ansatz,
    n: int,
    d: int,
    samples: int,
    rng: np.random.Generator,
    encoder_input=None,
) -> Mapping[str, float]:
    """Measure the noisy gradient variance at one grid point, one seed.

    Every noise level draws its own angles from the shared generator, so the
    p = 0 column is an independent re-estimate, not a copy of a G1 record.

    Args:
        ansatz: The family to measure.
        n: Number of qubits.
        d: Depth, in the unit of the family.
        samples: Random angle draws per noise level. Keep small (~70); every
            draw is two density-matrix simulations per angle index.
        rng: Generator from :mod:`..core.seeding`.
        encoder_input: Fixed input vector, or None.

    Returns:
        A mapping with, for every level index i: ``noise_si`` (the swept p),
        ``gradvar_global_si`` and ``gradabs_global_si``.
    """
    indices = _g5_angle_indices(ansatz.n_params(n, d))
    out: dict[str, float] = {}
    for i, p_noise in enumerate(NOISE_LEVELS):
        var, mean_abs = noisy_gradient_variance(
            ansatz, n, d, p_noise, samples, rng, indices, encoder_input)
        out[f"noise_s{i}"] = float(p_noise)
        out[f"gradvar_global_s{i}"] = var
        out[f"gradabs_global_s{i}"] = mean_abs
    return out


def reference(n: int, samples: int) -> Mapping[str, float]:
    """Return the finite-shot floor, the same convention G1 and G3 use."""
    del n, samples
    return {"shot_floor": SHOTS_FLOOR}


def noise_crossing(
    noise_levels: Sequence[float],
    variances: Sequence[float],
    floor: float = SHOTS_FLOOR,
) -> float | None:
    """Return the interpolated noise strength where the variance crosses the floor.

    Log-log interpolation between the last swept p above the floor and the
    first below it, so p_crit is not quantised to the sweep - the exact defect
    the old G5 audit named. The p = 0 control is excluded from interpolation.

    Args:
        noise_levels: Swept per-gate probabilities (may include 0).
        variances: Matching global gradient variances.
        floor: The finite-shot floor.

    Returns:
        The crossing p, or ``None`` when the variance stays above the floor
        for every finite swept p (report that as "not crossed in range",
        never as robustness - the old scorecard's silent 100).
    """
    pairs = sorted((p, v) for p, v in zip(noise_levels, variances) if p > 0)
    ps = np.array([p for p, _ in pairs])
    vs = np.array([v for _, v in pairs])
    below = vs < floor
    if not below.any():
        return None
    first = int(np.argmax(below))
    if first == 0:
        return float(ps[0])
    lp_hi, lp_lo = np.log(ps[first - 1]), np.log(ps[first])
    lv_hi, lv_lo = np.log(vs[first - 1] + 1e-300), np.log(vs[first] + 1e-300)
    f = np.log(floor)
    return float(np.exp(lp_hi + (lp_lo - lp_hi) * (lv_hi - f) / (lv_hi - lv_lo)))


def noise_depth_rate(depths: Sequence[int], variances: Sequence[float]) -> float:
    """Return the fitted depth-decay rate beta of ``Var ~ exp(-beta d)``.

    The axis Wang's theorem names: at fixed noise, the log-variance must fall
    linearly in depth. Positive beta is decay; the noiseless control should
    give beta near 0 by comparison (G2's flat retention curve).

    Args:
        depths: Depths, in any order.
        variances: Matching global gradient variances at one noise level.

    Returns:
        The fitted beta.
    """
    order = np.argsort(depths)
    x = np.asarray(depths, dtype=float)[order]
    y = np.log(np.asarray(variances, dtype=float)[order] + 1e-300)
    return float(-np.polyfit(x, y, 1)[0])


G5 = Metric(
    tag="G5",
    name="Noise-induced barren plateau (gradient variance under depolarizing noise)",
    unit="variance of one finite-difference cost derivative over random angles, "
         "per noise level",
    source="Wang et al., Nat. Commun. 12, 6961 (2021); closed form per Larocca "
           "et al. 2025, Box 8",
    measure=measure,
    reference=reference,
    reference_kind="none",
    deviations=(
        "PER-GATE NOISE, NOT PER-LAYER. The source model is one local Pauli "
        "channel per layer; here a DepolarizingChannel(p) follows every gate, "
        "because 'layer' is not the same object across the four families. "
        "Gate-heavy circuits are therefore noisier at equal (n, d) - the "
        "coupling to P6 is deliberate and NISQ-faithful. The depth decay "
        "remains exponential; only the effective rate changes.",
        "Depolarizing only. Bit-flip and phase-flip channels are not swept.",
        "The rotation angles are noiseless and there is no readout or shot "
        "noise; the simulation is analytic on a density matrix.",
        "p_crit is interpolated between swept noise levels (log-log), and an "
        "uncrossed floor is reported as None, never as a robustness score - "
        "both fixes to defects named on the old G5 slide.",
        "The angle pool is 3 indices, not G1's 6, to bound the density-matrix "
        "cost. The p = 0 column must match G1; that agreement is a test.",
    ),
)
