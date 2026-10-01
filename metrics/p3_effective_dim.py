# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-18  Created. Ports P3 of the old metrics_part1.py. The import-time
#               generator is gone, the A1-only limit is gone, and the arbitrary
#               rank tolerance is replaced by a fixed threshold reported next to
#               an analytic ceiling.

"""P3 - effective dimension, from the spectrum of the quantum Fisher matrix.

Definition
----------
Build the quantum Fisher information matrix ``F`` at a random parameter point
(see :mod:`..probes.qgt`) and read two counts of how many angle directions
actually move the state:

* ``qfim_rank`` - the numerical rank of ``F``, the number of independent
  directions. A rank far below the angle count ``p`` means the extra angles are
  redundant.
* ``qfim_pr`` - the participation ratio ``(sum lambda)**2 / sum lambda**2`` of
  the spectrum, a curvature-weighted count that is never larger than the rank.

Both are averaged over ``samples`` random parameter points.

The ceiling is analytic
-----------------------
The rank cannot exceed the real dimension of the pure-state manifold, which is
``2**(n+1) - 2`` for ``n`` qubits, and it cannot exceed ``p``. So the achievable
ceiling is ``min(p, 2**(n+1) - 2)``. The manifold term is reported by
:func:`reference`; ``p`` is already in every record as ``n_params``. When the
rank reaches this ceiling the circuit is over-parameterised: every direction the
state manifold allows is reached, and adding angles cannot add capacity.

Why it matters for the study
----------------------------
The over-parameterisation onset - the width or depth at which the rank saturates
- is where the loss landscape loses its spurious minima (Larocca et al. 2023).
P3 therefore locates that onset per family, and it complements P1: P1 says how
uniformly the space is covered, P3 says how many independent knobs reach it.

Cost
----
Each parameter point costs one broadcast device call of ``2p + 1`` states and a
dense ``p`` by ``p`` eigendecomposition. Run it with a small ``--samples`` (the
rank is nearly constant across points); ``30`` is enough.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..ansatze.base import Ansatz
from ..probes.qgt import effective_dimension, quantum_fisher, state_jacobian
from .base import Metric

RANK_TOL: float = 1e-5
"""Relative eigenvalue threshold for the numerical rank. It sits above the
finite-difference floor of the Jacobian and below every active eigenvalue seen
in the sweep, so the rank is stable against both."""


def measure(
    ansatz: Ansatz,
    n: int,
    d: int,
    samples: int,
    rng: np.random.Generator,
    encoder_input=None,
) -> Mapping[str, float]:
    """Measure P3 once, from one seed.

    Args:
        ansatz: The family to measure.
        n: Number of qubits.
        d: Depth, in the unit of the family.
        samples: Number of random parameter points at which the QFIM is built.
        rng: Generator from :mod:`..core.seeding`.
        encoder_input: Fixed input vector, or None.

    Returns:
        A mapping with three values, each averaged over the parameter points:

        * qfim_rank - the mean numerical rank of the QFIM. This is P3, and it is
          Haug's effective quantum dimension D_C = rank(F) at random angles.
        * qfim_pr - the mean participation ratio of the QFIM spectrum.
        * qfim_redundancy - the fraction of angles that do no independent work,
          R = (p - rank) / p = 1 - D_C / M in the notation of Haug's Table I,
          with M = p the angle count. R = 0 means every angle is used; R -> 1
          means most are redundant. It is the normalised reading of the rank.
    """
    ranks, participations = [], []
    for _ in range(samples):
        theta = ansatz.sample_params(rng, 1, n, d)[0]
        psi, jac = state_jacobian(ansatz, n, d, theta, encoder_input)
        rank, participation, _ = effective_dimension(quantum_fisher(psi, jac), RANK_TOL)
        ranks.append(rank)
        participations.append(participation)
    p = ansatz.n_params(n, d)
    mean_rank = float(np.mean(ranks))
    return {
        "qfim_rank": mean_rank,
        "qfim_pr": float(np.mean(participations)),
        "qfim_redundancy": (1.0 - mean_rank / p) if p else 0.0,
    }


def reference(n: int, samples: int) -> Mapping[str, float]:
    """Return the analytic rank ceiling set by the state manifold.

    The pure-state manifold of ``n`` qubits has real dimension
    ``2**(n+1) - 2``. No QFIM on ``n`` qubits can have a larger rank, whatever the
    circuit. The value is exact, so P3 needs no sampled floor.

    Args:
        n: Number of qubits.
        samples: Unused. It is accepted so that every metric has one signature.

    Returns:
        A mapping with qfim_rank_ceiling.
    """
    del samples  # P3 has no sampling-dependent reference.
    return {"qfim_rank_ceiling": float(2 ** (n + 1) - 2)}


P3 = Metric(
    tag="P3",
    name="Effective dimension (QFIM rank)",
    unit="count of independent parameter directions; ceiling min(p, 2^(n+1)-2)",
    source="Haug, Bharti & Kim, PRX Quantum 2, 040309 (2021); over-parameterisation "
           "reading from Larocca et al., Quantum 7, 1116 (2023)",
    measure=measure,
    reference=reference,
    reference_kind="analytic",
    deviations=(
        "The Jacobian of the state is a central finite difference with step 1e-3, "
        "not an exact parameter shift. The O(step**2) error is near 1e-6 and is "
        "validated against a closed-form single-rotation QFIM in the tests.",
        "The rank uses a fixed relative threshold of 1e-5 on the QFIM spectrum. "
        "Report the threshold with any rank, because a rank is threshold-defined.",
        "The reference is the state-manifold dimension 2^(n+1)-2. The full "
        "ceiling is min(p, 2^(n+1)-2); p is the n_params column of the record.",
        "Redundancy R = 1 - rank/p is reported with the rank (Haug Table I). At "
        "shallow depth R is design redundancy; once the rank saturates the "
        "manifold at larger depth, R also counts over-parameterisation, so read "
        "R at d = 1 for the redundancy of the generators alone.",
        "The encoder input is held fixed, so P3 describes the ansatz and not the "
        "encoder.",
    ),
)
