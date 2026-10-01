# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-11  Migrated from the p1_expressibility package. The 0-100 score is
#               removed. The sampled Haar floor is kept, and reported as a
#               reference rather than used to rescale.

"""P1 - expressibility, by KL divergence against the Haar law.

Definition
----------
Draw two independent parameter sets, build both states, and take the fidelity
F = |<a|b>|^2. Repeat, and histogram F. Then

    Expr = D_KL( P_circuit(F) || P_Haar(F) ),

with P_Haar(F) = (D - 1)(1 - F)^(D - 2) and D = 2**n.

0 means the circuit is indistinguishable from Haar. Larger means the circuit
reaches a smaller part of the space.

The estimator is biased, and the bias does not average out
----------------------------------------------------------
With a finite number of pairs the circuit histogram is noisy, and noise can only
raise the KL divergence. A perfect Haar-random circuit therefore returns a value
above zero. More seeds do not remove this. Only more pairs do.

The size of that offset is measured by :func:`reference` and reported next to
every value as kl_haar_floor. Any P1 value at or below its floor is not
distinguishable from Haar, whatever the number itself says.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..ansatze.base import Ansatz
from ..core.seeding import generator
from ..probes.fidelity import sample_fidelities
from ..probes.haar import fidelity_bin_probabilities
from ..probes.haar import sample_fidelities as sample_haar_fidelities
from .base import Metric

DEFAULT_BINS: int = 75
"""Histogram bins on [0, 1]. The value used by Sim et al. 2019."""

REFERENCE_SEEDS: int = 5
"""Seeds averaged when measuring the Haar floor. One seed is not enough."""


def kl_against_haar(fidelities: np.ndarray, n: int, bins: int = DEFAULT_BINS) -> float:
    """Return the KL divergence of a fidelity sample against the Haar law.

    Args:
        fidelities: Sample of fidelities in [0, 1].
        n: Number of qubits. It sets the dimension 2**n.
        bins: Number of equal-width bins on [0, 1].

    Returns:
        The divergence in nats. It is finite and never negative.
    """
    counts, _ = np.histogram(np.clip(fidelities, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    p = counts / counts.sum()
    q = fidelity_bin_probabilities(bins, 2 ** n)
    occupied = p > 0
    # Empty bins contribute nothing, because p log p tends to zero with p.
    # The floor on q prevents a division by zero in the far tail at large n.
    return float(np.sum(p[occupied] * np.log(p[occupied] / np.maximum(q[occupied], 1e-300))))


def measure(
    ansatz: Ansatz,
    n: int,
    d: int,
    samples: int,
    rng: np.random.Generator,
    encoder_input=None,
) -> Mapping[str, float]:
    """Measure P1 once, from one seed.

    Args:
        ansatz: The family to measure.
        n: Number of qubits.
        d: Depth, in the unit of the family.
        samples: Number of fidelity pairs.
        rng: Generator from :mod:`..core.seeding`.
        encoder_input: Fixed input vector, or None.

    Returns:
        A mapping with kl_nats, the divergence for this seed.
    """
    fidelities = sample_fidelities(ansatz, n, d, samples, rng, encoder_input)
    return {"kl_nats": kl_against_haar(fidelities, n)}


def reference(n: int, samples: int) -> Mapping[str, float]:
    """Return the measured sampling floor of the estimator.

    True Haar-random states are drawn and passed through the same estimator, at
    the same qubit count, pair count and bin count. The result is the smallest
    value any circuit can reach. It is averaged over :data:`REFERENCE_SEEDS`
    seeds, and its own spread is returned as well.

    Args:
        n: Number of qubits.
        samples: Number of fidelity pairs. The floor depends on it strongly.

    Returns:
        A mapping with kl_haar_floor and kl_haar_floor_std.
    """
    floors = []
    for repeat in range(REFERENCE_SEEDS):
        rng = generator("P1", "haar-floor", n, samples, repeat)
        floors.append(kl_against_haar(sample_haar_fidelities(samples, 2 ** n, rng), n))
    return {
        "kl_haar_floor": float(np.mean(floors)),
        "kl_haar_floor_std": float(np.std(floors, ddof=1)),
    }


P1 = Metric(
    tag="P1",
    name="Expressibility (KL divergence against Haar)",
    unit="nats, 0 = indistinguishable from Haar, larger = less uniform coverage",
    source="Sim, Johnson & Aspuru-Guzik, Adv. Quantum Technol. 2, 1900070 (2019)",
    measure=measure,
    reference=reference,
    reference_kind="sampled",
    deviations=(
        "The Haar term is integrated exactly per bin through the cumulative "
        "form, not read at bin centres. This is more accurate than the "
        "reference implementation.",
        "The estimator is positively biased at finite pair count. Compare every "
        "value against kl_haar_floor before reading it as physics.",
        "A fixed unitary at the end of a circuit cancels inside the fidelity, "
        "so P1 is blind to a terminal entangling layer.",
        "The encoder input is held fixed, so P1 describes the ansatz and not "
        "the encoder.",
    ),
)
