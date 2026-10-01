# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-11  Created. Replaces p2_entangling of the old metrics_part1.py.
#               The n = 2 only anchor of 0.4 is replaced by the analytic Haar
#               expectation at every n.

"""P2 - entangling capability, by the Meyer-Wallach measure.

Definition
----------
For a pure state of n qubits,

    Q = 2 * ( 1 - (1/n) * sum_k Tr[rho_k^2] ),

where rho_k is the reduced state of qubit k. This is the form of Brennen
2003, which is equal to the original measure of Meyer and Wallach 2002 but needs
no explicit wedge product.

Q = 0 means every qubit is in a product with the rest. Q = 1 means every
qubit is maximally mixed, which is the strongest single-qubit entanglement the
measure can see.

P2 is the average of Q over random parameter draws, as in Sim et al. 2019.

Why the spread matters as much as the mean
------------------------------------------
The old scorecard reported Q = 0.384 at n = 2 with a standard deviation
of 0.323. A spread almost as large as the mean means the ansatz produces
states from almost unentangled to almost maximally entangled, depending on the
angles. The mean alone hides that. Both are returned here.

What Q cannot see
---------------------
Q is built from single-qubit reduced states only. It is blind to genuine
multipartite structure. A W state and a GHZ state of three qubits have different
entanglement structure and the same Q. Do not read P2 as a complete
description of entanglement.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..ansatze.base import Ansatz
from ..probes.haar import expected_meyer_wallach
from ..probes.purity import single_qubit_purities
from ..probes.statevector import sample_states
from .base import Metric


def meyer_wallach(states: np.ndarray, n: int) -> np.ndarray:
    """Return the Meyer-Wallach measure of every state in a batch.

    Args:
        states: Array of shape (batch, 2**n). Each row must be normalised.
        n: Number of qubits.

    Returns:
        Array of shape (batch,) with values in [0, 1].

    Examples:
        >>> import numpy as np
        >>> bell = np.zeros((1, 4), dtype=complex)
        >>> bell[0, 0] = bell[0, 3] = 1 / np.sqrt(2)
        >>> float(np.round(meyer_wallach(bell, 2)[0], 6))
        1.0
    """
    purities = single_qubit_purities(states, n)
    return 2.0 * (1.0 - purities.mean(axis=1))


def measure(
    ansatz: Ansatz,
    n: int,
    d: int,
    samples: int,
    rng: np.random.Generator,
    encoder_input=None,
) -> Mapping[str, float]:
    """Measure P2 once, from one seed.

    Args:
        ansatz: The family to measure.
        n: Number of qubits.
        d: Depth, in the unit of the family.
        samples: Number of random parameter draws.
        rng: Generator from :mod:`..core.seeding`.
        encoder_input: Fixed input vector, or None.

    Returns:
        A mapping with three values:

        * mw_mean - the average of Q over the draws. This is P2.
        * mw_std - the spread of Q over the draws.
        * mw_p05, mw_p95 - the 5th and 95th percentiles, which describe
          the spread without assuming it is symmetric.
    """
    states = sample_states(ansatz, n, d, samples, rng, encoder_input)
    q = meyer_wallach(states, n)
    return {
        "mw_mean": float(q.mean()),
        "mw_std": float(q.std(ddof=1)),
        "mw_p05": float(np.percentile(q, 5)),
        "mw_p95": float(np.percentile(q, 95)),
    }


def reference(n: int, samples: int) -> Mapping[str, float]:
    """Return the exact Haar expectation of Q, and the product-state value.

    Neither number is sampled, so neither carries seed noise. This is the reason
    P2 needs no measured floor, unlike P1.

    Args:
        n: Number of qubits.
        samples: Unused. It is accepted so that every metric has one signature.

    Returns:
        A mapping with mw_haar and mw_product.
    """
    del samples  # P2 has no sampling-dependent reference.
    return {"mw_haar": expected_meyer_wallach(n), "mw_product": 0.0}


P2 = Metric(
    tag="P2",
    name="Entangling capability (Meyer-Wallach)",
    unit="dimensionless, 0 = product state, 1 = every qubit maximally mixed",
    source="Meyer & Wallach, J. Math. Phys. 43, 4273 (2002); Brennen, QIC 3, 619 "
           "(2003); averaging protocol from Sim et al., Adv. Quantum Technol. 2, "
           "1900070 (2019)",
    measure=measure,
    reference=reference,
    reference_kind="analytic",
    deviations=(
        "The reference is the analytic Haar expectation at each n. The old "
        "scorecard used 0.4 at every n, which is the correct value at n = 2 "
        "only; the true value rises towards 1 as n grows.",
        "Q uses single-qubit reduced states only, so it cannot distinguish GHZ "
        "from W type entanglement.",
        "The encoder input is held fixed, so P2 describes the ansatz and not "
        "the encoder.",
    ),
)
