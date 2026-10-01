# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-18  Created. Parameter count. The simplest descriptor and the
#               denominator of P3's redundancy.

"""P5 - number of variational parameters of the circuit.

Definition
----------
``M = n_params(n, d)``, the number of free angles the family exposes at a grid
point. It is exact and needs no measurement.

Why it is here
--------------
P5 is not a capability, it is a cost. It sets the classical optimisation load and
it is the denominator of Haug's redundancy R = (M - D_C) / M, where D_C is the
effective dimension of P3. Two families with the same effective dimension but
different M differ only in wasted parameters, and P5 is what makes that visible.
Read it next to P3, never on its own.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..ansatze.base import Ansatz
from .base import Metric


def measure(ansatz: Ansatz, n: int, d: int, samples: int, rng=None,
            encoder_input=None) -> Mapping[str, float]:
    """Return the parameter count. Exact; independent of seed and samples.

    Args:
        ansatz: The family to measure.
        n: Number of qubits.
        d: Depth, in the unit of the family.
        samples: Unused. Accepted so every metric shares one signature.
        rng: Unused. Accepted for the common signature.
        encoder_input: Unused. Accepted for the common signature.

    Returns:
        A mapping with n_params_p, the number of variational angles.
    """
    del samples, rng, encoder_input
    return {"n_params_p": float(ansatz.n_params(n, d))}


P5 = Metric(
    tag="P5",
    name="Parameter count",
    unit="number of variational angles M = n_params(n, d)",
    source="Circuit definition; used as the denominator of Haug's redundancy "
           "R = (M - D_C) / M, Haug, Bharti & Kim, PRX Quantum 2, 040309 (2021)",
    measure=measure,
    reference=None,
    reference_kind="none",
    deviations=(
        "P5 is a cost control, not a capability. It is reported so that P3's "
        "effective dimension can be normalised into a redundancy.",
        "The value duplicates the n_params column that every record already "
        "carries; it is exposed as a metric so it can be run and correlated "
        "on its own.",
    ),
)
