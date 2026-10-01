# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-18  Created. Ports P4 of the old metrics_part1.py. The generators are
#               no longer hardcoded for A1: each family declares its own set in
#               ansatze/, and the closure is grown by the shared probes.lie.

"""P4 - dimension of the dynamical Lie algebra of the ansatz.

Definition
----------
Close the family's gate generators under commutation (see :mod:`..probes.lie`).
The dimension of that Lie algebra is ``dla_dim``. It counts every direction the
circuit can reach at any depth, so it is the reachable set of the family. The
full special unitary algebra ``su(2**n)`` has dimension ``4**n - 1``, reported by
:func:`reference`.

Why it matters for the study
----------------------------
P4 is the sharpest structural predictor of a barren plateau in the whole battery.
The gradient variance of a cost falls like ``1 / dim(g)`` (Ragone et al. 2024;
Larocca et al. 2022). A DLA that grows to ``4**n - 1`` marks the exponentially
untrainable regime; a DLA that grows as a polynomial in ``n`` does not. P4 gives
this verdict exactly, from the circuit structure alone, with no sampling and no
gradient run.

Properties the reader must know
-------------------------------
* P4 is exact. It does not use the seed, the sample count, the depth or the
  encoder. Two runs give the same integer.
* P4 does not depend on depth. Repeating the same layer adds nothing to the Lie
  closure. Sweep depth for the other metrics; for P4 one depth is enough, and
  the record stores the value at that depth.
* P4 is exponential in ``n`` to compute whenever the DLA is full. The width is
  capped by :data:`DIM_CAP`. Above the cap the value is a lower bound, marked by
  ``dla_capped``.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..ansatze.base import Ansatz
from ..probes.lie import lie_closure_dimension
from .base import Metric

DIM_CAP: int = 2000
"""Largest Lie-closure dimension the sweep will grow before it stops. It bounds
both memory and time. A full su(2**n) exceeds it from n = 6, so full-algebra
families are exact only up to n = 5 and are lower-bounded above that."""


def measure(
    ansatz: Ansatz,
    n: int,
    d: int,
    samples: int,
    rng=None,
    encoder_input=None,
) -> Mapping[str, float]:
    """Measure P4 once. The result does not depend on the seed or the depth.

    Args:
        ansatz: The family to measure. It must declare a generator set through
            ``ansatz.generators``.
        n: Number of qubits.
        d: Depth. Unused; the DLA does not depend on it. Accepted for one common
            signature.
        samples: Unused. Accepted for one common signature.
        rng: Unused. Accepted for one common signature.
        encoder_input: Unused. Accepted for one common signature.

    Returns:
        A mapping with two values:

        * dla_dim - the dimension of the dynamical Lie algebra. This is P4.
        * dla_capped - 1.0 when the search stopped at :data:`DIM_CAP`, so
          dla_dim is a lower bound; 0.0 when the algebra closed exactly.

    Raises:
        ValueError: If the family declares no generator set.
    """
    del d, samples, rng, encoder_input  # P4 depends only on the family and n.
    if ansatz.generators is None:
        raise ValueError(f"family {ansatz.tag} declares no DLA generators")
    dimension, capped = lie_closure_dimension(ansatz.generators(n), dim_cap=DIM_CAP)
    return {"dla_dim": float(dimension), "dla_capped": 1.0 if capped else 0.0}


def reference(n: int, samples: int) -> Mapping[str, float]:
    """Return the dimension of the full special unitary algebra su(2**n).

    Args:
        n: Number of qubits.
        samples: Unused. Accepted so that every metric has one signature.

    Returns:
        A mapping with dla_su_full = 4**n - 1.
    """
    del samples
    return {"dla_su_full": float(4 ** n - 1)}


P4 = Metric(
    tag="P4",
    name="Dynamical Lie algebra dimension",
    unit="dimension of the reachable algebra; full su(2^n) is 4^n - 1",
    source="Larocca et al., Quantum 6, 824 (2022); barren-plateau scaling from "
           "Ragone et al., Nat. Commun. 15, 7172 (2024)",
    measure=measure,
    reference=reference,
    reference_kind="analytic",
    deviations=(
        "The generator set of each family is declared in its ansatze/ file. A "
        "fixed entangler is represented by the non-trivial Pauli terms of its "
        "generator; for the CNOT chain of A1 this is Z_q X_(q+1), which is also "
        "the native cross-resonance generator of the Kandala hardware.",
        "The Lie-closure dimension is invariant under the fixed single-qubit "
        "Clifford gates of a circuit, so those gates are not dressed onto the "
        "generators. Only the entangling structure they induce is included.",
        "The dimension is exact only up to DIM_CAP. Above it the value is a lower "
        "bound, marked by dla_capped, because a full algebra is exponential in n.",
    ),
)
