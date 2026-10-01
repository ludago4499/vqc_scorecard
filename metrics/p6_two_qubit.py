# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-18  Created. Two-qubit gate count, read from the circuit tape so it
#               follows any change to a family's apply() with no separate formula.
#   2026-10-01  Operations on more than two wires are decomposed before the
#               count (probes.gates.elementary), so the amplitude encoding of
#               A5 is counted as its Mottonen CNOTs. Families built from one-
#               and two-wire gates only (A1-A4, A6, A7) give the same numbers.

"""P6 - number of two-qubit gates in the circuit.

Definition
----------
Queue the family's gates on a tape and count the operations that act on two
wires. This is read from the circuit itself, not from a hand formula, so it
cannot drift away from what apply() actually builds.

For the four families the count is:

* A1 HEA: a CNOT chain per layer, ``d (n - 1)``.
* A2 SEL: a CNOT ring per layer, ``d n``.
* A3 re-uploading: a CNOT ring between repeats, ``(d - 1) n``.
* A4 DisCoCat: a CRZ ring per word box plus the cup CNOTs, ``d n + n / 2``.
* A5 circuit-centric: ``n`` CRot per block, ``d n``, plus the CNOTs of the
  amplitude encoding (Mottonen decomposition: ``2**n - 2`` at a generic
  input for ``n >= 3``, one CNOT at ``n = 2``).
* A6 IQP: the ZZ phases of the feature map, ``2 (n - 1)``, plus a CZ chain per
  layer, ``d (n - 1)``.
* A7 Farhi-Neven: two Pauli rotations per data wire per layer, ``2 d (n - 1)``.

An operation on more than two wires (the ``StatePrep`` of an amplitude
encoding) is decomposed first, with :func:`~vqc_scorecard.probes.gates.elementary`.

Why it is here
--------------
Two-qubit gates are the dominant error source on NISQ hardware, so P6 is the
metric that connects a circuit to its noise budget. Like P5 it is a cost, not a
capability: a family that reaches the same expressibility with fewer two-qubit
gates is cheaper to run, and P6 is what quantifies that.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pennylane as qml

from ..ansatze.base import Ansatz
from ..probes.gates import elementary
from .base import Metric


def two_qubit_count(ansatz: Ansatz, n: int, d: int, encoder_input=None) -> int:
    """Return the number of two-qubit gates the family queues at (n, d).

    The circuit is built once on a light-weight tape with zero angles. Only the
    gate structure matters for the count, so the angle values are irrelevant.

    Args:
        ansatz: The family to inspect.
        n: Number of qubits.
        d: Depth, in the unit of the family.
        encoder_input: Fixed input vector for families that encode data, or None.

    Returns:
        The count of operations acting on exactly two wires.
    """
    params = np.zeros(ansatz.n_params(n, d))
    script = qml.tape.make_qscript(lambda: ansatz.apply(params, n, d, encoder_input))()
    return sum(1 for op in elementary(script.operations) if len(op.wires) == 2)


def measure(ansatz: Ansatz, n: int, d: int, samples: int, rng=None,
            encoder_input=None) -> Mapping[str, float]:
    """Return the two-qubit gate count. Exact; independent of seed and samples.

    Args:
        ansatz: The family to measure.
        n: Number of qubits.
        d: Depth, in the unit of the family.
        samples: Unused. Accepted so every metric shares one signature.
        rng: Unused. Accepted for the common signature.
        encoder_input: Fixed input vector, or None.

    Returns:
        A mapping with two_qubit_gates, the count.
    """
    del samples, rng
    return {"two_qubit_gates": float(two_qubit_count(ansatz, n, d, encoder_input))}


P6 = Metric(
    tag="P6",
    name="Two-qubit gate count",
    unit="number of two-qubit gates on the tape",
    source="Circuit definition; two-qubit gates are the dominant NISQ error source",
    measure=measure,
    reference=None,
    reference_kind="none",
    deviations=(
        "P6 is a cost control, not a capability. It stands in for the hardware "
        "error budget, since two-qubit gates dominate NISQ error rates.",
        "The count is read from the tape, so it reflects the CNOT chain of A1, "
        "the CNOT rings of A2 and A3 and the CRZ ring plus cups of A4 exactly as "
        "apply() queues them, with no separate formula to keep in step.",
    ),
)
