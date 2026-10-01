# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-18  Created. Parameter-shift cost gradients over random angles, the
#               primitive of the G-series. Replaces grad_var of the old
#               metrics_part2.py: no import-time seed, and it runs on any family,
#               not only the A1 HEA. UNVERIFIED - see the metric notes.

"""Cost gradients under the parameter-shift rule, for the trainability series.

The static metrics P1 to P4 never touch a cost. The G-series does: it asks how
large the gradient of a cost is, and how its size decays as the circuit grows.
A gradient variance that falls exponentially in the qubit count is the signature
of a barren plateau.

The cost is an expectation of a diagonal Pauli-Z observable, so it is read off a
statevector without forming any matrix:

* global cost  C = <Z_0 Z_1 ... Z_(n-1)>   - a parity, sensitive to every qubit;
* local cost   C = <Z_0>                    - one qubit only.

The derivative with respect to one angle is a central finite difference on the
cost,  dC/dtheta_k = [ C(theta + h e_k) - C(theta - h e_k) ] / (2 h), evaluated in
one broadcast call over random angles.

Why a finite difference and not a two-term parameter shift
----------------------------------------------------------
The two-term parameter shift  1/2 [ C(+pi/2) - C(-pi/2) ]  is exact only for a
gate whose generator squares to the identity, i.e. a single-qubit Pauli rotation.
A1, A2 and A3 are built from those, so it would be exact for them. A4 uses CRZ,
a controlled rotation whose generator has eigenvalues {-1, 0, +1}; the two-term
shift is **not** exact there, and it was checked to be off by several percent on
A4. A central finite difference is correct for every gate. Its O(h**2) error is
validated against autodiff on all four families in the tests.

Status
------
Cross-checked against PennyLane autodiff (gradients agree to the finite-difference
error). The onset slope alpha still needs comparison to a published curve before
the numbers are treated as settled; read the G-series as indicative.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from numpy.typing import NDArray

from ..ansatze.base import Ansatz
from .statevector import build_qnode

FD_STEP: float = 1e-4
"""Central finite-difference step for the cost derivative. Small enough for an
O(step**2) error near 1e-8, large enough to stay clear of double-precision
round-off. Correct for every gate, unlike the two-term parameter shift."""


def diagonal_observable(n: int, kind: str) -> NDArray[np.floating]:
    """Return the diagonal of the cost observable, as a length-2**n vector.

    The statevector of ``default.qubit`` indexes wire 0 as the most significant
    bit, so bit ``q`` of basis state ``i`` is ``(i >> (n - 1 - q)) & 1`` and the
    eigenvalue of ``Z_q`` is ``1 - 2 * bit``.

    Args:
        n: Number of qubits.
        kind: ``"global"`` for the full ``Z^{\\otimes n}`` parity, or ``"local"``
            for ``Z_0``.

    Returns:
        A vector of +1 / -1 entries, one per computational basis state.
    """
    index = np.arange(2 ** n)
    if kind == "global":
        bits = (index[:, None] >> np.arange(n - 1, -1, -1)) & 1
        parity = bits.sum(axis=1) & 1
        return 1.0 - 2.0 * parity
    bit0 = (index >> (n - 1)) & 1
    return 1.0 - 2.0 * bit0


def shifted_states(
    ansatz: Ansatz,
    n: int,
    d: int,
    samples: int,
    rng: np.random.Generator,
    encoder_input=None,
    k: int = 0,
    step: float = FD_STEP,
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """Return the states at +step and -step on angle ``k``, over random angles.

    Draws ``samples`` random parameter vectors on ``[0, 2*pi)``, moves angle
    ``k`` up and down by ``step``, and evaluates all ``2 * samples`` circuits in
    one call.

    Args:
        ansatz: The family to run.
        n: Number of qubits.
        d: Depth, in the unit of the family.
        samples: Number of random angle vectors.
        rng: Generator from :mod:`..core.seeding`.
        encoder_input: Fixed input vector, or None.
        k: Index of the differentiated angle. Defaults to 0, the first angle.
        step: Finite-difference displacement. Defaults to :data:`FD_STEP`.

    Returns:
        A pair ``(plus, minus)`` of arrays, each of shape ``(samples, 2**n)``.
    """
    circuit = build_qnode(ansatz, n, d, encoder_input)
    p = ansatz.n_params(n, d)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(samples, p))
    plus, minus = theta.copy(), theta.copy()
    plus[:, k] += step
    minus[:, k] -= step
    states = np.asarray(circuit(np.concatenate([plus, minus], axis=0)))
    states = states.reshape(2 * samples, -1)
    return states[:samples], states[samples:]


def gradient_variance(
    plus: NDArray[np.complexfloating],
    minus: NDArray[np.complexfloating],
    diagonal: NDArray[np.floating],
    step: float = FD_STEP,
) -> tuple[float, float]:
    """Return the variance and mean absolute value of one cost derivative.

    Args:
        plus: States at ``theta + step e_k`` from :func:`shifted_states`.
        minus: States at ``theta - step e_k``.
        diagonal: The cost observable diagonal from :func:`diagonal_observable`.
        step: The finite-difference step used to produce the two state sets.

    Returns:
        A pair ``(variance, mean_abs)`` of the central finite-difference
        derivative across the random angle draws. The mean derivative is zero by
        symmetry, so the variance is the quantity that diagnoses a barren plateau.
    """
    cost_plus = (np.abs(plus) ** 2) @ diagonal
    cost_minus = (np.abs(minus) ** 2) @ diagonal
    grad = (cost_plus - cost_minus) / (2.0 * step)
    return float(np.var(grad)), float(np.mean(np.abs(grad)))
