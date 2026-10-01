# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-18  Created. State Jacobian and quantum Fisher information matrix,
#               the primitive of P3. Reuses the batched statevector device of
#               probes.statevector, so P1, P2 and P3 all start from one circuit.
#   2026-09-01  state_jacobian passes an optional fixed prelayer through to
#               build_qnode, for the G6 capacity reading. Default None; P3 is
#               unchanged.

"""The quantum geometric tensor of the ansatz, and the state Jacobian it needs.

Every static metric so far takes the ideal state of the ansatz. P1 takes pairs
of states. P2 takes single-qubit reduced states. P3 takes the **derivative** of
the state with respect to the angles, which is the object this probe adds.

The quantity
------------
For a pure state ``|psi(theta)>`` the quantum Fisher information matrix is

    F_ij = 4 Re[ <d_i psi | d_j psi> - <d_i psi | psi> <psi | d_j psi> ],

where ``|d_k psi>`` is the derivative of the state with respect to angle ``k``.
``F`` is real, symmetric and positive semidefinite. It is four times the
Fubini-Study metric of the state manifold. Its spectrum is what P3 reads.

How the derivative is taken
---------------------------
The derivative is estimated by a central finite difference on the statevector,

    |d_k psi>  ~  ( |psi(theta + eps e_k)> - |psi(theta - eps e_k)> ) / (2 eps).

The error of a central difference is of order ``eps**2``. With ``eps = 1e-3``
that is about ``1e-6``, which is far below the rank threshold of P3 and is
validated against a closed-form single-rotation case in the tests.

The whole Jacobian of ``p`` angles needs ``2p + 1`` states. They are produced in
one broadcast call of the shared statevector device, so one grid point costs one
device call and not ``2p + 1`` of them.

Why a finite difference and not a parameter shift
-------------------------------------------------
The parameter-shift rule is exact for the derivative of an **expectation value**,
not of a **state**. A state-level shift would need the generator of each gate to
be inserted at its own position on the tape, which the family ``apply`` functions
do not expose. The finite difference reuses the existing broadcast primitive
without opening up every circuit, and its error is measured rather than assumed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..ansatze.base import Ansatz
from .statevector import build_qnode

DEFAULT_STEP: float = 1e-3
"""Finite-difference step. Small enough for an O(step**2) error near 1e-6, large
enough to stay well above double-precision round-off. Fixed, not a free knob."""


def state_jacobian(
    ansatz: Ansatz,
    n: int,
    d: int,
    theta: NDArray[np.floating],
    encoder_input=None,
    step: float = DEFAULT_STEP,
    prelayer=None,
    postlayer=None,
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """Return the state and its derivative with respect to every angle.

    The state and the ``2p`` shifted states are evaluated in one broadcast call.
    The derivative of each angle is then a central difference of two rows.

    Args:
        ansatz: The family to run.
        n: Number of qubits.
        d: Depth, in the unit of the family.
        theta: Angle vector of length ``p = ansatz.n_params(n, d)``.
        encoder_input: Fixed input vector, or None.
        step: Central-difference step.
        prelayer: Optional fixed (n, 3) pre-layer angles, passed through to
            :func:`..probes.statevector.build_qnode`. The pre-layer is a fixed
            local unitary, so it changes the point on the manifold but not the
            geometry of the QFIM construction.
        postlayer: Optional fixed (n, 3) post-layer angles, likewise passed
            through. A fixed unitary after the circuit leaves the QFIM
            invariant (it preserves every inner product of state derivatives);
            it is accepted here only so G6 can reuse one circuit builder.

    Returns:
        A pair ``(psi, jac)``. ``psi`` has shape ``(2**n,)``. ``jac`` has shape
        ``(p, 2**n)``; row ``k`` is ``|d_k psi>``.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    p = theta.size
    shifts = np.eye(p) * step
    batch = np.empty((2 * p + 1, p), dtype=float)
    batch[0] = theta
    batch[1:p + 1] = theta + shifts       # forward shifts
    batch[p + 1:] = theta - shifts        # backward shifts

    circuit = build_qnode(ansatz, n, d, encoder_input, prelayer=prelayer,
                          postlayer=postlayer)
    states = np.asarray(circuit(batch)).reshape(2 * p + 1, -1)

    psi = states[0]
    forward, backward = states[1:p + 1], states[p + 1:]
    jac = (forward - backward) / (2.0 * step)
    return psi, jac


def quantum_fisher(
    psi: NDArray[np.complexfloating], jac: NDArray[np.complexfloating]
) -> NDArray[np.floating]:
    """Return the quantum Fisher information matrix from a state and its Jacobian.

    Implements ``F_ij = 4 Re[ <d_i|d_j> - <d_i|psi><psi|d_j> ]`` in matrix form.
    The result is symmetrised to remove the small asymmetry that finite-precision
    arithmetic leaves behind.

    Args:
        psi: State of shape ``(2**n,)``. It must be normalised.
        jac: Jacobian of shape ``(p, 2**n)`` from :func:`state_jacobian`.

    Returns:
        A real symmetric array of shape ``(p, p)``.

    Examples:
        >>> import numpy as np
        >>> # A single RX(theta) on one qubit has the known QFIM value 1.
        >>> theta = np.array([0.7])
        >>> eps = 1e-3
        >>> def state(t):
        ...     c, s = np.cos(t / 2), np.sin(t / 2)
        ...     return np.array([c, -1j * s])
        >>> psi = state(theta[0])
        >>> jac = ((state(theta[0] + eps) - state(theta[0] - eps)) / (2 * eps))[None, :]
        >>> float(np.round(quantum_fisher(psi, jac)[0, 0], 4))
        1.0
    """
    overlap = jac.conj() @ jac.T                 # <d_i | d_j>
    bias = jac.conj() @ psi                       # <d_i | psi>
    fisher = 4.0 * np.real(overlap - np.outer(bias, bias.conj()))
    return 0.5 * (fisher + fisher.T)


def effective_dimension(
    fisher: NDArray[np.floating], tol: float
) -> tuple[int, float, NDArray[np.floating]]:
    """Return the numerical rank, the participation ratio and the spectrum.

    Two complementary counts of how many angle directions move the state:

    * The numerical rank counts eigenvalues above ``tol`` times the largest one.
      It is a hard threshold, so it answers "how many directions move the state
      at all".
    * The participation ratio ``(sum lambda)**2 / sum lambda**2`` weights each
      direction by its curvature instead of thresholding. It answers "how many
      directions carry most of the geometry", and it is never larger than the
      rank.

    Args:
        fisher: A real symmetric QFIM of shape ``(p, p)``.
        tol: Relative eigenvalue threshold for the rank.

    Returns:
        A triple ``(rank, participation_ratio, eigenvalues)``. The eigenvalues
        are clipped at zero and returned in ascending order.
    """
    eigenvalues = np.linalg.eigvalsh(fisher)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    largest = float(eigenvalues.max()) if eigenvalues.size else 0.0
    if largest <= 0.0:
        return 0, 0.0, eigenvalues
    rank = int((eigenvalues > tol * largest).sum())
    total, total_sq = float(eigenvalues.sum()), float((eigenvalues ** 2).sum())
    participation = total * total / total_sq if total_sq > 0.0 else 0.0
    return rank, participation, eigenvalues
