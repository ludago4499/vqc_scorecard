# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-11  Created. Single-qubit purities, for the P2 Meyer-Wallach measure.

"""Single-qubit reduced purities of a batch of pure states.

For a pure state of n qubits, trace out every qubit except one. The result
is a 2x2 reduced density matrix. Its purity Tr[rho^2] is 1 when that
qubit is in a product with the rest, and 0.5 when it is maximally mixed,
which means maximally entangled with the rest.

The whole of P2 is built on this one number, averaged over qubits and samples.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def single_qubit_purities(
    states: NDArray[np.complexfloating], n: int
) -> NDArray[np.floating]:
    """Return Tr[rho_k^2] for every qubit k of every state in a batch.

    The reduced density matrix of qubit k is obtained by viewing the
    statevector as a three-index array (left, k, right) and contracting the
    left and right indices. No 2**n by 2**n matrix is ever formed, so the cost
    is linear in the size of the statevector.

    Args:
        states: Array of shape (batch, 2**n). Each row must be normalised.
        n: Number of qubits.

    Returns:
        Array of shape (batch, n) with values in [0.5, 1.0].

    Raises:
        ValueError: If the trailing dimension of states is not 2**n.

    Examples:
        >>> import numpy as np
        >>> product = np.zeros((1, 4), dtype=complex); product[0, 0] = 1.0
        >>> np.round(single_qubit_purities(product, 2), 6)
        array([[1., 1.]])
        >>> bell = np.zeros((1, 4), dtype=complex)
        >>> bell[0, 0] = bell[0, 3] = 1 / np.sqrt(2)
        >>> np.round(single_qubit_purities(bell, 2), 6)
        array([[0.5, 0.5]])
    """
    batch, dim = states.shape
    if dim != 2 ** n:
        raise ValueError(f"expected trailing dimension {2 ** n}, found {dim}")

    purities = np.empty((batch, n), dtype=float)
    for k in range(n):
        left, right = 2 ** k, 2 ** (n - k - 1)
        block = states.reshape(batch, left, 2, right)
        # rho[b, i, j] = sum over the traced indices of psi psi*.
        rho = np.einsum("blir,bljr->bij", block, np.conj(block), optimize=True)
        # rho is Hermitian, so Tr[rho^2] is the sum of the squared moduli.
        purities[:, k] = np.einsum("bij,bij->b", rho, np.conj(rho), optimize=True).real
    return purities
