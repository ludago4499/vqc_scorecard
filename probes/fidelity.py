# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-11  Moved from p1_expressibility.fidelity. State generation now
#               comes from probes.statevector, which P2 also uses.

"""Pairwise state fidelities, the input of P1.

The fidelity of two pure states is the squared modulus of their overlap. P1
compares the distribution of that quantity against the Haar law.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..ansatze.base import Ansatz
from .statevector import DEFAULT_CHUNK, sample_states


def sample_fidelities(
    ansatz: Ansatz,
    n: int,
    d: int,
    n_pairs: int,
    rng: np.random.Generator,
    encoder_input=None,
    chunk: int = DEFAULT_CHUNK,
) -> NDArray[np.floating]:
    """Return n_pairs independent fidelities of the ansatz.

    Two independent batches are drawn and matched element by element. The two
    batches come from the same generator, so a single seed fixes the whole
    measurement.

    Args:
        ansatz: The family to measure.
        n: Number of qubits.
        d: Depth, in the unit of the family.
        n_pairs: Number of independent parameter pairs.
        rng: Generator from :mod:`..core.seeding`.
        encoder_input: Fixed input vector, or None.
        chunk: Samples per broadcast call.

    Returns:
        Array of shape (n_pairs,) with values in [0, 1].
    """
    out = np.empty(n_pairs, dtype=float)
    done = 0
    while done < n_pairs:
        size = min(chunk, n_pairs - done)
        left = sample_states(ansatz, n, d, size, rng, encoder_input, chunk)
        right = sample_states(ansatz, n, d, size, rng, encoder_input, chunk)
        out[done:done + size] = np.abs(np.sum(np.conj(left) * right, axis=1)) ** 2
        done += size
    return out
