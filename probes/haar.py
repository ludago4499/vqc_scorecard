# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-11  Moved from p1_expressibility.haar. Added the analytic Haar
#               expectation of the Meyer-Wallach measure, used by P2.

"""The Haar reference, in the two forms the metrics need.

P1 needs the Haar **distribution** of the pairwise fidelity, because it compares
histograms. That distribution has a closed form, and its cumulative form lets
each bin be integrated exactly instead of approximated at bin centres.

P2 needs only the Haar **expectation** of a single scalar, the Meyer-Wallach
measure. That expectation also has a closed form, so P2 needs no sampled
reference at all. This is worth stating plainly: the two metrics need different
kinds of reference, and forcing one scheme on both is what produced the
unsourced constants of the old scorecard.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def hilbert_dim(n: int) -> int:
    """Return 2**n, the Hilbert space dimension of n qubits."""
    return 2 ** n


def fidelity_cdf(f: NDArray[np.floating], dim: int) -> NDArray[np.floating]:
    """Return the Haar cumulative fidelity 1 - (1 - F)^(dim - 1).

    Args:
        f: Fidelity values in [0, 1].
        dim: Hilbert space dimension.

    Returns:
        Array of the same shape as f.
    """
    return 1.0 - np.power(1.0 - np.clip(f, 0.0, 1.0), dim - 1)


def fidelity_bin_probabilities(bins: int, dim: int) -> NDArray[np.floating]:
    """Return the exact Haar probability of each equal-width bin on [0, 1].

    Each bin is integrated through the cumulative form. Reading the density at
    bin centres instead adds an error of its own, and biases the KL divergence
    of P1.

    Args:
        bins: Number of bins. Must be at least 2.
        dim: Hilbert space dimension.

    Returns:
        Array of shape (bins,). It sums to 1.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    return np.diff(fidelity_cdf(edges, dim))


def sample_fidelities(n_pairs: int, dim: int, rng: np.random.Generator) -> NDArray[np.floating]:
    """Return fidelities between pairs of truly Haar-random states.

    A complex Gaussian vector, once normalised, is exactly Haar distributed on
    the unit sphere. This gives the ideal control: no circuit can cover Hilbert
    space more evenly than this.

    Args:
        n_pairs: Number of pairs to draw.
        dim: Hilbert space dimension.
        rng: Generator from :mod:`..core.seeding`.

    Returns:
        Array of shape (n_pairs,).
    """
    left = _random_states(n_pairs, dim, rng)
    right = _random_states(n_pairs, dim, rng)
    return np.abs(np.sum(np.conj(left) * right, axis=1)) ** 2


def expected_meyer_wallach(n: int) -> float:
    """Return the Haar expectation of the Meyer-Wallach measure at n qubits.

    For a random pure state on a bipartition of dimensions m and k, the
    mean reduced purity is (m + k) / (m k + 1). With one qubit against the
    rest, m = 2 and k = 2**(n-1), so

        E[Q] = 2 * (1 - (2 + 2^(n-1)) / (2^n + 1)).

    This is exact. It needs no sampling.

    It also explains the 0.4 constant of the old s_entangling function.
    That value is not editorial: it is this expression at n = 2. The defect
    was applying it at every n, where the correct value rises towards 1.

    Args:
        n: Number of qubits. Must be at least 2.

    Returns:
        The Haar expectation of Q, in [0, 1].

    Examples:
        >>> round(expected_meyer_wallach(2), 4)
        0.4
        >>> round(expected_meyer_wallach(8), 4)
        0.9883
    """
    if n < 2:
        raise ValueError("the Meyer-Wallach measure needs at least two qubits")
    mean_purity = (2 + 2 ** (n - 1)) / (2 ** n + 1)
    return 2.0 * (1.0 - mean_purity)


def _random_states(count: int, dim: int, rng: np.random.Generator):
    """Return count normalised complex Gaussian vectors of length dim."""
    vectors = rng.normal(size=(count, dim)) + 1j * rng.normal(size=(count, dim))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
