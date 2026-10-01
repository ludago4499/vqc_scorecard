"""Common interface for all ansatz families.

Every ansatz family in this package is one :class:`Ansatz` record. The record
holds five things:

* how many angles the circuit needs,
* how to put the circuit on a PennyLane tape,
* which (n, d) points the family permits,
* the generators of its dynamical Lie algebra, for P4,
* a short text note about deviations from the source paper.

Nothing in this file builds a circuit. It only defines the shape that every
family must have, so that the sweep driver can loop over families without
knowing anything about them.

The generator set lives here, with the circuit, and not in the P4 metric. The
generators are a property of the family, so keeping them next to ``apply`` is
what stops the two from drifting apart. This is the same rule the old scorecard
broke when it defined A1 in three separate files.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# A parameter vector. Always flat, always radians.
Params = Sequence[float] | NDArray[np.floating]


def pauli_word(n: int, terms: Sequence[tuple[int, str]]) -> str:
    """Build a length-``n`` Pauli string with identity on the untouched wires.

    Args:
        n: Number of qubits, which is the length of the string.
        terms: Pairs ``(wire, letter)`` that set a non-identity Pauli. A wire not
            listed carries an identity.

    Returns:
        A string of ``n`` letters over I, X, Y, Z.

    Examples:
        >>> pauli_word(3, [(0, "Z"), (1, "X")])
        'ZXI'
    """
    letters = ["I"] * n
    for wire, letter in terms:
        letters[wire] = letter
    return "".join(letters)


@dataclass(frozen=True)
class Ansatz:
    """One ansatz family.

    Attributes:
        tag: Short code used in output files, for example ``"A1"``.
        name: Human name, for example ``"HEA (Kandala 2017)"``.
        n_params: Function ``(n, d) -> int``. It gives the number of angles.
        apply: Function ``(params, n, d, x) -> None``. It queues gates on the
            active PennyLane tape. ``x`` is the fixed input vector for families
            that encode data. Families without an encoder ignore ``x``.
        n_encoder: Function ``(n, d) -> int``. It gives the length of ``x``.
            Return ``0`` when the family has no data encoder.
        permits: Function ``(n, d) -> bool``. It says if the point is inside the
            protocol of the family. Points outside are still computed, but they
            are marked ``off_protocol`` in the output.
        source: Citation of the paper the circuit comes from.
        generators: Function ``(n) -> list[str]``. It returns the Pauli-string
            generators of the family's dynamical Lie algebra, which P4 closes
            under commutation. Each string has length ``n`` over the letters
            I, X, Y, Z. ``None`` means the family has not declared a set, and P4
            cannot run on it.
        deviations: Text list of every difference from that paper.
    """

    tag: str
    name: str
    n_params: Callable[[int, int], int]
    apply: Callable[..., None]
    n_encoder: Callable[[int, int], int]
    permits: Callable[[int, int], bool]
    source: str
    generators: Callable[[int], list[str]] | None = None
    deviations: tuple[str, ...] = field(default=())

    def sample_params(self, rng: np.random.Generator, batch: int, n: int, d: int) -> NDArray[np.floating]:
        """Draw ``batch`` parameter vectors uniformly on ``[0, 2*pi)``.

        This is the sampling rule of Sim et al. 2019. Do not change it without
        also changing the Haar reference, because the two must agree.
        """
        return rng.uniform(0.0, 2.0 * np.pi, size=(batch, self.n_params(n, d)))
