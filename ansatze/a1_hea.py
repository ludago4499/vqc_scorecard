"""A1 - Hardware-efficient ansatz (HEA).

Source: A. Kandala et al., *Nature* **549**, 242 (2017), Eq. (2).

Circuit:

1. Rotation layer 0: ``RX`` then ``RZ`` on every qubit. The leading ``Z`` of the
   Euler triple is dropped, because ``|0>`` is already a ``Z`` eigenstate.
2. Then ``d`` repeats of: a CNOT chain, then a full Euler ``RZ-RX-RZ`` layer.

Angle count: ``p = n * (3d + 2)``.

This file is a copy of the circuit in ``kandala_hea.py``. It is copied on
purpose, so that this package has no import from the old scorecard folder and
can be moved on its own.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from .base import Ansatz, Params, pauli_word


def n_params(n: int, d: int) -> int:
    """Return the angle count ``n(3d + 2)``."""
    return n * (3 * d + 2)


def generators(n: int) -> list[str]:
    """Return the DLA generators of the HEA.

    The single-qubit ``RX`` and ``RZ`` rotations give ``X_q`` and ``Z_q`` on every
    wire. The fixed CNOT chain contributes the non-trivial two-body term of its
    generator, ``Z_q X_(q+1)``, on each nearest-neighbour bond. That term is also
    the native cross-resonance generator of the Kandala hardware, so the set is
    both a faithful reading of the CNOT chain and of the paper's real gate.

    With single-qubit ``X`` and ``Z`` on every wire and a connected chain of
    entanglers, this set closes on the whole of ``su(2**n)``.
    """
    gens = []
    for q in range(n):
        gens.append(pauli_word(n, [(q, "X")]))
        gens.append(pauli_word(n, [(q, "Z")]))
    for q in range(n - 1):
        gens.append(pauli_word(n, [(q, "Z"), (q + 1, "X")]))
    return gens


def apply(params: Params, n: int, d: int, x=None) -> None:
    """Queue the HEA on the active tape. """
    # p[..., i] works for one vector and for a batch of vectors alike.
    p = np.asarray(params)
    i = 0
    for q in range(n):
        qml.RX(p[..., i], wires=q); i += 1
        qml.RZ(p[..., i], wires=q); i += 1
    for _ in range(d):
        _cnot_chain(n)
        for q in range(n):
            qml.RZ(p[..., i], wires=q); i += 1
            qml.RX(p[..., i], wires=q); i += 1
            qml.RZ(p[..., i], wires=q); i += 1


def _cnot_chain(n: int) -> None:
    """Queue one nearest-neighbour CNOT chain of ``n - 1`` gates."""
    for q in range(n - 1):
        qml.CNOT(wires=[q, q + 1])


A1 = Ansatz(
    tag="A1",
    name="HEA (Kandala 2017)",
    n_params=n_params,
    apply=apply,
    n_encoder=lambda n, d: 0,
    permits=lambda n, d: n >= 2 and d >= 0,
    source="Kandala et al., Nature 549, 242 (2017), Eq. (2)",
    generators=generators,
    deviations=(
        "Entangler is a linear CNOT chain. The paper uses hardware-native "
        "cross-resonance gates.",
    ),
)
