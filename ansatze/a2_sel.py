"""A2 - Strongly entangling layers (SEL) with a classical embedding.

Source: M. Schuld et al., *Phys. Rev. A* **101**, 032308 (2020), "Circuit-centric
quantum classifiers". This is the circuit behind PennyLane
``StronglyEntanglingLayers``.

Circuit:

1. Angle embedding of the fixed input vector ``x`` (``RY`` on every qubit).
2. Then ``d`` repeats of: a full Euler ``RZ-RY-RZ`` layer, then a CNOT ring whose
   step grows with the layer index.

Angle count: ``p = 3 * n * d``.

Why the embedding is fixed
--------------------------
P1 measures how much of Hilbert space the *variational* angles reach. If the
input angles were also random, the number would mix two different sources of
coverage, and A2 could not be compared with A1. So ``x`` is drawn once from a
fixed seed and then held constant across every pair. The vector used is written
into the results file. Use ``--embedding random`` in the runner to average over
inputs instead.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from .base import Ansatz, Params, pauli_word


def n_params(n: int, d: int) -> int:
    """Return the angle count ``3nd``."""
    return 3 * n * d


def n_encoder(n: int, d: int) -> int:
    """Return the input length. One angle per qubit."""
    return n


def generators(n: int) -> list[str]:
    """Return the DLA generators of the SEL circuit.

    The Euler ``RZ-RY-RZ`` layer gives ``Y_q`` and ``Z_q`` on every wire, which
    already close on the full single-qubit ``su(2)`` (their commutator is ``X_q``).
    The CNOT ring contributes ``Z_q X_(q+1)`` on each ring bond. The ring step
    grows with the layer, but a nearest-neighbour ring already connects every
    wire, and adding the longer-range bonds does not change the closure. With a
    complete single-qubit algebra and a connected entangler, the set spans the
    whole of ``su(2**n)``.
    """
    gens = []
    for q in range(n):
        gens.append(pauli_word(n, [(q, "Y")]))
        gens.append(pauli_word(n, [(q, "Z")]))
    for q in range(n):
        gens.append(pauli_word(n, [(q, "Z"), ((q + 1) % n, "X")]))
    return gens


def apply(params: Params, n: int, d: int, x=None) -> None:
    """Queue the embedding and then ``d`` strongly entangling layers."""
    p = np.asarray(params)
    p = p.reshape(p.shape[:-1] + (d, n, 3))
    if x is not None:
        xv = np.asarray(x).reshape(-1)
        for q in range(n):
            qml.RY(xv[q], wires=q)
    for layer in range(d):
        for q in range(n):
            qml.RZ(p[..., layer, q, 0], wires=q)
            qml.RY(p[..., layer, q, 1], wires=q)
            qml.RZ(p[..., layer, q, 2], wires=q)
        _cnot_ring(n, step=(layer % max(1, n - 1)) + 1)


def _cnot_ring(n: int, step: int) -> None:
    """Queue a CNOT ring with the given control-target distance."""
    if n < 2:
        return
    for q in range(n):
        qml.CNOT(wires=[q, (q + step) % n])


A2 = Ansatz(
    tag="A2",
    name="SEL + classical embedding (Schuld 2020)",
    n_params=n_params,
    apply=apply,
    n_encoder=n_encoder,
    permits=lambda n, d: n >= 2 and d >= 1,
    source="Schuld et al., Phys. Rev. A 101, 032308 (2020)",
    generators=generators,
    deviations=(
        "The embedding vector is fixed, not swept over a dataset. P1 therefore "
        "measures the ansatz, not the encoder.",
        "Ring range grows as (layer mod (n-1)) + 1, as in PennyLane "
        "StronglyEntanglingLayers.",
    ),
)
