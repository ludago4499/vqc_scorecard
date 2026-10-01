"""A3 - Data re-uploading.

Source: A. Perez-Salinas et al., *Quantum* **4**, 226 (2020).

Circuit: ``d`` repeats of "encode the input, then rotate with free angles".
Re-uploading the same input many times raises the Fourier degree of the model,
so depth, not width, is the capacity knob. That is why the protocol fixes
``n = 4`` and sweeps ``d``.

Per repeat, per qubit: three free angles. Between repeats there is a CNOT ring,
so that a multi-qubit model still correlates its wires.

Angle count: ``p = 3 * n * d``.

Protocol note
-------------
``permits`` returns ``True`` only for ``n = 4``. Every other width is still
computed, but the results file marks it ``off_protocol``. Read those rows as a
width control, not as part of the declared sweep.
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
    """Return the DLA generators of the re-uploading circuit.

    The variational ``RZ-RY-RZ`` block gives ``Y_q`` and ``Z_q`` on every wire,
    which close on the full single-qubit ``su(2)``. The ``RY`` re-upload of the
    fixed input is a single-qubit rotation that lives inside that same ``su(2)``,
    so it changes no dimension. The CNOT ring between repeats contributes
    ``Z_q X_(q+1)`` on each ring bond. As for A1 and A2, a complete single-qubit
    algebra with a connected entangler spans ``su(2**n)``.

    Depth does not enter: re-uploading the same block only repeats these
    generators, and a repeated generator adds nothing to a Lie closure.
    """
    gens = []
    for q in range(n):
        gens.append(pauli_word(n, [(q, "Y")]))
        gens.append(pauli_word(n, [(q, "Z")]))
    for q in range(n):
        gens.append(pauli_word(n, [(q, "Z"), ((q + 1) % n, "X")]))
    return gens


def apply(params: Params, n: int, d: int, x=None) -> None:
    """Queue ``d`` blocks of "re-upload the input, then rotate"."""
    p = np.asarray(params)
    p = p.reshape(p.shape[:-1] + (d, n, 3))
    xv = np.zeros(n) if x is None else np.asarray(x).reshape(-1)
    for layer in range(d):
        for q in range(n):
            qml.RY(xv[q], wires=q)          # re-upload the same input
        for q in range(n):
            qml.RZ(p[..., layer, q, 0], wires=q)
            qml.RY(p[..., layer, q, 1], wires=q)
            qml.RZ(p[..., layer, q, 2], wires=q)
        if layer < d - 1:
            _cnot_ring(n)


def _cnot_ring(n: int) -> None:
    """Queue a nearest-neighbour CNOT ring."""
    if n < 2:
        return
    for q in range(n):
        qml.CNOT(wires=[q, (q + 1) % n])


A3 = Ansatz(
    tag="A3",
    name="Data re-uploading (Perez-Salinas 2020)",
    n_params=n_params,
    apply=apply,
    n_encoder=n_encoder,
    permits=lambda n, d: n == 4 and d >= 1,
    source="Perez-Salinas et al., Quantum 4, 226 (2020)",
    generators=generators,
    deviations=(
        "The paper is single-qubit. The CNOT ring between repeats is an "
        "extension, needed to make the multi-qubit grid meaningful.",
        "The input is fixed, as in A2, so P1 measures the ansatz.",
        "Only n = 4 is inside the declared protocol. Other widths are controls.",
    ),
)
