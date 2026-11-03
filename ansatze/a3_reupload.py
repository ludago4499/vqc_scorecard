# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below.
#
# History:
#   2026-08-11  Moved from the p1_expressibility package, unchanged.
#   2026-10-08  Documentation only (no gate, order or count changed). Corrected
#               the deviation "the paper is single-qubit": the paper also has
#               2- and 4-qubit classifiers entangled with CZ. Added the
#               paper-to-code map, the task-path note and the full deviations
#               list. See docs/IMPLEMENTATION.md, section A3.

"""A3 - Data re-uploading.

Source: A. Perez-Salinas et al., *Quantum* **4**, 226 (2020).

Circuit: ``d`` repeats of "encode the input, then rotate with free angles".
Re-uploading the same input many times raises the Fourier degree of the model,
so depth, not width, is the capacity knob. That is why the protocol fixes
``n = 4`` and sweeps ``d``.

Per repeat, per qubit: three free angles. Between repeats there is a CNOT ring,
so that a multi-qubit model still correlates its wires.

Angle count: ``p = 3 * n * d``.

Paper-to-code map
-----------------
* Processing layer, Eq. (3): ``L(i) = U(phi_i) U(x)``. Here ``U(x)`` is
  ``RY(x_q)`` on qubit ``q`` (one feature per qubit) and ``U(phi_i)`` is
  ``RZ, RY, RZ`` in time order, the paper's ``e^{i phi_2 Z} e^{i phi_1 Y}
  e^{i phi_3 Z}`` (Eq. 12) up to the sign convention of the angles.
* Entangler: the paper's multi-qubit classifiers apply ``CZ`` after every
  layer except the last (4 qubits: pairs (1,2),(3,4) and (2,3),(1,4) in
  alternation). Here a nearest-neighbour ``CNOT`` ring ``(q, q+1 mod n)`` sits
  between layers, none after the last. At ``d = 1`` there is no entangler
  (README 7.4).
* The weighted layer of Eq. (5), ``U(theta_i + w_i o x)``, is not used.

Protocol note
-------------
``permits`` returns ``True`` only for ``n = 4``. It governs the descriptor
sweep (:mod:`vqc_scorecard.cli` marks other widths ``off_protocol``). The
pure-v1 family arm runs A3 at every width of the protocol (4, 6, 8) and does
not read ``permits``.

Task path (pure-v1)
-------------------
:func:`vqc_scorecard.tasks.model._reupload_body` re-implements this ``apply``
with a batched ``x``; the two give the same state (``tests/test_conventions.py``).

Not A2's entangler
------------------
A2 rings after every layer with a range that grows with the layer. This file
rings between layers at range 1. The encoding-arm contrast A2 against A3 is
therefore encoding plus entangler, not encoding alone.
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
        "Entangler: a nearest-neighbour CNOT ring between repeats. The paper's "
        "multi-qubit classifiers (2 and 4 qubits) use CZ after every layer but "
        "the last, on alternating pairs.",
        "Encoding: one feature per qubit through RY(x_q). The paper uploads x "
        "through a general SU(2) gate, up to three components per gate (Eq. 3).",
        "Unweighted layers (Eq. 3); the weighted form U(theta + w o x) of "
        "Eq. (5) is not used.",
        "Readout and loss: the paper trains a fidelity (Eq. 7) or weighted "
        "fidelity (Eq. 9) cost against label states; the task head is "
        "sigmoid(a <Z_0> + b) with a class-balanced BCE.",
        "Not A2's entangler: A2 rings after every layer with a growing range; "
        "A3 rings between layers at range 1.",
        "The input is fixed, as in A2, so P1 measures the ansatz.",
        "Only n = 4 is inside the declared descriptor protocol. Other widths "
        "are controls there; the pure-v1 task arm runs n = 4, 6, 8.",
    ),
)
