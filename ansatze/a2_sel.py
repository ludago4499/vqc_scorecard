# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below.
#
# History:
#   2026-08-11  Moved from the p1_expressibility package, unchanged.
#   2026-10-08  Documentation only (no gate, order or count changed). Paper-to-
#               code map, the exact PennyLane equivalence, the entangler
#               difference with A3, and the full deviations list. See
#               docs/IMPLEMENTATION.md, section A2.

"""A2 - Strongly entangling layers (SEL) with a classical embedding.

Source: M. Schuld et al., *Phys. Rev. A* **101**, 032308 (2020), "Circuit-centric
quantum classifiers". This is the circuit behind PennyLane
``StronglyEntanglingLayers``.

Circuit:

1. Angle embedding of the fixed input vector ``x`` (``RY`` on every qubit).
2. Then ``d`` repeats of: a full Euler ``RZ-RY-RZ`` layer, then a CNOT ring whose
   step grows with the layer index.

Angle count: ``p = 3 * n * d``.

Exact equivalence
-----------------
``apply(theta, n, d)`` produces the same state as
``qml.StronglyEntanglingLayers(theta.reshape(d, n, 3), wires=range(n))`` with
PennyLane's default ranges and ``CNOT`` (checked in
``tests/test_conventions.py``). Each ``Rot(phi, theta, omega)`` is queued as
``RZ(phi), RY(theta), RZ(omega)`` in time order. The ring of layer ``l`` is
``CNOT(q, (q + r_l) mod n)`` for ``q = 0 .. n-1``, with
``r_l = (l mod (n - 1)) + 1``; a ring follows **every** layer, the last one
included.

What differs from Schuld et al. 2020
------------------------------------
The paper's code block is a layer of general single-qubit gates followed by
``n / gcd(n, r)`` **trainable** controlled general gates ``C(G)``, control
``c_j = j r mod n``, target ``t_j = (j - 1) r mod n``, on an **amplitude**
encoded input. PennyLane's SEL keeps the layer layout and replaces the
controlled gates by fixed CNOTs with control ``q`` and target ``q + r``. A5
(:mod:`.a5_circuit_centric`) restores the trainable controlled gates and the
amplitude encoding.

Not the same entangler as A3
----------------------------
A3 (:mod:`.a3_reupload`) uses the same Euler block but a range-1 ring placed
only **between** layers. So A2 (angle) against A3 (re-upload) changes the
encoding **and** the entangler. Read the encoding-arm contrast with that.

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
        "Entanglers are fixed CNOTs (PennyLane SEL). The paper uses trainable "
        "controlled general gates C(G), n / gcd(n, r) per block, control "
        "j r mod n and target (j - 1) r mod n.",
        "Encoding is one RY(x_q) per qubit. The paper amplitude-encodes the "
        "input (A5 restores that).",
        "Task path: the task model queues RY(x_q) itself and calls apply with "
        "x=None; this apply's own x branch is the descriptor path and does not "
        "broadcast over a batch.",
        "Readout and loss: the paper uses pi = p(q_0 = 1) + b with a least-"
        "squares cost and SGD; the task head is sigmoid(a <Z_0> + b) with a "
        "class-balanced BCE and Adam.",
    ),
)
