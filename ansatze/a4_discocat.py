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
#               the word-box deviation (the CRZ ring spans the sentence
#               register, it is not a per-word lambeq box), added the readout
#               and task-path notes. See docs/IMPLEMENTATION.md, section A4.

"""A4 - DisCoCat sentence circuit.

Source: B. Coecke et al., arXiv:2005.04147 (2020); D. Kartsaklis et al.,
*lambeq*, arXiv:2110.04236 (2021).

In DisCoCat, grammar picks the circuit. Every word is a box on one or more
wires. The parse tree then joins wires with "cups", which contract a noun wire
against the matching wire of the verb.

Template used here
------------------
A fixed alternating parse, ``noun verb noun verb ...``, one qubit per word. The
noun wires are the even qubits and the verb wires are the odd qubits. Each word
box is an IQP-style block: a Hadamard layer, then a ring of ``CRZ`` gates with
free angles. Each box is repeated ``d`` times.

Angle count: ``p = n * d`` (one ``CRZ`` angle per qubit per repeat), plus ``n``
single-qubit ``RY`` angles for the word states, so ``p = n * (d + 1)``.

Deviation you must state in any write-up
----------------------------------------
A real DisCoCat cup is a Bell effect. It needs post-selection, and the output
state is subnormalized. This module queues the **unitary part only**: the
``CNOT`` and ``H`` of the cup, with no measurement. The state stays pure on all
``n`` wires. So this measures the expressibility of the sentence state *before*
contraction. It is an upper bound on the contracted circuit, not the same
quantity. Do not report it as "DisCoCat expressibility" without this sentence.

What the IQP layer is, and is not
---------------------------------
With one qubit per word, the ``CRZ`` ring ``(q, q+1 mod n)`` couples
**different words** directly. In DisCoCat a word box acts only on its own
word's wires, and only cups connect words. lambeq's ``IQPAnsatz`` (lambeq
source, ``lambeq/ansatz/circuit.py``) gives a one-qubit box the Euler sequence
``Rx, Rz, Rx`` and a multi-qubit box a ``CRz`` ladder on ``n - 1`` adjacent
pairs (no wrap-around), each followed by a Hadamard layer. So this module is an
IQP-style **sentence** circuit on a DisCoCat cup template, not lambeq's word
boxes.

Readout
-------
For even ``n`` the cups pair every wire, ``(0, 1), (2, 3), ...``, so no wire is
left open for the sentence type. The task readout ``<Z_0>`` reads a cup wire
that a contracted circuit would post-select.

Task path (pure-v1)
-------------------
The task model (encoding ``"angle"``) queues ``RY(x_q)`` and then this
``apply`` with ``x=None``. The first layer here is ``RY(theta_q)`` on the same
wire, so the two compose to ``RY(x_q + theta_q)``: the word-state angles act as
trainable offsets on the features. The parse-driven word-table route
(:func:`vqc_scorecard.tasks.model.build_compositional_model`) is the legacy v10
task axis and is not part of pure-v1.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from .base import Ansatz, Params, pauli_word


def n_params(n: int, d: int) -> int:
    """Return the angle count ``n(d + 1)``."""
    return n * (d + 1)


def n_encoder(n: int, d: int) -> int:
    """Return ``0``. Word angles are variational, so there is no separate input."""
    return 0


def generators(n: int) -> list[str]:
    """Return the DLA generators of the DisCoCat sentence circuit.

    The word-state ``RY`` layer gives ``Y_q`` on every wire. Each ``CRZ`` of the
    IQP ring has generator ``(Z_t - Z_c Z_t) / 2``, so it contributes the single
    term ``Z_(q+1)`` and the two-body term ``Z_q Z_(q+1)`` on every ring bond. The
    unitary cup adds ``Z_q X_(q+1)`` on the even bonds it joins. The fixed
    Hadamard layers are Clifford, so they leave the closure dimension unchanged
    and add no generator.

    The IQP core on its own is a commuting, sub-universal family. It is the
    ``RY`` word states that make ``Y_q`` and ``Z_q`` a complete single-qubit set
    on each wire, which, with the ``ZZ`` ring, lifts the circuit to a universal
    generating set. So the restriction of the word boxes is not enough to hold
    the DLA below ``su(2**n)``; the sentence-state layer removes it.
    """
    gens = []
    for q in range(n):
        gens.append(pauli_word(n, [(q, "Y")]))            # RY word states
    for q in range(n):
        target = (q + 1) % n
        gens.append(pauli_word(n, [(target, "Z")]))       # CRZ single-body term
        gens.append(pauli_word(n, [(q, "Z"), (target, "Z")]))  # CRZ two-body term
    for q in range(0, n - 1, 2):
        gens.append(pauli_word(n, [(q, "Z"), (q + 1, "X")]))   # unitary cup CNOT
    return gens


def apply(params: Params, n: int, d: int, x=None) -> None:
    """Queue the word boxes, then the unitary part of the cup layer."""
    p = np.asarray(params)
    i = 0
    # word states: one RY per wire
    for q in range(n):
        qml.RY(p[..., i], wires=q); i += 1
    # IQP-style word boxes, repeated d times
    for _ in range(d):
        for q in range(n):
            qml.Hadamard(wires=q)
        for q in range(n):
            qml.CRZ(p[..., i], wires=[q, (q + 1) % n]); i += 1
    _cup_layer(n)


def _cup_layer(n: int) -> None:
    """Queue the unitary part of every cup. No post-selection. See module note.

    A cup between wires ``a`` and ``b`` is ``H(a)`` after ``CNOT(a, b)``, then a
    measurement of both wires in the computational basis. Only the two gates are
    queued here.
    """
    for q in range(0, n - 1, 2):
        qml.CNOT(wires=[q, q + 1])
        qml.Hadamard(wires=q)


A4 = Ansatz(
    tag="A4",
    name="DisCoCat sentence circuit (pre-contraction)",
    n_params=n_params,
    apply=apply,
    n_encoder=n_encoder,
    permits=lambda n, d: n >= 2 and n % 2 == 0 and d >= 1,
    source="Coecke et al. arXiv:2005.04147; Kartsaklis et al. arXiv:2110.04236",
    generators=generators,
    deviations=(
        "CUPS ARE NOT POST-SELECTED. Only the CNOT and H are queued. This is an "
        "upper bound on the contracted sentence circuit.",
        "The parse is a fixed alternating noun-verb template, not a real parser "
        "output. Sentence length maps to n as one qubit per word.",
        "The IQP layer (H layer + CRZ ring with wrap-around) spans the whole "
        "sentence register and couples different words directly. lambeq's "
        "IQPAnsatz uses Rx Rz Rx for one-qubit boxes and a CRz ladder of n - 1 "
        "gates per layer inside a multi-qubit box; it is not reproduced here.",
        "Every wire is cupped for even n, so no sentence wire stays open; the "
        "readout <Z_0> reads a cup wire that contraction would post-select.",
        "Task path: RY(x_q) is queued before the word-state RY(theta_q) on the "
        "same wire, so the pair is RY(x_q + theta_q).",
    ),
)
