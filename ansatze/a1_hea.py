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
#               code map, task-path note and the full deviations list, for
#               reproduction. See docs/IMPLEMENTATION.md, section A1.

"""A1 - Hardware-efficient ansatz (HEA).

Source: A. Kandala et al., *Nature* **549**, 242 (2017), Eq. (2).

Circuit:

1. Rotation layer 0: ``RX`` then ``RZ`` on every qubit. The leading ``Z`` of the
   Euler triple is dropped, because ``|0>`` is already a ``Z`` eigenstate.
2. Then ``d`` repeats of: a CNOT chain, then a full Euler ``RZ-RX-RZ`` layer.

Angle count: ``p = n * (3d + 2)``, the count Kandala et al. state with Eq. (2).

Paper-to-code map
-----------------
* Euler rotation ``U^{q,i} = Z_{theta1} X_{theta2} Z_{theta3}`` (operator
  order): in time order the code queues ``RZ, RX, RZ``.
* Layer 0 drops the first-in-time ``Z`` because ``|0>`` is a ``Z``
  eigenstate, as the paper does; so it queues ``RX, RZ`` (2 angles per qubit).
* ``U_ENT``: the paper uses the device drift ``exp(-i H_0 tau)`` (cross-resonance
  gates); the code uses a fixed nearest-neighbour CNOT chain ``(q, q+1)``.

Task path (pure-v1)
-------------------
Kandala's ansatz is a VQE trial state and takes no data. In the task model
(:mod:`vqc_scorecard.tasks.model`, encoding ``"angle"``) one ``RY(x_q)`` per
qubit is queued first, then this ``apply`` with ``x=None``. The readout is
``<Z_0>`` through ``sigmoid(a <Z_0> + b)``, not an energy.

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
        "Task path: the paper's ansatz takes no data (VQE). The task model "
        "prepends one RY(x_q) per qubit (encoding 'angle'); the descriptor "
        "sweep runs the bare ansatz from |0>.",
        "Readout is <Z_0> (task head sigmoid(a <Z_0> + b)), not a Hamiltonian "
        "energy.",
        "Rotation signs follow PennyLane (RX(t) = exp(-i t X / 2)); the paper's "
        "Z_theta, X_theta convention can differ by the sign of theta, which "
        "changes nothing under uniform sampling or training.",
    ),
)
