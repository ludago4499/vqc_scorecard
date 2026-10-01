# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-11  Created. Split out of the P1-only fidelity sampler, so that P2
#               and later metrics reuse the same state generation.
#   2026-09-01  build_qnode accepts an optional fixed pre-layer and post-layer
#               of single-qubit rotations, needed by G6 so that the small-angle
#               limit of an ansatz is a generic point of the cost. The pre-layer
#               moves the start state off the cost's eigenstate; the post-layer
#               is a fixed local basis rotation of the observable, without
#               which the theta = 0 circuit of a rotation-plus-CNOT family is
#               Clifford and keeps a Z-string cost diagonal, zeroing every
#               Z-generator gradient exactly. Default None; every existing
#               caller is unchanged.

"""Batched statevector generation. The one expensive primitive.

Every static metric in this package starts from the same object: a batch of
statevectors produced by an ansatz at random parameters. P1 then takes pairwise
overlaps of that batch. P2 takes single-qubit reduced states of it. Later
metrics will take derivatives of it.

Producing the batch is the slow step, so it lives here once and is shared.

Speed
-----
PennyLane accepts a parameter array of shape (batch, n_params) and returns
an array of shape (batch, 2**n) in a single call. That is roughly two orders
of magnitude faster than one call per sample. Batches are chunked, so memory
stays bounded at large qubit counts.
"""

from __future__ import annotations

import functools

import numpy as np
import pennylane as qml
from numpy.typing import NDArray

from ..ansatze.base import Ansatz

DEFAULT_CHUNK: int = 2048
"""Samples per broadcast call. Lower it if memory is short at large n."""


@functools.lru_cache(maxsize=None)
def device(n: int) -> qml.devices.Device:
    """Return a cached exact statevector device on n wires.

    The device has no shot noise and no error model. Static metrics are defined
    on the ideal state. Noise belongs to the G-series diagnostics, not here.

    Args:
        n: Number of wires.

    Returns:
        A PennyLane default.qubit device.
    """
    return qml.device("default.qubit", wires=n)


def build_qnode(ansatz: Ansatz, n: int, d: int, encoder_input=None,
                prelayer=None, postlayer=None):
    """Return a QNode that maps a parameter batch to a statevector batch.

    Args:
        ansatz: The family to run.
        n: Number of qubits.
        d: Depth, in the unit of the family.
        encoder_input: Fixed input vector for families that encode data, or
            None for families that do not.
        prelayer: Optional fixed angles of shape (n, 3). When given, one
            ``qml.Rot`` per wire is queued **before** the ansatz, so the
            circuit starts from a fixed product state instead of ``|0...0>``.
        postlayer: Optional fixed angles of shape (n, 3), one ``qml.Rot`` per
            wire **after** the ansatz. Reading a diagonal cost off the result
            is the same as measuring the ansatz state in a fixed random local
            basis - a fixed local rotation of the observable. Zhang et al.
            2022 close their circuit with trailing RX and RY layers for the
            same reason.

    Both layers are constants, not parameters: nothing differentiates them,
    the QFIM is invariant under the fixed post-unitary, and the Haar and
    manifold references are unchanged by fixed local rotations. Used by G6,
    where without them the ``theta = 0`` limit of a rotation-plus-CNOT family
    is a Clifford circuit whose Z-string cost stays diagonal, so Z-generator
    gradients vanish exactly for structural reasons that have nothing to do
    with plateaus.

    Returns:
        A callable QNode. It accepts an array of shape (batch, n_params) and
        returns an array of shape (batch, 2**n).
    """

    @qml.qnode(device(n), interface=None)
    def circuit(parameters):
        if prelayer is not None:
            for q in range(n):
                qml.Rot(prelayer[q][0], prelayer[q][1], prelayer[q][2], wires=q)
        ansatz.apply(parameters, n, d, encoder_input)
        if postlayer is not None:
            for q in range(n):
                qml.Rot(postlayer[q][0], postlayer[q][1], postlayer[q][2], wires=q)
        return qml.state()

    return circuit


def sample_states(
    ansatz: Ansatz,
    n: int,
    d: int,
    count: int,
    rng: np.random.Generator,
    encoder_input=None,
    chunk: int = DEFAULT_CHUNK,
) -> NDArray[np.complexfloating]:
    """Return count statevectors at parameters drawn uniformly on [0, 2pi).

    The sampling rule is the one of Sim et al. 2019. Every metric that compares
    against a Haar reference depends on it, so it must not be changed for one
    metric alone.

    Args:
        ansatz: The family to run.
        n: Number of qubits.
        d: Depth, in the unit of the family.
        count: Number of statevectors to produce.
        rng: Generator from :mod:`..core.seeding`.
        encoder_input: Fixed input vector, or None.
        chunk: Samples per broadcast call.

    Returns:
        Array of shape (count, 2**n) with complex entries.
    """
    circuit = build_qnode(ansatz, n, d, encoder_input)
    out = np.empty((count, 2 ** n), dtype=complex)
    done = 0
    while done < count:
        size = min(chunk, count - done)
        parameters = ansatz.sample_params(rng, size, n, d)
        out[done:done + size] = np.asarray(circuit(parameters)).reshape(size, -1)
        done += size
    return out
