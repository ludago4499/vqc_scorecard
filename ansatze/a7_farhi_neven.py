# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below.
#
# History:
#   2026-10-01  Created. A7, the quantum neural network of Farhi and Neven
#               2018: data qubits and one readout qubit, layers of trainable
#               two-body rotations exp(i theta Sigma) that all touch the readout,
#               and a Y measurement of the readout. Replaces the retired A7
#               (quantum attention), which has classical layers built into its
#               architecture and is kept in the taxonomy only.
#   2026-10-08  Documentation only (no gate, order or count changed). Corrected
#               "readout in |1>, as in the paper": the paper's |z, 1> uses
#               +1/-1 valued bits, so its readout is the Z = +1 state |0>; the
#               |1> used here flips the sign of every output exactly. Added the
#               product-form identity, the basis-encoding limit and the loss.
#               See docs/IMPLEMENTATION.md, section A7.

"""A7 - Farhi-Neven quantum neural network (readout qubit).

Source: E. Farhi and H. Neven, "Classification with quantum neural networks on
near term processors", arXiv:1802.06002 (2018), Secs. 2-4.

Why this family is in the study
-------------------------------
It is the first VQC classifier proposed for near-term hardware, and the only
one in the study whose every trainable gate couples a data qubit to one
dedicated readout qubit. Its gate set is not a complete single-qubit set plus an
entangler, so its dynamical Lie algebra is **not** forced to the full
``su(2**n)``. It is the structural control of the barren-plateau hypotheses: if
its gradients do not decay like A1-A6, the algebra is why.

Wire layout
-----------
``n`` counts **all** qubits, so every family runs on the same hardware budget:
wire 0 is the readout, wires ``1 .. n-1`` hold ``n - 1`` data features.

Circuit
-------
1. Readout prepared in ``|1>`` (``X`` on wire 0). The paper writes
   ``|z, 1>`` with bits valued ``+1`` or ``-1``, so its readout is the
   ``Z = +1`` state, ``|0>`` here (its Eqs. 10-11 and 19-20 need
   ``<Z_readout> = +1``). ``X_0`` commutes with every gate of this circuit, so
   ``|1>`` flips the sign of the output exactly and nothing else; the head
   ``a`` absorbs it. Kept as is to leave pure-v1 results unchanged.
2. Encoding: ``RY(x_j)`` on data wire ``j``. The paper loads binary data as a
   computational basis state; a continuous angle is the declared deviation
   that puts A7 in Class I next to A1, A2 and A4 (same angle features).
   ``RY(0)|0> = |0>`` and ``RY(pi)|0> = |1>``, so the endpoints of the
   ``[0, pi]`` feature range are the paper's basis states.
3. ``d`` layers. One layer: ``exp(i theta Z_j X_0)`` for every data wire ``j``,
   then ``exp(i theta X_j X_0)`` for every data wire: the paper's alternating
   ``ZX`` and ``XX`` sweeps. Queued as ``PauliRot(-2 theta)``, because
   ``PauliRot(t, P) = exp(-i t P / 2)``.
4. Readout: the paper measures ``Y_0``. A fixed ``RX(pi/2)`` on wire 0 maps
   ``Z_0`` onto ``Y_0`` (``RX(pi/2)^dagger Z RX(pi/2) = Y``), so the package's
   local cost ``<Z_0>`` **is** the paper's readout, for the descriptors and the
   task head alike.

Angle count: ``p = 2 (n - 1) d``. At ``n = 17``, ``d = 3`` this is 96, the
paper's MNIST circuit (16 data qubits, 3 ZX and 3 XX sweeps).

Product form (classically simulable)
------------------------------------
Write the readout as ``(|+> - |->) / sqrt(2)`` in the ``X_0`` basis. Each gate
``exp(i t P_j X_0)`` acts on the data as ``exp(+i t P_j)`` in the ``+`` branch
and ``exp(-i t P_j)`` in the ``-`` branch, and gates on different data wires act
on different qubits. With ``u_j^+`` and ``u_j^-`` the products of a wire's gates
in each branch and ``|phi_j> = RY(x_j)|0>``::

    <Y_0> = Im prod_j <phi_j| (u_j^+)^dagger u_j^- |phi_j>

So the output is the imaginary part of a product of ``n - 1`` single-qubit
overlaps, computable in ``O(n d)``. This is the circuit-level form of the
algebra ``(n - 1)`` copies of ``su(2)`` (P4). Checked in
``tests/test_conventions.py``.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from .base import Ansatz, Params, pauli_word

READOUT: int = 0
"""Wire of the readout qubit."""


def n_params(n: int, d: int) -> int:
    """Return the angle count ``2(n - 1)d``."""
    return 2 * (n - 1) * d


def n_encoder(n: int, d: int) -> int:
    """Return the input length ``n - 1``: one feature per data wire."""
    return n - 1


def data_wires(n: int) -> list[int]:
    """Return the data wires ``1 .. n-1``."""
    return list(range(1, n))


def embed(x, n: int) -> None:
    """Prepare the readout in ``|1>`` and load ``x`` on the data wires.

    Args:
        x: Features, ``n - 1`` per row (one vector or a batch).
        n: Number of qubits, readout included.
    """
    qml.PauliX(wires=READOUT)
    for k, wire in enumerate(data_wires(n)):
        qml.RY(x[..., k], wires=wire)


def generators(n: int) -> list[str]:
    """Return the DLA generators: ``Z_j X_0`` and ``X_j X_0`` on every data wire.

    These are the exact generators of the trainable gates (each gate is
    ``exp(i theta Sigma)`` with ``Sigma`` a Pauli string). There is no fixed
    entangler to convert, so no convention is involved. Every generator carries
    ``X`` on the readout, which keeps the closure far below ``su(2**n)``; P4
    measures how far.

    Args:
        n: Number of qubits, readout included.

    Returns:
        The Pauli-string generators.
    """
    gens = []
    for j in data_wires(n):
        gens.append(pauli_word(n, [(j, "Z"), (READOUT, "X")]))
        gens.append(pauli_word(n, [(j, "X"), (READOUT, "X")]))
    return gens


def apply(params: Params, n: int, d: int, x=None) -> None:
    """Queue the readout preparation and encoding (when ``x`` is given), the layers and the readout basis change.

    The readout ``X`` is part of the encoding step. Without ``x`` the readout
    still starts in ``|1>``, because that preparation belongs to the model, not
    to the data.

    Args:
        params: The ``2(n-1)d`` angles, flat, or a batch with the angles on the
            last axis.
        n: Number of qubits, readout included.
        d: Number of layers.
        x: Data features (length ``n - 1``, or a batch), or ``None``.
    """
    p = params if hasattr(params, "shape") else np.asarray(params)
    if x is not None:
        embed(x if hasattr(x, "shape") else np.asarray(x, dtype=float), n)
    else:
        qml.PauliX(wires=READOUT)
    i = 0
    for _ in range(d):
        for word in ("ZX", "XX"):
            for j in data_wires(n):
                qml.PauliRot(-2.0 * p[..., i], word, wires=[j, READOUT])
                i += 1
    qml.RX(np.pi / 2, wires=READOUT)


A7 = Ansatz(
    tag="A7",
    name="QNN with readout qubit (Farhi-Neven 2018)",
    n_params=n_params,
    apply=apply,
    n_encoder=n_encoder,
    permits=lambda n, d: n >= 2 and d >= 1,
    source="Farhi & Neven, arXiv:1802.06002 (2018)",
    generators=generators,
    deviations=(
        "Continuous RY angle encoding on the data wires; the paper loads binary "
        "data as a computational basis state.",
        "n counts the readout qubit, so A7 reads n - 1 features where A1, A2, A4 "
        "and A6 read n (equal hardware width, not equal feature count).",
        "One layer is one ZX sweep plus one XX sweep over the data wires; the paper "
        "studies several gate orders, of which this alternation is one.",
        "The Y readout is implemented as a fixed RX(pi/2) on the readout followed "
        "by Z_0, an exact basis change.",
        "Readout prepared in |1>. The paper's |z, 1> (bits valued +1/-1) is the "
        "Z = +1 state |0>. Exact global sign flip of the output, absorbed by the "
        "head scale a.",
        "Training: class-balanced BCE on sigmoid(a <Y_0> + b) with Adam. The "
        "paper uses loss = 1 - l(z) <Y> (Eq. 6) with the loss-normalised SGD "
        "step of Eq. (32).",
    ),
)
