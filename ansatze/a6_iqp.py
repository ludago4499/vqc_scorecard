# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below.
#
# History:
#   2026-10-01  Created. A6, the variational quantum classifier of Havlicek et
#               al. 2019: the second-order IQP (ZZ) feature map, then a
#               trainable layered circuit of local Y/Z rotations and CZ
#               entanglers. Replaces the retired A6 (quantum LSTM), which has
#               classical layers built into its architecture and is kept in
#               the taxonomy only.
#   2026-10-08  Documentation only (no gate, order or count changed). Declared
#               the input range (the task angles lie on [0, pi], half the
#               paper's (0, 2 pi], so every ZZ phase is non-negative), the
#               readout and loss, and that the kernel method (QKE) is not
#               implemented. See docs/IMPLEMENTATION.md, section A6.

"""A6 - IQP (ZZ) feature map with a trainable variational circuit.

Source: V. Havlicek, A. D. Corcoles, K. Temme, A. W. Harrow, A. Kandala, J. M.
Chow and J. M. Gambetta, "Supervised learning with quantum-enhanced feature
spaces", *Nature* **567**, 209 (2019), the "quantum variational classifier"
protocol (feature map ``U_Phi(x)``, then ``W(theta)``, then measurement).

Why this family is in the study
-------------------------------
The feature map is the one the paper conjectures to be hard to simulate
classically: an instantaneous-quantum-polynomial (IQP) circuit with
data-dependent ``Z`` and ``ZZ`` phases, applied twice. A6 is the encoding level
of Class II that is quantum by construction, against the single angle (A2), the
re-upload (A3) and the amplitude encoding (A5).

Circuit
-------
1. Feature map, applied twice: ``U_Phi(x) = U_Z(x) H^n U_Z(x) H^n`` with
   ``U_Z(x) = exp(i sum_S phi_S(x) Z_S)``, ``phi_q = x_q`` on every qubit and
   ``phi_qr = (pi - x_q)(pi - x_r)`` on every edge ``(q, r)``. The edges are the
   nearest-neighbour chain. Queued as ``RZ(-2 phi_q)`` and ``IsingZZ(-2 phi_qr)``,
   because ``RZ(t) = exp(-i t Z / 2)`` and ``IsingZZ(t) = exp(-i t ZZ / 2)``.
2. ``W(theta)``: a local layer ``RY`` then ``RZ`` on every qubit, then ``d``
   repeats of [``CZ`` on every chain edge, then a local layer].
3. ``<Z_0>``.

Angle count: ``p = 2 n (d + 1)``, the paper's ``theta in R^{2n(l+1)}`` with
``l = d``.

Input range
-----------
The paper draws ``x`` from ``(0, 2 pi]``. The task features are angles on
``[0, pi]`` (standardised PCA, clipped at 3 sigma, mapped by
``(s + 3) / 6 * pi``). So ``phi_q = x_q`` covers half the paper's range, and
``(pi - x_q)(pi - x_r)`` lies in ``[0, pi^2]``: the ZZ phase never changes
sign, where the paper's spans ``[-pi^2, pi^2]``. A mean feature maps to
``pi / 2``.

Readout and decision rule
-------------------------
The paper measures a parity ``f`` of all qubits and assigns
``sign(<f> + b)``; its cost smooths the binomial error with a sigmoid whose
sharpness is set by the shot count ``R``. The task head
``sigmoid(a <Z_0> + b)`` keeps the bias ``b`` and learns the sharpness ``a``.
The paper's second protocol, quantum kernel estimation (fidelity kernel and a
classical SVM), is not implemented.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from .base import Ansatz, Params, pauli_word

FEATURE_MAP_REPS: int = 2
"""Repetitions of ``U_Z H^n`` in the feature map (the paper uses two)."""


def n_params(n: int, d: int) -> int:
    """Return the angle count ``2n(d + 1)``."""
    return 2 * n * (d + 1)


def n_encoder(n: int, d: int) -> int:
    """Return the input length. One feature per qubit."""
    return n


def edges(n: int) -> list[tuple[int, int]]:
    """Return the interaction graph: the nearest-neighbour chain.

    Args:
        n: Number of qubits.

    Returns:
        The ``n - 1`` bonds ``(q, q + 1)``.

    Examples:
        >>> edges(4)
        [(0, 1), (1, 2), (2, 3)]
    """
    return [(q, q + 1) for q in range(n - 1)]


def embed(x, n: int) -> None:
    """Queue the feature map ``U_Phi(x)`` on the active tape.

    ``x`` may be one vector of length ``n`` or a batch ``(batch, n)``; the last
    axis is the qubit, so the map broadcasts over a dataset.

    Args:
        x: Features, ``n`` per row. In the paper they lie in ``(0, 2 pi]``; the
            task side passes the package angles on ``[0, pi]``.
        n: Number of qubits.
    """
    for _ in range(FEATURE_MAP_REPS):
        for q in range(n):
            qml.Hadamard(wires=q)
        for q in range(n):
            qml.RZ(-2.0 * x[..., q], wires=q)
        for q, r in edges(n):
            qml.IsingZZ(-2.0 * (np.pi - x[..., q]) * (np.pi - x[..., r]), wires=[q, r])


def generators(n: int) -> list[str]:
    """Return the DLA generators of ``W(theta)``.

    The local layers give ``Y_q`` and ``Z_q`` on every wire, which close on the
    full single-qubit ``su(2)``. A ``CZ`` is ``exp(i pi (I - Z_q - Z_r + Z_q Z_r) / 4)``;
    its non-trivial two-body term is ``Z_q Z_r``. With a complete single-qubit
    algebra and the connected chain, the closure is ``su(2**n)``. The feature
    map acts on the fixed input and is not part of the trainable algebra.

    Args:
        n: Number of qubits.

    Returns:
        The Pauli-string generators.
    """
    gens = []
    for q in range(n):
        gens.append(pauli_word(n, [(q, "Y")]))
        gens.append(pauli_word(n, [(q, "Z")]))
    for q, r in edges(n):
        gens.append(pauli_word(n, [(q, "Z"), (r, "Z")]))
    return gens


def _local_layer(p, i: int, n: int) -> int:
    for q in range(n):
        qml.RY(p[..., i], wires=q)
        qml.RZ(p[..., i + 1], wires=q)
        i += 2
    return i


def apply(params: Params, n: int, d: int, x=None) -> None:
    """Queue the feature map when ``x`` is given, then ``W(theta)``.

    Args:
        params: The ``2n(d + 1)`` angles, flat, or a batch with the angles on
            the last axis.
        n: Number of qubits.
        d: Number of entangling layers of ``W``.
        x: Input features (length ``n``, or a batch), or ``None``.
    """
    p = params if hasattr(params, "shape") else np.asarray(params)
    if x is not None:
        embed(x if hasattr(x, "shape") else np.asarray(x, dtype=float), n)
    i = _local_layer(p, 0, n)
    for _ in range(d):
        for q, r in edges(n):
            qml.CZ(wires=[q, r])
        i = _local_layer(p, i, n)


A6 = Ansatz(
    tag="A6",
    name="IQP feature map + VQC (Havlicek 2019)",
    n_params=n_params,
    apply=apply,
    n_encoder=n_encoder,
    permits=lambda n, d: n >= 2 and d >= 1,
    source="Havlicek et al., Nature 567, 209 (2019)",
    generators=generators,
    deviations=(
        "The paper runs n = 2, where the only pair is the single edge. Here the "
        "interaction graph is the nearest-neighbour chain at every n, for the "
        "feature map and the CZ entangler alike (hardware-native connectivity).",
        "The paper reads a parity of the measured bits; the study reads the local "
        "<Z_0> for every family (G4: the global cost flattens faster).",
        "Rotation signs follow PennyLane (RZ(t) = exp(-i t Z/2)); the paper's "
        "exp(+i theta Z/2) differs by the sign of theta, which changes nothing "
        "under uniform sampling or training.",
        "The descriptor sweep holds x fixed (package convention).",
        "Input range: task angles lie on [0, pi], half the paper's (0, 2 pi]; "
        "the ZZ phase (pi - x_q)(pi - x_r) is then in [0, pi^2] and never "
        "changes sign.",
        "Training: class-balanced BCE on sigmoid(a <Z_0> + b), Adam, exact "
        "expectations. The paper minimises a sigmoid-smoothed binomial error "
        "with SPSA on hardware shots, with zero-noise extrapolation.",
        "Only the variational classifier is implemented; the paper's quantum "
        "kernel estimation protocol is not.",
    ),
)
