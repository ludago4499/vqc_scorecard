# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below.
#
# History:
#   2026-10-01  Created. A5 is now the circuit-centric quantum classifier of
#               Schuld, Bocharov, Svore and Wiebe 2020 (amplitude encoding,
#               code blocks of general single-qubit gates and controlled
#               general gates). It replaces the retired A5 (the dressed
#               transfer-learning head of Mari et al. 2020), which carried
#               trainable classical layers and so broke the pure-v1 rule that
#               the circuit is the only trainable component.
#   2026-10-08  Documentation only (no gate, order or count changed). Declared
#               the controlled-gate layout against the paper's Eq. (9) rule
#               (orientation, order, and the extra gates at n = 6), and the
#               readout and loss. See docs/IMPLEMENTATION.md, section A5.

"""A5 - Circuit-centric quantum classifier (amplitude encoding).

Source: M. Schuld, A. Bocharov, K. M. Svore and N. Wiebe, "Circuit-centric
quantum classifiers", *Phys. Rev. A* **101**, 032308 (2020), Secs. III-IV and
Fig. 4.

Why this family is in the study
-------------------------------
A2 (SEL) is PennyLane's simplification of the same paper: it keeps the layout
but replaces the trainable controlled gates by fixed CNOTs and replaces the
amplitude encoding by one angle per qubit. A5 restores both. A2 and A5
therefore differ in exactly the two things the paper proposes, the encoding and
the trainable entangler, which makes the pair a clean Class II contrast.

Circuit
-------
1. Amplitude encoding: the input ``x`` of length ``2**n`` is normalised and
   loaded as the amplitudes of the ``n``-qubit state. ``n`` qubits carry
   ``2**n`` features, the logarithmic compression the paper is built on.
2. ``d`` code blocks. Block ``l`` holds a general single-qubit gate
   ``Rot(phi, theta, omega)`` on every qubit, then ``n`` controlled general
   gates ``CRot`` with control ``q`` and target ``(q + r_l) mod n``. The range
   alternates ``r = 1`` (even blocks) and ``r = 3`` (odd blocks), the two
   ranges of the paper's Fig. 4; ``r = 1`` is used when ``n <= 3``.
3. A final ``Rot`` on qubit 0, then ``<Z_0>`` (the paper reads the first qubit
   and adds a trainable bias, which is the task head's ``b``).

Angle count: ``p = 6 n d + 3``.

Controlled-gate layout against the paper
----------------------------------------
The paper's block (Sec. III A) holds ``n / gcd(n, r)`` controlled gates; gate
``j = 1 .. n / gcd(n, r)`` has control ``c_j = j r mod n`` and target
``t_j = (j - 1) r mod n``. This module places ``n`` gates, control ``q`` and
target ``(q + r) mod n``, in the order ``q = 0 .. n - 1``. Consequences:

* When ``gcd(n, r) = 1`` (``n = 4, 8`` with ``r = 1, 3``) the unordered pairs
  and the gate count are the paper's, but the orientation is reversed (target
  ahead of control by ``r`` here, behind it in the paper) and, for ``r = 3``,
  the order differs.
* At ``n = 6``, ``r = 3`` the paper's rule gives 2 gates, ``3 -> 0`` and
  ``0 -> 3``. This module places 6, every pair ``{q, q + 3}`` twice in both
  directions. The angle count is therefore above the paper's rule at ``n = 6``
  and ``d >= 2``: 111 against 99 at ``d = 3``, 183 against 159 at ``d = 5``.

Cost of the encoding
--------------------
Amplitude encoding is not free. Exact state preparation of an arbitrary real
vector (Mottonen et al. 2004) needs ``O(2**n)`` CNOTs. P6 counts them, because
:func:`~vqc_scorecard.metrics.p6_two_qubit.two_qubit_count` decomposes any
operation on more than two wires before counting, and the task noise model
does the same before it inserts channels. The NISQ cost of this family is
therefore reported honestly, not hidden inside one ``StatePrep`` operation.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from .base import Ansatz, Params, pauli_word


def n_params(n: int, d: int) -> int:
    """Return the angle count ``6nd + 3``."""
    return 6 * n * d + 3


def n_encoder(n: int, d: int) -> int:
    """Return the input length ``2**n``: one amplitude per basis state."""
    return 2 ** n


def block_range(n: int, layer: int) -> int:
    """Return the control-target range ``r`` of code block ``layer``.

    Args:
        n: Number of qubits.
        layer: Block index, from 0.

    Returns:
        ``1`` on even blocks, ``3`` on odd blocks, and ``1`` whenever ``n <= 3``.

    Examples:
        >>> [block_range(8, l) for l in range(4)]
        [1, 3, 1, 3]
    """
    if n <= 3:
        return 1
    return 1 if layer % 2 == 0 else 3


def embed(x, n: int) -> None:
    """Queue the amplitude encoding of ``x`` on the active tape.

    ``x`` may be one vector of length ``2**n`` or a batch ``(batch, 2**n)``. Each
    row is normalised to unit length. A row of zeros has no direction and
    cannot be encoded.

    Args:
        x: Features, ``2**n`` per row.
        n: Number of qubits.
    """
    qml.AmplitudeEmbedding(x, wires=range(n), normalize=True)


def generators(n: int) -> list[str]:
    """Return the DLA generators of the circuit-centric classifier.

    ``Rot = RZ RY RZ`` gives ``Y_q`` and ``Z_q`` on every wire. A controlled
    rotation ``C-R_sigma(t) = exp(-i t (I - Z_c) sigma_t / 4)`` has one generator,
    ``sigma_t - Z_c sigma_t``. Its first term is already in the algebra (the
    single-qubit set), so listing the two Pauli strings ``sigma_t`` and
    ``Z_c sigma_t`` separately gives the same closure. With ``sigma`` in
    ``{Y, Z}`` and a connected set of control-target bonds the closure is
    ``su(2**n)``. Both ranges are included, so the set is the union over blocks.

    Args:
        n: Number of qubits.

    Returns:
        The Pauli-string generators.
    """
    gens = []
    for q in range(n):
        gens.append(pauli_word(n, [(q, "Y")]))
        gens.append(pauli_word(n, [(q, "Z")]))
    ranges = {1} if n <= 3 else {1, 3}
    bonds = sorted({(q, (q + r) % n) for r in ranges for q in range(n) if (q + r) % n != q})
    for control, target in bonds:
        gens.append(pauli_word(n, [(control, "Z"), (target, "Y")]))
        gens.append(pauli_word(n, [(control, "Z"), (target, "Z")]))
    return gens


def apply(params: Params, n: int, d: int, x=None) -> None:
    """Queue the amplitude encoding when ``x`` is given, then ``d`` code blocks.

    Args:
        params: The ``6nd + 3`` angles, flat, or a batch with the angles on the
            last axis.
        n: Number of qubits.
        d: Number of code blocks.
        x: Features of length ``2**n`` (or a batch of them), or ``None`` to
            queue the variational part only.
    """
    p = params if hasattr(params, "shape") else np.asarray(params)
    if x is not None:
        embed(x if hasattr(x, "shape") else np.asarray(x, dtype=float), n)
    i = 0
    for layer in range(d):
        for q in range(n):
            qml.Rot(p[..., i], p[..., i + 1], p[..., i + 2], wires=q)
            i += 3
        r = block_range(n, layer)
        for q in range(n):
            t = (q + r) % n
            if t == q:
                continue
            qml.CRot(p[..., i], p[..., i + 1], p[..., i + 2], wires=[q, t])
            i += 3
    qml.Rot(p[..., i], p[..., i + 1], p[..., i + 2], wires=0)


A5 = Ansatz(
    tag="A5",
    name="Circuit-centric classifier (Schuld 2020)",
    n_params=n_params,
    apply=apply,
    n_encoder=n_encoder,
    permits=lambda n, d: n >= 2 and d >= 1,
    source="Schuld, Bocharov, Svore, Wiebe, Phys. Rev. A 101, 032308 (2020)",
    generators=generators,
    deviations=(
        "Ranges alternate r = 1, 3 as in the paper's 8-qubit Fig. 4; the paper "
        "gives no rule for other widths, so the same alternation is used at every "
        "n (r = 1 only when n <= 3).",
        "General gates are PennyLane Rot / CRot (RZ RY RZ Euler form). The paper "
        "parameterises the same SU(2) group differently; the reachable set is equal.",
        "The paper pads x with a constant and normalises; here x is normalised "
        "without padding (task side: standardised PCA(2^n) scores).",
        "The descriptor sweep holds x fixed (package convention), so P1 to P4 "
        "describe the ansatz at one encoded state.",
        "Controlled gates: n per block, control q, target (q + r) mod n. The "
        "paper places n / gcd(n, r) per block, control j r mod n, target "
        "(j - 1) r mod n. Same pairs and count when gcd(n, r) = 1, reversed "
        "orientation; at n = 6, r = 3 this module has 6 gates where the paper "
        "has 2 (p = 111 vs 99 at d = 3, 183 vs 159 at d = 5).",
        "Readout and loss: the paper uses pi = p(q_0 = 1) + b with a least-"
        "squares cost and single-sample SGD; the task head is "
        "sigmoid(a <Z_0> + b) with a class-balanced BCE and Adam.",
    ),
)
