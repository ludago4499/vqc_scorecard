# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-09-01  Created. Initialisation-scale sweep per Haug, Bharti & Kim
#               2021. Replaces the old G6 of metrics_part2.py entirely: that
#               design measured a cost (<Z...Z> from |0...0>) whose maximum
#               sits exactly at theta = 0, so the small-angle gradient
#               vanished by construction, and its 0-100 score was hardcoded
#               to 100 with keys the measurement never wrote. Neither the
#               number nor the score survives here. Fixes: a fixed random
#               product pre-layer moves the small-angle limit off the cost's
#               eigenstate; the reported quantity is the second moment (the
#               mean is nonzero at small scale, so the variance alone would
#               under-read the gradient); and the capacity side (QFIM rank)
#               is measured jointly, which is Haug's actual protocol.

"""G6 - initialisation sensitivity: the trainability/capacity trade-off.

The claims being tested
-----------------------
* Haug, Bharti & Kim, PRX Quantum 2, 040309 (2021): draw the angles as
  ``theta = a * theta_random`` and sweep the scale ``a``. Small ``a`` keeps
  gradients large but collapses the effective quantum dimension G_C (the rank
  of the QFIM); ``a -> 1`` restores full capacity and lands in the plateau.
  Near ``log10(a) ~ -2.5`` both are near-optimal for their circuits.
* Zhang, Liu, Hsieh & Tao, NeurIPS 2022: Gaussian initialisation with
  variance ``O(1/L)`` lower-bounds the expected squared gradient by
  ``1/poly(N, L)`` - polynomial, not exponential. The guarantee is nontrivial
  only when the gradient at ``theta = 0`` is itself non-negligible, which is
  exactly the condition the old G6 violated.
* Grant et al., Quantum 3, 214 (2019): identity-block initialisation removes
  the plateau at the *first step only*; nothing pins the trajectory there.
  (Not implemented; cited because it bounds what any init strategy can claim.)

The design faults this file exists to avoid
-------------------------------------------
Two separate degeneracies make the naive protocol vanish at small angles for
reasons that have nothing to do with plateaus:

* **State side.** From ``|0...0>``, both package costs are extremal at small
  angles - ``Z...Z`` and ``Z_0`` take their maximum value 1 there - so every
  derivative dies as ``a -> 0``. The old G6 measured precisely this artefact
  and read it as "Gaussian init is 110x worse". Fix: a fixed, seeded layer of
  single-qubit ``Rot`` gates **before** the ansatz, so the start is a generic
  product state.
* **Observable side.** At exactly ``theta = 0`` a rotation-plus-CNOT family
  is a Clifford circuit, and CNOT conjugation maps a Z-string to a Z-string.
  The effective observable seen by every Z-generator rotation is therefore
  still diagonal, its commutator is exactly zero, and the small-angle
  gradient vanishes structurally even with the pre-layer in place (measured:
  this alone leaves E[(dC)^2] ~ 1e-6 at a = 0.01). Fix: a fixed, seeded
  ``Rot`` layer **after** the ansatz, which is the same as measuring the
  Z-string in a fixed random local basis. Zhang et al. 2022 end their circuit
  with trailing RX and RY layers for exactly this reason.

Both layers are fixed local unitaries derived from the work's identity
(family, n, d): every seed and scale sees the same start state and the same
measurement basis, the QFIM is invariant under the post-layer, and no Haar or
manifold reference moves.

What is measured at one grid point
----------------------------------
For every scale in :data:`SCALES`: the **second moment** ``E[(dC)^2]`` of the
central finite-difference derivative (global and local cost, pooled over the
same angle spread G1 uses) - the quantity Zhang bounds, and equal to the
variance at ``a = 1`` where the mean vanishes by symmetry - and the mean QFIM
rank over :data:`RANK_POINTS` base points, computed with P3's probe and P3's
threshold, so the ``a = 1`` column must agree with P3. The two curves against
``a`` are the trade-off; where they cross is the finding.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..ansatze.base import Ansatz
from ..core.seeding import generator
from ..probes.gradient import FD_STEP, diagonal_observable, gradient_variance
from ..probes.qgt import quantum_fisher, state_jacobian
from ..probes.qgt import effective_dimension as qfim_effective_dimension
from ..probes.statevector import build_qnode
from .base import Metric
from .g1_gradient import _angle_indices
from .p3_effective_dim import RANK_TOL

SCALES: tuple[float, ...] = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
"""Initialisation scales a of theta = a * theta_random, roughly half-decade
steps from the uniform baseline a = 1 down to 1e-3. The grid brackets Haug's
near-optimal region log10(a) ~ -2.5 (between s5 = 0.003 and s6 = 0.001)."""

RANK_POINTS: int = 3
"""Base points per scale for the QFIM rank. The rank at fixed scale varies
little between points (it is a counting quantity); three points expose a
disagreement without paying P3's full 30-point protocol at every scale."""


def prelayer_angles(ansatz: Ansatz, n: int, d: int) -> np.ndarray:
    """Return the fixed (n, 3) pre-layer angles of one grid point.

    Derived from the identity of the work, never from the sweep's generator,
    so every seed, scale and re-run sees the same start state - the pre-layer
    is part of the protocol, exactly like a fixed encoder input.

    Args:
        ansatz: The family, whose tag enters the derivation.
        n: Number of qubits.
        d: Depth.

    Returns:
        An array of shape (n, 3) of Rot angles in [0, 2*pi).
    """
    return generator("G6-prelayer", ansatz.tag, n, d).uniform(0.0, 2.0 * np.pi, size=(n, 3))


def postlayer_angles(ansatz: Ansatz, n: int, d: int) -> np.ndarray:
    """Return the fixed (n, 3) post-layer angles of one grid point.

    A different derivation tag than the pre-layer, so the two fixed layers are
    independent. See the module note: without the post-layer, the theta = 0
    Clifford limit keeps the Z-string cost diagonal and zeroes the Z-generator
    gradients exactly.

    Args:
        ansatz: The family, whose tag enters the derivation.
        n: Number of qubits.
        d: Depth.

    Returns:
        An array of shape (n, 3) of Rot angles in [0, 2*pi).
    """
    return generator("G6-postlayer", ansatz.tag, n, d).uniform(0.0, 2.0 * np.pi, size=(n, 3))


def measure(
    ansatz: Ansatz,
    n: int,
    d: int,
    samples: int,
    rng: np.random.Generator,
    encoder_input=None,
) -> Mapping[str, float]:
    """Measure the init-scale trade-off at one grid point, from one seed.

    Args:
        ansatz: The family to measure.
        n: Number of qubits.
        d: Depth, in the unit of the family.
        samples: Random draws of theta_random behind each second moment.
        rng: Generator from :mod:`..core.seeding`.
        encoder_input: Fixed input vector, or None.

    Returns:
        A mapping with, for every scale index i: ``scale_si``,
        ``grad2_global_si``, ``grad2_local_si`` (second moments of the cost
        derivative) and ``qfim_rank_si`` (mean rank over the base points).
    """
    p = ansatz.n_params(n, d)
    pre = prelayer_angles(ansatz, n, d)
    post = postlayer_angles(ansatz, n, d)
    circuit = build_qnode(ansatz, n, d, encoder_input, prelayer=pre, postlayer=post)
    diag_global = diagonal_observable(n, "global")
    diag_local = diagonal_observable(n, "local")
    indices = _angle_indices(p)

    out: dict[str, float] = {}
    for i, scale in enumerate(SCALES):
        theta = scale * rng.uniform(0.0, 2.0 * np.pi, size=(samples, p))

        second_g, second_l = [], []
        for k in indices:
            plus, minus = theta.copy(), theta.copy()
            plus[:, k] += FD_STEP
            minus[:, k] -= FD_STEP
            states = np.asarray(
                circuit(np.concatenate([plus, minus], axis=0))).reshape(2 * samples, -1)
            upper, lower = states[:samples], states[samples:]
            # E[g^2] = Var[g] + (E[g])^2; gradient_variance returns (var, E|g|),
            # so recompute the moments directly from the derivative here.
            grad_g = ((np.abs(upper) ** 2) @ diag_global
                      - (np.abs(lower) ** 2) @ diag_global) / (2.0 * FD_STEP)
            grad_l = ((np.abs(upper) ** 2) @ diag_local
                      - (np.abs(lower) ** 2) @ diag_local) / (2.0 * FD_STEP)
            second_g.append(float(np.mean(grad_g ** 2)))
            second_l.append(float(np.mean(grad_l ** 2)))

        ranks = []
        for base in theta[:RANK_POINTS]:
            psi, jac = state_jacobian(ansatz, n, d, base, encoder_input,
                                      prelayer=pre, postlayer=post)
            rank, _, _ = qfim_effective_dimension(quantum_fisher(psi, jac), RANK_TOL)
            ranks.append(rank)

        out[f"scale_s{i}"] = float(scale)
        out[f"grad2_global_s{i}"] = float(np.mean(second_g))
        out[f"grad2_local_s{i}"] = float(np.mean(second_l))
        out[f"qfim_rank_s{i}"] = float(np.mean(ranks))
    return out


def reference(n: int, samples: int) -> Mapping[str, float]:
    """Return the capacity ceiling the rank column is read against.

    The state-manifold dimension ``2**(n+1) - 2``, the same ceiling P3
    reports; the rank at any scale can never exceed it.
    """
    del samples
    return {"manifold_dim": float(2 ** (n + 1) - 2)}


G6 = Metric(
    tag="G6",
    name="Initialisation sensitivity (gradient second moment and QFIM rank vs "
         "init scale)",
    unit="E[(dC)^2] of one finite-difference cost derivative, and QFIM rank, "
         "per initialisation scale a",
    source="Haug, Bharti & Kim, PRX Quantum 2, 040309 (2021); Zhang et al., "
           "NeurIPS 2022 (arXiv:2203.09376); first-step caveat per Grant et "
           "al., Quantum 3, 214 (2019)",
    measure=measure,
    reference=reference,
    reference_kind="analytic",
    deviations=(
        "A FIXED RANDOM PRODUCT PRE-LAYER precedes the ansatz and a FIXED "
        "POST-LAYER follows it (seeded Rot per wire, derived from family/n/d). "
        "Without the pre-layer, |0...0> is a maximum of both package costs and "
        "the small-angle derivative vanishes by construction - the design "
        "fault of the old G6. Without the post-layer, the theta = 0 circuit "
        "of a rotation-plus-CNOT family is Clifford, the Z-string cost stays "
        "diagonal, and every Z-generator gradient is exactly zero - so the "
        "post-layer measures the same cost in a fixed random local basis, as "
        "Zhang et al.'s trailing RX/RY layers do. Both are fixed local "
        "unitaries: they move the start state and the basis, not the geometry.",
        "The reported gradient quantity is the SECOND MOMENT E[(dC)^2], not "
        "the variance: at small scale the mean derivative is nonzero, so the "
        "variance alone would under-read trainability. At a = 1 the mean "
        "vanishes by symmetry and the second moment equals G1's variance.",
        "Scales are drawn as theta = a * U[0, 2*pi), Haug's protocol, rather "
        "than the N(0, sigma^2) of Zhang; Zhang's sigma^2 = O(1/L) guarantee "
        "is quoted on the definition slide but the swept variable is a.",
        "The QFIM rank uses P3's probe and threshold at RANK_POINTS base "
        "points per scale, not P3's full 30-point protocol; the a = 1 column "
        "is checked against P3 in the tests.",
        "Identity-block initialisation (Grant 2019) is cited, not implemented.",
    ),
)
