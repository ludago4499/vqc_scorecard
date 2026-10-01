# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-09-01  Created. Noisy cost evaluation for G5: per-gate depolarizing
#               noise on a density-matrix device, family-agnostic via
#               qml.transforms.insert. The noiseless limit (p = 0) must and
#               does reproduce the statevector fast path of probes.gradient;
#               that agreement is a permanent test.
#   2026-10-01  Operations on more than two wires are decomposed before the
#               channels are inserted (probes.gates.elementary), so A5's
#               amplitude encoding is charged per Mottonen gate, as P6 counts
#               it and as the task noise model charges it. Families of one- and
#               two-wire gates (A1-A4, A6, A7) are unchanged.

"""Noisy cost evaluation, the primitive of the noise-induced BP diagnostic (G5).

What this probe adds
--------------------
Every other probe in this package runs the ideal circuit. This one runs the
same circuit under **depolarizing noise** and returns the same object the
G-series reads everywhere else: the central finite-difference derivative of a
cost over random angles.

The noise model, and why it is per-gate
---------------------------------------
A ``DepolarizingChannel(p)`` is inserted after **every gate** of the family's
circuit, on every wire that gate touches (``qml.transforms.insert``,
``position="all"``). The source model of Wang et al. 2021 places a channel per
*layer*; a layer is not the same object in the four families of this package
(entangling block, re-upload, word box), so a per-layer channel cannot be
declared once and applied identically to all of them. A per-gate channel can.
Two consequences, both deliberate and both declared in the G5 metric record:

* Gate-heavy circuits are noisier at equal (n, d). This couples G5 to the cost
  control P6, which is the NISQ-faithful reading: on hardware, more gates *is*
  more noise.
* The decay of the gradient stays exponential in depth, as in Wang's bound
  ``|dC| <= F(n) q^(L+1)``, because the per-gate channels compound layer by
  layer exactly as per-layer channels do, only faster.

Cost
----
Noise means density matrices: ``default.mixed`` stores ``4**n`` amplitudes and
does not broadcast over parameters, so every sample is one call. Keep ``n``
and the sample counts small; this is the one probe where the simulation, not
the estimator, is the budget.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from numpy.typing import NDArray

from ..ansatze.base import Ansatz
from .gates import elementary
from .gradient import FD_STEP


def build_noisy_cost(ansatz: Ansatz, n: int, d: int, noise: float,
                     kind: str, encoder_input=None):
    """Return a callable that evaluates the noisy cost at one angle vector.

    Args:
        ansatz: The family to run.
        n: Number of qubits.
        d: Depth, in the unit of the family.
        noise: Depolarizing probability of the per-gate channel, in [0, 1).
            ``0.0`` still runs the mixed-state simulator, so the p = 0 column
            is a control that must match the statevector path.
        kind: ``"global"`` for ``<Z...Z>``, anything else for ``<Z_0>``.
        encoder_input: Fixed input vector, or None.

    Returns:
        A function mapping one flat angle vector to a float cost.
    """
    dev = qml.device("default.mixed", wires=n)
    if kind == "global":
        observable = qml.prod(*[qml.PauliZ(q) for q in range(n)])
    else:
        observable = qml.PauliZ(0)

    @qml.qnode(dev, interface=None)
    def circuit(parameters):
        script = qml.tape.make_qscript(ansatz.apply)(parameters, n, d, encoder_input)
        for op in elementary(script.operations):
            qml.apply(op)
        return qml.expval(observable)

    noisy = qml.transforms.insert(circuit, qml.DepolarizingChannel, noise,
                                  position="all")
    return lambda theta: float(noisy(np.asarray(theta, dtype=float)))


def noisy_gradient_variance(
    ansatz: Ansatz,
    n: int,
    d: int,
    noise: float,
    samples: int,
    rng: np.random.Generator,
    angle_indices: NDArray[np.integer] | list[int],
    encoder_input=None,
    step: float = FD_STEP,
) -> tuple[float, float]:
    """Return the variance and mean |grad| of the noisy global cost derivative.

    Draws ``samples`` random angle vectors on ``[0, 2*pi)``, takes the central
    finite difference of the **global** cost on each index in
    ``angle_indices``, and pools the derivatives of all sampled angles, the
    same estimator G1 uses on the ideal state.

    Args:
        ansatz: The family to run.
        n: Number of qubits.
        d: Depth, in the unit of the family.
        noise: Per-gate depolarizing probability.
        samples: Random angle draws.
        rng: Generator from :mod:`..core.seeding`.
        angle_indices: Which angles to differentiate; the results are pooled.
        encoder_input: Fixed input vector, or None.
        step: Finite-difference displacement.

    Returns:
        A pair ``(variance, mean_abs)`` over all draws and angle indices.
    """
    cost = build_noisy_cost(ansatz, n, d, noise, "global", encoder_input)
    p = ansatz.n_params(n, d)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(samples, p))
    grads = np.empty((len(angle_indices), samples))
    for row, k in enumerate(angle_indices):
        for s in range(samples):
            plus, minus = theta[s].copy(), theta[s].copy()
            plus[k] += step
            minus[k] -= step
            grads[row, s] = (cost(plus) - cost(minus)) / (2.0 * step)
    per_angle_var = grads.var(axis=1)
    per_angle_abs = np.abs(grads).mean(axis=1)
    return float(per_angle_var.mean()), float(per_angle_abs.mean())
