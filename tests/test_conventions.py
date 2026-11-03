# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-10-08  Created. One check per convention that docs/IMPLEMENTATION.md
#               states about how a family's code relates to its source paper
#               (gate signs, operator order, counts, readout state, input
#               range, exact library equivalences, the known departures). A
#               statement in the document without a check here is a claim, not
#               a reproduction note.

"""Paper-to-code conventions of A1-A7, each checked numerically.

Every section of ``docs/IMPLEMENTATION.md`` that says "the code does X where
the paper does Y" has a check here, so the document cannot drift from the
code. Departures are checked too: a check that confirms a known departure
(for example A5 at ``n = 6``) is labelled ``departure`` and passes when the
departure is still exactly as documented.

Run::

    python -m vqc_scorecard.tests.test_conventions
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import reduce
from math import gcd

import numpy as np
import pennylane as qml
from scipy.linalg import expm

from ..ansatze import REGISTRY as ANSATZE
from ..ansatze import a5_circuit_centric, a6_iqp, a7_farhi_neven
from ..core.seeding import generator
from ..tasks.features import _to_angles
from ..tasks.model import _reupload_body, circuit_body

OUTCOMES: list[tuple[bool, str]] = []

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1.0, -1.0]).astype(complex)


def check(passed: bool, label: str, detail: str = "") -> None:
    """Record one check and print its outcome.

    Args:
        passed: Whether the check passed.
        label: One line naming the convention.
        detail: Optional numbers behind the verdict.
    """
    OUTCOMES.append((bool(passed), label))
    print(f"[{'PASS' if passed else 'FAIL'}] {label}" + (f"  --  {detail}" if detail else ""))


def _kron(*mats: np.ndarray) -> np.ndarray:
    return reduce(np.kron, mats)


def _state(fn: Callable[[], None], n: int) -> np.ndarray:
    dev = qml.device("default.qubit", wires=n)
    return np.asarray(qml.QNode(lambda: (fn(), qml.state())[1], dev)())


def _z0(fn: Callable[[], None], n: int) -> float:
    dev = qml.device("default.qubit", wires=n)
    return float(qml.QNode(lambda: (fn(), qml.expval(qml.PauliZ(0)))[1], dev)())


def _same_ray(a: np.ndarray, b: np.ndarray) -> float:
    """Return ``1 - |<a|b>|``: zero when the states agree up to a global phase."""
    return float(1.0 - abs(np.vdot(a, b)))


def _two_qubit_ops(fn: Callable[[], None]) -> list[qml.operation.Operator]:
    tape = qml.tape.make_qscript(fn)()
    return [op for op in tape.operations if len(op.wires) == 2]


# --------------------------------------------------------------------------- #
# A1  Kandala et al. 2017                                                      #
# --------------------------------------------------------------------------- #

def test_a1_count_and_dropped_z() -> None:
    """Layer 0 is RX, RZ: the first-in-time Z of the Euler triple acts on |0>
    and only adds a global phase, so dropping it loses nothing (Kandala Eq. 2)."""
    n, d = 3, 2
    check(ANSATZE["A1"].n_params(n, d) == n * (3 * d + 2), "A1: p = n(3d + 2)",
          f"{ANSATZE['A1'].n_params(n, d)}")
    rng = generator("test-conventions", "a1")
    a, b, c = rng.uniform(0, 2 * np.pi, 3)
    full = _state(lambda: (qml.RZ(c, 0), qml.RX(b, 0), qml.RZ(a, 0)), 1)
    dropped = _state(lambda: (qml.RX(b, 0), qml.RZ(a, 0)), 1)
    check(_same_ray(full, dropped) < 1e-12, "A1: a leading RZ on |0> is a global phase",
          f"1 - |overlap| = {_same_ray(full, dropped):.1e}")
    names = [op.name for op in qml.tape.make_qscript(
        lambda: ANSATZE["A1"].apply(np.zeros(ANSATZE["A1"].n_params(n, 1)), n, 1))().operations]
    want = ["RX", "RZ"] * n + ["CNOT"] * (n - 1) + ["RZ", "RX", "RZ"] * n
    check(names == want, "A1: time order RX RZ | CNOT chain | RZ RX RZ")


# --------------------------------------------------------------------------- #
# A2  Schuld et al. 2020 via PennyLane StronglyEntanglingLayers                #
# --------------------------------------------------------------------------- #

def test_a2_is_pennylane_sel() -> None:
    """A2 equals qml.StronglyEntanglingLayers with default ranges and CNOT."""
    rng = generator("test-conventions", "a2")
    worst = 0.0
    for n, d in ((2, 2), (4, 3), (5, 5)):
        th = rng.uniform(0, 2 * np.pi, ANSATZE["A2"].n_params(n, d))
        got = _state(lambda: ANSATZE["A2"].apply(th, n, d), n)
        want = _state(lambda: qml.StronglyEntanglingLayers(th.reshape(d, n, 3), wires=range(n)), n)
        worst = max(worst, float(np.abs(got - want).max()))
    check(worst < 1e-12, "A2: identical to qml.StronglyEntanglingLayers (n, d) in "
          "{(2,2), (4,3), (5,5)}", f"max err {worst:.1e}")


# --------------------------------------------------------------------------- #
# A3  Perez-Salinas et al. 2020                                                #
# --------------------------------------------------------------------------- #

def test_a3_task_body_and_entangler() -> None:
    """The task body re-implements apply exactly; the ring sits only between
    layers, range 1, so d = 1 has no two-qubit gate."""
    rng = generator("test-conventions", "a3")
    n, d = 4, 3
    th = rng.uniform(0, 2 * np.pi, ANSATZE["A3"].n_params(n, d))
    x = rng.uniform(0, np.pi, n)
    a = _state(lambda: ANSATZE["A3"].apply(th, n, d, x), n)
    b = _state(lambda: _reupload_body(th, x, n, d), n)
    check(np.abs(a - b).max() < 1e-12, "A3: tasks.model._reupload_body == ansatz apply",
          f"max err {np.abs(a - b).max():.1e}")
    ops = _two_qubit_ops(lambda: ANSATZE["A3"].apply(th, n, d, x))
    pairs = [tuple(op.wires) for op in ops]
    ring = [(q, (q + 1) % n) for q in range(n)]
    check(pairs == ring * (d - 1), "A3: (d - 1) range-1 CNOT rings, none after the last layer",
          f"{len(pairs)} CNOTs")
    th1 = th[: ANSATZE["A3"].n_params(n, 1)]
    check(not _two_qubit_ops(lambda: ANSATZE["A3"].apply(th1, n, 1, x)),
          "A3: d = 1 has no entangler (README 7.4)")


def test_a2_a3_entanglers_differ() -> None:
    """Departure: A2 (angle) and A3 (re-upload) at d = 1 and equal angles are
    different circuits, because A2 rings after its layer and A3 does not."""
    rng = generator("test-conventions", "a2a3")
    n = 4
    th = rng.uniform(0, 2 * np.pi, 3 * n)
    x = rng.uniform(0, np.pi, n)
    a2 = _state(lambda: circuit_body("A2", n, 1)(th, x), n)
    a3 = _state(lambda: circuit_body("A3", n, 1)(th, x), n)
    gap = _same_ray(a2, a3)
    check(gap > 1e-3, "A2 vs A3 (departure): same Euler block, different entangler",
          f"1 - |overlap| = {gap:.3f}")


# --------------------------------------------------------------------------- #
# A4  DisCoCat template                                                        #
# --------------------------------------------------------------------------- #

def test_a4_cup_and_task_offset() -> None:
    """The cup's CNOT + H maps the Bell state to |00> (the post-selected
    outcome); the task's RY(x) before the word-state RY(theta) is RY(x + theta)."""
    bell = _state(lambda: (qml.Hadamard(0), qml.CNOT([0, 1]), qml.CNOT([0, 1]), qml.Hadamard(0)), 2)
    check(abs(abs(bell[0]) - 1.0) < 1e-12, "A4: cup unitary sends the Bell state to |00>")
    rng = generator("test-conventions", "a4")
    n, d = 4, 2
    th = rng.uniform(0, 2 * np.pi, ANSATZE["A4"].n_params(n, d))
    x = rng.uniform(0, np.pi, n)
    shifted = th.copy()
    shifted[:n] += x
    a = _state(lambda: circuit_body("A4", n, d)(th, x), n)
    b = _state(lambda: ANSATZE["A4"].apply(shifted, n, d), n)
    check(np.abs(a - b).max() < 1e-12, "A4: task path = word-state angles offset by x",
          f"max err {np.abs(a - b).max():.1e}")
    ops = _two_qubit_ops(lambda: ANSATZE["A4"].apply(th, n, d))
    crz = [tuple(op.wires) for op in ops if op.name == "CRZ"]
    check(crz[:n] == [(q, (q + 1) % n) for q in range(n)],
          "A4: CRZ ring with wrap-around across word qubits (not a lambeq ladder)")


# --------------------------------------------------------------------------- #
# A5  Schuld et al. 2020, circuit-centric                                      #
# --------------------------------------------------------------------------- #

def _paper_block_gates(n: int, r: int) -> list[tuple[int, int]]:
    """Return the paper's (control, target) list of one code block (Sec. III A)."""
    return [((j * r) % n, ((j - 1) * r) % n) for j in range(1, n // gcd(n, r) + 1)]


def test_a5_controlled_layout() -> None:
    """Same unordered pairs and count as the paper when gcd(n, r) = 1, with the
    orientation reversed; at n = 6, r = 3 six gates where the paper has two."""
    ok = True
    for n in (4, 8):
        for r in (1, 3):
            code = [(q, (q + r) % n) for q in range(n)]
            paper = _paper_block_gates(n, r)
            ok &= (len(code) == len(paper)
                   and {frozenset(p) for p in code} == {frozenset(p) for p in paper}
                   and {(t, c) for c, t in paper} == set(code))
    check(ok, "A5: n = 4, 8 -- paper's pairs and count, orientation reversed")
    n, d = 6, 3
    th = np.zeros(ANSATZE["A5"].n_params(n, d))
    odd = [tuple(op.wires) for op in _two_qubit_ops(lambda: ANSATZE["A5"].apply(th, n, 2))
           if op.name == "CRot"][n:]
    paper = _paper_block_gates(6, 3)
    check(len(odd) == 6 and len(paper) == 2,
          "A5 (departure): n = 6, r = 3 places 6 CRot, the paper's rule 2",
          f"code {odd}, paper {paper}")
    ranges = [a5_circuit_centric.block_range(n, l) for l in range(d)]
    paper_p = 3 * (n * d + 1) + 3 * sum(n // gcd(n, r) for r in ranges)
    check((ANSATZE["A5"].n_params(6, 3), paper_p) == (111, 99),
          "A5 (departure): p = 111 at n = 6, d = 3; the paper's rule gives 99")


# --------------------------------------------------------------------------- #
# A6  Havlicek et al. 2019                                                     #
# --------------------------------------------------------------------------- #

def test_a6_signs_and_range() -> None:
    """RZ(-2x) = exp(i x Z), IsingZZ(-2 phi) = exp(i phi ZZ); the task angles lie
    on [0, pi], so (pi - x_q)(pi - x_r) is never negative."""
    x, phi = 0.83, 1.71
    rz = qml.matrix(qml.RZ(-2 * x, wires=0))
    zz = qml.matrix(qml.IsingZZ(-2 * phi, wires=[0, 1]))
    err = max(np.abs(rz - expm(1j * x * _Z)).max(), np.abs(zz - expm(1j * phi * _kron(_Z, _Z))).max())
    check(err < 1e-12, "A6: RZ(-2x) = exp(ixZ) and IsingZZ(-2phi) = exp(i phi ZZ)", f"{err:.1e}")
    z = np.array([[-50.0, 0.0, 50.0]])
    ang = _to_angles(z, np.zeros(3), np.ones(3))
    check(np.allclose(ang, [[0.0, np.pi / 2, np.pi]]),
          "A6: task angles = (clip(s, -3, 3) + 3) / 6 * pi, on [0, pi]")
    grid = np.linspace(0.0, np.pi, 61)
    prod = (np.pi - grid)[:, None] * (np.pi - grid)[None, :]
    check(prod.min() >= 0.0 and abs(prod.max() - np.pi ** 2) < 1e-12,
          "A6 (departure): ZZ phase on [0, pi^2], never negative")
    check(ANSATZE["A6"].n_params(2, 4) == 2 * 2 * (4 + 1), "A6: p = 2n(d + 1)")


# --------------------------------------------------------------------------- #
# A7  Farhi and Neven 2018                                                     #
# --------------------------------------------------------------------------- #

def _a7_paper_readout(theta: np.ndarray, x: np.ndarray, n: int, d: int) -> None:
    """The A7 circuit with the paper's readout |0> (no X on wire 0)."""
    for k, wire in enumerate(a7_farhi_neven.data_wires(n)):
        qml.RY(x[k], wires=wire)
    i = 0
    for _ in range(d):
        for word in ("ZX", "XX"):
            for j in a7_farhi_neven.data_wires(n):
                qml.PauliRot(-2.0 * theta[i], word, wires=[j, 0])
                i += 1
    qml.RX(np.pi / 2, wires=0)


def test_a7_gate_sign_and_count() -> None:
    """PauliRot(-2t, P) = exp(i t P), the paper's Eq. (8); n = 17, d = 3 is 96."""
    t = 0.47
    m = qml.matrix(qml.PauliRot(-2 * t, "ZX", wires=[0, 1]))
    err = np.abs(m - expm(1j * t * _kron(_Z, _X))).max()
    check(err < 1e-12, "A7: PauliRot(-2t, ZX) = exp(i t Z X)", f"{err:.1e}")
    check(ANSATZE["A7"].n_params(17, 3) == 96, "A7: n = 17, d = 3 gives the paper's 96 angles")


def test_a7_readout_sign() -> None:
    """The paper's Eq. (20): with readout |0>, U = exp(i beta/2 sum Z_j X_0)
    gives <Y> = sin(beta sum z_j). The package's |1> gives exactly minus that."""
    n, d = 4, 1
    beta = 0.9 * np.pi / 3
    x = np.array([0.0, np.pi, 0.0])
    z = np.where(x == 0.0, 1.0, -1.0)
    th = np.concatenate([np.full(n - 1, beta / 2), np.zeros(n - 1)])
    paper = _z0(lambda: _a7_paper_readout(th, x, n, d), n)
    package = _z0(lambda: ANSATZE["A7"].apply(th, n, d, x), n)
    want = np.sin(beta * z.sum())
    check(abs(paper - want) < 1e-12, "A7: readout |0> reproduces Farhi-Neven Eq. (20)",
          f"{paper:.6f} vs {want:.6f}")
    check(abs(package + want) < 1e-12, "A7: readout |1> (package) is the exact negative",
          f"{package:.6f}")
    rng = generator("test-conventions", "a7sign")
    n, d = 5, 3
    th = rng.uniform(0, 2 * np.pi, ANSATZE["A7"].n_params(n, d))
    x = rng.uniform(0, np.pi, n - 1)
    a = _z0(lambda: _a7_paper_readout(th, x, n, d), n)
    b = _z0(lambda: ANSATZE["A7"].apply(th, n, d, x), n)
    check(abs(a + b) < 1e-12, "A7: sign flip holds at random angles and features",
          f"{a:.6f} vs {b:.6f}")


def test_a7_product_form() -> None:
    """<Y_0> = Im prod_j <phi_j| (u_j^+)^dagger u_j^- |phi_j> (package readout |1>)."""
    rng = generator("test-conventions", "a7prod")
    n, d = 5, 3
    th = rng.uniform(0, 2 * np.pi, ANSATZE["A7"].n_params(n, d))
    x = rng.uniform(0, np.pi, n - 1)
    circuit = _z0(lambda: ANSATZE["A7"].apply(th, n, d, x), n)
    prod = 1.0 + 0.0j
    for k in range(n - 1):
        phi = np.array([np.cos(x[k] / 2), np.sin(x[k] / 2)], dtype=complex)
        up, um = np.eye(2, dtype=complex), np.eye(2, dtype=complex)
        i = 0
        for _ in range(d):
            for pauli in (_Z, _X):
                for j in range(n - 1):
                    if j == k:
                        up = expm(1j * th[i] * pauli) @ up
                        um = expm(-1j * th[i] * pauli) @ um
                    i += 1
        prod *= np.vdot(phi, up.conj().T @ um @ phi)
    check(abs(circuit - prod.imag) < 1e-12, "A7: output is Im of a product of n - 1 "
          "single-qubit overlaps", f"{circuit:.10f} vs {prod.imag:.10f}")


def test_a7_basis_endpoints() -> None:
    """RY(0)|0> = |0> and RY(pi)|0> = |1>: the feature endpoints are basis states."""
    s0 = _state(lambda: qml.RY(0.0, 0), 1)
    s1 = _state(lambda: qml.RY(np.pi, 0), 1)
    check(np.allclose(np.abs(s0), [1, 0]) and np.allclose(np.abs(s1), [0, 1]),
          "A7: x = 0 and x = pi load |0> and |1>")


def main() -> int:
    """Run every check and return the process exit code."""
    test_a1_count_and_dropped_z()
    test_a2_is_pennylane_sel()
    test_a3_task_body_and_entangler()
    test_a2_a3_entanglers_differ()
    test_a4_cup_and_task_offset()
    test_a5_controlled_layout()
    test_a6_signs_and_range()
    test_a7_gate_sign_and_count()
    test_a7_readout_sign()
    test_a7_product_form()
    test_a7_basis_endpoints()
    failures = [label for passed, label in OUTCOMES if not passed]
    print(f"\n{len(OUTCOMES) - len(failures)}/{len(OUTCOMES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
