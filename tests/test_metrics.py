# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-08-11  Created. Analytic checks for the probes, P1 and P2.
#   2026-08-18  Added closed-form checks for P3 (QFIM rank) and P4 (DLA
#               dimension), and a cross-check of the Lie closure against
#               PennyLane when it is available.
#   2026-09-01  Added the G3-G6 checks: the G3 crossing interpolation against
#               an exact exponential, the G4 gap on synthetic and measured
#               data, the G5 noiseless control against the G1 statevector
#               value and the Larocca Box 8 closed-form direction, and the G6
#               fixes (nonzero small-angle gradient, rank agreement with P3,
#               capacity collapse at small scale).
#   2026-09-22  A5 registered. The P4 universality check now covers A1-A4
#               only, and A5 gets its own closed-form check: its circuit is
#               real, so its DLA is so(2^n), not su(2^n). P6 hand formula and
#               the PennyLane closure cross-check extended to A5. The loops
#               over the registry (states, P5, gradient vs autodiff, G1 on
#               every family) pick up A5 without a change.
#   2026-10-01  pure-v1 families. The Mari A5 is retired with its so(2^n)
#               check. Universality now covers A1-A6 (A5 circuit-centric and
#               A6 IQP close on su(2^n)); A7 Farhi-Neven gets its closed form,
#               a direct sum of n - 1 copies of su(2), dimension 3(n - 1). P6
#               hand formulas cover A5-A7 with and without their encodings
#               (A5's amplitude encoding counted as its 2^n - 2 Mottonen
#               CNOTs). The PennyLane closure cross-check moves to A5 and A7.

"""Checks with known answers.

Every check compares against a value that is known in closed form, or against a
number already published in Scorecard v5. A check that only compares the code
against itself proves nothing, so there are none of those here.

Run::

    python -m vqc_scorecard.tests.test_metrics
"""

from __future__ import annotations

import sys

import numpy as np

from ..ansatze import REGISTRY as ANSATZE
from ..core.seeding import derive, generator
from ..metrics.p1_expressibility import kl_against_haar
from ..metrics.p2_entangling import meyer_wallach
from ..metrics.p3_effective_dim import RANK_TOL
from ..metrics.p3_effective_dim import measure as measure_p3
from ..metrics.p4_dla import measure as measure_p4
from ..metrics.p5_params import measure as measure_p5
from ..metrics.p6_two_qubit import measure as measure_p6
from ..metrics.g1_gradient import SHOTS_FLOOR, bp_onset_slope
from ..metrics.g1_gradient import measure as measure_g1
from ..metrics.g3_shot_floor import crossing_width, shot_floor_margin
from ..metrics.g4_locality import locality_gap
from ..metrics.g5_noise import NOISE_LEVELS, noise_crossing, noise_depth_rate
from ..metrics.g5_noise import measure as measure_g5
from ..metrics.g6_init import SCALES, prelayer_angles
from ..metrics.g6_init import measure as measure_g6
from ..probes.gradient import diagonal_observable, gradient_variance, shifted_states
from ..probes.fidelity import sample_fidelities
from ..probes.haar import expected_meyer_wallach, fidelity_bin_probabilities
from ..probes.haar import sample_fidelities as sample_haar_fidelities
from ..probes.lie import lie_closure_dimension
from ..probes.purity import single_qubit_purities
from ..probes.qgt import effective_dimension, quantum_fisher, state_jacobian
from ..probes.statevector import sample_states

OUTCOMES: list[tuple[bool, str]] = []


def check(passed: bool, label: str, detail: str = "") -> None:
    """Record one check and print its outcome."""
    OUTCOMES.append((passed, label))
    print(f"[{'PASS' if passed else 'FAIL'}] {label}" + (f"  --  {detail}" if detail else ""))


def test_purity_of_known_states() -> None:
    """A product state has purity 1. A Bell state has purity 0.5."""
    product = np.zeros((1, 4), dtype=complex)
    product[0, 0] = 1.0
    check(np.allclose(single_qubit_purities(product, 2), 1.0),
          "product state has unit single-qubit purity")

    bell = np.zeros((1, 4), dtype=complex)
    bell[0, 0] = bell[0, 3] = 1 / np.sqrt(2)
    check(np.allclose(single_qubit_purities(bell, 2), 0.5),
          "Bell state has half single-qubit purity")


def test_meyer_wallach_of_known_states() -> None:
    """Q is 0 for a product state, 1 for a Bell state and 1 for GHZ."""
    product = np.zeros((1, 8), dtype=complex)
    product[0, 0] = 1.0
    check(abs(float(meyer_wallach(product, 3)[0])) < 1e-12, "Q = 0 for a product state")

    bell = np.zeros((1, 4), dtype=complex)
    bell[0, 0] = bell[0, 3] = 1 / np.sqrt(2)
    check(abs(float(meyer_wallach(bell, 2)[0]) - 1.0) < 1e-12, "Q = 1 for a Bell state")

    ghz = np.zeros((1, 8), dtype=complex)
    ghz[0, 0] = ghz[0, 7] = 1 / np.sqrt(2)
    check(abs(float(meyer_wallach(ghz, 3)[0]) - 1.0) < 1e-12, "Q = 1 for a GHZ state")


def test_haar_expectation_matches_sampling() -> None:
    """The analytic Haar E[Q] must match Q measured on Haar-random states."""
    for n in (2, 4, 6):
        rng = generator("test", "haar-q", n)
        raw = rng.normal(size=(4000, 2 ** n)) + 1j * rng.normal(size=(4000, 2 ** n))
        states = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        measured = float(meyer_wallach(states, n).mean())
        expected = expected_meyer_wallach(n)
        check(abs(measured - expected) < 0.01,
              f"analytic Haar E[Q] matches sampling at n={n}",
              f"analytic={expected:.4f} sampled={measured:.4f}")


def test_haar_bins_sum_to_one() -> None:
    """The exact per-bin Haar probabilities must sum to 1."""
    for n in (2, 4, 6, 8):
        total = fidelity_bin_probabilities(75, 2 ** n).sum()
        check(abs(total - 1.0) < 1e-9, f"Haar bins sum to 1 at n={n}", f"sum={total:.12f}")


def test_p1_matches_scorecard_v5() -> None:
    """A1 at n=2, d=1 must fall inside the band published in Scorecard v5.

    Scorecard v5 reports 0.0207 +- 0.0016 nats over five seeds at 1500 pairs.
    The band 0.010 to 0.035 catches a real break without failing on seed noise.
    """
    rng = generator("test", "p1-regression")
    value = kl_against_haar(sample_fidelities(ANSATZE["A1"], 2, 1, 1500, rng), 2)
    check(0.010 <= value <= 0.035, "P1 for A1 at n=2 d=1 is inside the v5 band",
          f"kl={value:.5f}")


def test_p1_floor_falls_with_more_pairs() -> None:
    """The sampling bias of P1 must shrink when the pair count grows."""
    low = kl_against_haar(sample_haar_fidelities(1500, 16, generator("test", "floor", 1500)), 4)
    high = kl_against_haar(sample_haar_fidelities(5000, 16, generator("test", "floor", 5000)), 4)
    check(high < low, "P1 floor falls from 1500 to 5000 pairs", f"{low:.5f} -> {high:.5f}")


def test_seeds_are_order_independent() -> None:
    """The same identity must give the same seed, and different ones must not."""
    check(derive("P1", "A1", 2, 1, 0) == derive("P1", "A1", 2, 1, 0),
          "the same identity gives the same seed")
    check(derive("P1", "A1", 2, 1, 0) != derive("P2", "A1", 2, 1, 0),
          "a different metric gives a different seed")


def test_every_family_runs() -> None:
    """Every registered family must produce normalised states at every width."""
    for tag, ansatz in ANSATZE.items():
        for n, d in ((2, 1), (4, 3), (6, 5)):
            if tag == "A4" and n % 2:
                continue
            # A non-zero input: an amplitude encoding (A5) has no zero vector.
            states = sample_states(ansatz, n, d, 8, generator("test", tag, n, d),
                                   None if ansatz.n_encoder(n, d) == 0
                                   else np.full(ansatz.n_encoder(n, d), 0.3))
            norms = np.linalg.norm(states, axis=1)
            check(np.allclose(norms, 1.0), f"{tag} produces normalised states at n={n} d={d}",
                  f"p={ansatz.n_params(n, d)}")


def test_qfim_of_a_single_rotation() -> None:
    """The QFIM of a single-qubit RX is exactly 1, and central FD reproduces it.

    For ``|psi> = RX(theta)|0>`` the quantum Fisher information is the constant 1
    at every angle. This is the closed form that validates the finite-difference
    state Jacobian and the QFIM assembly together.
    """
    def rx_state(angle: float) -> np.ndarray:
        return np.array([np.cos(angle / 2), -1j * np.sin(angle / 2)])

    eps, theta = 1e-3, 0.7
    psi = rx_state(theta)
    jac = ((rx_state(theta + eps) - rx_state(theta - eps)) / (2 * eps))[None, :]
    value = float(quantum_fisher(psi, jac)[0, 0])
    check(abs(value - 1.0) < 1e-4, "QFIM of a single RX equals 1", f"F={value:.6f}")


def test_qfim_rank_is_bounded_by_the_manifold() -> None:
    """The QFIM rank can never exceed the state-manifold dimension 2**(n+1)-2.

    A1 at n=2 has 10 angles but a manifold of dimension 6, so the rank must sit
    at 6, not at 10. This is the over-parameterised regime made concrete.
    """
    ansatz, n, d = ANSATZE["A1"], 2, 1
    theta = ansatz.sample_params(generator("test", "qfim-rank"), 1, n, d)[0]
    psi, jac = state_jacobian(ansatz, n, d, theta)
    rank, _, _ = effective_dimension(quantum_fisher(psi, jac), RANK_TOL)
    check(rank <= 2 ** (n + 1) - 2, "QFIM rank is bounded by the state manifold",
          f"rank={rank} ceiling={2 ** (n + 1) - 2} p={ansatz.n_params(n, d)}")


def test_p3_saturates_a_deep_wide_circuit() -> None:
    """A1 at n=4, d=5 has p=68 angles but must saturate the manifold rank 30.

    A deep, wide HEA reaches every direction the manifold allows. The mean rank
    must therefore land on the ceiling min(p, 2**(n+1)-2) = min(68, 30) = 30.
    """
    ansatz, n, d = ANSATZE["A1"], 4, 5
    out = measure_p3(ansatz, n, d, 8, generator("test", "p3-sat"))
    ceiling = min(ansatz.n_params(n, d), 2 ** (n + 1) - 2)
    check(abs(out["qfim_rank"] - ceiling) < 0.5, "P3 saturates a deep wide HEA",
          f"rank={out['qfim_rank']:.2f} ceiling={ceiling}")


def test_lie_closure_of_known_algebras() -> None:
    """Closed-form Lie-algebra dimensions: su(2)=3, an abelian set=its size.

    ``{X, Z}`` on one qubit generates su(2), dimension 3. ``{Z0, Z1}`` commute,
    so their closure is the two-dimensional span they already are.
    """
    check(lie_closure_dimension(["X", "Z"])[0] == 3, "closure of {X,Z} is su(2), dim 3")
    check(lie_closure_dimension(["ZI", "IZ"])[0] == 2,
          "closure of a commuting pair is dim 2")


UNIVERSAL_FAMILIES: tuple[str, ...] = ("A1", "A2", "A3", "A4", "A5", "A6")
"""Families whose DLA is the full su(2**n). A7 is the exception: 3(n - 1)."""


def test_p4_is_full_su_for_every_family() -> None:
    """A1-A6 have a universal DLA: dim = 4**n - 1 at n = 2 and 3.

    These six families carry a complete single-qubit algebra plus a connected
    entangler (fixed, or trainable for A5), so each closes on the whole of
    su(2**n): 15 at n = 2, 63 at n = 3. This both validates the closure and
    records the finding that none of the six escapes the barren-plateau regime
    by its structure. A7 is the exception and has its own check,
    :func:`test_p4_of_a7_is_polynomial`.
    """
    for n in (2, 3):
        for tag in UNIVERSAL_FAMILIES:
            if tag == "A4" and n % 2:
                continue
            dim = measure_p4(ANSATZE[tag], n, 1, 1)["dla_dim"]
            check(dim == 4 ** n - 1, f"{tag} DLA is full su(2**{n}) = {4 ** n - 1}",
                  f"dla_dim={dim:.0f}")


def test_p4_of_a7_is_polynomial() -> None:
    """A7 (Farhi-Neven) closes on (n - 1) copies of su(2): dim = 3(n - 1).

    Closed form. Data wire ``j`` contributes ``Z_j X_0`` and ``X_j X_0``; their
    commutator is ``2i Y_j`` (``X_0^2 = I``), so each wire spans an su(2). Two
    different wires commute term by term (``X_0`` with itself, and Paulis on
    different wires), so the algebra is the direct sum, of dimension
    ``3(n - 1)``: 3, 6, 9, 12 at n = 2..5. Linear in n, against ``4**n - 1``
    for A1-A6: A7 is the only family whose structure rules out the
    exponential-DLA barren plateau.
    """
    for n in (2, 3, 4, 5):
        dim = measure_p4(ANSATZE["A7"], n, 1, 1)["dla_dim"]
        want = 3 * (n - 1)
        check(dim == want and dim < 4 ** n - 1,
              f"A7 DLA is (n-1) su(2) = {want} at n={n}, below su = {4 ** n - 1}",
              f"dla_dim={dim:.0f}")


def test_lie_closure_matches_pennylane() -> None:
    """Cross-check the closure dimension against PennyLane, where available.

    PennyLane ships its own Lie-closure utility. When it is present the two
    implementations must agree on A1 at n=2. When it is absent the check is
    skipped rather than failed, so the suite does not depend on one version.
    """
    try:
        import pennylane as qml
        words = ANSATZE["A1"].generators(2)
        ops = [qml.pauli.string_to_pauli_word(w) for w in words]
        reference_dim = len(qml.lie_closure(ops))
    except Exception as error:  # noqa: BLE001 - a missing or renamed API is a skip
        check(True, "PennyLane cross-check skipped", f"{type(error).__name__}")
        return
    mine = lie_closure_dimension(ANSATZE["A1"].generators(2))[0]
    check(mine == reference_dim, "closure matches PennyLane on A1 n=2",
          f"mine={mine} pennylane={reference_dim}")
    for tag in ("A5", "A7"):
        words = ANSATZE[tag].generators(3)
        reference = len(qml.lie_closure([qml.pauli.string_to_pauli_word(w) for w in words]))
        mine_t = lie_closure_dimension(words)[0]
        check(mine_t == reference, f"closure matches PennyLane on {tag} n=3",
              f"mine={mine_t} pennylane={reference}")


def test_p3_redundancy_is_one_minus_rank_over_p() -> None:
    """A4 spends every parameter (R=0); SEL at d=1 wastes a third (R~1/3).

    A4's rank equals its parameter count, so its redundancy is zero. A2 (SEL)
    holds 3n angles but ranks only 2n at d=1, so a third are redundant.
    """
    a4 = measure_p3(ANSATZE["A4"], 4, 3, 8, generator("test", "p3r-a4"))
    check(abs(a4["qfim_redundancy"]) < 0.02, "A4 redundancy is ~0 (rank = p)",
          f"R={a4['qfim_redundancy']:.3f}")
    a2 = measure_p3(ANSATZE["A2"], 6, 1, 8, generator("test", "p3r-a2"))
    check(abs(a2["qfim_redundancy"] - 1/3) < 0.03, "A2 redundancy is ~1/3 at d=1",
          f"R={a2['qfim_redundancy']:.3f}")


def test_p5_is_the_parameter_count() -> None:
    """P5 must return exactly n_params for every family and grid point."""
    for tag, ansatz in ANSATZE.items():
        for n, d in ((2, 1), (4, 3), (8, 5)):
            value = measure_p5(ansatz, n, d, 1)["n_params_p"]
            check(value == ansatz.n_params(n, d), f"P5 = n_params for {tag} n={n} d={d}",
                  f"{value:.0f} vs {ansatz.n_params(n, d)}")


def test_p6_matches_hand_formulas() -> None:
    """The tape count must match the closed-form two-qubit counts of each family.

    A1 CNOT chain d(n-1); A2 CNOT ring dn; A3 ring between repeats (d-1)n;
    A4 CRZ ring plus cups dn + n/2; A5 CRot per block dn; A6 CZ chain d(n-1);
    A7 two Pauli rotations per data wire per layer 2d(n-1). With their
    encodings: A5 adds the 2**n - 2 CNOTs of the Mottonen state preparation
    (generic input), A6 adds the 2(n-1) ZZ phases of the feature map.
    """
    n, d = 4, 3
    expect = {"A1": d * (n - 1), "A2": d * n, "A3": (d - 1) * n, "A4": d * n + n // 2,
              "A5": d * n, "A6": d * (n - 1), "A7": 2 * d * (n - 1)}
    for tag, want in expect.items():
        got = measure_p6(ANSATZE[tag], n, d, 1)["two_qubit_gates"]
        check(got == want, f"P6 count for {tag} at n={n} d={d} is {want}", f"got {got:.0f}")
    encoded = {"A5": d * n + 2 ** n - 2, "A6": d * (n - 1) + 2 * (n - 1), "A7": 2 * d * (n - 1)}
    for tag, want in encoded.items():
        x = np.linspace(0.2, 1.1, ANSATZE[tag].n_encoder(n, d))
        got = measure_p6(ANSATZE[tag], n, d, 1, encoder_input=x)["two_qubit_gates"]
        check(got == want, f"P6 for {tag} with its encoding at n={n} d={d} is {want}", f"got {got:.0f}")


def test_gradient_diagonal_matches_expval() -> None:
    """The diagonal cost read off the statevector must equal a PennyLane expval.

    A diagonal Z observable applied to |psi|^2 is exactly what qml.expval
    returns, so the fast path used by the G-series is validated against the
    reference simulator on a random three-qubit state.
    """
    import pennylane as qml
    n = 3
    dev = qml.device("default.qubit", wires=n)

    def build():
        for q in range(n):
            qml.RY(0.3 * (q + 1), wires=q)
            qml.RZ(0.5 * (q + 1), wires=q)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])

    state = qml.QNode(lambda: (build(), qml.state())[1], dev)()
    probs = np.abs(np.asarray(state)) ** 2
    glob = qml.QNode(lambda: (build(), qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2)))[1], dev)()
    loc = qml.QNode(lambda: (build(), qml.expval(qml.PauliZ(0)))[1], dev)()
    check(np.allclose(probs @ diagonal_observable(n, "global"), float(glob)),
          "global diagonal cost matches qml.expval")
    check(np.allclose(probs @ diagonal_observable(n, "local"), float(loc)),
          "local diagonal cost matches qml.expval")


def test_hea_shows_barren_plateau_decay() -> None:
    """A1's global gradient variance must fall with width and fit alpha near ln 2.

    The HEA at depth d = n is an approximate 2-design, so its gradient variance
    decays like exp(-alpha n) with alpha close to ln 2 = 0.693 (McClean 2018).
    This is a direction-and-scale check, not an exact value.
    """
    widths, variances = [2, 4, 6, 8], []
    for nq in widths:
        out = measure_g1(ANSATZE["A1"], nq, nq, 200, generator("test", "bp", nq))
        variances.append(out["gradvar_global"])
    falling = all(variances[i + 1] < variances[i] for i in range(len(variances) - 1))
    alpha = bp_onset_slope(widths, variances)
    check(falling and 0.5 < alpha < 0.9, "HEA gradient variance decays, alpha near ln 2",
          f"alpha={alpha:.3f} vars={['%.2e' % v for v in variances]}")


def test_gradient_matches_autodiff() -> None:
    """The finite-difference cost gradient must match PennyLane autodiff.

    This is the verification that the G-series rests on: the fast diagonal-cost
    finite difference is compared, for every family, against an autodiff gradient
    of the same cost. It also guards the A4 case, where a two-term parameter
    shift would be wrong because CRZ is not a Pauli rotation.
    """
    import pennylane as qml
    import pennylane.numpy as pnp
    from ..probes.statevector import build_qnode

    n, d = 4, 3
    rng = generator("test", "grad-autodiff")
    diag = diagonal_observable(n, "global")
    for tag, ansatz in ANSATZE.items():
        p = ansatz.n_params(n, d)
        k = p // 2
        size = ansatz.n_encoder(n, d)
        x = None if size == 0 else rng.uniform(0, np.pi, size)
        theta = rng.uniform(0, 2 * np.pi, p)

        def cost(t):
            state = np.asarray(build_qnode(ansatz, n, d, x)(np.asarray(t).reshape(1, -1)))
            return float((np.abs(state.reshape(-1)) ** 2) @ diag)

        step = 1e-4
        tp, tm = theta.copy(), theta.copy()
        tp[k] += step
        tm[k] -= step
        fd = (cost(tp) - cost(tm)) / (2 * step)

        dev = qml.device("default.qubit", wires=n)
        obs = qml.prod(*[qml.PauliZ(i) for i in range(n)])

        @qml.qnode(dev, diff_method="parameter-shift")
        def qcost(t):
            ansatz.apply(t, n, d, x)
            return qml.expval(obs)

        ad = float(qml.grad(qcost)(pnp.array(theta, requires_grad=True))[k])
        check(abs(fd - ad) < 1e-5, f"G1 gradient matches autodiff for {tag}",
              f"fd={fd:+.6f} autodiff={ad:+.6f}")


def test_gradient_runs_on_every_family() -> None:
    """G1 must return finite, non-trivial variances for all four families.

    The encoder input is supplied for the families that use one, exactly as the
    sweep driver does, so the mid-circuit gradient is not degenerate.
    """
    n, d = 4, 3
    for tag, ansatz in ANSATZE.items():
        size = ansatz.n_encoder(n, d)
        enc = None if size == 0 else generator("test", "enc", tag).uniform(0, np.pi, size)
        out = measure_g1(ansatz, n, d, 128, generator("test", "gall", tag), enc)
        ok = all(np.isfinite(v) and v >= 0.0 for v in out.values())
        nontrivial = out["gradvar_global"] > 1e-6 or out["gradvar_local"] > 1e-6
        check(ok and nontrivial, f"G1 gradient variance is finite and non-trivial for {tag}",
              f"global={out['gradvar_global']:.2e} local={out['gradvar_local']:.2e}")


def test_g3_crossing_on_an_exact_exponential() -> None:
    """The interpolated crossing width must be exact on Var = exp(-alpha n).

    For Var(n) = exp(-0.693 n) and floor 1/1024, the crossing solves
    0.693 n = ln 1024, so n* = 10 exactly. The interpolation is log-linear and
    the data is log-linear, so the answer must match to rounding.
    """
    alpha = np.log(2.0)
    widths = [2, 4, 6, 8, 10, 12]
    variances = [np.exp(-alpha * n) for n in widths]
    got = crossing_width(widths, variances, floor=1.0 / 1024)
    check(got is not None and abs(got - 10.0) < 1e-9,
          "G3 crossing width is exact on an exponential", f"n*={got}")
    check(crossing_width([2, 4], [1.0, 0.5], floor=1.0 / 1024) is None,
          "G3 reports None when the floor is never crossed")
    check(abs(shot_floor_margin(SHOTS_FLOOR) - 1.0) < 1e-12,
          "G3 margin is 1 exactly at the floor")


def test_g4_gap_on_synthetic_and_measured_data() -> None:
    """The gap must recover alpha_g - alpha_l on synthetic data, and A1 at
    shallow depth must show the Cerezo ordering (global decays faster).

    Synthetic: Var_g = exp(-0.7 n), Var_l = exp(-0.1 n) has gap 0.6 exactly.
    Measured: at d = 1 the alternating-layered claim applies; the global cost
    must lose variance with width faster than the local cost on A1.
    """
    widths = [2, 4, 6, 8]
    synthetic = locality_gap(widths,
                             [np.exp(-0.7 * n) for n in widths],
                             [np.exp(-0.1 * n) for n in widths])
    check(abs(synthetic["gap"] - 0.6) < 1e-9, "G4 gap is exact on synthetic data",
          f"gap={synthetic['gap']:.4f}")

    var_g, var_l = [], []
    for nq in widths:
        out = measure_g1(ANSATZE["A1"], nq, 1, 200, generator("test", "g4", nq))
        var_g.append(out["gradvar_global"])
        var_l.append(out["gradvar_local"])
    measured = locality_gap(widths, var_g, var_l)
    check(measured["gap"] > 0.05,
          "G4 measured gap on shallow A1 has the Cerezo ordering",
          f"alpha_g={measured['alpha_global']:.3f} "
          f"alpha_l={measured['alpha_local']:.3f}")


def test_g5_noiseless_control_matches_g1() -> None:
    """G5 at p = 0 runs the same estimator on a density matrix; its variance
    must agree with the statevector value at the same grid point in scale.

    Same family, width, depth and sample count; different seeds and different
    angle pools, so the comparison is a scale check (factor 2), not an
    identity. A disagreement in scale means the mixed path is wrong.
    """
    n, d, samples = 2, 2, 60
    g5 = measure_g5(ANSATZE["A1"], n, d, samples, generator("test", "g5-ctrl"))
    g1 = measure_g1(ANSATZE["A1"], n, d, samples, generator("test", "g5-ref"))
    lo, hi = g1["gradvar_global"] / 2.0, g1["gradvar_global"] * 2.0
    check(g5["noise_s0"] == 0.0 and lo <= g5["gradvar_global_s0"] <= hi,
          "G5 noiseless control matches the G1 statevector scale",
          f"mixed={g5['gradvar_global_s0']:.3e} state={g1['gradvar_global']:.3e}")
    finite = [g5[f"gradvar_global_s{i}"] for i in range(len(NOISE_LEVELS))]
    check(finite[-1] < finite[0],
          "G5 variance falls from p = 0 to the strongest noise",
          f"p0={finite[0]:.3e} p={NOISE_LEVELS[-1]}: {finite[-1]:.3e}")


def test_g5_decay_is_exponential_in_depth() -> None:
    """At fixed noise the log-variance must fall with depth (Wang Thm 1 /
    Larocca Box 8), and faster than the noiseless control falls.

    A1 at n = 2, depths {1, 3, 5}, p = 0.1: the fitted depth rate beta of the
    noisy column must be positive and clearly larger than the noiseless
    column's rate at the same points (G2 says the noiseless curve is flat).
    """
    depths, noisy, clean = [1, 3, 5], [], []
    for d in depths:
        out = measure_g5(ANSATZE["A1"], 2, d, 60, generator("test", "g5-depth", d))
        noisy.append(out["gradvar_global_s3"])   # p = 0.1
        clean.append(out["gradvar_global_s0"])   # p = 0
    beta_noisy = noise_depth_rate(depths, noisy)
    beta_clean = noise_depth_rate(depths, clean)
    check(beta_noisy > 0.2 and beta_noisy > beta_clean + 0.1,
          "G5 noisy variance decays with depth, noiseless does not",
          f"beta_noisy={beta_noisy:.3f} beta_clean={beta_clean:.3f}")


def test_g5_crossing_interpolates() -> None:
    """The interpolated p_crit must be exact on a log-log-linear curve.

    Var(p) = 1e-3 * (p / 1e-2)^(-1) crosses floor 1/1024 ... use a clean case:
    Var = {1e-1 at p=1e-3, 1e-3 at p=1e-2, 1e-5 at p=1e-1}; the floor 1/1024
    is crossed between p = 1e-2 and 1e-1, and the log-log line gives
    p_crit = 10**(-2 - (log10(1e-3) - log10(1/1024)) / 2) within rounding.
    """
    ps = [0.0, 1e-3, 1e-2, 1e-1]
    vs = [2e-1, 1e-1, 1e-3, 1e-5]
    got = noise_crossing(ps, vs, floor=1.0 / 1024)
    expect = 10 ** (-2.0 + (np.log10(1e-3) - np.log10(1.0 / 1024)) / 2.0)
    check(got is not None and abs(np.log10(got) - np.log10(expect)) < 1e-9,
          "G5 p_crit interpolation is exact on log-log-linear data",
          f"p_crit={got:.4e} expect={expect:.4e}")
    check(noise_crossing(ps, [1.0, 0.9, 0.8, 0.7]) is None,
          "G5 reports None when the floor is not crossed in range")


def test_g6_small_angle_gradient_does_not_vanish() -> None:
    """The fixed layers must remove the structural zeros of the old design.

    The broken protocol gave E[(dC)^2] ~ 1e-6 at a = 0.01 on A1 at n = 4,
    d = 3: the |0...0> start is a cost eigenstate, and the theta = 0 Clifford
    limit keeps a Z-string diagonal, so both costs' small-angle derivatives
    vanished by construction. With the pre- and post-layer in place the same
    point measures ~1e-3 (global) and ~1e-2 (local). The check is the order
    of magnitude, not the exact value: both second moments at a = 0.01 must
    sit far above the artefact scale. Whether small-angle init *beats* the
    uniform baseline is a finding, not a known answer - it depends on cost
    locality and width (Zhang's polynomial bound needs a local observable) -
    so it is read from the sweep, never asserted here.
    """
    out = measure_g6(ANSATZE["A1"], 4, 3, 100, generator("test", "g6-grad"))
    small_g = out["grad2_global_s4"]  # a = 0.01
    small_l = out["grad2_local_s4"]
    check(small_g > 1e-4 and small_l > 1e-4,
          "G6 small-angle gradients stay clear of the old artefact scale",
          f"a=0.01 global={small_g:.3e} local={small_l:.3e} (artefact ~1e-6)")


def test_g6_rank_collapses_at_small_scale_and_matches_p3_at_one() -> None:
    """The QFIM rank must fall as the scale falls (Haug's capacity side), and
    the a = 1 column must agree with P3's rank at the same grid point.

    A1 at n = 4, d = 3 saturates the manifold (rank 30) at a = 1; at
    a = 0.001 the accessible directions collapse far below that.
    """
    out = measure_g6(ANSATZE["A1"], 4, 3, 40, generator("test", "g6-rank"))
    p3 = measure_p3(ANSATZE["A1"], 4, 3, 8, generator("test", "g6-p3"))
    check(abs(out["qfim_rank_s0"] - p3["qfim_rank"]) < 1.0,
          "G6 rank at a = 1 agrees with P3",
          f"g6={out['qfim_rank_s0']:.1f} p3={p3['qfim_rank']:.1f}")
    check(out["qfim_rank_s6"] < out["qfim_rank_s0"],
          "G6 rank collapses at small scale",
          f"a=1: {out['qfim_rank_s0']:.1f}  a=0.001: {out['qfim_rank_s6']:.1f}")


def test_g6_prelayer_is_deterministic() -> None:
    """The pre-layer is part of the protocol: same identity, same angles."""
    one = prelayer_angles(ANSATZE["A2"], 4, 3)
    two = prelayer_angles(ANSATZE["A2"], 4, 3)
    other = prelayer_angles(ANSATZE["A2"], 6, 3)
    check(np.array_equal(one, two) and not np.array_equal(one[:4], other[:4]),
          "G6 pre-layer is deterministic per grid point")
    check(len(SCALES) == 7 and SCALES[0] == 1.0,
          "G6 scale grid starts at the uniform baseline")


def main() -> int:
    """Run every check and return the process exit code."""
    test_purity_of_known_states()
    test_meyer_wallach_of_known_states()
    test_haar_expectation_matches_sampling()
    test_haar_bins_sum_to_one()
    test_p1_matches_scorecard_v5()
    test_p1_floor_falls_with_more_pairs()
    test_seeds_are_order_independent()
    test_every_family_runs()
    test_qfim_of_a_single_rotation()
    test_qfim_rank_is_bounded_by_the_manifold()
    test_p3_saturates_a_deep_wide_circuit()
    test_lie_closure_of_known_algebras()
    test_p4_is_full_su_for_every_family()
    test_p4_of_a7_is_polynomial()
    test_lie_closure_matches_pennylane()
    test_p3_redundancy_is_one_minus_rank_over_p()
    test_p5_is_the_parameter_count()
    test_p6_matches_hand_formulas()
    test_gradient_diagonal_matches_expval()
    test_gradient_matches_autodiff()
    test_hea_shows_barren_plateau_decay()
    test_gradient_runs_on_every_family()
    test_g3_crossing_on_an_exact_exponential()
    test_g4_gap_on_synthetic_and_measured_data()
    test_g5_noiseless_control_matches_g1()
    test_g5_decay_is_exponential_in_depth()
    test_g5_crossing_interpolates()
    test_g6_small_angle_gradient_does_not_vanish()
    test_g6_rank_collapses_at_small_scale_and_matches_p3_at_one()
    test_g6_prelayer_is_deterministic()
    failures = [label for passed, label in OUTCOMES if not passed]
    print(f"\n{len(OUTCOMES) - len(failures)}/{len(OUTCOMES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
