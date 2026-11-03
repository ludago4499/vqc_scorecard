# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-10-01  Created. Checks for the pure-v1 families (A5 circuit-centric,
#               A6 IQP, A7 Farhi-Neven) against independent NumPy
#               constructions, the pure model (circuit-only training, P6
#               agreement, zero-noise identity, gradients), the harness split,
#               the runner, and the frozen decision rules and categories of
#               the report on planted data.
#   2026-10-08  Docstring only: the A7 check builds the package's model
#               (readout |1>), not the paper's (readout |0>).

"""Checks with known answers for pure-v1.

Closed forms (amplitude encoding read out by ``<Z_0>``, the Havlicek feature
map as an explicit diagonal-and-Hadamard product, the Farhi-Neven layer as
explicit matrix exponentials), identities (zero noise is no noise, the model's
two-qubit count is the family's P6), and planted outcomes (a synthetic record
set with a known effect must give the pre-registered verdict).

Run::

    python -m vqc_scorecard.tests.test_pure
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from functools import reduce
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ..ansatze import REGISTRY as ANSATZE
from ..ansatze import a6_iqp, a7_farhi_neven
from ..core.seeding import BASE_SEED, generator
from ..metrics.p6_two_qubit import two_qubit_count
from ..noise import NoiseModel
from ..probes.gates import elementary
from ..tasks import data as data_mod
from ..tasks import harness, pure, pure_report
from ..tasks import stats as st
from ..tasks.model import MODELS, body_two_qubit_count, build_pure_model
from .fixtures import synthetic_cache, toy_movie_csv

OUTCOMES: list[tuple[bool, str]] = []

_I = np.eye(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1.0, -1.0]).astype(complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def check(passed: bool, label: str, detail: str = "") -> None:
    """Record one check and print its outcome."""
    OUTCOMES.append((bool(passed), label))
    print(f"[{'PASS' if passed else 'FAIL'}] {label}" + (f"  --  {detail}" if detail else ""))


def _kron(*mats):
    return reduce(np.kron, mats)


def _state(fn, n: int) -> np.ndarray:
    dev = qml.device("default.qubit", wires=n)
    return np.asarray(qml.QNode(lambda: (fn(), qml.state())[1], dev)())


# --------------------------------------------------------------------------- #
# 1. The new families against independent constructions                       #
# --------------------------------------------------------------------------- #

def test_a5_amplitude_readout_closed_form() -> None:
    """At theta = 0 every Rot and CRot is the identity, so <Z_0> of A5 is the
    amplitude-encoded vector's weight on bit0 = 0 minus its weight on bit0 = 1."""
    n = 3
    rng = generator("test-pure", "a5")
    x = rng.normal(size=(4, 2 ** n))
    m = build_pure_model("A5", n, 2)
    params = pnp.array(np.concatenate([np.zeros(m.p_circuit), [1.0, 0.0]]))
    got = np.asarray(m.forward(params, x))
    w = x ** 2 / (x ** 2).sum(axis=1, keepdims=True)
    z0 = w[:, : 2 ** (n - 1)].sum(axis=1) - w[:, 2 ** (n - 1):].sum(axis=1)
    want = 1.0 / (1.0 + np.exp(-z0))
    check(np.allclose(got, want, atol=1e-12), "A5: amplitude encoding read by <Z_0> at theta = 0",
          f"max err {np.abs(got - want).max():.1e}")


def test_a6_feature_map_matches_havlicek() -> None:
    """The IQP map equals (U_Z(x) H^n)^2 |0> with U_Z = exp(i sum phi_S Z_S),
    phi_q = x_q, phi_qr = (pi - x_q)(pi - x_r) on chain edges, built as matrices."""
    n = 3
    x = np.array([0.4, 1.9, 2.7])
    zs = [_kron(*[_Z if k == q else _I for k in range(n)]) for q in range(n)]
    phase = sum(x[q] * zs[q] for q in range(n))
    for q, r in a6_iqp.edges(n):
        phase = phase + (np.pi - x[q]) * (np.pi - x[r]) * zs[q] @ zs[r]
    u_z = np.diag(np.exp(1j * np.diag(phase)))
    hn = _kron(*[_H] * n)
    psi0 = np.zeros(2 ** n, complex)
    psi0[0] = 1.0
    want = u_z @ hn @ u_z @ hn @ psi0
    got = _state(lambda: a6_iqp.embed(x, n), n)
    check(np.allclose(got, want, atol=1e-10), "A6: feature map equals Havlicek's (U_Z H^n)^2",
          f"max err {np.abs(got - want).max():.1e}")


def test_a7_layer_and_readout_match_farhi_neven() -> None:
    """A7 at n = 3, d = 1: <Z_0> of the circuit equals <Y_readout> of
    exp(i t Sigma) ... RY(x)|0>|1>, the package's model, built with expm.

    The readout here is |1>, as in the package. The paper's readout is |0>; the
    sign relation between the two is checked in test_conventions."""
    from scipy.linalg import expm
    n, d = 3, 1
    rng = generator("test-pure", "a7")
    theta = rng.uniform(0, 2 * np.pi, ANSATZE["A7"].n_params(n, d))
    x = np.array([0.7, 2.1])

    def ry(a):
        return np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]])

    one = np.array([0.0, 1.0])
    psi = _kron(one, ry(x[0]) @ np.array([1.0, 0.0]), ry(x[1]) @ np.array([1.0, 0.0])).astype(complex)
    i = 0
    for word in (_Z, _X):
        for j in (1, 2):
            ops = [_X if k == 0 else (word if k == j else _I) for k in range(n)]
            psi = expm(1j * theta[i] * _kron(*ops)) @ psi
            i += 1
    y0 = _kron(_Y, _I, _I)
    want = float(np.real(np.conj(psi) @ y0 @ psi))
    dev = qml.device("default.qubit", wires=n)
    got = float(qml.QNode(lambda: (ANSATZE["A7"].apply(theta, n, d, x),
                                   qml.expval(qml.PauliZ(0)))[1], dev)())
    check(abs(got - want) < 1e-10, "A7: <Z_0> after RX(pi/2) is the paper's <Y> readout",
          f"{got:.10f} vs {want:.10f}")


def test_mottonen_cost_is_counted() -> None:
    """elementary() turns a StatePrep into its Mottonen gates: 2**n - 2 CNOTs (n >= 3).

    At n = 2 PennyLane's decomposition of a real state needs a single CNOT; from
    n = 3 on the count is the generic ``2**n - 2``.
    """
    for n in (3, 4, 5):
        x = np.linspace(0.3, 1.0, 2 ** n)
        script = qml.tape.make_qscript(lambda: qml.AmplitudeEmbedding(x, wires=range(n), normalize=True))()
        ops = elementary(script.operations)
        cnots = sum(1 for op in ops if len(op.wires) == 2)
        check(cnots == 2 ** n - 2 and all(len(op.wires) <= 2 for op in ops),
              f"amplitude encoding at n={n} costs 2**n - 2 = {2 ** n - 2} CNOTs", f"got {cnots}")


# --------------------------------------------------------------------------- #
# 2. The pure model                                                            #
# --------------------------------------------------------------------------- #

def test_only_the_head_is_classical() -> None:
    """Every model trains its circuit angles and exactly two classical numbers."""
    for tag, spec in MODELS.items():
        m = build_pure_model(tag, 4, 2)
        check(m.p_classical == 2 and m.p_circuit == ANSATZE[spec.ansatz].n_params(4, 2)
              and m.layout["head"] == (m.p_circuit, m.p_circuit + 2),
              f"{tag}: p_classical = 2 (readout scale and bias), circuit = P5 of {spec.ansatz}")


def test_two_qubit_count_is_p6() -> None:
    """A family's model count equals its P6 with its encoding; A3 re-uploads RY only."""
    n, d = 4, 3
    for tag in ("A1", "A2", "A4", "A5", "A6", "A7"):
        a = ANSATZE[tag]
        x = np.linspace(0.1, 1.0, a.n_encoder(n, d)) if a.n_encoder(n, d) else None
        want = two_qubit_count(a, n, d, x)
        got = body_two_qubit_count(tag, n, d)
        check(got == want, f"{tag}: model two-qubit count = P6 with encoding", f"{got} vs {want}")
    check(body_two_qubit_count("A3", n, d) == (d - 1) * n, "A3: ring between re-uploads, (d-1) n")
    check(body_two_qubit_count("A2:amp", n, d) == d * n + 2 ** n - 2
          and body_two_qubit_count("A2:iqp", n, d) == d * n + 2 * (n - 1),
          "encoding arm: SEL ring plus the encoding's own two-qubit gates")


def test_zero_noise_is_noiseless() -> None:
    """A p = 0 noise model on the mixed-state device reproduces every model."""
    rng = generator("test-pure", "p0")
    for tag, spec in MODELS.items():
        x = rng.uniform(0.1, np.pi, (3, spec.n_features(4)))
        clean = build_pure_model(tag, 4, 2)
        zero = build_pure_model(tag, 4, 2, noise=NoiseModel("depolarizing", 0.0, 0.0))
        params = pnp.array(np.concatenate([rng.uniform(0, 2 * np.pi, clean.p_circuit), [1.0, 0.0]]))
        diff = float(np.abs(np.asarray(clean.forward(params, x)) - np.asarray(zero.forward(params, x))).max())
        check(diff < 1e-10, f"{tag}: zero-rate noise equals the noiseless model", f"{diff:.1e}")


def test_gradient_matches_finite_difference() -> None:
    """The backprop gradient of the loss w.r.t. one circuit angle equals a central difference."""
    from ..tasks.train import _bce
    rng = generator("test-pure", "grad")
    for tag, spec in MODELS.items():
        m = build_pure_model(tag, 4, 2)
        x = rng.uniform(0.1, np.pi, (5, spec.n_features(4)))
        y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
        v = np.concatenate([rng.uniform(0, 2 * np.pi, m.p_circuit), [1.3, -0.2]])
        g = np.asarray(qml.grad(lambda w: _bce(m.forward(w, x), y))(pnp.array(v, requires_grad=True)))
        k, h = int(np.argmax(np.abs(g[:m.p_circuit]))), 1e-6
        up, dn = v.copy(), v.copy()
        up[k] += h
        dn[k] -= h
        fd = (float(_bce(m.forward(pnp.array(up), x), y)) - float(_bce(m.forward(pnp.array(dn), x), y))) / (2 * h)
        check(abs(g[k] - fd) < 1e-6 and abs(g[k]) > 1e-4,
              f"{tag}: dL/dtheta_k matches the central difference (largest circuit component)",
              f"{g[k]:+.6f} vs {fd:+.6f}")


# --------------------------------------------------------------------------- #
# 3. Protocol, harness and runner                                              #
# --------------------------------------------------------------------------- #

MINI = replace(pure.PROTOCOL, name="mini", widths=(2, 4), depths=(1, 3), n_seeds=3, epochs=2,
               batch=8, shots_eval=64, imdb_per_class=20, encoding_width=2)


def test_protocol_shape() -> None:
    """63 family cells and 6 encoding cells; the digest moves with any setting."""
    p = pure.PROTOCOL
    check(len(p.cells("family")) == 7 * 3 * 3 and len(p.cells("encoding")) == 2 * 3
          and p.digest() == pure.PureProtocol().digest() and replace(p, sesoi=0.05).digest() != p.digest()
          and p.seeds()[0] == BASE_SEED, "pure-v1 grid: 63 + 6 cells, stable digest, seeds from BASE_SEED")


def test_harness_split_is_the_v10_split() -> None:
    """Train/validation/test are disjoint and deterministic; test is the v10 test split."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = toy_movie_csv(Path(tmp), per_class=20)
        s1 = harness.task_split("imdb1k", BASE_SEED, protocol=MINI, movie_csv=csv_path)
        s2 = harness.task_split("imdb1k", BASE_SEED, protocol=MINI, movie_csv=csv_path)
        sets = [set(s1[k][0]) for k in ("tr", "va", "te")]
        docs, labels = data_mod.load_movie_reviews(csv_path, seed=BASE_SEED)
        docs, labels = data_mod.subsample_balanced(docs, labels, n_per_class=20, seed=BASE_SEED)
        _, te = data_mod.stratified_split(labels, seed=BASE_SEED)
        check(not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
              and all(np.array_equal(s1[k][0], s2[k][0]) for k in s1)
              and list(s1["te"][0]) == list(docs[te]) and len(s1["va"][0]) == 6,
              "harness split: disjoint, deterministic, test identical to the v10 split")


def test_runner_end_to_end() -> None:
    """Train both arms, no duplicates on re-run, like-for-like baselines, noise-eval, status."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        csv_path = toy_movie_csv(folder, per_class=20)
        docs, labels = data_mod.load_movie_reviews(csv_path, seed=BASE_SEED)
        cache = synthetic_cache(folder, docs, labels, dim=16)
        root = folder / "pure"
        args = ["--task", "imdb1k", "--cache", str(cache), "--movie-csv", str(csv_path)]
        pure.main(["train", *args, "--arm", "all"], root=root, protocol=MINI)
        path = harness.checkpoint_path("imdb1k", "clean", root)
        n1 = len(harness.load(path))
        pure.main(["train", *args, "--arm", "all"], root=root, protocol=MINI)
        n2 = len(harness.load(path))
        want = (7 * 2 * 2 + 2 * 2) * 3
        check(n1 == n2 == want, "runner: every cell x seed once; a re-run adds nothing", f"{n1} -> {n2}")
        recs = harness.load(path)
        a5 = next(r for r in recs if r["family"] == "A5" and r["n"] == 4)
        a7 = next(r for r in recs if r["family"] == "A7" and r["n"] == 4)
        check(a5["n_features"] == 16 and a5["encoding"] == "amplitude" and a7["n_features"] == 3
              and all(r["p_classical"] == 2 for r in recs),
              "records: A5 reads PCA(2^n), A7 reads PCA(n-1), every model trains 2 classical numbers")
        base = harness.load(harness.checkpoint_path("imdb1k", "baselines", root))
        keys4 = set(next(b for b in base if b["n"] == 4))
        check({"logreg_pca3", "logreg_pca4", "logreg_pca16", "logreg_frozen", "logreg_tfidf"} <= keys4,
              "baselines: logistic regression on the exact features of every encoding")
        pure.main(["noise-eval", *args, "--levels", "nisq", "--channels", "depolarizing",
                   "--width", "2", "--models", "A1", "A5"], root=root, protocol=MINI)
        nz = harness.load(harness.checkpoint_path("imdb1k", "noise_eval", root))
        check(len(nz) == 2 * 2 * 3 and all(0.0 <= r["prediction_fidelity"] <= 1.0 for r in nz),
              "noise-eval: selected models x seeds, fidelity in [0, 1]", f"{len(nz)} records")
        status = pure.status("imdb1k", protocol=MINI, root=root)
        check("A2:amp" in status and " 3/3" in status, "status grid lists both arms")


# --------------------------------------------------------------------------- #
# 4. Decision rules and categories on planted data                             #
# --------------------------------------------------------------------------- #

def test_verdict_rules() -> None:
    """The three verdicts on hand-made contrasts."""
    big = st.contrast(np.full(10, 0.5) + np.linspace(0, 0.02, 10), np.full(10, 0.4))
    none = st.contrast(np.full(10, 0.4) + np.linspace(-0.005, 0.005, 10), np.full(10, 0.4))
    check(st.greater(big, 0.03, big["p_greater"]) == "supported"
          and st.greater(none, 0.03, none["p_greater"]) == "refuted",
          "greater: a 0.1 gain is supported; a null difference is refuted (CI below SESOI)")
    check(st.not_greater(none, 0.03) == "supported" and st.not_greater(big, 0.03) == "refuted",
          "not_greater: null supported, 0.1 gain refuted")
    check(st.equivalent(none, 0.03) == "supported" and st.equivalent(big, 0.03) == "refuted",
          "equivalent: null inside +-SESOI, 0.1 gain outside")
    check(st.combine(["supported", "inconclusive"]) == "inconclusive"
          and st.combine(["supported", "refuted"]) == "refuted",
          "combine: all supported -> supported; any refuted -> refuted")
    fr = st.friedman(np.tile([0.1, 0.2, 0.3], (8, 1)) + np.random.default_rng(0).normal(0, 1e-3, (8, 3)))
    check(abs(fr["W"] - 1.0) < 1e-9 and fr["p"] < 0.01, "Kendall W = 1 for a perfectly consistent order")


def _planted(tag, n, d, seed, mcc):
    return {"task": "imdb1k", "family": tag, "n": n, "d": d, "seed": seed,
            "selected": {"test": {"mcc": mcc, "macro_f1": 0.5 + mcc / 2}}}


def test_hypotheses_on_planted_data() -> None:
    """H3 and H6 return the pre-registered verdict for a planted effect; a gap is incomplete."""
    p = replace(pure.PROTOCOL, families=("A1",), widths=(4,))
    seeds = p.seeds()
    jitter = np.linspace(-0.01, 0.01, len(seeds))
    clean = []
    for i, s in enumerate(seeds):
        clean += [_planted("A1", 4, 1, s, 0.30 + jitter[i]), _planted("A1", 4, 3, s, 0.40 + jitter[i] * 0.5),
                  _planted("A1", 4, 5, s, 0.40 + jitter[i] * 0.2)]
        for d in p.depths:
            clean += [_planted("A2", 4, d, s, 0.30 + jitter[i]), _planted("A3", 4, d, s, 0.42 + jitter[i] * 0.3),
                      _planted("A2:amp", 4, d, s, 0.30 - jitter[i] * 0.4),
                      _planted("A2:iqp", 4, d, s, 0.30 + jitter[i] * 0.1)]
    cells = pure_report.per_seed(clean)
    res3 = pure_report.h3(cells, p)
    check(res3["verdict"] == "supported", "H3: planted +0.10 from d=1 to 3 and a plateau to d=5 is supported",
          res3["verdict"])
    res6 = pure_report.h6(cells, p)
    check(res6["verdict_a"] == "supported" and res6["verdict_b"] == "refuted" and res6["verdict_c"] == "supported",
          "H6: re-upload gain supported; amplitude null refuted; IQP no-gain supported",
          f"{res6['verdict_a']}, {res6['verdict_b']}, {res6['verdict_c']}")
    holed = [r for r in clean if not (r["family"] == "A1" and r["d"] == 5 and r["seed"] == seeds[0])]
    check(pure_report.h3(pure_report.per_seed(holed), p)["verdict"] == "incomplete",
          "a missing seed makes the hypothesis incomplete, never scored")


def test_categories_on_planted_cells() -> None:
    """Floor, Trainability-limited, Performing, Noise-limited and Capacity-limited cells."""
    p = replace(pure.PROTOCOL, n_seeds=10)
    seeds = p.seeds()
    j = np.linspace(-0.02, 0.02, 10)
    spec = {("A1", 4, 1): 0.00 + 5 * j, ("A2", 4, 1): np.r_[[0.05] * 4, [0.5] * 6],
            ("A3", 4, 1): 0.45 + j, ("A4", 4, 1): 0.45 + j, ("A6", 4, 1): 0.30 + j}
    clean = [_planted(t, n, d, s, float(v[i])) for (t, n, d), v in spec.items() for i, s in enumerate(seeds)]
    base = [{"n": 4, "seed": s, "logreg_pca4": {"mcc": 0.46}} for s in seeds]
    noise = [{"family": "A4", "n": 4, "d": 1, "noise": {"level": "nisq"},
              "clean_test": {"mcc": 0.45}, "test": {"mcc": 0.38}}]
    cats = pure_report.categorise({"clean": clean, "baselines": base, "noise_eval": noise}, {}, p)
    got = {k[0]: v["category"] for k, v in cats.items()}
    want = {"A1": "Floor", "A2": "Trainability-limited", "A3": "Performing", "A4": "Noise-limited",
            "A6": "Capacity-limited"}
    check(got == want, "categories follow the frozen order of rules", f"{got}")


def main() -> int:
    """Run every check and return the process exit code."""
    test_a5_amplitude_readout_closed_form()
    test_a6_feature_map_matches_havlicek()
    test_a7_layer_and_readout_match_farhi_neven()
    test_mottonen_cost_is_counted()
    test_only_the_head_is_classical()
    test_two_qubit_count_is_p6()
    test_zero_noise_is_noiseless()
    test_gradient_matches_finite_difference()
    test_protocol_shape()
    test_harness_split_is_the_v10_split()
    test_runner_end_to_end()
    test_verdict_rules()
    test_hypotheses_on_planted_data()
    test_categories_on_planted_cells()
    failures = [label for passed, label in OUTCOMES if not passed]
    print(f"\n{len(OUTCOMES) - len(failures)}/{len(OUTCOMES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
