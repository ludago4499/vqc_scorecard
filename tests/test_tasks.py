# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-10-01  Created. The protocol-independent checks of the retired A5 and
#               fair-v1 suites, moved here: the loss, the training loop
#               (legacy lift, optimiser schedule, validation selection, the
#               bit-for-bit legacy record), features and extraction, the
#               gate-noise models, the classification metrics and the
#               statistics helpers. The A5 and dressed-interface checks are
#               archived with their code in old/fair_v1_archive.

"""Checks with known answers for the task machinery shared by every protocol.

Run::

    python -m vqc_scorecard.tests.test_tasks
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ..core.seeding import BASE_SEED, generator
from ..noise import CHANNELS, LEVELS, NoiseModel, add_noise, preset
from ..tasks import data as data_mod
from ..tasks import stats as st
from ..tasks.features import FrozenFeatures, FrozenPca, TfidfPca, doc_key
from ..tasks.get_features import pool_windows, task_texts, window_token_ids
from ..tasks.metrics import classification_metrics
from ..tasks.model import TaskModel, as_task_model, build_feature_model, build_pure_model
from ..tasks.train import _bce, train_one
from ..ansatze import REGISTRY as ANSATZE
from .fixtures import MOVIE_CSV, synthetic_cache, tiny_bert, toy_movie_csv

OUTCOMES: list[tuple[bool, str]] = []


def check(passed: bool, label: str, detail: str = "") -> None:
    """Record one check and print its outcome."""
    OUTCOMES.append((bool(passed), label))
    print(f"[{'PASS' if passed else 'FAIL'}] {label}" + (f"  --  {detail}" if detail else ""))


def skip(label: str, reason: str) -> None:
    """Record a skipped check as passed, with the reason, like test_metrics does."""
    check(True, f"{label} (skipped)", reason)


def test_bce_is_two_class_cross_entropy() -> None:
    """BCE on P(y=1) = sigmoid(l_1 - l_0) equals the softmax cross-entropy, term by term."""
    rng = generator("test-a5", "eq15")
    logits = rng.normal(0.0, 2.0, (50, 2))
    y = rng.integers(0, 2, 50)
    log_soft = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    eq15 = -np.mean(log_soft[np.arange(50), y])
    p1 = 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))
    ours = float(_bce(pnp.array(p1), pnp.array(y.astype(float))))
    check(abs(ours - eq15) < 1e-9, "BCE on sigmoid(logit_1 - logit_0) equals the softmax CE",
          f"{ours:.10f} vs {eq15:.10f}")


def test_legacy_models_are_lifted_unchanged() -> None:
    """A legacy triple becomes a TaskModel with a 2-parameter head and the legacy init."""
    tm = as_task_model(build_feature_model("A1", 4, 2))
    check(isinstance(tm, TaskModel) and tm.p_classical == 2 and tm.init is None
          and tm.p_circuit == ANSATZE["A1"].n_params(4, 2),
          "legacy (forward, p, 2q) triple lifts to TaskModel(p_classical=2, init=None)")
    refused = False
    try:
        build_feature_model("A5", 4, 3)
    except ValueError:
        refused = True
    check(refused, "the legacy feature route refuses non-legacy families (A5-A7 run in pure-v1)")


def test_step_schedule_and_betas() -> None:
    """StepLR multiplies the step size by gamma every lr_step epochs; betas reach Adam."""
    rng = generator("test-a5", "sched")
    x = rng.normal(0.0, 1.0, (16, 4))
    y = (x[:, 0] > 0).astype(int)
    x = np.mod(x, np.pi)
    rec = train_one(build_model=lambda s: build_pure_model("A2", 4, 1, shots=s),
                    feats_tr=x, y_tr=y, feats_te=x, y_te=y, epochs=4, batch=8, lr=0.4,
                    seed=1, shots_eval=None, adam_betas=(0.9, 0.999), lr_step=2, lr_gamma=0.5)
    lrs = [h["lr"] for h in rec["history"]]
    check(np.allclose(lrs, [0.4, 0.4, 0.2, 0.2]), "StepLR(step=2, gamma=0.5) schedule", f"{lrs}")
    opt = rec["optimizer"]
    check(opt["beta2"] == 0.999 and opt["lr_step"] == 2 and rec["p_classical"] == 2
          and "uniform" in rec["init"], "record carries optimizer, init rule and p_classical = 2")


def test_legacy_record_reproduces_pre_a5_harness() -> None:
    """A2 on a fixed 120-document split reproduces the numbers of the pre-A5 harness.

    The values below were produced by the task harness as it stood before A5
    (2026-09-14 build) on this exact configuration. A change to the A1-A4
    initialisation, RNG order, features or loss would move them.
    """
    if not MOVIE_CSV.exists():
        skip("pre-A5 A2 record reproduces", f"{MOVIE_CSV.name} not found")
        return
    docs, labels = data_mod.load_movie_reviews(MOVIE_CSV, seed=BASE_SEED)
    docs, labels = data_mod.subsample_balanced(docs, labels, n_per_class=60, seed=BASE_SEED)
    tr, te = data_mod.stratified_split(labels, seed=BASE_SEED)
    feat = TfidfPca(n_dims=4).fit(docs[tr])
    rec = train_one(build_model=lambda s: build_feature_model("A2", 4, 2, shots=s),
                    feats_tr=feat.angles(docs[tr]), y_tr=labels[tr],
                    feats_te=feat.angles(docs[te]), y_te=labels[te],
                    epochs=2, batch=16, lr=0.05, seed=BASE_SEED, shots_eval=None)
    final = rec["final"]
    check(final["f1"] == 0.5555555555555556 and final["mcc"] == 0.3849001794597505
          and final["accuracy"] == 0.6666666666666666,
          "A2 record reproduces the pre-A5 harness bit for bit",
          f"f1={final['f1']!r} mcc={final['mcc']!r}")


def test_doc_key() -> None:
    """The cache key is the SHA-1 of the UTF-8 text (FIPS 180 test vector)."""
    check(doc_key("abc") == "a9993e364706816aba3e25717850c26c9cd0d89d", "doc_key is SHA-1 of the text")


def test_frozen_cache_lookup() -> None:
    """Lookup follows the requested order, resolves negative layers, refuses unknown text."""
    docs = np.array([f"document number {i}" for i in range(6)], dtype=object)
    labels = np.array([0, 1, 0, 1, 0, 1])
    with tempfile.TemporaryDirectory() as tmp:
        path = synthetic_cache(Path(tmp), docs, labels)
        with np.load(path) as z:
            raw = z["layers"]
        last = FrozenFeatures(path)
        first = FrozenFeatures(path, layer=0)
        order = [4, 1, 5]
        check(last.layer == 2 and np.allclose(last.lookup(docs[order]), raw[2, order])
              and np.allclose(first.lookup(docs[order]), raw[0, order])
              and last.model == "toy-encoder",
              "frozen cache: rows by content key, layer -1 = last, layer 0 = embeddings")
        refused = 0
        try:
            last.lookup(np.array(["not in the cache"], dtype=object))
        except KeyError:
            refused += 1
        try:
            FrozenFeatures(path, layer=3)
        except ValueError:
            refused += 1
        check(refused == 2, "frozen cache refuses unknown documents and out-of-range layers")


def test_frozen_pca_fits_on_train_only() -> None:
    """PCA(n) of frozen features: train-only fit, [0, pi] angles, unit-spread pca_std."""
    docs = np.array([f"doc {i}" for i in range(40)], dtype=object)
    labels = np.arange(40) % 2
    with tempfile.TemporaryDirectory() as tmp:
        frozen = FrozenFeatures(synthetic_cache(Path(tmp), docs, labels))
        rep = FrozenPca(frozen, n_dims=4).fit(docs[:30])
        ang = rep.angles(docs)
        std_tr = rep.pca_std(docs[:30])
        refit = FrozenPca(frozen, n_dims=4).fit(docs[:30])
        check(ang.min() >= 0 and ang.max() <= np.pi and ang.shape == (40, 4),
              "FrozenPca angles lie in [0, pi]")
        check(np.allclose(std_tr.mean(0), 0, atol=1e-9) and np.allclose(std_tr.std(0), 1, atol=1e-9),
              "FrozenPca pca_std is standardised on the training rows")
        check(np.array_equal(refit.angles(docs[30:]), rep.angles(docs[30:]))
              and np.array_equal(rep.raw(docs[:3]), frozen.lookup(docs[:3])),
              "FrozenPca transform depends on the training rows only")
        expected = (np.clip(rep.pca_std(docs), -3, 3) + 3) / 6 * np.pi
        check(np.array_equal(expected, ang), "angles are the clipped pca_std mapped to [0, pi]")


def test_tfidf_views_are_consistent() -> None:
    """TfidfPca: angles = clipped pca_std on [0, pi]; raw() is the TF-IDF matrix."""
    docs = np.array(["good great fun film", "bad awful boring film", "great acting good plot",
                     "awful plot bad acting", "fun good film great", "boring bad awful plot"] * 3,
                    dtype=object)
    feat = TfidfPca(n_dims=2, min_df=1).fit(docs)
    expected = (np.clip(feat.pca_std(docs), -3, 3) + 3) / 6 * np.pi
    check(np.array_equal(expected, feat.angles(docs)) and
          np.array_equal(feat.raw(docs), feat.tfidf_raw(docs)) and feat.raw_dim == feat.vocab_size,
          "TfidfPca views agree: angles, pca_std, raw")


def test_windowing_and_pooling() -> None:
    """Windows keep every token once, wrapped in [CLS]/[SEP]; pooling is token-weighted."""
    wins = window_token_ids(list(range(10, 23)), window=6, cls_id=1, sep_id=2)
    inner = [t for w in wins for t in w[1:-1]]
    check(inner == list(range(10, 23)) and all(w[0] == 1 and w[-1] == 2 and len(w) <= 6 for w in wins)
          and len(wins) == 4, "windows cover the document once, each [CLS] ... [SEP], <= window")
    check(window_token_ids([], window=6, cls_id=1, sep_id=2) == [[1, 2]],
          "an empty document is one [CLS] [SEP] window")
    try:
        window_token_ids([5], window=2, cls_id=1, sep_id=2)
        check(False, "a window with no room for content is refused")
    except ValueError:
        check(True, "a window with no room for content is refused")
    rng = generator("test-a5", "pool")
    hidden = rng.normal(0.0, 1.0, (3, 2, 5, 4))
    mask = np.array([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
    tokens = np.concatenate([hidden[:, 0, :5], hidden[:, 1, :2]], axis=1)
    check(np.allclose(pool_windows(hidden, mask), tokens.mean(axis=1)),
          "pooling is the mean over every real token of every window")


def test_task_texts_match_the_loader() -> None:
    """The extractor hashes exactly the text the runner loads, deduplicated."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = toy_movie_csv(Path(tmp), per_class=5)
        texts = task_texts("imdb1k", movie_csv=csv_path)
        docs, _ = data_mod.load_movie_reviews(csv_path, seed=BASE_SEED)
        check(sorted(doc_key(t) for t in texts) == sorted({doc_key(t) for t in docs})
              and len(texts) == 10, "task_texts keys equal the loader's documents")


def test_extract_matches_direct_pooling() -> None:
    """End to end on a tiny random BERT: one window = the model's own mean pooling,
    several windows = the token-weighted mean of the windows run one by one."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        skip("extraction on a tiny BERT", f"{type(error).__name__}: extraction env only")
        return
    from ..tasks.get_features import extract
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        words = tiny_bert(folder)
        short = "good film"
        long = " ".join(words * 2)                                  # 20 tokens
        result = extract([short, long], folder, window=8, batch=3, verbose=False)
        tok = AutoTokenizer.from_pretrained(folder)
        model = AutoModel.from_pretrained(folder, output_hidden_states=True).eval()
        with torch.no_grad():
            enc = tok(short, return_tensors="pt")
            states = model(**enc).hidden_states
            direct = np.stack([s[0].mean(0).numpy() for s in states])
            ids = tok(long, add_special_tokens=False)["input_ids"]
            total, count = 0.0, 0
            for w in window_token_ids(ids, window=8, cls_id=tok.cls_token_id,
                                      sep_id=tok.sep_token_id):
                hs = model(input_ids=torch.tensor([w]), attention_mask=torch.ones(1, len(w)),
                           token_type_ids=torch.zeros(1, len(w), dtype=torch.long)).hidden_states
                total = total + np.stack([h[0].sum(0).numpy() for h in hs])
                count += len(w)
        manual = total / count
        check(result["layers"].shape == (3, 2, 16) and result["n_windows"].tolist() == [1, 4],
              "extract: (states, docs, dim) layout and window counts")
        check(np.abs(result["layers"][:, 0] - direct).max() < 1e-5,
              "extract: a one-window document equals the model's own mean pooling")
        check(np.abs(result["layers"][:, 1] - manual).max() < 1e-5,
              "extract: a long document is the token-weighted mean of its windows")
        import hashlib
        sha = hashlib.sha256((folder / result["meta"]["weights_file"]).read_bytes()).hexdigest()
        check(result["meta"]["weights_sha256"] == sha and result["keys"][0] == doc_key(short),
              "extract: meta records the weight file's SHA-256; keys are doc_key")
        (folder / "config.json").unlink()
        try:
            extract([short], folder, verbose=False)
            check(False, "extract refuses a model folder without config.json")
        except FileNotFoundError:
            check(True, "extract refuses a model folder without config.json")


def test_single_qubit_channel_closed_forms() -> None:
    """RY(t)|0> then one channel: <Z> = (1-2p)cos t (bit flip), cos t (phase flip),
    (1-4p/3)cos t (depolarizing)."""
    t, p = 0.83, 0.07
    dev = qml.device("default.mixed", wires=1)
    want = {"bitflip": (1 - 2 * p) * np.cos(t), "phaseflip": np.cos(t),
            "depolarizing": (1 - 4 * p / 3) * np.cos(t)}
    for ch in CHANNELS:
        qnode = add_noise(qml.QNode(lambda: (qml.RY(t, wires=0), qml.expval(qml.PauliZ(0)))[1], dev),
                          NoiseModel(ch, p, 0.0))
        got = float(qnode())
        check(abs(got - want[ch]) < 1e-12, f"{ch} after RY: <Z> matches its closed form",
              f"{got:.6f} vs {want[ch]:.6f}")
    qnode = add_noise(qml.QNode(lambda: (qml.RY(t, wires=0), qml.expval(qml.PauliX(0)))[1], dev),
                      NoiseModel("phaseflip", p, 0.0))
    check(abs(float(qnode()) - (1 - 2 * p) * np.sin(t)) < 1e-12,
          "phase flip shrinks <X> by 1 - 2p (it attacks coherences)")


def test_two_rate_placement() -> None:
    """p1 after every 1-qubit gate, p2 on both wires after every 2-qubit gate."""
    model = NoiseModel("depolarizing", 0.001, 0.01)

    def circ():
        qml.RY(0.1, wires=0)
        qml.RY(0.2, wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    tape = qml.tape.make_qscript(circ)()
    [noisy], _ = add_noise(tape, model)
    chans = [op for op in noisy.operations if isinstance(op, qml.DepolarizingChannel)]
    ps = sorted(float(op.parameters[0]) for op in chans)
    check(len(chans) == 4 and ps == [0.001, 0.001, 0.01, 0.01],
          "two-rate model: one p1 channel per 1q gate, p2 on both wires of a CNOT", f"{ps}")


def test_levels_and_validation() -> None:
    """Every level keeps p2 = 10 p1; bad channels and probabilities are refused."""
    ok = all(abs(p2 - 10 * p1) < 1e-15 for p1, p2 in LEVELS.values())
    check(ok and preset("bitflip", "nisq").p2 == 0.01, "noise levels keep p2 = 10 p1 (nisq: 1e-3 / 1e-2)")
    refused = 0
    for kwargs in ({"channel": "amplitude", "p1": 0.1, "p2": 0.1}, {"channel": "bitflip", "p1": 1.2, "p2": 0}):
        try:
            NoiseModel(**kwargs)
        except ValueError:
            refused += 1
    check(refused == 2, "noise model refuses unknown channels and probabilities outside [0, 1)")


def test_validation_selection() -> None:
    """The selected epoch is the best validation epoch; its test and parameters are kept."""
    rng = generator("test-fair", "val")
    x = rng.uniform(0.0, np.pi, size=(90, 2))
    y = (x[:, 0] > np.pi / 2).astype(int)
    rec = train_one(build_model=lambda s: build_pure_model("A1", 2, 1, shots=s),
                    feats_tr=x[:50], y_tr=y[:50], feats_va=x[50:70], y_va=y[50:70],
                    feats_te=x[70:], y_te=y[70:], epochs=5, batch=10, lr=0.1, seed=3,
                    shots_eval=None, keep_params=True)
    vals = [h["val_mcc"] for h in rec["history"]]
    ep = rec["selected"]["epoch"]
    first_best = int(np.argmax(vals)) + 1
    same_test = rec["selected"]["test"]["mcc"] == rec["history"][ep - 1]["test_mcc"]
    check(ep == first_best and same_test and len(rec["params_selected"]) == 2 * 5 + 2,
          "validation selects the first best epoch; its test metrics and parameters are recorded",
          f"epoch {ep}, val {np.round(vals, 3)}")
    legacy = train_one(build_model=lambda s: build_pure_model("A1", 2, 1, shots=s),
                       feats_tr=x[:50], y_tr=y[:50], feats_te=x[70:], y_te=y[70:], epochs=1,
                       batch=10, seed=3, shots_eval=None)
    check("selected" not in legacy and "params_selected" not in legacy,
          "without a validation split the record is the legacy one")


def test_new_metrics() -> None:
    """Macro-F1 and balanced accuracy: a constant predictor cannot score well."""
    y = np.array([1] * 7 + [0] * 3)
    m = classification_metrics(y, np.ones(10, int))
    f1_pos = 2 * 7 / (2 * 7 + 3)
    check(abs(m["macro_f1"] - f1_pos / 2) < 1e-12 and m["balanced_accuracy"] == 0.5 and m["mcc"] == 0.0,
          "constant predictor: macro-F1 = F1_pos / 2, balanced accuracy = 0.5, MCC = 0")


def test_statistics_helpers() -> None:
    """Holm adjustment and the t-interval against hand computations."""
    adj = st.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    check(np.allclose([adj["a"], adj["c"], adj["b"]], [0.03, 0.06, 0.06]), "Holm adjustment", f"{adj}")
    s = st.summary([1.0, 2.0, 3.0])
    check(abs(s["ci95"] - 4.302652729911275 * 1.0 / np.sqrt(3)) < 1e-9 and s["sd"] == 1.0,
          "95% t-interval half-width")


def main() -> int:
    """Run every check and return the process exit code."""
    test_bce_is_two_class_cross_entropy()
    test_legacy_models_are_lifted_unchanged()
    test_step_schedule_and_betas()
    test_legacy_record_reproduces_pre_a5_harness()
    test_doc_key()
    test_frozen_cache_lookup()
    test_frozen_pca_fits_on_train_only()
    test_tfidf_views_are_consistent()
    test_windowing_and_pooling()
    test_task_texts_match_the_loader()
    test_extract_matches_direct_pooling()
    test_single_qubit_channel_closed_forms()
    test_two_rate_placement()
    test_levels_and_validation()
    test_validation_selection()
    test_new_metrics()
    test_statistics_helpers()
    failures = [label for passed, label in OUTCOMES if not passed]
    print(f"\n{len(OUTCOMES) - len(failures)}/{len(OUTCOMES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
