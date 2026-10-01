# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-09-09  Created. CLI that runs a whole task (IMDb-1k or CoLA) for a set
#               of families, depths and seeds, and writes one JSONL record per
#               (family, route, n, d, seed) plus the classical baseline.
#   2026-09-22  A5. New route "dressed" (the A5 default): the dressed quantum
#               circuit of Mari et al. 2020, with --prelayer dressed|pca. New
#               representation switch --features tfidf|frozen: "frozen" reads a
#               get_features cache (--encoder-cache, --layer) and serves every
#               family, so A1-A4 can run on the same pre-trained features (the
#               Class II control). Frozen runs add two baselines, logreg on the
#               full frozen vector and on its PCA(n); the TF-IDF baselines are
#               always kept. Optimiser flags of the paper: --adam-beta1/2,
#               --lr-step, --lr-gamma; --q-delta for the A5 initial angles.
#               All defaults keep the previous behaviour for A1-A4. main()
#               takes an optional argv, for tests.
#   2026-10-01  pure-v1. The A5 dressed route and its flags (--prelayer,
#               --q-delta) are removed with the retired Mari A5 (archived in
#               old/fair_v1_archive). This runner stays the legacy v10 task
#               axis for A1-A4 (TF-IDF or frozen features, embedding or
#               compositional route); pure-v1 runs go through tasks.pure.

"""Run the task axis end to end.

Examples (run from the folder that contains ``vqc_scorecard/``)::

    python -m vqc_scorecard.tasks.run --task imdb1k --families A1 A2 A3 A4
    python -m vqc_scorecard.tasks.run --task cola --depths 1 2 3 --seeds 3

Legacy runner (v10 task axis, A1-A4). The pre-registered pure-v1 protocol for
A1-A7 is :mod:`~vqc_scorecard.tasks.pure`; use this runner only to reproduce the
v10 tables.

Routes (``--route auto``, the default): A1/A2/A3 use the embedding route; A4 is
compositional on CoLA (sentence-level) and embedding on IMDb (long documents).

Representations (``--features``): ``tfidf`` (default) or ``frozen``, a cached
pre-trained encoder (build it with :mod:`~vqc_scorecard.tasks.get_features`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..core.seeding import BASE_SEED
from . import data as data_mod
from .baseline import logreg
from .features import CompVocab, FrozenFeatures, FrozenPca, TfidfPca
from .get_features import default_cache_path
from .model import LEGACY_FAMILIES, build_compositional_model, build_feature_model
from .train import train_one

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DEFAULT_CSV = PKG.parent / "old" / "HEA_scorecard" / "archive" / "movie_review.csv"
DEFAULT_COLA = PKG.parent / "data" / "cola"


def _route_for(family: str, task: str, route: str) -> str:
    if family not in LEGACY_FAMILIES:
        raise SystemExit(f"{family} is not a legacy family {LEGACY_FAMILIES}; "
                         "run it with vqc_scorecard.tasks.pure")
    if route != "auto":
        return route
    if family == "A4" and task == "cola":
        return "compositional"
    return "embedding"


def _representation(args, frozen: FrozenFeatures | None, tfidf: TfidfPca,
                    train_docs):
    """Return the fitted representation every quantum route reads."""
    if frozen is None:
        return tfidf
    return FrozenPca(frozen, n_dims=args.width).fit(train_docs)


def _baselines(tfidf: TfidfPca, rep, docs_tr, y_tr, docs_te, y_te) -> dict:
    """Logistic-regression references on identical splits.

    ``pca`` and ``tfidf`` are always present, so every table carries the same
    two references. A frozen representation adds ``frozen_pca`` (the
    like-for-like control: the same n numbers the circuit sees) and ``frozen``
    (the full encoder vector, standardised on the training split).
    """
    base = {
        "pca": logreg(tfidf.pca_raw(docs_tr), y_tr, tfidf.pca_raw(docs_te), y_te),
        "tfidf": logreg(tfidf.tfidf_raw(docs_tr), y_tr, tfidf.tfidf_raw(docs_te), y_te),
    }
    if isinstance(rep, FrozenPca):
        base["frozen_pca"] = logreg(rep.pca_raw(docs_tr), y_tr, rep.pca_raw(docs_te), y_te)
        x_tr, x_te = rep.raw(docs_tr), rep.raw(docs_te)
        mu, sd = x_tr.mean(axis=0), np.maximum(x_tr.std(axis=0), 1e-12)
        base["frozen"] = logreg((x_tr - mu) / sd, y_tr, (x_te - mu) / sd, y_te)
    return base


def _quantum_run(args, fam: str, route: str, d: int, rep, docs_tr, docs_te,
                 y_tr, y_te, seed: int, cw, extra: dict | None = None) -> dict:
    """Train one (family, route, depth, seed) on the embedding route."""
    x_tr, x_te = rep.angles(docs_tr), rep.angles(docs_te)
    d_in = x_tr.shape[1]
    builder = lambda shots, f=fam, dd=d: build_feature_model(  # noqa: E731
        f, args.width, dd, shots=shots)
    rec = train_one(
        build_model=builder, feats_tr=x_tr, y_tr=y_tr, feats_te=x_te, y_te=y_te,
        epochs=args.epochs, batch=args.batch, lr=args.lr, seed=seed,
        shots_eval=args.shots_eval, class_weight=cw,
        adam_betas=(args.adam_beta1, args.adam_beta2),
        lr_step=args.lr_step or None, lr_gamma=args.lr_gamma)
    rec["d_in"] = int(d_in)
    rec.update(extra or {})
    return rec


COMPOSITIONAL_TAGS: dict = {"features": "word-ids", "encoder": None, "layer": None}
"""Representation fields of a compositional (A4 word-box) record: it reads word
ids, never the TF-IDF or frozen features of the run."""


def _representation_tags(frozen: FrozenFeatures | None) -> dict:
    """Return the record fields that name the representation."""
    if frozen is None:
        return {"features": "tfidf", "encoder": None, "layer": None}
    return {"features": "frozen", "encoder": frozen.model, "layer": frozen.layer,
            "encoder_sha256": frozen.meta.get("weights_sha256")}


def _load_frozen(args) -> FrozenFeatures | None:
    if args.features != "frozen":
        return None
    path = Path(args.encoder_cache) if args.encoder_cache else default_cache_path(args.task)
    frozen = FrozenFeatures(path, layer=args.layer)
    print(f"frozen features: {frozen.path.name} layer {frozen.layer}/"
          f"{frozen.n_states - 1} dim {frozen.dim}", flush=True)
    return frozen


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def run_imdb(args, seeds: list[int], out: Path) -> None:
    cw = None if args.class_weight == "none" else args.class_weight
    docs, labels = data_mod.load_movie_reviews(args.movie_csv, seed=BASE_SEED)
    docs, labels = data_mod.subsample_balanced(
        docs, labels, n_per_class=args.n_per_class, seed=BASE_SEED)
    frozen = _load_frozen(args)
    tags = _representation_tags(frozen)
    for seed in seeds:
        tr, te = data_mod.stratified_split(labels, seed=seed)
        feat = TfidfPca(n_dims=args.width).fit(docs[tr])
        rep = _representation(args, frozen, feat, docs[tr])
        y_tr, y_te = labels[tr], labels[te]
        base = _baselines(feat, rep, docs[tr], y_tr, docs[te], y_te)
        for fam in args.families:
            route = _route_for(fam, "imdb1k", args.route)
            if route == "compositional":
                raise SystemExit("the compositional route is sentence-level; IMDb has none")
            for d in args.depths:
                rec = _quantum_run(args, fam, route, d, rep, docs[tr], docs[te],
                                   y_tr, y_te, seed, cw)
                rec.update(task="imdb1k", family=fam, route=route,
                           n=args.width, d=d, n_train=len(tr), n_test=len(te),
                           vocab_size=feat.vocab_size, classical=base, **tags)
                _append(out, rec)
                _log(rec)


def run_cola(args, seeds: list[int], out: Path) -> None:
    cw = None if args.class_weight == "none" else args.class_weight
    s_tr, y_tr = data_mod.load_cola(args.cola_dir, split="train")
    s_te, y_te = data_mod.load_cola(args.cola_dir, split="dev")
    feat = TfidfPca(n_dims=args.width).fit(s_tr)
    frozen = _load_frozen(args)
    tags = _representation_tags(frozen)
    rep = _representation(args, frozen, feat, s_tr)
    base = _baselines(feat, rep, s_tr, y_tr, s_te, y_te)
    vocab = CompVocab(n_words=args.width).fit(s_tr)
    oh_tr, oh_te = vocab.onehot(s_tr), vocab.onehot(s_te)
    for seed in seeds:
        for fam in args.families:
            route = _route_for(fam, "cola", args.route)
            for d in args.depths:
                if route == "compositional":
                    rec = train_one(
                        build_model=lambda shots, dd=d: build_compositional_model(
                            args.width, dd, vocab.vocab_size, shots=shots),
                        feats_tr=oh_tr, y_tr=y_tr, feats_te=oh_te, y_te=y_te,
                        epochs=args.epochs, batch=args.batch, lr=args.lr,
                        seed=seed, shots_eval=args.shots_eval, class_weight=cw)
                    rec["comp_vocab"] = int(vocab.vocab_size)
                else:
                    rec = _quantum_run(args, fam, route, d, rep, s_tr, s_te,
                                       y_tr, y_te, seed, cw)
                rec_tags = COMPOSITIONAL_TAGS if route == "compositional" else tags
                rec.update(task="cola", family=fam, route=route, n=args.width, d=d,
                           n_train=len(s_tr), n_test=len(s_te),
                           vocab_size=feat.vocab_size, classical=base, **rec_tags)
                _append(out, rec)
                _log(rec)


def _log(rec: dict) -> None:
    f = rec["final"]
    se = rec.get("shot_eval", {})
    print(f"  {rec['task']:6s} {rec['family']} {rec['route']:13s} d={rec['d']} "
          f"seed={rec['seed']} | F1 {f['f1']:.3f} acc {f['accuracy']:.3f} "
          f"mcc {f['mcc']:+.3f} | shot F1 {se.get('f1', float('nan')):.3f} | "
          f"p={rec['p_params']}+{rec.get('p_classical', 2)}cl "
          f"2q={rec['two_qubit_gates']} {rec['wall_time_s']}s",
          flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Legacy task axis runner (A1-A4).")
    ap.add_argument("--task", choices=("imdb1k", "cola"), required=True)
    ap.add_argument("--families", nargs="+", default=["A1", "A2", "A3", "A4"])
    ap.add_argument("--route", choices=("auto", "embedding", "compositional"),
                    default="auto")
    ap.add_argument("--width", type=int, default=4, help="qubit count n")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--shots-eval", type=int, default=1024)
    ap.add_argument("--class-weight", choices=("none", "balanced"), default="none",
                    help="'balanced' weights the BCE by inverse class frequency "
                         "(CoLA rescue: removes the majority-class minimum)")
    ap.add_argument("--n-per-class", type=int, default=500, help="IMDb-1k per class")
    ap.add_argument("--movie-csv", default=str(DEFAULT_CSV))
    ap.add_argument("--cola-dir", default=str(DEFAULT_COLA))
    ap.add_argument("--out", default=None)
    # representation (the Class II control for A1-A4)
    ap.add_argument("--features", choices=("tfidf", "frozen"), default="tfidf",
                    help="tfidf (default) or frozen pre-trained encoder features")
    ap.add_argument("--encoder-cache", default=None,
                    help="get_features .npz (default data/features/"
                         "{task}__all-MiniLM-L6-v2.npz)")
    ap.add_argument("--layer", type=int, default=-1,
                    help="encoder hidden state to read: 0 = embeddings, -1 = last "
                         )
    # optimiser (defaults = the A1-A4 protocol; Mari 2020: 0.999, 10, 0.1)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.99,
                    help="PennyLane default 0.99; torch / Mari 2020 0.999")
    ap.add_argument("--lr-step", type=int, default=0,
                    help="decay the step size every N epochs (0 = constant)")
    ap.add_argument("--lr-gamma", type=float, default=0.1)
    args = ap.parse_args(argv)

    seeds = [BASE_SEED + i for i in range(args.seeds)]
    out = Path(args.out) if args.out else (PKG / "results" / f"task_{args.task}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"task={args.task} families={args.families} depths={args.depths} "
          f"seeds={seeds} -> {out}", flush=True)

    if args.task == "imdb1k":
        run_imdb(args, seeds, out)
    else:
        run_cola(args, seeds, out)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
