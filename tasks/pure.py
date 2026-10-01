# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-10-01  Created. The pure-v1 task protocol: the circuit is the only
#               trainable component (fixed PCA features in, the family's own
#               encoding, <Z_0>, a two-parameter head). Families A1-A7 swept
#               over width and depth (family arm), the A2 ansatz under four
#               encodings (encoding arm, Class II), noise evaluation of the
#               trained models, and like-for-like classical references.
#               Replaces fair-v1 (archived), whose shared dressed interface
#               trained ~1550 classical parameters around every circuit.

"""The pure-v1 task protocol: the circuit is the only trainable component.

Why this module exists
----------------------
fair-v1 gave every family the same trainable classical input layer
``L(384 -> n)`` and output layer ``L(n -> 2)``. Those layers carried the
learning: all families tied, the classical control C0 tied with them, and depth
and width barely moved the result. fair-v1 is therefore a hybrid reference, not
evidence about variational circuits. pure-v1 removes the dressing.

The model (:func:`~.model.build_pure_model`)
-------------------------------------------
* **Features, fixed and unsupervised.** Frozen MiniLM vectors, PCA fitted on the
  training documents of each seed only. Rotation encodings read the scores
  standardised, clipped at 3 sigma and mapped onto ``[0, pi]``; the amplitude
  encoding reads the standardised scores and normalises them. No gradient
  reaches this step.
* **Encoding and circuit.** The family's own (:data:`~.model.MODELS`).
* **Readout.** ``p = sigmoid(a <Z_0> + b)``: the circuit's local readout and two
  classical parameters, the readout scale and bias. Nothing else is trained.
* **Initialisation.** Circuit angles uniform on ``[0, 2 pi)``, head ``(1, 0)``:
  the ensemble under which G1 measures gradient variance.

Design
------
* **Family arm.** A1-A7, widths 4, 6, 8, depths 1, 3, 5, 10 paired seeds.
* **Encoding arm (Class II).** The A2 ansatz at width 4 under the angle (A2),
  re-upload (A3), amplitude (A2:amp) and IQP (A2:iqp) encodings, depths 1, 3, 5.
  The A2 and A3 records are shared with the family arm.
* **Depth and width are factors, not tuning knobs.** Every (n, d) cell is
  reported; nothing is selected across depths. Only the epoch is selected, on
  validation MCC.
* **Classical references** on the same split and the same numbers the circuit
  reads: logistic regression on PCA(n - 1), PCA(n) and PCA(2**n), on the full
  frozen vector and on TF-IDF.

Commands (from the folder that holds ``vqc_scorecard/``)::

    python -m vqc_scorecard.tasks.pure status     --task imdb1k
    python -m vqc_scorecard.tasks.pure train      --task imdb1k --arm family --widths 4
    python -m vqc_scorecard.tasks.pure train      --task imdb1k --arm encoding
    python -m vqc_scorecard.tasks.pure noise-eval --task imdb1k --width 4

See ``RUN_PURE.md`` for the full run order and :mod:`.pure_report` for the
pre-registered hypothesis tests and the result categories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pennylane import numpy as pnp

from ..core.seeding import BASE_SEED
from ..noise import CHANNELS, LEVELS, REPORTED_LEVELS, NoiseModel, preset
from .baseline import logreg
from .features import TfidfPca
from .harness import (DEFAULT_COLA, DEFAULT_CSV, PKG, TASKS, TaskData, append,
                      checkpoint_path, load, run_key)
from .metrics import classification_metrics
from .model import MODELS, build_pure_model
from .train import _predict_proba, train_one

RESULTS = PKG / "results" / "pure"
ARMS: tuple[str, ...] = ("family", "encoding")


@dataclass(frozen=True)
class PureProtocol:
    """The single configuration every pure-v1 run uses.

    Attributes:
        name: Version label of the protocol.
        families: Models of the family arm (each family as published).
        encoding_models: Extra models of the encoding arm (A2 ansatz, swapped
            encoding); the arm also reads the ``A2`` and ``A3`` records.
        widths: Qubit counts of the family arm.
        encoding_width: Qubit count of the encoding arm.
        depths: Circuit depths of both arms.
        n_seeds: Seeds per cell; seed ``i`` is ``BASE_SEED + i``.
        epochs: Training epochs.
        batch: Minibatch size.
        lr: Adam step size.
        adam_betas: Adam ``(beta1, beta2)``.
        init: Initialisation rule of the circuit angles (declared, not a switch).
        class_weight: Loss weighting.
        val_frac: Fraction of the training split held out for validation.
        select_on: Validation metric that selects the epoch.
        headline: The two headline test metrics.
        encoder: Frozen encoder whose cache every model reads.
        layer: Encoder hidden state (``-1`` = last).
        shots_eval: Shots of the finite-shot evaluation of the selected model.
        imdb_per_class: Reviews per class in IMDb-1k.
        noise_width: Width at which trained models are evaluated under noise.
        sesoi: Smallest effect size of interest on test MCC.
        fail_mcc: A seed whose test MCC is below this value is a failed run.
    """

    name: str = "pure-v1"
    families: tuple[str, ...] = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")
    encoding_models: tuple[str, ...] = ("A2:amp", "A2:iqp")
    widths: tuple[int, ...] = (4, 6, 8)
    encoding_width: int = 4
    depths: tuple[int, ...] = (1, 3, 5)
    n_seeds: int = 10
    epochs: int = 20
    batch: int = 32
    lr: float = 0.05
    adam_betas: tuple[float, float] = (0.9, 0.999)
    init: str = "circuit uniform on [0, 2pi); head (a, b) = (1, 0)"
    class_weight: str = "balanced"
    val_frac: float = 0.2
    select_on: str = "mcc"
    headline: tuple[str, ...] = ("mcc", "macro_f1")
    encoder: str = "all-MiniLM-L6-v2"
    layer: int = -1
    shots_eval: int = 1024
    imdb_per_class: int = 500
    noise_width: int = 4
    sesoi: float = 0.03
    fail_mcc: float = 0.2

    def seeds(self) -> list[int]:
        """Return the protocol seeds, ``BASE_SEED + i`` for ``i < n_seeds``."""
        return [BASE_SEED + i for i in range(self.n_seeds)]

    def digest(self) -> str:
        """Return a 12-hex fingerprint of the whole configuration."""
        text = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    def cells(self, arm: str) -> list[tuple[str, int, int]]:
        """Return the ``(model, n, d)`` cells of one arm, in a stable order.

        Args:
            arm: ``"family"`` or ``"encoding"``.

        Returns:
            The cells. The encoding arm lists only its extra models; its A2 and
            A3 cells are family-arm cells at ``encoding_width``.
        """
        if arm == "family":
            return [(m, n, d) for m in self.families for n in self.widths for d in self.depths]
        if arm == "encoding":
            return [(m, self.encoding_width, d) for m in self.encoding_models for d in self.depths]
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")


PROTOCOL = PureProtocol()


# --------------------------------------------------------------------------- #
# Features of one model                                                        #
# --------------------------------------------------------------------------- #

def model_inputs(data: TaskData, tag: str, n: int, seed: int) -> dict:
    """Return the fixed input of one model at width ``n``: ``{part: (x, y)}``.

    Args:
        data: The task's shared features.
        tag: A key of :data:`~.model.MODELS`.
        n: Qubit count.
        seed: Protocol seed.

    Returns:
        ``{"tr": (x, y), "va": (x, y), "te": (x, y)}`` with ``x`` of shape
        ``(rows, MODELS[tag].n_features(n))``.
    """
    spec = MODELS[tag]
    view = data.pca_view(seed, spec.n_features(n), spec.feature_view)
    split = data.split(seed)
    return {part: (view[part], split[part][1]) for part in ("tr", "va", "te")}


# --------------------------------------------------------------------------- #
# Training and evaluation                                                      #
# --------------------------------------------------------------------------- #

def train_run(data: TaskData, tag: str, n: int, d: int, seed: int, *,
              noise: NoiseModel | None = None) -> dict:
    """Train one (model, n, d, seed) under the protocol and return its record.

    Args:
        data: The task's shared features.
        tag: A key of :data:`~.model.MODELS`.
        n: Qubit count.
        d: Depth.
        seed: Protocol seed.
        noise: Train (and evaluate) under this gate-noise model, or ``None``.

    Returns:
        The record: protocol digest, model spec, representation, split sizes,
        parameter and gate counts, the validation-selected epoch with its
        validation and test metrics, the selected parameters, the finite-shot
        test metrics (noiseless runs) and the per-epoch history.
    """
    p = data.protocol
    inp = model_inputs(data, tag, n, seed)
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = inp["tr"], inp["va"], inp["te"]
    rec = train_one(
        build_model=lambda shots: build_pure_model(tag, n, d, noise=noise, shots=shots),
        feats_tr=x_tr, y_tr=y_tr, feats_te=x_te, y_te=y_te, feats_va=x_va, y_va=y_va,
        epochs=p.epochs, batch=p.batch, lr=p.lr, seed=seed,
        shots_eval=p.shots_eval if noise is None else None,
        class_weight=p.class_weight, adam_betas=p.adam_betas, select_on=p.select_on,
        keep_params=True)
    spec = MODELS[tag]
    rec.update(task=data.task, family=tag, ansatz=spec.ansatz, encoding=spec.encoding,
               arm=spec.arm, n=n, d=d, n_features=spec.n_features(n), seed=seed,
               seed_index=seed - BASE_SEED, protocol=p.name, protocol_digest=p.digest(),
               n_train=len(y_tr), n_val=len(y_va), n_test=len(y_te),
               noise=noise.record() if noise else None, **data.representation())
    return rec


def evaluate_under_noise(data: TaskData, rec: dict, noise: NoiseModel) -> dict:
    """Evaluate a trained clean model on its test split under a noise model.

    The selected parameters of ``rec`` are loaded into the same model on a
    mixed-state device with ``noise``; nothing is retrained ("trained in
    simulation, deployed on noisy hardware"). Views:

    * ``test`` - metrics from the exact noisy expectation values;
    * ``test_shots`` - metrics from a finite-shot noisy run;
    * ``prediction_fidelity`` - ``1 - mean |p_noisy - p_clean|`` over the test
      set (1 = the noisy model reproduces the clean function);
    * ``decision_flips`` - test decisions that change.

    Args:
        data: The task's shared features.
        rec: A clean training record with ``params_selected``.
        noise: The noise model.

    Returns:
        A record with the noisy views next to the clean test metrics.
    """
    p = data.protocol
    tag, n, d = rec["family"], rec["n"], rec["d"]
    x_te, y_te = model_inputs(data, tag, n, rec["seed"])["te"]
    params = pnp.array(rec["params_selected"], requires_grad=False)
    t0 = time.time()
    clean = _predict_proba(build_pure_model(tag, n, d).forward, params, x_te)
    noisy = _predict_proba(build_pure_model(tag, n, d, noise=noise).forward, params, x_te)
    shot = _predict_proba(build_pure_model(tag, n, d, noise=noise, shots=p.shots_eval).forward,
                          params, x_te)
    return {"task": rec["task"], "family": tag, "n": n, "d": d, "seed": rec["seed"],
            "protocol": rec["protocol"], "protocol_digest": rec["protocol_digest"],
            "mode": "eval", "noise": noise.record(),
            "test": classification_metrics(y_te, (noisy >= 0.5).astype(int)),
            "test_shots": classification_metrics(y_te, (shot >= 0.5).astype(int)),
            "shots": p.shots_eval, "clean_test": rec["selected"]["test"],
            "mean_abs_dp": float(np.mean(np.abs(noisy - clean))),
            "prediction_fidelity": float(1.0 - np.mean(np.abs(noisy - clean))),
            "decision_flips": int(np.sum((noisy >= 0.5) != (clean >= 0.5))),
            "two_qubit_gates": rec["two_qubit_gates"], "p_params": rec["p_params"],
            "wall_time_s": round(time.time() - t0, 2)}


def baseline_run(data: TaskData, seed: int, n: int) -> dict:
    """Classical references on the same split as the models of width ``n``.

    Logistic regression on the exact numbers each encoding reads (PCA(n - 1) for
    A7, PCA(n) for the rotation encodings, PCA(2**n) for the amplitude
    encoding), on the full standardised frozen vector, and on TF-IDF.

    Args:
        data: The task's shared features.
        seed: Protocol seed.
        n: Qubit count.

    Returns:
        The baseline record of (seed, n).
    """
    split = data.split(seed)
    (x_tr, y_tr, d_tr), (x_te, y_te, d_te) = split["tr"], split["te"]
    out = {"task": data.task, "family": "baselines", "n": n, "d": 0, "seed": seed,
           "protocol": data.protocol.name, "protocol_digest": data.protocol.digest()}
    for k in sorted({n - 1, n, 2 ** n}):
        if k > data.frozen.dim or k > len(y_tr):
            continue
        view = data.pca_view(seed, k, "pca_std")
        out[f"logreg_pca{k}"] = logreg(view["tr"], y_tr, view["te"], y_te)
    mu, sd = x_tr.mean(axis=0), np.maximum(x_tr.std(axis=0), 1e-12)
    out["logreg_frozen"] = logreg((x_tr - mu) / sd, y_tr, (x_te - mu) / sd, y_te)
    tfidf = TfidfPca(n_dims=n).fit(d_tr)
    out["logreg_tfidf"] = logreg(tfidf.tfidf_raw(d_tr), y_tr, tfidf.tfidf_raw(d_te), y_te)
    return out


# --------------------------------------------------------------------------- #
# Planning and status                                                          #
# --------------------------------------------------------------------------- #

def plan(cells, seeds, task: str, noise: NoiseModel | None = None) -> list[tuple]:
    """Return the run keys of a request, in a stable order."""
    tag = "clean" if noise is None else noise.tag
    return [(task, m, n, d, s, tag) for (m, n, d) in cells for s in seeds]


def status(task: str, *, protocol: PureProtocol = PROTOCOL, root: Path = RESULTS) -> str:
    """Return a completeness grid of the clean checkpoint, one row per (model, n)."""
    done = {run_key(r) for r in load(checkpoint_path(task, "clean", root))
            if r.get("protocol_digest") == protocol.digest()}
    lines = [f"task {task}  protocol {protocol.name} [{protocol.digest()}]",
             f"{'model':8s} {'n':>2s}" + "".join(f"  d={d:<3d}" for d in protocol.depths)]
    rows = sorted({(m, n) for arm in ARMS for (m, n, _) in protocol.cells(arm)},
                  key=lambda mn: (list(MODELS).index(mn[0]), mn[1]))
    for m, n in rows:
        cells = []
        for d in protocol.depths:
            k = sum((task, m, n, d, s, "clean") in done for s in protocol.seeds())
            cells.append(f"  {k:2d}/{protocol.n_seeds}")
        lines.append(f"{m:8s} {n:2d}" + "".join(cells))
    noise = load(checkpoint_path(task, "noise_eval", root))
    lines.append(f"noise-eval records: {len(noise)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Command line                                                                 #
# --------------------------------------------------------------------------- #

def _seed_slice(protocol: PureProtocol, spec: str | None) -> list[int]:
    seeds = protocol.seeds()
    if not spec:
        return seeds
    lo, _, hi = spec.partition("-")
    return seeds[int(lo):int(hi or lo) + 1]


def _log(rec: dict) -> None:
    sel = rec.get("selected", {})
    te = sel.get("test", rec.get("test", {}))
    noise = rec.get("noise")
    tag = "clean" if not noise else f"{noise['channel']}@{noise['level']}"
    print(f"  {rec['task']:6s} {rec['family']:7s} n={rec['n']} d={rec['d']} seed={rec['seed']} "
          f"[{tag}] | test MCC {te.get('mcc', float('nan')):+.3f} "
          f"macroF1 {te.get('macro_f1', float('nan')):.3f} | ep {sel.get('epoch', '-')} "
          f"| {rec.get('wall_time_s', 0)}s", flush=True)


def _requested_cells(p: PureProtocol, args) -> list[tuple[str, int, int]]:
    arms = ARMS if args.arm == "all" else (args.arm,)
    cells = [c for arm in arms for c in p.cells(arm)]
    if args.models:
        unknown = [m for m in args.models if m not in MODELS]
        if unknown:
            raise SystemExit(f"unknown models {unknown}; choose from {list(MODELS)}")
        cells = [c for c in cells if c[0] in args.models]
    if args.widths:
        cells = [c for c in cells if c[1] in args.widths]
    if args.depths:
        cells = [c for c in cells if c[2] in args.depths]
    return cells


def main(argv: list[str] | None = None, *, root: Path = RESULTS,
         protocol: PureProtocol | None = None) -> None:
    """Command-line entry point. See the module docstring."""
    p = protocol or PROTOCOL
    ap = argparse.ArgumentParser(description="pure-v1 task protocol (A1-A7, circuit-only training).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--task", choices=TASKS, required=True)
        sp.add_argument("--cache", default=None, help="frozen-feature cache (default by task)")
        sp.add_argument("--movie-csv", default=str(DEFAULT_CSV))
        sp.add_argument("--cola-dir", default=str(DEFAULT_COLA))

    sp = sub.add_parser("status", help="completeness grid of the checkpoints")
    sp.add_argument("--task", choices=TASKS, required=True)

    sp = sub.add_parser("train", help="train protocol cells (skips finished keys)")
    common(sp)
    sp.add_argument("--arm", choices=(*ARMS, "all"), default="family")
    sp.add_argument("--models", nargs="+", default=None, help="subset of the arm's models")
    sp.add_argument("--widths", nargs="+", type=int, default=None, help="subset of the widths")
    sp.add_argument("--depths", nargs="+", type=int, default=None, help="subset of the depths")
    sp.add_argument("--seeds", default=None, help="seed indices, e.g. 0-4 (default all)")

    sp = sub.add_parser("noise-eval", help="evaluate trained clean models under noise")
    common(sp)
    sp.add_argument("--channels", nargs="+", choices=CHANNELS, default=list(CHANNELS))
    sp.add_argument("--levels", nargs="+", choices=list(LEVELS), default=list(REPORTED_LEVELS))
    sp.add_argument("--width", type=int, default=p.noise_width)
    sp.add_argument("--models", nargs="+", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "status":
        print(status(args.task, protocol=p, root=root))
        return

    data = TaskData(args.task, protocol=p, cache=args.cache, movie_csv=args.movie_csv,
                    cola_dir=args.cola_dir)
    print(f"protocol {p.name} [{p.digest()}]  task {args.task}  features "
          f"{data.frozen.model} L{data.frozen.layer}", flush=True)

    if args.cmd == "noise-eval":
        out = checkpoint_path(args.task, "noise_eval", root)
        done = {run_key(r) for r in load(out)}
        clean = [r for r in load(checkpoint_path(args.task, "clean", root))
                 if r.get("protocol_digest") == p.digest() and r["n"] == args.width
                 and (not args.models or r["family"] in args.models)]
        for level in args.levels:
            for channel in args.channels:
                model = preset(channel, level)
                for rec in clean:
                    key = run_key({**rec, "noise": model.record()})
                    if key in done:
                        continue
                    res = evaluate_under_noise(data, rec, model)
                    append(out, res)
                    done.add(key)
                    _log(res)
        print(f"done -> {out}")
        return

    cells = _requested_cells(p, args)
    out = checkpoint_path(args.task, "clean", root)
    done = {run_key(r) for r in load(out)}
    seeds = _seed_slice(p, args.seeds)
    todo = [k for k in plan(cells, seeds, args.task) if k not in done]
    print(f"{len(todo)} runs to do ({len(done)} already in {out.name})", flush=True)

    base_out = checkpoint_path(args.task, "baselines", root)
    have = {(r["seed"], r["n"]) for r in load(base_out) if r.get("protocol_digest") == p.digest()}
    for n in sorted({c[1] for c in cells}):
        for seed in seeds:
            if (seed, n) not in have:
                append(base_out, baseline_run(data, seed, n))

    for _, tag, n, d, seed, _ in todo:
        rec = train_run(data, tag, n, d, seed)
        append(out, rec)
        _log(rec)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
