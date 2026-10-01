# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-09-09  Created. Adam training loop for the task axis, generic over the
#               model builder and the feature representation (angles or one-hot).
#               Trains analytically (backprop); logs a finite-shot evaluation
#               pass so the shots column is real.
#   2026-09-22  A5. The loop accepts a TaskModel (model.as_task_model lifts the
#               legacy triples unchanged) and initialises with the model's own
#               rule when it has one; the legacy rule and its RNG order are
#               untouched, so A1-A4 records reproduce bit for bit. Added the
#               optimiser settings of Mari et al. 2020: Adam betas (torch's
#               (0.9, 0.999) vs PennyLane's default (0.9, 0.99)) and a StepLR
#               schedule (lr x gamma every lr_step epochs). Defaults keep the
#               old behaviour. Records gain p_classical, init, optimizer and a
#               per-epoch lr in the history.
#   2026-09-23  Fair protocol. Optional validation split: every epoch is
#               scored on it, and the epoch with the best validation metric
#               (select_on, default MCC) is the selected model; its test
#               metrics, its parameters (keep_params) and its finite-shot
#               evaluation go into the record under "selected". This removes
#               the test-set model selection flagged in the v10.1 review
#               (item 2). Without a validation split nothing changes.

"""Training loop and per-run record for the task axis."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from numpy.typing import NDArray

from .metrics import classification_metrics
from .model import as_task_model


def _bce(p, y, w=None):
    p = pnp.clip(p, 1e-7, 1 - 1e-7)
    terms = y * pnp.log(p) + (1 - y) * pnp.log(1 - p)
    if w is not None:
        return -pnp.mean(w * terms)
    return -pnp.mean(terms)


def _balanced_weights(y: NDArray) -> tuple[float, float]:
    """Return (w_pos, w_neg) inverse-frequency class weights (sklearn 'balanced').

    Weights normalise so a constant predictor no longer minimises the loss:
    ``w_c = N / (2 * N_c)``, giving each class equal total mass. Their mean over a
    class-balanced batch is 1, so the effective step size is unchanged.

    Args:
        y: Training labels (0/1).

    Returns:
        The ``(w_pos, w_neg)`` weight pair.
    """
    y = np.asarray(y, dtype=float)
    n = float(len(y))
    n_pos = float(y.sum())
    n_neg = n - n_pos
    w_pos = n / (2.0 * n_pos) if n_pos else 0.0
    w_neg = n / (2.0 * n_neg) if n_neg else 0.0
    return w_pos, w_neg


def _predict_proba(forward: Callable, params, feats: NDArray, *, batch: int = 256) -> NDArray:
    out = []
    for i in range(0, len(feats), batch):
        fb = np.asarray(feats[i:i + batch], dtype=float)
        out.append(np.asarray(forward(params, fb)))
    return np.concatenate(out) if out else np.array([])


def _predict_labels(forward: Callable, params, feats: NDArray, *, batch: int = 256) -> NDArray:
    return (_predict_proba(forward, params, feats, batch=batch) >= 0.5).astype(int)


def train_one(
    *,
    build_model: Callable[[int | None], tuple[Callable, int, int]],
    feats_tr: NDArray,
    y_tr: NDArray,
    feats_te: NDArray,
    y_te: NDArray,
    epochs: int = 20,
    batch: int = 32,
    lr: float = 0.05,
    seed: int = 20260811,
    shots_eval: int | None = 1024,
    class_weight: str | None = None,
    adam_betas: tuple[float, float] = (0.9, 0.99),
    lr_step: int | None = None,
    lr_gamma: float = 0.1,
    feats_va: NDArray | None = None,
    y_va: NDArray | None = None,
    select_on: str = "mcc",
    keep_params: bool = False,
) -> dict:
    """Train one variational classifier and return its record.

    Args:
        build_model: ``shots -> model``, where the model is a
            :class:`~vqc_scorecard.tasks.model.TaskModel` or the legacy
            ``(forward, n_circuit_params, two_qubit_gates)`` triple. Called with
            ``None`` for analytic training and with ``shots_eval`` for the
            finite-shot evaluation pass.
        feats_tr: Training features (angles ``(N, n)`` or one-hot
            ``(N, n, V)``).
        y_tr: Training labels.
        feats_te: Test features.
        y_te: Test labels.
        epochs: Training epochs.
        batch: Minibatch size.
        lr: Adam step size.
        seed: Seed for the parameter initialization and minibatch order.
        shots_eval: Shots for the finite-shot evaluation pass, or ``None`` to
            skip it.
        class_weight: ``"balanced"`` to weight the BCE by inverse class
            frequency, which removes the majority-class minimum on imbalanced
            tasks such as CoLA; ``None`` for the standard unweighted loss.
        adam_betas: Adam ``(beta1, beta2)``. The default is PennyLane's own,
            the setting every A1-A4 run used; ``(0.9, 0.999)`` is torch's
            default, the setting of Mari et al. 2020.
        lr_step: Multiply the step size by ``lr_gamma`` after every
            ``lr_step`` epochs (torch ``StepLR``); ``None`` keeps it constant.
            Mari et al. 2020 use ``lr_step=10, lr_gamma=0.1``.
        lr_gamma: Decay factor of the schedule.
        feats_va: Validation features, or ``None``. When given, the model is
            selected by validation (see Returns) and the finite-shot pass runs
            on the selected parameters.
        y_va: Validation labels.
        select_on: Validation metric that selects the epoch (a key of
            :func:`~vqc_scorecard.tasks.metrics.classification_metrics`); ties
            keep the earlier epoch.
        keep_params: Store the selected parameters in the record (needed to
            re-evaluate a trained model later, e.g. under noise).

    Returns:
        A record dict with ``p_params`` (circuit angles), ``p_classical``,
        ``two_qubit_gates``, ``init``, ``optimizer``, per-epoch ``history``,
        ``final`` and ``best`` (by F1) test metrics, ``wall_time_s``, ``shots``
        and (if evaluated) ``shot_eval`` metrics. With a validation split it
        also holds ``selected``: ``{"epoch", "val", "test"}`` for the epoch with
        the best validation ``select_on``, and, with ``keep_params``,
        ``params_selected``.

    Note:
        PennyLane's Adam folds the bias correction into the step size, so its
        ``eps`` enters as ``eps`` where torch's enters as
        ``eps * sqrt(1 - beta2**t)``. The two updates differ by ``O(eps)`` only.
    """
    model = as_task_model(build_model(None))
    forward, p_circ, two_q = model.forward, model.p_circuit, model.two_qubit_gates
    rng = np.random.default_rng(seed)
    if model.init is None:
        init = np.concatenate([rng.uniform(0, 2 * np.pi, p_circ), [1.0, 0.0]])
    else:
        init = np.asarray(model.init(rng), dtype=float)
    params = pnp.array(init, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=lr, beta1=adam_betas[0], beta2=adam_betas[1])

    w_pos, w_neg = _balanced_weights(y_tr) if class_weight == "balanced" else (1.0, 1.0)

    n_batches = int(np.ceil(len(feats_tr) / batch))
    history, best = [], {"f1": -1.0}
    use_val = feats_va is not None
    chosen = {"epoch": 0, "val": {select_on: -np.inf}, "test": {}}
    chosen_params = np.array(params)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(feats_tr))
        for bi in range(n_batches):
            sel = order[bi * batch:(bi + 1) * batch]
            fb = np.asarray(feats_tr[sel], dtype=float)
            yb = np.asarray(y_tr[sel], dtype=float)
            wb = np.where(yb >= 0.5, w_pos, w_neg) if class_weight == "balanced" else None
            params = opt.step(lambda v: _bce(forward(v, fb), yb, wb), params)
        te_pred = _predict_labels(forward, params, feats_te)
        mte = classification_metrics(y_te, te_pred)
        entry = {"epoch": ep, "lr": float(opt.stepsize),
                 **{f"test_{k}": v for k, v in mte.items()}}
        if use_val:
            mva = classification_metrics(y_va, _predict_labels(forward, params, feats_va))
            entry.update({f"val_{k}": v for k, v in mva.items()})
            if mva[select_on] > chosen["val"][select_on]:
                chosen = {"epoch": ep, "val": mva, "test": mte}
                chosen_params = np.array(params)
        history.append(entry)
        if mte["f1"] > best["f1"]:
            best = {**mte, "epoch": ep}
        if lr_step and ep % lr_step == 0:
            opt.stepsize = opt.stepsize * lr_gamma
    wall = round(time.time() - t0, 2)

    record = {
        "p_params": int(p_circ),
        "p_classical": int(model.p_classical),
        "two_qubit_gates": int(two_q),
        "init": model.init_rule,
        "optimizer": {"name": "adam", "beta1": float(adam_betas[0]),
                      "beta2": float(adam_betas[1]), "eps": float(opt.eps),
                      "lr_step": int(lr_step) if lr_step else None,
                      "lr_gamma": float(lr_gamma) if lr_step else None},
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "class_weight": class_weight or "none",
        "seed": seed,
        "shots": "analytic",
        "final": classification_metrics(y_te, _predict_labels(forward, params, feats_te)),
        "best": best,
        "history": history,
        "wall_time_s": wall,
    }

    if use_val:
        record["selected"] = {**chosen, "select_on": select_on}
        if keep_params:
            record["params_selected"] = [float(v) for v in chosen_params]

    if shots_eval:
        fwd_s = as_task_model(build_model(shots_eval)).forward
        eval_params = pnp.array(chosen_params) if use_val else params
        shot_pred = _predict_labels(fwd_s, eval_params, feats_te, batch=batch)
        record["shots"] = shots_eval
        record["shot_eval"] = classification_metrics(y_te, shot_pred)

    return record
