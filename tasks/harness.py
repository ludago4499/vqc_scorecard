# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-10-01  Created. The protocol-independent parts of the fair-v1 driver
#               (keyed JSONL checkpoints, the seeded train / validation / test
#               split, the shared frozen-feature store), moved here unchanged
#               when fair-v1 was archived, so the pure-v1 driver reads the same
#               splits and the same features without importing an archived
#               protocol. The split arithmetic is byte-for-byte the fair-v1
#               arithmetic: seed i gives the same documents in both protocols.

"""Run harness shared by every task protocol.

Three jobs, none of which knows what a circuit is:

* **Checkpoints.** One JSONL file per (task, kind). A record is identified by
  :func:`run_key`; a driver skips a key that is already present, so re-running a
  command never duplicates a seed.
* **Splits.** :func:`task_split` gives the train / validation / test documents
  of one seed. IMDb-1k: the balanced 1000 reviews, 80/20 train/test by seed (the
  v10 split), then a validation share of train. CoLA: the 1k train file split
  train/validation by seed; the in-domain dev set is the test set.
* **Features.** :class:`TaskData` holds the frozen-encoder cache of one task and
  serves every split and every PCA view a protocol asks for, cached per seed.

A protocol object passed to these helpers needs ``val_frac``,
``imdb_per_class``, ``encoder`` and ``layer``; nothing else is read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..core.seeding import BASE_SEED
from . import data as data_mod
from .features import FrozenFeatures, FrozenPca
from .get_features import default_cache_path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DEFAULT_CSV = PKG.parent / "old" / "HEA_scorecard" / "archive" / "movie_review.csv"
DEFAULT_COLA = PKG.parent / "data" / "cola"
TASKS: tuple[str, ...] = ("imdb1k", "cola")


# --------------------------------------------------------------------------- #
# Checkpoints                                                                  #
# --------------------------------------------------------------------------- #

def checkpoint_path(task: str, kind: str, root: Path) -> Path:
    """Return the checkpoint file of one task and kind.

    Args:
        task: Task name.
        kind: ``clean`` (noiseless training), ``noise_eval`` (clean models
            evaluated under noise), ``noise_train`` or ``baselines``.
        root: Results folder of the protocol.

    Returns:
        ``root / "<task>.jsonl"`` for ``clean``, else ``root / "<task>__<kind>.jsonl"``.
    """
    name = task if kind == "clean" else f"{task}__{kind}"
    return root / f"{name}.jsonl"


def run_key(rec: dict) -> tuple:
    """Return the identity of one record: (task, model, n, d, seed, noise tag)."""
    noise = rec.get("noise")
    tag = "clean" if not noise else f"{noise['channel']}@{noise['level']}"
    return (rec["task"], rec["family"], int(rec["n"]), int(rec["d"]), int(rec["seed"]), tag)


def load(path: Path) -> list[dict]:
    """Read a JSONL checkpoint; a missing file is an empty list."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def append(path: Path, rec: dict) -> None:
    """Append one record to a JSONL checkpoint, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- #
# Splits and features                                                          #
# --------------------------------------------------------------------------- #

def task_split(task: str, seed: int, *, protocol, movie_csv: str | Path = DEFAULT_CSV,
               cola_dir: str | Path = DEFAULT_COLA) -> dict:
    """Return the train / validation / test documents and labels of one seed.

    Args:
        task: ``"imdb1k"`` or ``"cola"``.
        seed: Protocol seed.
        protocol: Supplies ``val_frac`` and ``imdb_per_class``.
        movie_csv: Movie-review CSV.
        cola_dir: CoLA folder.

    Returns:
        Dict with ``tr``, ``va``, ``te``: each a ``(docs, labels)`` pair.

    Raises:
        ValueError: If ``task`` is unknown.
    """
    if task == "imdb1k":
        docs, labels = data_mod.load_movie_reviews(movie_csv, seed=BASE_SEED)
        docs, labels = data_mod.subsample_balanced(docs, labels, n_per_class=protocol.imdb_per_class,
                                                   seed=BASE_SEED)
        tr, te = data_mod.stratified_split(labels, seed=seed)
        docs_tr, y_tr, docs_te, y_te = docs[tr], labels[tr], docs[te], labels[te]
    elif task == "cola":
        docs_tr, y_tr = data_mod.load_cola(cola_dir, split="train")
        docs_te, y_te = data_mod.load_cola(cola_dir, split="dev")
    else:
        raise ValueError(f"unknown task {task!r}")
    fit, va = data_mod.stratified_split(y_tr, seed=seed + 7919, test_frac=protocol.val_frac)
    return {"tr": (docs_tr[fit], y_tr[fit]), "va": (docs_tr[va], y_tr[va]), "te": (docs_te, y_te)}


class TaskData:
    """Frozen features of one task, shared by every model and seed of a run.

    Besides the raw frozen vectors of each split, it serves fixed PCA views of
    them: :meth:`pca_view` fits PCA(k) on the training documents of one seed only
    (never on validation or test) and returns the requested view of every split.
    Fits are cached per (seed, k), so seven models at one width share one fit.
    """

    def __init__(self, task: str, *, protocol, cache: str | Path | None = None,
                 movie_csv: str | Path = DEFAULT_CSV, cola_dir: str | Path = DEFAULT_COLA) -> None:
        self.task, self.protocol = task, protocol
        self.movie_csv, self.cola_dir = movie_csv, cola_dir
        path = Path(cache) if cache else default_cache_path(task, protocol.encoder)
        self.frozen = FrozenFeatures(path, layer=protocol.layer)
        self._splits: dict[int, dict] = {}
        self._pca: dict[tuple[int, int], FrozenPca] = {}

    def split(self, seed: int) -> dict:
        """Return the split of one seed with features: ``{part: (x, y, docs)}``."""
        if seed not in self._splits:
            parts = task_split(self.task, seed, protocol=self.protocol,
                               movie_csv=self.movie_csv, cola_dir=self.cola_dir)
            self._splits[seed] = {k: (self.frozen.lookup(docs), y, docs)
                                  for k, (docs, y) in parts.items()}
        return self._splits[seed]

    def pca_view(self, seed: int, k: int, view: str) -> dict[str, NDArray]:
        """Return one fixed PCA(k) view of every split of one seed.

        Args:
            seed: Protocol seed.
            k: PCA components.
            view: ``"angles"`` (standardised, clipped at 3 sigma, mapped onto
                ``[0, pi]``), ``"pca_std"`` (standardised, unclipped) or
                ``"pca_raw"`` (raw scores).

        Returns:
            ``{"tr": ..., "va": ..., "te": ...}``, each of shape ``(rows, k)``.
        """
        if (seed, k) not in self._pca:
            docs_tr = self.split(seed)["tr"][2]
            self._pca[(seed, k)] = FrozenPca(self.frozen, n_dims=k).fit(docs_tr)
        pca = self._pca[(seed, k)]
        fn = getattr(pca, view)
        return {part: np.asarray(fn(self.split(seed)[part][2]), dtype=float)
                for part in ("tr", "va", "te")}

    def representation(self) -> dict:
        """Return the record fields naming the representation."""
        return {"encoder": self.frozen.model, "layer": self.frozen.layer,
                "encoder_sha256": self.frozen.meta.get("weights_sha256")}
