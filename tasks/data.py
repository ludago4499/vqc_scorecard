# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-09-09  Created. Loaders and splits for the two task datasets.

"""Dataset loaders and splits for the task axis.

Two datasets:

* **IMDb-1k stand-in** - the Pang & Lee polarity v2.0 corpus shipped in
  ``old/HEA_scorecard/archive/movie_review.csv`` (sentence-level rows grouped
  into 2000 documents), subsampled to a balanced 1000 (500 pos / 500 neg).
* **CoLA** - the Corpus of Linguistic Acceptability (GLUE), TSV files with
  columns ``source, label, orig_label, sentence``. Fetch with
  :mod:`vqc_scorecard.tasks.get_cola`.

All randomness is seeded, and the chosen document / row indices are returned so
a run is reproducible.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_movie_reviews(csv_path: str | Path, *, seed: int) -> tuple[NDArray, NDArray]:
    """Load the movie-review corpus as documents, shuffled into a random mix.

    The CSV is sentence-level; rows are grouped by ``(tag, cv_tag)`` into one
    document each and the sentences joined in ``sent_id`` order.

    Args:
        csv_path: Path to ``movie_review.csv``.
        seed: Seed for the pos/neg shuffle.

    Returns:
        A pair ``(docs, labels)`` of object and int arrays; label 1 = pos,
        0 = neg.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"movie_review.csv not found: {csv_path}")
    sentences: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    label_of: dict[tuple[str, str], int] = {}
    with csv_path.open(encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["tag"], row["cv_tag"])
            sentences[key].append((int(row["sent_id"]), row["text"]))
            label_of[key] = 1 if row["tag"] == "pos" else 0
    docs, labels = [], []
    for key, sents in sentences.items():
        sents.sort(key=lambda t: t[0])
        docs.append(" ".join(text for _, text in sents))
        labels.append(label_of[key])
    docs = np.array(docs, dtype=object)
    labels = np.array(labels, dtype=int)
    perm = np.random.default_rng(seed).permutation(len(docs))
    return docs[perm], labels[perm]


def subsample_balanced(
    docs: NDArray, labels: NDArray, *, n_per_class: int, seed: int
) -> tuple[NDArray, NDArray]:
    """Return a class-balanced subsample of ``n_per_class`` documents per class.

    Args:
        docs: Document array.
        labels: Matching int labels (0/1).
        n_per_class: Documents to keep from each class.
        seed: Seed for the per-class choice.

    Returns:
        A pair ``(docs_sub, labels_sub)``, shuffled.

    Raises:
        ValueError: If a class has fewer than ``n_per_class`` documents.
    """
    rng = np.random.default_rng(seed + 3)
    keep = []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        if len(idx) < n_per_class:
            raise ValueError(f"class {cls} has {len(idx)} < {n_per_class} docs")
        keep.append(rng.choice(idx, size=n_per_class, replace=False))
    keep = np.concatenate(keep)
    keep = keep[rng.permutation(len(keep))]
    return docs[keep], labels[keep]


def stratified_split(
    labels: NDArray, *, seed: int, test_frac: float = 0.2
) -> tuple[NDArray, NDArray]:
    """Return class-balanced train / test index arrays.

    Args:
        labels: Int labels (0/1).
        seed: Seed for the shuffle.
        test_frac: Fraction held out for test, per class.

    Returns:
        A pair ``(train_idx, test_idx)`` of int arrays, each internally
        shuffled so the stream is not class-blocked.
    """
    rng = np.random.default_rng(seed + 1)
    tr, te = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        idx = idx[rng.permutation(len(idx))]
        cut = int(round(len(idx) * test_frac))
        te.append(idx[:cut])
        tr.append(idx[cut:])
    tr = np.concatenate(tr)
    te = np.concatenate(te)
    return tr[rng.permutation(len(tr))], te[rng.permutation(len(te))]


def load_cola(
    data_dir: str | Path, *, split: str = "train"
) -> tuple[NDArray, NDArray]:
    """Load one CoLA split as sentences and acceptability labels.

    Args:
        data_dir: Directory holding the CoLA TSVs (``in_domain_train.tsv`` /
            ``in_domain_dev.tsv``, or the 1k subsample ``in_domain_train_1k.tsv``).
        split: ``"train"`` or ``"dev"``. ``"train"`` prefers the 1k subsample if
            present, else the full train file.

    Returns:
        A pair ``(sentences, labels)``; label 1 = acceptable, 0 = not.

    Raises:
        FileNotFoundError: If no file for the requested split is found.
    """
    data_dir = Path(data_dir)
    if split == "train":
        candidates = ["in_domain_train_1k.tsv", "in_domain_train.tsv"]
    else:
        candidates = ["in_domain_dev.tsv"]
    path = next((data_dir / c for c in candidates if (data_dir / c).exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"no CoLA {split} file in {data_dir} (looked for {candidates}); "
            "run `python -m vqc_scorecard.tasks.get_cola` first"
        )
    sents, labels = [], []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            labels.append(int(parts[1]))
            sents.append(parts[3])
    return np.array(sents, dtype=object), np.array(labels, dtype=int)
