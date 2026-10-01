# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-10-01  Created. The synthetic-data fixtures of the retired A5 suite
#               (a get_features-style cache whose features carry the label, a
#               movie_review.csv in the corpus layout, a tiny random BERT),
#               moved here unchanged so every suite can share them.

"""Synthetic fixtures shared by the task suites. No real data, no network."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..core.seeding import generator
from ..tasks.features import doc_key
from ..tasks.get_features import write_cache

PKG = Path(__file__).resolve().parent.parent
MOVIE_CSV = PKG.parent / "old" / "HEA_scorecard" / "archive" / "movie_review.csv"


def synthetic_cache(folder: Path, docs, labels, *, dim: int = 8, states: int = 3) -> Path:
    """Write a get_features-style cache whose features carry the label."""
    rng = generator("test-a5", "cache")
    layers = rng.normal(0.0, 1.0, (states, len(docs), dim))
    layers[:, :, 0] += 2.0 * (np.asarray(labels) - 0.5)[None, :]
    result = {"keys": [doc_key(t) for t in docs], "layers": layers.astype(np.float32),
              "n_tokens": np.ones(len(docs), int), "n_windows": np.ones(len(docs), int),
              "meta": {"model": "toy-encoder", "weights_sha256": "0" * 64}}
    return write_cache(folder / "toy.npz", result)


def toy_movie_csv(folder: Path, *, per_class: int) -> Path:
    """Write a movie_review.csv in the corpus's own layout (sentence rows)."""
    rng = generator("test-a5", "csv")
    pos, neg = ["good", "great", "fun", "moving"], ["bad", "awful", "dull", "boring"]
    filler = ["film", "plot", "actor", "scene", "story", "camera"]
    path = folder / "movie_review.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold_id", "cv_tag", "html_id",
                                                    "sent_id", "text", "tag"])
        writer.writeheader()
        for tag, bank in (("pos", pos), ("neg", neg)):
            for i in range(per_class):
                for sent in range(3):
                    words = list(rng.choice(bank, 3)) + list(rng.choice(filler, 4))
                    writer.writerow({"fold_id": 0, "cv_tag": f"cv{i:03d}", "html_id": i,
                                     "sent_id": sent, "text": " ".join(words), "tag": tag})
    return path


def tiny_bert(folder: Path):
    """Save a random 2-layer BERT and a word-level vocab; return the tokenizer words."""
    from transformers import BertConfig, BertModel, BertTokenizer
    words = ["good", "bad", "film", "plot", "great", "awful", "fun", "dull", "the", "a"]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    (folder / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    BertTokenizer(str(folder / "vocab.txt")).save_pretrained(folder)
    import torch
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)
    BertModel(config).eval().save_pretrained(folder)
    return words
