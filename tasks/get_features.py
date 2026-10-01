# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-09-22  Created. One-shot extraction of frozen pre-trained encoder
#               features, the feature extractor A' of the A5 transfer-learning
#               family (Mari et al. 2020). Companion to get_cola.py. Runs in a
#               separate extraction environment (torch + transformers); the
#               pinned scorecard environment only ever reads the .npz it writes.
#   2026-10-01  Docstrings only: the Mari A5 is retired; the cache is now the
#               fixed representation of pure-v1, read through PCA by every
#               family. Extraction unchanged.

"""Extract frozen encoder features: the fixed representation of pure-v1.

A pre-trained sentence encoder is run once, frozen, and its pooled hidden states
are cached. Nothing downstream trains it: pure-v1 reads these vectors through a
fixed PCA, so the circuit stays the only trainable component. Frozen means the
features of a document never change, so they are computed once here, for every
hidden state of the encoder at the same time, and cached. Every family, depth,
seed and truncation layer then reads the same file.

Run it in the extraction environment, from the folder that holds
``vqc_scorecard/``::

    python -m vqc_scorecard.tasks.get_features --task imdb1k
    python -m vqc_scorecard.tasks.get_features --task cola

What is computed
----------------
* The encoder is ``models/all-MiniLM-L6-v2`` by default (6 transformer layers,
  hidden size 384), loaded with ``transformers.AutoModel`` and
  ``output_hidden_states=True``: state 0 is the embedding output, state ``L``
  the last layer. Choosing a state below ``L`` at run time is the truncation
  ``A -> A'``.
* Pooling is the mean over every real token, ``[CLS]`` and ``[SEP]`` included,
  which is the sentence-transformers convention the model was trained with. No
  L2 normalisation is applied, at any layer: the cache holds the pooled states
  as they are, so every layer is treated the same way.
* Documents longer than one window (the IMDb reviews are) are split into
  consecutive windows of :data:`WINDOW_TOKENS` tokens, each wrapped in
  ``[CLS] ... [SEP]``, and pooled over all tokens of all windows. For a document
  that fits one window this is exactly the model's own mean pooling.

What is written
---------------
``data/features/{task}__{model}.npz`` with ``keys`` (the
:func:`~vqc_scorecard.tasks.features.doc_key` of each document), ``layers``
``(n_states, n_docs, dim)`` float32, ``n_tokens`` and ``n_windows`` per document,
and ``meta``: a JSON record with the model, the SHA-256 of its weight file, the
window, the pooling rule and the library versions. The runner looks documents
up by content key and refuses a cache that misses any of them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..core.seeding import BASE_SEED
from . import data as data_mod
from .features import doc_key

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DEFAULT_MODEL = PKG.parent / "models" / "all-MiniLM-L6-v2"
DEFAULT_OUT_DIR = PKG.parent / "data" / "features"
DEFAULT_CSV = PKG.parent / "old" / "HEA_scorecard" / "archive" / "movie_review.csv"
DEFAULT_COLA = PKG.parent / "data" / "cola"

WINDOW_TOKENS: int = 256
"""Tokens per window, [CLS] and [SEP] included. 256 is the ``max_seq_length`` of
all-MiniLM-L6-v2 (its ``sentence_bert_config.json``), the length it was trained
at, so no window is longer than anything the model saw in training."""

WEIGHT_FILES: tuple[str, ...] = ("model.safetensors", "pytorch_model.bin")


def task_texts(task: str, *, movie_csv: str | Path = DEFAULT_CSV,
               cola_dir: str | Path = DEFAULT_COLA) -> list[str]:
    """Return every distinct document the runner can ask for, in a stable order.

    The same loaders as :mod:`~vqc_scorecard.tasks.run` are used, so the text
    hashed here is the text hashed there. IMDb returns all 2000 reviews, not the
    balanced 1k subsample, so a change of ``--n-per-class`` does not invalidate
    the cache. CoLA returns the train split the runner uses plus the dev split.

    Args:
        task: ``"imdb1k"`` or ``"cola"``.
        movie_csv: The movie-review CSV.
        cola_dir: The CoLA TSV folder.

    Returns:
        The distinct documents, first occurrence kept.

    Raises:
        ValueError: If ``task`` is unknown.
    """
    if task == "imdb1k":
        docs, _ = data_mod.load_movie_reviews(movie_csv, seed=BASE_SEED)
        texts = list(docs)
    elif task == "cola":
        s_tr, _ = data_mod.load_cola(cola_dir, split="train")
        s_te, _ = data_mod.load_cola(cola_dir, split="dev")
        texts = list(s_tr) + list(s_te)
    else:
        raise ValueError(f"unknown task {task!r}")
    seen: set[str] = set()
    unique = []
    for text in texts:
        key = doc_key(text)
        if key not in seen:
            seen.add(key)
            unique.append(str(text))
    return unique


def window_token_ids(ids: Sequence[int], *, window: int, cls_id: int,
                     sep_id: int) -> list[list[int]]:
    """Split one document's token ids into ``[CLS] ... [SEP]`` windows.

    Args:
        ids: Token ids of the document, without special tokens.
        window: Tokens per window including the two special tokens; at least 3.
        cls_id: Id of ``[CLS]``.
        sep_id: Id of ``[SEP]``.

    Returns:
        One id list per window. An empty document gives one ``[CLS] [SEP]``
        window, as the tokenizer itself would.

    Raises:
        ValueError: If ``window`` leaves no room for content.

    Examples:
        >>> window_token_ids([5, 6, 7], window=4, cls_id=1, sep_id=2)
        [[1, 5, 6, 2], [1, 7, 2]]
    """
    content = window - 2
    if content < 1:
        raise ValueError(f"window={window} leaves no room between [CLS] and [SEP]")
    ids = list(ids)
    if not ids:
        return [[cls_id, sep_id]]
    return [[cls_id, *ids[i:i + content], sep_id] for i in range(0, len(ids), content)]


def pool_windows(hidden: NDArray, mask: NDArray) -> NDArray:
    """Mean-pool one document over every real token of every window, per state.

    Args:
        hidden: Hidden states of the document's windows, shape
            ``(n_states, n_windows, n_tokens, dim)``.
        mask: Attention mask ``(n_windows, n_tokens)``, 1 on real tokens.

    Returns:
        Array ``(n_states, dim)``: the token-weighted mean over all windows. For
        a single window this is the model's own mean pooling.
    """
    m = np.asarray(mask, dtype=float)[None, :, :, None]
    total = (np.asarray(hidden, dtype=float) * m).sum(axis=(1, 2))
    return total / max(float(m.sum()), 1.0)


def _weights_sha256(model_dir: Path) -> tuple[str, str]:
    """Return ``(file name, SHA-256)`` of the model's weight file."""
    for name in WEIGHT_FILES:
        path = model_dir / name
        if path.exists():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            return name, digest.hexdigest()
    raise FileNotFoundError(f"no weight file ({', '.join(WEIGHT_FILES)}) in {model_dir}")


def extract(
    texts: Sequence[str],
    model_dir: str | Path,
    *,
    window: int = WINDOW_TOKENS,
    batch: int = 32,
    verbose: bool = True,
) -> dict:
    """Run the frozen encoder over every document and pool every hidden state.

    Imports ``torch`` and ``transformers`` here, not at module level, so the
    module can be imported (and its pure functions tested) without them.

    Args:
        texts: Documents to encode.
        model_dir: Local folder of a Hugging Face encoder (``config.json``,
            tokenizer files, weights).
        window: Tokens per window, special tokens included.
        batch: Windows per forward pass.
        verbose: Print progress.

    Returns:
        A dict with ``keys`` (list of str), ``layers`` float32
        ``(n_states, n_docs, dim)``, ``n_tokens`` and ``n_windows`` (int arrays)
        and ``meta`` (dict).

    Raises:
        FileNotFoundError: If the model folder lacks ``config.json`` or weights.
    """
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"{model_dir / 'config.json'} is missing; download it from the model "
            "repo together with the weights and tokenizer files")
    weight_name, weight_sha = _weights_sha256(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, output_hidden_states=True).eval()
    cls_id, sep_id, pad_id = (tokenizer.cls_token_id, tokenizer.sep_token_id,
                              tokenizer.pad_token_id)

    windows: list[list[int]] = []
    owner: list[int] = []
    n_tokens = np.zeros(len(texts), dtype=np.int64)
    n_windows = np.zeros(len(texts), dtype=np.int64)
    for i, text in enumerate(texts):
        ids = tokenizer(text, add_special_tokens=False, truncation=False,
                        verbose=False)["input_ids"]
        pieces = window_token_ids(ids, window=window, cls_id=cls_id, sep_id=sep_id)
        windows += pieces
        owner += [i] * len(pieces)
        n_tokens[i] = sum(len(p) for p in pieces)
        n_windows[i] = len(pieces)

    n_states = model.config.num_hidden_layers + 1
    dim = model.config.hidden_size
    sums = np.zeros((n_states, len(texts), dim), dtype=np.float64)
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(windows), batch):
            chunk = windows[start:start + batch]
            width = max(len(w) for w in chunk)
            ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for r, w in enumerate(chunk):
                ids[r, :len(w)] = torch.tensor(w, dtype=torch.long)
                mask[r, :len(w)] = 1
            out = model(input_ids=ids, attention_mask=mask,
                        token_type_ids=torch.zeros_like(ids))
            states = torch.stack(out.hidden_states)            # (S, B, T, D)
            masked = (states * mask[None, :, :, None]).sum(dim=2)  # (S, B, D)
            masked = masked.double().numpy()
            for r in range(len(chunk)):
                sums[:, owner[start + r], :] += masked[:, r, :]
            if verbose and (start // batch) % 50 == 0:
                print(f"  windows {start + len(chunk)}/{len(windows)} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
    layers = (sums / n_tokens[None, :, None]).astype(np.float32)

    meta = {
        "model": model_dir.name,
        "model_path": str(model_dir),
        "weights_file": weight_name,
        "weights_sha256": weight_sha,
        "n_states": n_states,
        "hidden_size": dim,
        "window_tokens": window,
        "pooling": "mean over every real token of every window, [CLS]/[SEP] included",
        "normalisation": "none",
        "n_docs": len(texts),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__, "transformers": transformers.__version__},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return {"keys": [doc_key(t) for t in texts], "layers": layers,
            "n_tokens": n_tokens, "n_windows": n_windows, "meta": meta}


def write_cache(path: str | Path, result: dict) -> Path:
    """Write an :func:`extract` result as the ``.npz`` the runner reads.

    Args:
        path: Destination file.
        result: The dict returned by :func:`extract`, with ``meta["task"]`` set.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keys=np.asarray(result["keys"], dtype="U40"),
        layers=result["layers"],
        n_tokens=np.asarray(result["n_tokens"], dtype=np.int32),
        n_windows=np.asarray(result["n_windows"], dtype=np.int32),
        meta=np.asarray(json.dumps(result["meta"], sort_keys=True)),
    )
    return path


def default_cache_path(task: str, model: str = DEFAULT_MODEL.name) -> Path:
    """Return the conventional cache path ``data/features/{task}__{model}.npz``."""
    return DEFAULT_OUT_DIR / f"{task}__{model}.npz"


def main(argv: list[str] | None = None) -> None:
    """Extract and cache the frozen features of one task."""
    ap = argparse.ArgumentParser(description="Frozen-encoder features (fixed pure-v1 representation).")
    ap.add_argument("--task", choices=("imdb1k", "cola"), required=True)
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="local Hugging Face model folder")
    ap.add_argument("--out", default=None,
                    help="output .npz (default data/features/{task}__{model}.npz)")
    ap.add_argument("--window", type=int, default=WINDOW_TOKENS)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--movie-csv", default=str(DEFAULT_CSV))
    ap.add_argument("--cola-dir", default=str(DEFAULT_COLA))
    args = ap.parse_args(argv)

    texts = task_texts(args.task, movie_csv=args.movie_csv, cola_dir=args.cola_dir)
    model_dir = Path(args.model)
    print(f"task={args.task} docs={len(texts)} model={model_dir.name} "
          f"window={args.window}", flush=True)
    result = extract(texts, model_dir, window=args.window, batch=args.batch)
    result["meta"]["task"] = args.task
    out = Path(args.out) if args.out else default_cache_path(args.task, model_dir.name)
    write_cache(out, result)
    multi = int((result["n_windows"] > 1).sum())
    print(f"wrote {out}  states={result['layers'].shape[0]} dim={result['layers'].shape[2]} "
          f"docs={len(texts)} (multi-window: {multi})", flush=True)


if __name__ == "__main__":
    main()
