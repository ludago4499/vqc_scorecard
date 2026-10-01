# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-09-09  Created. TF-IDF -> PCA angle features (embedding route) and a
#               content-word vocabulary (compositional route). Pure NumPy; the
#               TF-IDF/PCA pipeline is ported from old task_sentiment.py.
#   2026-09-22  A5 support. Added the frozen-encoder representation: doc_key
#               (the SHA-1 text key shared with get_features), FrozenFeatures
#               (hash-keyed lookup into a get_features cache at one encoder
#               layer) and FrozenPca (PCA(n) of those features, same interface
#               as TfidfPca). The PCA fit and the [0, pi] angle map are now
#               module functions shared by both representations (_fit_pca,
#               _to_angles); TfidfPca's arithmetic is unchanged, checked
#               bit-for-bit against the previous version. Both classes gained
#               pca_std() (the standardised, unclipped PCA scores that feed the
#               fixed pre-layer of A5) and raw() (the full feature vector that
#               feeds the trainable pre-layer of A5).
#   2026-10-01  Docstrings only: the Mari A5 is retired. pca_std now feeds the
#               amplitude encoding of pure-v1 (A5 circuit-centric); raw feeds
#               the classical references. No arithmetic changed.

"""Feature builders for the task routes.

Two representations of a document, one interface:

* :class:`TfidfPca` - TF-IDF then PCA, fit on the training split only.
* :class:`FrozenPca` - a frozen pre-trained encoder (read from a
  :mod:`~vqc_scorecard.tasks.get_features` cache through
  :class:`FrozenFeatures`), then PCA, fit on the training split only. The
  fixed representation of pure-v1, read by every family.

Both expose the same four views, so the runner does not care which one it holds:

* ``angles(docs)`` - top-``n`` PCA scores standardised, clipped at 3 sigma and
  mapped onto ``[0, pi]``; the embedding route of every family.
* ``pca_std(docs)`` - the same standardised scores, unclipped; the input of the
  amplitude encoding (A5), which normalises each row.
* ``raw(docs)`` - the full feature vector (TF-IDF, or the encoder's pooled
  hidden state); the input of the full-vector classical reference.
* ``pca_raw(docs)`` - the raw PCA scores; the like-for-like classical control.

The compositional route of A4 uses :class:`CompVocab` instead: a content-word
vocabulary from the training sentences, each sentence encoded as ``n`` word ids
(one per qubit), returned as a one-hot tensor the compositional QNode consumes.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

TOKEN_RE = re.compile(r"[a-z']+")


def doc_key(text: str) -> str:
    """Return the cache key of one document: the SHA-1 hex digest of its UTF-8 text.

    The frozen-feature cache is keyed by content, not by row position. The IMDb
    loader shuffles with a seed and the runner subsamples, so a position-keyed
    cache would silently pair features with the wrong labels after any change
    upstream. A content key cannot.

    Args:
        text: The document exactly as the task loader returns it.

    Returns:
        A 40-character lowercase hex string.

    Examples:
        >>> doc_key("abc")
        'a9993e364706816aba3e25717850c26c9cd0d89d'
    """
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def _fit_pca(x_tr: NDArray, n_dims: int) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Fit PCA on training rows and return ``(mean, components, z_mean, z_std)``.

    ``z`` are the training scores on the top ``n_dims`` components; their mean
    and spread standardise every later transform, so test rows never touch the
    fit.
    """
    mu = x_tr.mean(axis=0)
    _, _, vt = np.linalg.svd(x_tr - mu, full_matrices=False)
    comps = vt[:n_dims]
    z_tr = (x_tr - mu) @ comps.T
    return mu, comps, z_tr.mean(axis=0), z_tr.std(axis=0)


def _standardise(z: NDArray, z_mean: NDArray, z_std: NDArray) -> NDArray:
    """Return PCA scores in units of the training spread."""
    return (z - z_mean) / np.maximum(z_std, 1e-12)


def _to_angles(z: NDArray, z_mean: NDArray, z_std: NDArray) -> NDArray:
    """Map PCA scores onto ``[0, pi]``: standardise, clip at 3 sigma, rescale."""
    s = np.clip(_standardise(z, z_mean, z_std), -3.0, 3.0)
    return (s + 3.0) / 6.0 * np.pi


def tokenize(doc: str, *, min_len: int = 3) -> list[str]:
    """Return lowercase word tokens of at least ``min_len`` characters."""
    return [t for t in TOKEN_RE.findall(doc.lower()) if len(t) >= min_len]


class TfidfPca:
    """TF-IDF then PCA, fit on train only, mapped to circuit angles.

    Attributes:
        n_dims: Number of PCA components / qubits / angles.
        vocab_size: Size of the fitted TF-IDF vocabulary.
    """

    def __init__(self, *, n_dims: int, max_features: int = 3000, min_df: int = 3) -> None:
        """Store the hyperparameters; call :meth:`fit` before transforming.

        Args:
            n_dims: PCA components to keep (= qubit count ``n``).
            max_features: TF-IDF vocabulary cap, by document frequency.
            min_df: Minimum document frequency for a term to be kept.
        """
        self.n_dims = n_dims
        self.max_features = max_features
        self.min_df = min_df
        self._fitted = False

    def fit(self, train_docs: NDArray) -> "TfidfPca":
        """Fit TF-IDF and PCA on the training documents.

        Args:
            train_docs: Training document array.

        Returns:
            This instance.
        """
        df: dict[str, int] = {}
        self._tok_tr: list[list[str]] = []
        for doc in train_docs:
            toks = tokenize(doc)
            self._tok_tr.append(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        kept = sorted(((t, c) for t, c in df.items() if c >= self.min_df),
                      key=lambda kv: (-kv[1], kv[0]))[: self.max_features]
        self._vocab = {t: i for i, (t, _) in enumerate(kept)}
        n_docs = len(train_docs)
        self._idf = np.zeros(len(self._vocab))
        for t, i in self._vocab.items():
            self._idf[i] = np.log((1.0 + n_docs) / (1.0 + df[t])) + 1.0
        x_tr = self._tfidf(train_docs, self._tok_tr)
        self._mu, self._comps, self._zmean, self._zstd = _fit_pca(x_tr, self.n_dims)
        self.vocab_size = len(self._vocab)
        self._fitted = True
        return self

    def _tfidf(self, docs: NDArray, tok: list[list[str]] | None = None) -> NDArray:
        x = np.zeros((len(docs), len(self._vocab)))
        for r, doc in enumerate(docs):
            toks = tok[r] if tok is not None else tokenize(doc)
            counts: dict[int, int] = {}
            for t in toks:
                j = self._vocab.get(t)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                x[r, j] = (1.0 + np.log(c)) * self._idf[j]
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norm, 1e-12)

    def _pca(self, docs: NDArray) -> NDArray:
        return (self._tfidf(docs) - self._mu) @ self._comps.T

    def angles(self, docs: NDArray) -> NDArray:
        """Return standardized, clipped PCA features mapped to ``[0, pi]``.

        Args:
            docs: Documents to transform.

        Returns:
            Array ``(n_docs, n_dims)`` of angles in ``[0, pi]``.
        """
        return _to_angles(self._pca(docs), self._zmean, self._zstd)

    def pca_std(self, docs: NDArray) -> NDArray:
        """Return the standardised PCA scores, unclipped (amplitude-encoding input)."""
        return _standardise(self._pca(docs), self._zmean, self._zstd)

    def pca_raw(self, docs: NDArray) -> NDArray:
        """Return raw PCA features (for the like-for-like classical control)."""
        return self._pca(docs)

    def tfidf_raw(self, docs: NDArray) -> NDArray:
        """Return full TF-IDF features (for the honest classical reference)."""
        return self._tfidf(docs)

    def raw(self, docs: NDArray) -> NDArray:
        """Return the full feature vector, here TF-IDF (classical reference input)."""
        return self._tfidf(docs)

    @property
    def raw_dim(self) -> int:
        """Length of :meth:`raw` rows: the fitted TF-IDF vocabulary size."""
        return self.vocab_size


class FrozenFeatures:
    """One layer of a frozen-encoder feature cache, looked up by document content.

    The cache is the ``.npz`` written by :mod:`~vqc_scorecard.tasks.get_features`:
    ``keys`` (one :func:`doc_key` per document), ``layers`` of shape
    ``(n_states, n_docs, dim)`` (hidden state 0 is the embedding output, state
    ``L`` the last transformer layer), and a JSON ``meta`` record. Choosing a
    layer below the last is the truncation of the pre-trained network ``A`` to
    ``A'`` in Mari et al. 2020 (Sec. 3, step 2).

    Attributes:
        path: The cache file.
        layer: The resolved, non-negative hidden-state index in use.
        n_states: Number of hidden states in the cache (layers + 1).
        dim: Feature length.
        meta: The extraction record (model, pooling, window, versions, hashes).
    """

    def __init__(self, cache_path: str | Path, *, layer: int = -1) -> None:
        """Load the cache and select one hidden state.

        Args:
            cache_path: Path to the ``.npz`` cache.
            layer: Hidden-state index; negative values count from the last layer,
                so ``-1`` is the final layer.

        Raises:
            FileNotFoundError: If the cache does not exist.
            ValueError: If ``layer`` is outside the cache.
        """
        self.path = Path(cache_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"frozen-feature cache not found: {self.path}; build it with "
                "`python -m vqc_scorecard.tasks.get_features` in the extraction env")
        with np.load(self.path, allow_pickle=False) as cache:
            keys = cache["keys"]
            layers = cache["layers"]
            self.meta = json.loads(str(cache["meta"]))
        self.n_states = int(layers.shape[0])
        index = layer if layer >= 0 else self.n_states + layer
        if not 0 <= index < self.n_states:
            raise ValueError(f"layer {layer} outside the cache (0..{self.n_states - 1})")
        self.layer = index
        self.dim = int(layers.shape[2])
        self._x = np.asarray(layers[index], dtype=float)
        self._row = {str(k): i for i, k in enumerate(keys)}

    @property
    def model(self) -> str:
        """Short name of the encoder, as recorded at extraction."""
        return str(self.meta.get("model", self.path.stem))

    def lookup(self, docs: NDArray) -> NDArray:
        """Return the ``(len(docs), dim)`` feature rows of the given documents.

        Args:
            docs: Documents exactly as the task loader returns them.

        Returns:
            Float64 array, one row per document, in the order given.

        Raises:
            KeyError: If any document is missing from the cache. Nothing is
                filled in or skipped: a partial cache is a stale cache.
        """
        rows = [self._row.get(doc_key(doc), -1) for doc in docs]
        missing = sum(1 for r in rows if r < 0)
        if missing:
            raise KeyError(
                f"{missing} of {len(rows)} documents are not in {self.path.name}; the "
                "cache was built from different text. Re-run get_features for this task.")
        return self._x[np.asarray(rows, dtype=int)]


class FrozenPca:
    """Frozen-encoder features then PCA, fit on train only: :class:`TfidfPca`'s twin.

    Attributes:
        n_dims: Number of PCA components / qubits / angles.
        frozen: The :class:`FrozenFeatures` layer the features come from.
    """

    def __init__(self, frozen: FrozenFeatures, *, n_dims: int) -> None:
        """Store the feature source; call :meth:`fit` before transforming.

        Args:
            frozen: The cache layer to read.
            n_dims: PCA components to keep (= qubit count ``n``).

        Raises:
            ValueError: If ``n_dims`` exceeds the feature length.
        """
        if n_dims > frozen.dim:
            raise ValueError(f"n_dims={n_dims} exceeds the feature length {frozen.dim}")
        self.frozen = frozen
        self.n_dims = n_dims

    def fit(self, train_docs: NDArray) -> "FrozenPca":
        """Fit PCA on the training documents' frozen features.

        Args:
            train_docs: Training document array.

        Returns:
            This instance.
        """
        x_tr = self.frozen.lookup(train_docs)
        self._mu, self._comps, self._zmean, self._zstd = _fit_pca(x_tr, self.n_dims)
        return self

    def _pca(self, docs: NDArray) -> NDArray:
        return (self.frozen.lookup(docs) - self._mu) @ self._comps.T

    def angles(self, docs: NDArray) -> NDArray:
        """Return standardised, clipped PCA features mapped to ``[0, pi]``."""
        return _to_angles(self._pca(docs), self._zmean, self._zstd)

    def pca_std(self, docs: NDArray) -> NDArray:
        """Return the standardised PCA scores, unclipped (amplitude-encoding input)."""
        return _standardise(self._pca(docs), self._zmean, self._zstd)

    def pca_raw(self, docs: NDArray) -> NDArray:
        """Return raw PCA features (for the like-for-like classical control)."""
        return self._pca(docs)

    def raw(self, docs: NDArray) -> NDArray:
        """Return the frozen feature vectors (classical reference input)."""
        return self.frozen.lookup(docs)

    @property
    def raw_dim(self) -> int:
        """Length of :meth:`raw` rows: the encoder's hidden size."""
        return self.frozen.dim


class CompVocab:
    """Content-word vocabulary and one-hot encoder for the compositional route.

    Each sentence maps to ``n`` word ids (the first ``n`` in-vocabulary content
    words; padded with a reserved zero id), one per qubit.

    Attributes:
        n_words: Words per sentence (= qubit count ``n``).
        vocab_size: Vocabulary size including the pad id at index 0.
    """

    def __init__(self, *, n_words: int, max_vocab: int = 200, min_df: int = 2) -> None:
        """Store hyperparameters; call :meth:`fit` before encoding.

        Args:
            n_words: Words kept per sentence (= qubit count ``n``).
            max_vocab: Cap on distinct content words (excluding the pad id).
            min_df: Minimum document frequency to keep a word.
        """
        self.n_words = n_words
        self.max_vocab = max_vocab
        self.min_df = min_df

    def fit(self, train_sents: NDArray) -> "CompVocab":
        """Build the vocabulary from the training sentences.

        Args:
            train_sents: Training sentence array.

        Returns:
            This instance.
        """
        df: dict[str, int] = {}
        for s in train_sents:
            for t in set(tokenize(s, min_len=2)):
                df[t] = df.get(t, 0) + 1
        kept = sorted(((t, c) for t, c in df.items() if c >= self.min_df),
                      key=lambda kv: (-kv[1], kv[0]))[: self.max_vocab]
        # id 0 is the pad / out-of-vocab token
        self._vocab = {t: i + 1 for i, (t, _) in enumerate(kept)}
        self.vocab_size = len(self._vocab) + 1
        return self

    def ids(self, sents: NDArray) -> NDArray:
        """Return the ``(n_sents, n_words)`` word-id matrix (0 = pad)."""
        out = np.zeros((len(sents), self.n_words), dtype=int)
        for r, s in enumerate(sents):
            picks = [self._vocab[t] for t in tokenize(s, min_len=2) if t in self._vocab]
            for c in range(min(self.n_words, len(picks))):
                out[r, c] = picks[c]
        return out

    def onehot(self, sents: NDArray) -> NDArray:
        """Return the ``(n_sents, n_words, vocab_size)`` one-hot tensor."""
        return np.eye(self.vocab_size)[self.ids(sents)]
