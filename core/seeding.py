# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-11  Created. Replaces the single module-level generator of the old
#               metrics_part1.py, whose results depended on call order.

"""Deterministic seeds, derived rather than stored.

The old scorecard seeded one generator when its module was imported. Every
metric then drew from that one stream, so the value returned by a function
depended on which functions had run before it. The numbers were reproducible
only if the whole script ran in one exact order.

This module removes that coupling. Every generator is derived from a fixed base
and the identity of the work it is doing. Two calls with the same arguments
return the same stream, whatever else has run.

Do not create a generator any other way inside this package.
"""

from __future__ import annotations

import hashlib

import numpy as np

BASE_SEED: int = 20260811
"""Base seed of the whole package. Change it only to re-run a study from scratch."""


def derive(*parts: object) -> int:
    """Return a stable 32-bit seed from any set of identifying parts.

    The parts are converted to text, joined, and hashed. The hash is stable
    across processes and across Python versions, which ``hash()`` is not.

    Args:
        *parts: Anything with a text form. Typical parts are the metric tag, the
            family tag, the qubit count, the depth and the repeat index.

    Returns:
        An integer in the range [0, 2**32).

    Examples:
        >>> derive("P1", "A1", 2, 1, 0) == derive("P1", "A1", 2, 1, 0)
        True
        >>> derive("P1", "A1", 2, 1, 0) == derive("P2", "A1", 2, 1, 0)
        False
    """
    text = "|".join(str(p) for p in (BASE_SEED, *parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def generator(*parts: object) -> np.random.Generator:
    """Return a fresh generator seeded by :func:`derive`.

    Args:
        *parts: The same identifying parts accepted by :func:`derive`.

    Returns:
        A NumPy generator. It is independent of every other generator in the run.
    """
    return np.random.default_rng(derive(*parts))
