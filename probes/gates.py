# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below.
#
# History:
#   2026-10-01  Created. Decomposition of multi-wire operations into gates of
#               at most two wires, shared by P6 and the gate-noise transform,
#               so an amplitude encoding (A5) is counted and charged as the
#               CNOTs it really needs, not as one free StatePrep.

"""Elementary gates: decompose every operation down to one or two wires.

A circuit can hold operations that act on many wires at once, such as the
``StatePrep`` behind an amplitude encoding. On hardware such an operation is a
sequence of one- and two-qubit gates (for an arbitrary real state, the Mottonen
et al. 2004 construction with ``O(2**n)`` CNOTs). Counting it as one operation
would make the encoding look free. :func:`elementary` replaces it by its
decomposition, recursively, until every operation acts on at most two wires.

Operations that already act on one or two wires are returned unchanged, so a
circuit made of such gates (A1 to A4, A6, A7) is left exactly as it was.
"""

from __future__ import annotations

from collections.abc import Iterable

MAX_WIRES: int = 2
"""Largest arity kept as one gate."""


def elementary(ops: Iterable, max_wires: int = MAX_WIRES) -> list:
    """Return ``ops`` with every operation on more than ``max_wires`` wires decomposed.

    Args:
        ops: Operations, in time order.
        max_wires: Largest arity to keep.

    Returns:
        A flat list of operations, in time order, each on at most ``max_wires``
        wires.

    Raises:
        ValueError: If an operation on more than ``max_wires`` wires has no
            decomposition.
    """
    out: list = []
    for op in ops:
        if len(op.wires) <= max_wires:
            out.append(op)
            continue
        if not op.has_decomposition:
            raise ValueError(f"{op.name} acts on {len(op.wires)} wires and has no decomposition")
        out.extend(elementary(op.decomposition(), max_wires))
    return out
