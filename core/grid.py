# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-11  Created. Grid point record and grid expansion.

"""Grid points of the homogeneous sweep.

A grid point is one combination of family, qubit count and depth. Every metric
in this package is measured at grid points, and the same grid is used for every
metric, so that the resulting table can be correlated column by column.

The declared grid of the experiment is n in {2, 4, 6, 8} crossed with
d in {1, 3, 5}. Families that cannot take a value of the grid are still
measured there, and the record is marked off_protocol. Those rows are
controls. They must not be mixed with the declared sweep in a regression.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

DEFAULT_WIDTHS: tuple[int, ...] = (2, 4, 6, 8)
DEFAULT_DEPTHS: tuple[int, ...] = (1, 3, 5)


@dataclass(frozen=True, order=True)
class GridPoint:
    """One combination of family, qubit count and depth.

    Attributes:
        family: Ansatz tag, for example "A1".
        n: Number of qubits.
        d: Depth, in the unit that the family defines. For A1 and A2 this is the
            number of entangling layers. For A3 it is the number of re-uploads.
            For A4 it is the number of repeats of each word box.
    """

    family: str
    n: int
    d: int

    def label(self) -> str:
        """Return a short text form, for example "A1/n2/d1"."""
        return f"{self.family}/n{self.n}/d{self.d}"


def expand(
    families: Sequence[str],
    widths: Sequence[int] = DEFAULT_WIDTHS,
    depths: Sequence[int] = DEFAULT_DEPTHS,
) -> Iterator[GridPoint]:
    """Yield every grid point of the cross product, in a stable order.

    The order is family, then width, then depth. It never changes between runs,
    so a checkpoint written by one run can be continued by the next.

    Args:
        families: Ansatz tags to sweep.
        widths: Qubit counts to sweep.
        depths: Depths to sweep.

    Yields:
        One :class:`GridPoint` per combination.

    Examples:
        >>> [p.label() for p in expand(["A1"], (2,), (1, 3))]
        ['A1/n2/d1', 'A1/n2/d3']
    """
    for family in families:
        for n in widths:
            for d in depths:
                yield GridPoint(family=family, n=n, d=d)
