# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-11  Created. Metric record and registry.

"""Common interface for every metric, and the registry that holds them.

A metric answers one question about one grid point. It declares what it needs,
it returns raw named values, and it states how it differs from its source paper.
It does **not** convert its value to a score. Scores are a separate concern and
are deliberately absent from this package for now: the study predicts task
performance from raw descriptors, and a monotone rescaling adds nothing to a
regression while destroying information at the clipped ends.

Add a metric in three steps:

1. Write pN_name.py or gN_name.py next to this file.
2. Build a :class:`Metric` record in it.
3. Import the record in metrics/__init__.py and add it to REGISTRY.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Metric:
    """One measurable property of an ansatz at one grid point.

    Attributes:
        tag: Short code used in output files, for example "P2".
        name: Human name, for example "Entangling capability".
        unit: Text description of the value range and its meaning.
        source: Citation of the paper that defines the quantity.
        measure: Function (ansatz, n, d, samples, rng, encoder_input) -> dict.
            It returns one measurement from one seed. Keys are value names.
        reference: Function (n, samples) -> dict giving the reference values
            of the metric at that grid point, or None when the metric has no
            reference. Reference values are reported, never used to rescale.
        reference_kind: One of "analytic", "sampled" or "none". This
            is recorded because it decides how much a reference can be trusted.
        deviations: Every difference from the source paper, in plain text.
    """

    tag: str
    name: str
    unit: str
    source: str
    measure: Callable[..., Mapping[str, float]]
    reference: Callable[[int, int], Mapping[str, float]] | None = None
    reference_kind: str = "none"
    deviations: tuple[str, ...] = field(default=())

    def provenance(self) -> dict[str, object]:
        """Return the record that must accompany any published number."""
        return {
            "name": self.name,
            "unit": self.unit,
            "source": self.source,
            "reference_kind": self.reference_kind,
            "deviations": list(self.deviations),
        }
