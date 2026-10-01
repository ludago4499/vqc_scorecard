# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-11  Created with P1 and P2. P3 to P6 and G1 to G6 are not yet ported.
#   2026-08-18  Added P3 (effective dimension, QFIM rank) and P4 (dynamical Lie
#               algebra dimension). P5, P6 and G1 to G6 are still not ported.
#   2026-08-18  Added P5 (parameter count) and P6 (two-qubit gate count), the two
#               cost controls. P3 now also reports the redundancy R = 1 - D_C/M.
#               G1 to G6 are still not ported.
#   2026-08-18  Added G1, the gradient-variance grid. G1 (barren-plateau onset)
#               and G2 (depth retention) are read from it. UNVERIFIED. G3 to G6
#               are still not ported.
#   2026-09-01  Ported the rest of the G-series. G3 (shot-floor margin) and G4
#               (cost-locality gap) are READINGS of the G1 grid - pure analysis
#               functions in g3_shot_floor.py and g4_locality.py, deliberately
#               not registered because they add no sweep of their own, exactly
#               like the G1/G2 readings inside g1_gradient.py. G5 (noise-induced
#               BP, per-gate depolarizing on a density matrix) and G6 (init-scale
#               sweep per Haug 2021, with the old design fault fixed) are new
#               measured metrics and are registered.

"""Registry of every metric.

Only the metrics listed here can be run. A metric that is written but not
registered is invisible to the command line, which prevents a half-finished
metric from reaching a result table by accident.
"""

from .base import Metric
from .g1_gradient import G1
from .g5_noise import G5
from .g6_init import G6
from .p1_expressibility import P1
from .p2_entangling import P2
from .p3_effective_dim import P3
from .p4_dla import P4
from .p5_params import P5
from .p6_two_qubit import P6

REGISTRY: dict[str, Metric] = {m.tag: m for m in (P1, P2, P3, P4, P5, P6, G1, G5, G6)}

__all__ = ["Metric", "REGISTRY", "P1", "P2", "P3", "P4", "P5", "P6", "G1", "G5", "G6"]
