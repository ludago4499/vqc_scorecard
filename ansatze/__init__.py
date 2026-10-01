# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-11  Moved from the p1_expressibility package, unchanged.
#   2026-09-22  Registered A5, the bare circuit of the dressed VQC of Mari et
#               al. 2020 (hybrid transfer learning, Pool 3).
#   2026-10-01  pure-v1. The Mari A5 is retired (its trainable classical layers
#               break the rule that the circuit is the only trainable part; the
#               module is archived in old/fair_v1_archive). New A5 circuit-centric
#               classifier (Schuld 2020), A6 IQP feature map + VQC (Havlicek
#               2019), A7 Farhi-Neven QNN (2018).

"""Registry of every ansatz family.

Add a family in three steps:

1. Write aN_yourfamily.py next to this file. Copy the shape of a1_hea.py.
2. Import its :class:`~.base.Ansatz` record below.
3. Add the record to REGISTRY.

Nothing else in the package needs a change.
"""

from .base import Ansatz
from .a1_hea import A1
from .a2_sel import A2
from .a3_reupload import A3
from .a4_discocat import A4
from .a5_circuit_centric import A5
from .a6_iqp import A6
from .a7_farhi_neven import A7

REGISTRY: dict[str, Ansatz] = {a.tag: a for a in (A1, A2, A3, A4, A5, A6, A7)}

__all__ = ["Ansatz", "REGISTRY", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
