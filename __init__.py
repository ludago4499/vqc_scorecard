# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-11  Created.
#   2026-09-23  Listed the tasks and noise packages.
#   2026-10-01  3.0.0, pure-v1: families A1-A7 (new A5 circuit-centric, A6 IQP,
#               A7 Farhi-Neven), circuit-only task protocol; fair-v1 and the
#               Mari A5 archived in old/fair_v1_archive.

"""VQC/VQN architecture scorecard.

Layout, in the order the data flows:

* :mod:`ansatze` - one file per family. Circuits only.
* :mod:`probes` - the expensive primitives. Physics only.
* :mod:`metrics` - one file per metric. Definitions and deviations.
* :mod:`core` - grid, seeding, checkpoint and output. No physics.
* :mod:`noise` - gate-noise models shared by every consumer.
* :mod:`tasks` - the task axis (NLP training, the pure-v1 protocol, its report).
* :mod:`cli` - the single entry point.

Every file has one reason to change. That rule is the whole point of the layout.
"""

__version__ = "3.0.0"
