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

"""Machinery that is shared by every metric.

* :mod:`grid` - which points are measured.
* :mod:`seeding` - how a generator is derived, so that order does not matter.
* :mod:`checkpoint` - how a sweep stops and continues.
* :mod:`io` - how results and the environment are written.

Nothing in this sub-package knows about quantum circuits.
"""
