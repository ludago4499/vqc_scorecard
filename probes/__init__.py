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
#   2026-08-18  Added qgt (state Jacobian and QFIM, for P3) and lie (Lie-closure
#               dimension, for P4).

"""Expensive primitives, shared by the metrics.

* :mod:`statevector` - batches of states. Every static metric starts here.
* :mod:`fidelity` - pairwise overlaps. Used by P1.
* :mod:`purity` - single-qubit reduced purities. Used by P2.
* :mod:`haar` - the Haar reference, as a distribution and as an expectation.
* :mod:`qgt` - state Jacobian and quantum Fisher information matrix. Used by P3.
* :mod:`lie` - dimension of a Lie closure of Pauli strings. Used by P4.
* :mod:`gates` - decomposition into one- and two-wire gates. Used by P6 and noise.

A probe computes a physical quantity. It never decides what that quantity means.
"""
