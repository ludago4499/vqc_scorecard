# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-09-01  Created. Circuit-diagram renderer for the four ansatz families.
#   2026-09-22  A5 added to the renderer.
#   2026-10-01  pure-v1: A5-A7 are the circuit-centric, IQP and Farhi-Neven
#               families; the Mari A5 diagram is retired.

"""Circuit diagrams for the ansatz families (A1-A7).

The figures are drawn from the *same* ``apply`` functions the scorecard runs, so
a diagram can never drift from the circuit that produced a number. See
``make_diagrams.py``.
"""
