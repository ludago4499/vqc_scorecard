# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-09-23  Created. Package-wide gate-noise models (depolarizing, bit
#               flip, phase flip) with two-rate levels, for the task-noise
#               section and any later consumer (G-series, hardware studies).

"""Gate-noise models shared across the package.

One place defines what "noise" means, so the task axis, a future G-series
extension and any hardware comparison all apply the same channels at the same
rates. See :mod:`.models`.
"""

from .models import CHANNELS, LEVELS, REPORTED_LEVELS, NoiseModel, add_noise, preset

__all__ = ["CHANNELS", "LEVELS", "REPORTED_LEVELS", "NoiseModel", "add_noise", "preset"]
