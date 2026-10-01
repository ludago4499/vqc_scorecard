# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-09-01  Created. G4, the cost-locality gap, as a reading of the G1
#               grid. Replaces the old G4 of metrics_part2.py, which compared
#               global and local variance at depth d = n - far outside the
#               O(log n) regime where Cerezo et al. make their claim, as its
#               own audit noted. The G1 grid holds both costs at shallow and
#               deep depth, so the claimed regime and the deep control are
#               both readable without a new sweep.

"""G4 - cost locality: how much of the plateau is the observable's fault.

The claim being tested
----------------------
Cerezo, Sone, Volkoff, Cincio & Coles, Nat. Commun. 12, 1791 (2021): with a
global observable (``Z...Z`` here), the gradient variance decays exponentially
in n **even for shallow circuits**; with a local observable (``Z_0``), the
decay is at worst polynomial **as long as the depth is O(log n)**. Locality is
therefore a free trainability lever: it changes the gradient without touching
the ansatz. The claim is regime-bound - at large depth both costs plateau and
the separation closes, which is exactly what the old d = n implementation
measured, and why it found a factor ~2 instead of an exponential gap.

Scope caveat, on the record
---------------------------
Larocca et al. 2025, Box 10: global-bad / local-good is not a law of nature.
For matchgate circuits the ordering reverses; what decides trainability is the
module the observable overlaps with, not its geometric locality. For the four
circuit families of this package (single-qubit-complete rotations plus a
connected entangler, universal DLA - the P4 finding), the Cerezo ordering is
the relevant one, and the measured gap should be read as evidence about this
package's families only.

The reading
-----------
From the finished G1 records, at a fixed shallow depth, fit the width-decay
rate of each cost with the same estimator G1 uses:

    Var_global ~ exp(-alpha_global n),   Var_local ~ exp(-alpha_local n)

and report ``locality_gap = alpha_global - alpha_local``. A gap near ln 2
means the observable alone accounts for a full 2-design worth of decay; a gap
near 0 (as at large depth) means both costs are already deep in the plateau
and locality buys nothing. Read it at d = 1 (shallowest arm; note A3's d = 1
column is a product-state control, see the family notes) and at d = 3, which
is O(log n) across the swept widths; d >= 5 is the deep control.
"""

from __future__ import annotations

from collections.abc import Sequence

from .g1_gradient import bp_onset_slope


def locality_gap(
    widths: Sequence[int],
    variances_global: Sequence[float],
    variances_local: Sequence[float],
) -> dict[str, float]:
    """Return the width-decay rates of both costs and their gap.

    Args:
        widths: Qubit counts, in any order.
        variances_global: Global-cost gradient variances at those widths.
        variances_local: Local-cost gradient variances at those widths.

    Returns:
        A dict with ``alpha_global``, ``alpha_local`` (each the fitted decay
        rate of ``Var ~ exp(-alpha n)``) and ``gap`` = their difference.
        Positive gap: the global cost decays faster, the Cerezo ordering.
    """
    alpha_g = bp_onset_slope(widths, variances_global)
    alpha_l = bp_onset_slope(widths, variances_local)
    return {
        "alpha_global": float(alpha_g),
        "alpha_local": float(alpha_l),
        "gap": float(alpha_g - alpha_l),
    }
