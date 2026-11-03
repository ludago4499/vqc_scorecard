# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-09-22  Created. Runs every check suite and reports one total.
#   2026-09-23  Added the fair-protocol and noise suite (test_fair).
#   2026-10-01  pure-v1. test_a5 and test_fair are retired with their code; their
#               protocol-independent checks live in test_tasks. Added test_pure.
#   2026-10-08  Added test_conventions (paper-to-code conventions, A1-A7).

"""Run every check suite: ``python -m vqc_scorecard.tests``.

The descriptor checks (:mod:`test_metrics`), the shared task machinery
(:mod:`test_tasks`), the pure-v1 families, model, runner and decision rules
(:mod:`test_pure`) and the paper-to-code conventions (:mod:`test_conventions`)
run in order, and the process exits non-zero if any single
check fails.
"""

from __future__ import annotations

import sys

from . import test_conventions, test_metrics, test_pure, test_tasks


def main() -> int:
    """Run every suite and print the combined count."""
    codes = [test_metrics.main(), test_tasks.main(), test_pure.main(),
             test_conventions.main()]
    outcomes = (test_metrics.OUTCOMES + test_tasks.OUTCOMES + test_pure.OUTCOMES
                + test_conventions.OUTCOMES)
    failed = sum(1 for passed, _ in outcomes if not passed)
    print(f"\nall suites: {len(outcomes) - failed}/{len(outcomes)} passed")
    return 1 if any(codes) else 0


if __name__ == "__main__":
    sys.exit(main())
