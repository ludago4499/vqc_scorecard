# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-11  Created. JSON and CSV writers, with environment metadata.

"""Output files, and the environment record that must go with them.

Two files are written. The JSON file holds the records, the environment and the
declared deviations of every family and metric used. The CSV file holds the same
records in a flat form, for a spreadsheet or for a correlation matrix.

The environment block exists because a number without a library version cannot
be reproduced. The old scorecard folder recorded no versions at all.
"""

from __future__ import annotations

import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml


def environment() -> dict[str, str]:
    """Return the versions that determine the numbers of this run.

    Returns:
        A dictionary with the run time in UTC and the version of every library
        that can change a result.
    """
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pennylane": qml.version(),
        "platform": platform.platform(),
    }


def write_json(path: Path, records: list[dict], provenance: dict[str, Any]) -> None:
    """Write the full result file.

    Args:
        path: Destination path. The suffix is forced to .json.
        records: Finished measurement records, in grid order.
        provenance: Declared sources and deviations of the families and metrics
            used in this run.
    """
    payload = {
        "environment": environment(),
        "provenance": provenance,
        "records": records,
    }
    path.with_suffix(".json").write_text(json.dumps(payload, indent=1), encoding="utf-8")


def write_csv(path: Path, records: list[dict]) -> None:
    """Write the flat table.

    Columns are the union of the keys of every record, with the identifying
    columns first. Missing values are written as empty cells, because different
    metrics return different value names.

    Args:
        path: Destination path. The suffix is forced to .csv.
        records: Finished measurement records, in grid order.
    """
    lead = ["metric", "family", "n", "d", "samples", "repeats", "off_protocol"]
    rest = sorted({k for r in records for k in r} - set(lead))
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=lead + rest, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
