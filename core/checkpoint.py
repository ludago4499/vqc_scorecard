# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below. A metric whose definition changes without
# a history entry cannot be traced back to the number it produced.
#
# History:
#   2026-08-11  Created. Line-per-record checkpoint with resume.

"""Checkpoint file, so that a long sweep can stop and continue.

One finished measurement is one line of JSON. The file is opened in append mode
and closed again after every line, so a stopped process loses at most the
measurement it was running.

A record is identified by metric, family, qubit count, depth and sample count.
Two records with the same identity are the same work, and the second one is
skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

RecordKey = tuple[str, str, int, int, int]


def key_of(record: dict) -> RecordKey:
    """Return the identity of a record.

    Args:
        record: A finished measurement record.

    Returns:
        The tuple (metric, family, n, d, samples).
    """
    return (
        record["metric"],
        record["family"],
        int(record["n"]),
        int(record["d"]),
        int(record["samples"]),
    )


def load(path: Path) -> dict[RecordKey, dict]:
    """Read every finished record from the checkpoint file.

    A missing file is not an error. It means no work has been done yet.

    Args:
        path: Path of the .jsonl checkpoint file.

    Returns:
        A dictionary from record identity to record. Later lines win, so a
        repeated measurement replaces the earlier one.
    """
    if not path.exists():
        return {}
    done: dict[RecordKey, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            done[key_of(record)] = record
    return done


def append(path: Path, record: dict) -> None:
    """Add one finished record to the checkpoint file.

    Args:
        path: Path of the .jsonl checkpoint file. It is created if absent.
        record: The record to write. It must be JSON serialisable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
