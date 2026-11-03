'''
Flatten results/pure/summary.json into flat CSV and JSON tables.

The report (tasks/pure_report.py) writes a nested summary. This module
reads that file only and writes one row per cell, so the numbers in the
tables, the deck and the GitHub release cannot drift from the report.

Run from the folder that contains vqc_scorecard/:

    python -m vqc_scorecard.figures.export_pure_tables
'''
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "pure"

CELL_FIELDS = (
    "task", "model", "n", "d", "mcc_mean", "mcc_sd", "mcc_ci95", "seeds",
    "ref_logreg_same_features", "failed_seeds", "noise_drop_nisq",
    "gradvar_local", "category",
)
REF_FIELDS = ("task", "n", "reference", "mcc_mean", "mcc_sd", "mcc_ci95", "seeds")


def flatten_cells(*, summary: dict) -> list[dict]:
    '''
    Build one row per (task, model, n, d) cell.

    Args:
        summary:  parsed content of results/pure/summary.json

    Returns:
        list of flat dictionaries with the keys in CELL_FIELDS

    Raises:
        KeyError if the summary does not have the pure-v1 layout
    '''
    rows: list[dict] = []
    for task, block in summary["tasks"].items():
        for key, cell in block["categories"].items():
            model, n, d = key.split("|")
            mcc = cell["mcc"]
            rows.append({
                "task": task, "model": model, "n": int(n), "d": int(d),
                "mcc_mean": round(mcc["mean"], 4), "mcc_sd": round(mcc["sd"], 4),
                "mcc_ci95": round(mcc["ci95"], 4), "seeds": mcc["n"],
                "ref_logreg_same_features": _r(cell.get("ref")),
                "failed_seeds": cell.get("failed"),
                "noise_drop_nisq": _r(cell.get("noise_drop")),
                "gradvar_local": cell.get("gradvar_local"),
                "category": cell["category"],
            })
    return rows


def flatten_references(*, summary: dict) -> list[dict]:
    '''
    Build one row per classical reference model.

    Args:
        summary:  parsed content of results/pure/summary.json

    Returns:
        list of flat dictionaries with the keys in REF_FIELDS

    Raises:
        KeyError if the summary does not have the pure-v1 layout
    '''
    rows: list[dict] = []
    for task, block in summary["tasks"].items():
        for key, ref in block["references"].items():
            n, name = key.split("|")
            rows.append({
                "task": task, "n": int(n), "reference": name,
                "mcc_mean": round(ref["mean"], 4), "mcc_sd": round(ref["sd"], 4),
                "mcc_ci95": round(ref["ci95"], 4), "seeds": ref["n"],
            })
    return rows


def hypothesis_verdicts(*, summary: dict) -> dict[str, str]:
    '''
    Collect the overall verdict of every pre-registered hypothesis.

    Args:
        summary:  parsed content of results/pure/summary.json

    Returns:
        mapping from hypothesis id (H1 to H8) to its verdict string

    Raises:
        KeyError if the summary has no hypotheses block
    '''
    out: dict[str, str] = {}
    for hid, block in summary["hypotheses"].items():
        out[hid] = block.get("verdict", "see sub-verdicts")
    return out


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), 4)


def _write_csv(*, rows: list[dict], fields: tuple[str, ...], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    '''
    Read summary.json and write cells.csv, references.csv and tables.json.

    Args:
        argv:  command-line arguments; None reads sys.argv

    Returns:
        None

    Raises:
        FileNotFoundError if summary.json does not exist yet
    '''
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    summary = json.loads((args.root / "summary.json").read_text(encoding="utf-8"))
    cells = flatten_cells(summary=summary)
    refs = flatten_references(summary=summary)
    _write_csv(rows=cells, fields=CELL_FIELDS, path=args.root / "cells.csv")
    _write_csv(rows=refs, fields=REF_FIELDS, path=args.root / "references.csv")
    tables = {
        "protocol": summary["protocol"], "digest": summary["digest"],
        "missing": {t: b["missing"] for t, b in summary["tasks"].items()},
        "hypotheses": hypothesis_verdicts(summary=summary),
        "cells": cells, "references": refs,
    }
    (args.root / "tables.json").write_text(json.dumps(tables, indent=1), encoding="utf-8")
    print(f"wrote {len(cells)} cells and {len(refs)} references to {args.root}")


if __name__ == "__main__":
    main()
