# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-09-09  Created. Markdown/CSV report over a task checkpoint, aggregating
#               across seeds. Companion to run.py; every number is read back
#               from the JSONL, never typed.
#   2026-09-22  A5. Records that carry a representation other than TF-IDF, or
#               an A5 pre-layer, switch the table to an extended layout: the
#               groups also split by representation (encoder and layer) and
#               pre-layer, and the classical-parameter count gets a column.
#               Every baseline in the record is printed. A checkpoint with
#               neither (every A1-A4 file so far) prints exactly as before.

"""Aggregate a task checkpoint into a Markdown or CSV table.

Run from the folder that contains ``vqc_scorecard/``::

    python -m vqc_scorecard.tasks.present_task_results --task imdb1k
    python -m vqc_scorecard.tasks.present_task_results --task cola --format csv --out RESULTS_cola.csv

For each (family, route, depth) it reports the mean +- std across seeds of the
final test F1, accuracy and MCC, the finite-shot F1, and the resource columns
(parameters, two-qubit gates, wall time). The classical baseline is printed once.

When a checkpoint holds frozen-encoder or A5 dressed runs, the groups also split
by representation (``tfidf`` or ``encoder:L<layer>``) and A5 pre-layer, and a
``p_cls`` column reports the classical parameters (the A5 pre/post layers, or
the 2-parameter head of A1-A4), which ``params`` (circuit angles, P5) excludes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no checkpoint: {path}")
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _ms(values: list[float]) -> str:
    a = np.asarray(values, float)
    return f"{a.mean():.3f}+-{a.std():.3f}" if len(a) > 1 else f"{a.mean():.3f}"


BASELINE_LABELS: dict[str, str] = {
    "pca": "PCA", "tfidf": "TFIDF", "frozen_pca": "FROZEN-PCA", "frozen": "FROZEN",
}


def _representation(r: dict) -> str:
    """Return ``tfidf``, ``word-ids`` or ``encoder:L<layer>`` for one record."""
    if r.get("route") == "compositional":
        return "word-ids"
    if r.get("features", "tfidf") == "tfidf":
        return "tfidf"
    return f"{r.get('encoder', 'frozen')}:L{r.get('layer')}"


def _extended(records: list[dict]) -> bool:
    """True when any record carries a non-TF-IDF representation or an A5 pre-layer."""
    return any(_representation(r) not in ("tfidf", "word-ids") or r.get("prelayer")
               for r in records)


def build(records: list[dict], fmt: str) -> str:
    extended = _extended(records)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if extended:
            groups[(r["family"], r["route"], _representation(r),
                    r.get("prelayer") or "-", r["d"])].append(r)
        else:
            groups[(r["family"], r["route"], r["d"])].append(r)
    header = ["family", "route", "d", "F1", "acc", "MCC", "shot F1",
              "params", "2q", "s/run", "seeds"]
    if extended:
        header = ["family", "route", "repr", "prelayer", "d", "F1", "acc", "MCC",
                  "shot F1", "params", "p_cls", "2q", "s/run", "seeds"]
    rows = []
    for key in sorted(groups):
        g = groups[key]
        stats = [
            _ms([r["final"]["f1"] for r in g]),
            _ms([r["final"]["accuracy"] for r in g]),
            _ms([r["final"]["mcc"] for r in g]),
            _ms([r.get("shot_eval", {}).get("f1", float("nan")) for r in g]),
        ]
        cost = [str(g[0]["p_params"])]
        if extended:
            cost.append(str(g[0].get("p_classical", 2)))
        cost.append(str(g[0]["two_qubit_gates"]))
        tail = [f"{np.mean([r['wall_time_s'] for r in g]):.1f}", str(len(g))]
        rows.append([*(str(k) for k in key), *stats, *cost, *tail])
    base = records[0].get("classical", {})
    lines = []
    task = records[0]["task"]
    n = records[0]["n"]
    legacy_base = set(base) <= {"pca", "tfidf"}
    if fmt == "csv":
        lines.append(",".join(header))
        lines += [",".join(r) for r in rows]
        for name, value in base.items():
            label = BASELINE_LABELS.get(name, name.upper())
            if extended:
                lines.append(f"baseline-{label},logreg,-,-,-,{value['f1']:.3f},"
                             f"{value['accuracy']:.3f},{value['mcc']:.3f},-,-,-,-,-,-")
            else:
                lines.append(f"baseline-{label},logreg,-,{value['f1']:.3f},"
                             f"{value['accuracy']:.3f},{value['mcc']:.3f},-,-,-,-,-")
        return "\n".join(lines) + "\n"
    lines.append(f"# Task results - {task} (n = {n})")
    lines.append("")
    lines.append("Mean +- std across seeds. MCC is the headline metric for CoLA.")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    if base and legacy_base:
        lines.append("")
        lines.append(f"Classical baseline (logreg): PCA({n}) F1 {base['pca']['f1']:.3f} / "
                     f"MCC {base['pca']['mcc']:.3f}; TF-IDF F1 {base['tfidf']['f1']:.3f} / "
                     f"MCC {base['tfidf']['mcc']:.3f}.")
    elif base:
        names = {"pca": f"PCA({n}) of TF-IDF", "tfidf": "full TF-IDF",
                 "frozen_pca": f"PCA({n}) of the frozen encoder",
                 "frozen": "full frozen encoder (standardised)"}
        parts = [f"{names.get(k, k)} F1 {v['f1']:.3f} / MCC {v['mcc']:.3f}"
                 for k, v in base.items()]
        lines.append("")
        lines.append("Classical baselines (logreg, first record's split): "
                     + "; ".join(parts) + ".")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Report a task checkpoint.")
    ap.add_argument("--task", choices=("imdb1k", "cola"), required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--format", choices=("md", "csv"), default="md")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path = Path(args.checkpoint) if args.checkpoint else (
        PKG / "results" / f"task_{args.task}.jsonl")
    report = build(load(path), args.format)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
