# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-09-09  Created. One-shot CoLA fetcher/preparer, stdlib only. Run this
#               once on a machine with internet; the sweep runner then reads the
#               TSVs it writes.

"""Fetch and prepare the CoLA dataset (GLUE), standard library only.

Run once, from the folder that contains ``vqc_scorecard/``::

    python -m vqc_scorecard.tasks.get_cola                # -> ./data/cola/
    python -m vqc_scorecard.tasks.get_cola --out data/cola --train-size 1000

Writes into the output directory:

* ``in_domain_train.tsv`` and ``in_domain_dev.tsv`` - the official CoLA splits
  (columns: source, label, orig_label, sentence).
* ``in_domain_train_1k.tsv`` - a subsample of the train split to ``--train-size``
  rows, preserving the class ratio; this is what the runner uses by default.

CoLA: Warstadt, Singh & Bowman, "Neural Network Acceptability Judgments",
TACL 2019. Distributed at https://nyu-mll.github.io/CoLA/.
"""

from __future__ import annotations

import argparse
import io
import random
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

URL = "https://nyu-mll.github.io/CoLA/cola_public_1.1.zip"


def _read_tsv(raw: bytes) -> list[list[str]]:
    rows = []
    for line in raw.decode("utf-8", errors="ignore").splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            rows.append(parts[:4])
    return rows


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write("\t".join(r) + "\n")


def subsample(rows: list[list[str]], *, size: int, seed: int) -> list[list[str]]:
    """Return a class-ratio-preserving subsample of ``size`` rows."""
    by_label: dict[str, list[list[str]]] = defaultdict(list)
    for r in rows:
        by_label[r[1]].append(r)
    rng = random.Random(seed)
    out: list[list[str]] = []
    total = len(rows)
    for label, group in by_label.items():
        k = max(1, round(size * len(group) / total))
        rng.shuffle(group)
        out.extend(group[:k])
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch and prepare CoLA.")
    ap.add_argument("--out", default="data/cola")
    ap.add_argument("--train-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.url} ...", flush=True)
    with urllib.request.urlopen(args.url, timeout=120) as resp:
        blob = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(blob))
    names = {Path(n).name: n for n in zf.namelist()}
    for fname in ("in_domain_train.tsv", "in_domain_dev.tsv"):
        if fname not in names:
            raise SystemExit(f"{fname} not found in the archive: {list(names)[:8]}")
        rows = _read_tsv(zf.read(names[fname]))
        _write_tsv(out / fname, rows)
        pos = sum(1 for r in rows if r[1] == "1")
        print(f"  {fname}: {len(rows)} rows ({pos} acceptable / {len(rows) - pos} not)",
              flush=True)

    train_rows = _read_tsv(zf.read(names["in_domain_train.tsv"]))
    sub = subsample(train_rows, size=args.train_size, seed=args.seed)
    _write_tsv(out / "in_domain_train_1k.tsv", sub)
    pos = sum(1 for r in sub if r[1] == "1")
    print(f"  in_domain_train_1k.tsv: {len(sub)} rows ({pos} acceptable / "
          f"{len(sub) - pos} not)", flush=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
