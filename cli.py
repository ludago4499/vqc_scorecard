# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below.
#
# History:
#   2026-08-11  Created. Single entry point for every metric and family.
#   2026-08-17  The flat .json and .csv now mirror the whole checkpoint, not
#               only the metrics of the current run, so a single-metric run no
#               longer drops other metrics from the tables.

"""Command line for the sweep.

Examples:
    Run P2 for all four families over the declared grid::

        python -m vqc_scorecard.cli --metrics P2

    Run both metrics for one family, with a time budget::

        python -m vqc_scorecard.cli --metrics P1 P2 --families A1 --budget 140

    A stopped run continues where it left off. Finished measurements are stored
    in the ``.jsonl`` checkpoint and are skipped on the next call.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .ansatze import REGISTRY as ANSATZE
from .core import checkpoint, io
from .core.grid import DEFAULT_DEPTHS, DEFAULT_WIDTHS, GridPoint, expand
from .core.seeding import generator
from .metrics import REGISTRY as METRICS

HERE = Path(__file__).parent
DEFAULT_OUTPUT = HERE / "results" / "scorecard"


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the sweep."""
    parser = argparse.ArgumentParser(description="VQC/VQN scorecard sweep.")
    parser.add_argument("--metrics", nargs="+", default=list(METRICS),
                        help="metric tags to run, for example P1 P2")
    parser.add_argument("--families", nargs="+", default=list(ANSATZE),
                        help="ansatz tags to run, for example A1 A2")
    parser.add_argument("--widths", nargs="+", type=int, default=list(DEFAULT_WIDTHS),
                        help="qubit counts")
    parser.add_argument("--depths", nargs="+", type=int, default=list(DEFAULT_DEPTHS),
                        help="depths, in the unit of each family")
    parser.add_argument("--samples", nargs="+", type=int, default=[5000],
                        help="parameter draws per measurement")
    parser.add_argument("--repeats", type=int, default=5,
                        help="independent seeds per grid point")
    parser.add_argument("--embedding", choices=["fixed", "random"], default="fixed",
                        help="fixed holds one input constant; random draws a new "
                             "input per seed")
    parser.add_argument("--budget", type=float, default=0.0,
                        help="stop after this many seconds; 0 means no limit")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT),
                        help="output path without a suffix")
    return parser


def encoder_input(family: str, n: int, d: int, rng: np.random.Generator):
    """Return the input vector of a family, or ``None`` if it has no encoder.

    Args:
        family: Ansatz tag.
        n: Number of qubits.
        d: Depth.
        rng: Generator that decides the vector.

    Returns:
        An array of encoder angles in ``[0, pi]``, or ``None``.
    """
    size = ANSATZE[family].n_encoder(n, d)
    return None if size == 0 else rng.uniform(0.0, np.pi, size=size)


def measure_point(metric_tag: str, point: GridPoint, samples: int,
                  repeats: int, embedding: str) -> dict:
    """Measure one metric at one grid point, over several seeds.

    Each seed produces one measurement. The seeds are derived from the identity
    of the work, so the result does not depend on what ran before it.

    Args:
        metric_tag: Metric to run, for example ``"P2"``.
        point: The grid point.
        samples: Parameter draws per measurement.
        repeats: Number of independent seeds.
        embedding: ``"fixed"`` or ``"random"``. See the command line help.

    Returns:
        One record, ready for the checkpoint file. Every value name returned by
        the metric appears twice, as ``<name>_mean`` and ``<name>_std``, and once
        more as ``<name>_seeds`` with the individual values.
    """
    metric, ansatz = METRICS[metric_tag], ANSATZE[point.family]
    started = time.perf_counter()
    per_seed: dict[str, list[float]] = {}

    for repeat in range(repeats):
        rng = generator(metric_tag, point.family, point.n, point.d, samples, repeat)
        source = generator("encoder", point.family, point.n, point.d) \
            if embedding == "fixed" else rng
        values = metric.measure(ansatz, point.n, point.d, samples, rng,
                                encoder_input(point.family, point.n, point.d, source))
        for name, value in values.items():
            per_seed.setdefault(name, []).append(float(value))

    record = {
        "metric": metric_tag,
        "family": point.family,
        "n": point.n,
        "d": point.d,
        "samples": samples,
        "repeats": repeats,
        "n_params": ansatz.n_params(point.n, point.d),
        "off_protocol": not ansatz.permits(point.n, point.d),
        "seconds": round(time.perf_counter() - started, 2),
    }
    record.update(_aggregate(per_seed))
    if metric.reference is not None:
        record.update({k: float(v) for k, v in metric.reference(point.n, samples).items()})
    return record


def _aggregate(per_seed: dict[str, list[float]]) -> dict[str, object]:
    """Return the mean, the spread and the raw values of every measured name."""
    out: dict[str, object] = {}
    for name, values in per_seed.items():
        array = np.asarray(values, dtype=float)
        out[f"{name}_mean"] = round(float(array.mean()), 8)
        out[f"{name}_std"] = round(float(array.std(ddof=1)) if array.size > 1 else 0.0, 8)
        out[f"{name}_seeds"] = [round(v, 8) for v in values]
    return out


def main(argv: list[str] | None = None) -> None:
    """Run the sweep, then write the JSON and CSV result files."""
    args = build_parser().parse_args(argv)
    output = Path(args.out)
    ckpt = output.with_suffix(".jsonl")
    done = checkpoint.load(ckpt)
    deadline = time.time() + args.budget if args.budget > 0 else float("inf")

    planned = [(m, p, s)
               for m in args.metrics
               for p in expand(args.families, args.widths, args.depths)
               for s in args.samples
               if not (p.family == "A4" and p.n % 2)]
    pending = [job for job in planned
               if (job[0], job[1].family, job[1].n, job[1].d, job[2]) not in done]
    print(f"{len(planned) - len(pending)} done, {len(pending)} pending", flush=True)

    for metric_tag, point, samples in pending:
        if time.time() > deadline:
            print("time budget reached; run the command again to continue", flush=True)
            break
        record = measure_point(metric_tag, point, samples, args.repeats, args.embedding)
        checkpoint.append(ckpt, record)
        done[checkpoint.key_of(record)] = record
        _report(record)

    # Write every measurement in the checkpoint, not only this run's slice, so a
    # run of one metric does not drop another metric from the flat tables. The
    # .jsonl checkpoint is the accumulation; the .json and .csv mirror it.
    ordered = sorted(
        done.values(),
        key=lambda r: (r["metric"], r["family"], int(r["n"]), int(r["d"]),
                       int(r["samples"])),
    )
    metrics_present = sorted({r["metric"] for r in ordered})
    families_present = sorted({r["family"] for r in ordered})
    provenance = {
        "metrics": {t: METRICS[t].provenance() for t in metrics_present
                    if t in METRICS},
        "families": {t: {"name": ANSATZE[t].name, "source": ANSATZE[t].source,
                         "deviations": list(ANSATZE[t].deviations)}
                     for t in families_present if t in ANSATZE},
        "embedding_mode": args.embedding,
    }
    io.write_json(output, ordered, provenance)
    io.write_csv(output, ordered)
    print(f"\nwrote {output.with_suffix('.json')}")
    print(f"wrote {output.with_suffix('.csv')}")
    print(f"{len(ordered)} measurements in the tables "
          f"({len(planned)} planned this run)")


def _report(record: dict) -> None:
    """Print one finished measurement as a single line."""
    primary = next(k for k in record if k.endswith("_mean"))
    flag = "  off-protocol" if record["off_protocol"] else ""
    print(f"{record['metric']} {record['family']} n={record['n']} d={record['d']} "
          f"samples={record['samples']:5d}  "
          f"{primary[:-5]}={record[primary]:.5f}+-{record[primary[:-5] + '_std']:.5f}  "
          f"({record['seconds']}s){flag}", flush=True)


if __name__ == "__main__":
    main()
