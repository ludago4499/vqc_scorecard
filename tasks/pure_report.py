# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-10-01  Created. Report of the pure-v1 protocol: cell tables, the
#               pre-registered hypothesis tests H1-H8 with their frozen
#               decision rules, the classical references, and the result
#               category of every (model, n, d) cell.

"""Report of the pure-v1 protocol: hypotheses, references and result categories.

Nothing here is tuned after seeing data. The hypotheses, their tests and their
decision rules are the ones frozen in the pre-registration
(``claude/pure_v1_prereg`` in the project; the deck's pre-registration part).
A hypothesis is only scored when every cell it reads is complete; otherwise its
verdict is ``incomplete`` and the missing cells are listed.

Inputs
------
* the pure-v1 checkpoints of each task (``results/pure``),
* the descriptor checkpoint ``results/scorecard.jsonl`` (G1 gradient grid), for
  H1, H2 and the gradient part of H5 and of the categories,
* optionally the archived fair-v1 checkpoint, for the bridge hypothesis H8.

Result categories (one per (task, model, n, d) cell), checked in this order:

1. **Floor** - the 95% interval of the mean test MCC includes 0.
2. **Trainability-limited** - the G1 local-cost gradient variance at the cell is
   below the shot floor ``1/1024``, or at least 3 of 10 seeds failed
   (test MCC below ``fail_mcc``).
3. **Performing** - mean MCC within the SESOI of the like-for-like logistic
   regression (same PCA features) and at most 1 failed seed; becomes
   **Noise-limited** when the mean MCC drop at the ``nisq`` level exceeds the
   SESOI (cells with noise data only).
4. **Capacity-limited** - everything else: it trains, but stays below the
   reference by more than the SESOI.

Command::

    python -m vqc_scorecard.tasks.pure_report --tasks imdb1k cola
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..core import checkpoint as desc_checkpoint
from ..metrics.g1_gradient import SHOTS_FLOOR, bp_onset_slope
from . import stats as st
from .harness import PKG, TASKS, checkpoint_path, load
from .model import MODELS
from .pure import PROTOCOL, RESULTS, PureProtocol

DEFAULT_SCORECARD = PKG / "results" / "scorecard.jsonl"
CLASS_I: tuple[str, ...] = ("A1", "A2", "A4", "A7")
"""Single-angle families: the ansatz axis at a fixed encoding (Class I)."""
LN2: float = float(np.log(2.0))
H1_WIDTHS: tuple[int, ...] = (2, 4, 6, 8, 10)
H2_WIDTH: int = 6
PRIMARY_TASK: str = "imdb1k"


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

def load_task(task: str, protocol: PureProtocol = PROTOCOL, root: Path = RESULTS) -> dict:
    """Return the current-protocol records of one task, by kind."""
    digest = protocol.digest()
    out = {}
    for kind in ("clean", "noise_eval", "baselines"):
        recs = load(checkpoint_path(task, kind, root))
        out[kind] = [r for r in recs if r.get("protocol_digest") == digest]
        out[f"{kind}_foreign"] = len(recs) - len(out[kind])
    return out


def per_seed(clean: list[dict], metric: str = "mcc") -> dict:
    """Return ``{(model, n, d): {seed: selected test metric}}``."""
    out: dict = defaultdict(dict)
    for r in clean:
        out[(r["family"], int(r["n"]), int(r["d"]))][int(r["seed"])] = r["selected"]["test"][metric]
    return out


def paired(cells: dict, a: tuple, b: tuple, seeds: list[int]):
    """Return the two per-seed arrays of cells ``a`` and ``b``, or ``None`` if incomplete."""
    ca, cb = cells.get(a, {}), cells.get(b, {})
    if not all(s in ca and s in cb for s in seeds):
        return None
    return np.array([ca[s] for s in seeds]), np.array([cb[s] for s in seeds])


def load_g1(path: Path) -> dict:
    """Return ``{(family, n, d): {"global": [...], "local": [...]}}`` from the descriptor checkpoint.

    The per-seed lists are the repeats of the G1 measurement. When several
    sample counts exist for a point, the largest is used.
    """
    if not Path(path).exists():
        return {}
    best: dict = {}
    for rec in desc_checkpoint.load(Path(path)).values():
        if rec.get("metric") != "G1":
            continue
        key = (rec["family"], int(rec["n"]), int(rec["d"]))
        if key in best and best[key]["samples"] >= rec["samples"]:
            continue
        best[key] = {"samples": rec["samples"],
                     "global": list(rec.get("gradvar_global_seeds", [rec["gradvar_global_mean"]])),
                     "local": list(rec.get("gradvar_local_seeds", [rec["gradvar_local_mean"]]))}
    return best


# --------------------------------------------------------------------------- #
# Hypotheses                                                                   #
# --------------------------------------------------------------------------- #

def _alpha_with_ci(g1: dict, family: str, d: int, cost: str) -> dict | None:
    """Fit the onset slope per G1 repeat over :data:`H1_WIDTHS`; return its summary."""
    rows = [g1.get((family, n, d)) for n in H1_WIDTHS]
    if any(r is None for r in rows):
        return None
    reps = min(len(r[cost]) for r in rows)
    alphas = [bp_onset_slope(list(H1_WIDTHS), [r[cost][i] for r in rows]) for i in range(reps)]
    s = st.summary(alphas)
    return {**s, "lo": s["mean"] - s["ci95"], "hi": s["mean"] + s["ci95"], "values": alphas}


def h1(g1: dict) -> dict:
    """H1 width x observable. A1-A6: global cost decays exponentially at d = 5
    (alpha interval above 0) and the local cost at d = 1 decays at most a third
    as fast. A7 (polynomial DLA): global alpha at d = 5 below ln 2 / 2."""
    parts, rows = [], {}
    for fam in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        glob = _alpha_with_ci(g1, fam, 5, "global")
        loc = _alpha_with_ci(g1, fam, 1, "local")
        if glob is None or loc is None:
            rows[fam] = {"verdict": "incomplete"}
            parts.append("incomplete")
            continue
        if fam == "A7":
            v = ("supported" if glob["hi"] < LN2 / 2 else
                 "refuted" if glob["lo"] >= LN2 / 2 else "inconclusive")
        else:
            va = ("supported" if glob["lo"] > 0 else
                  "refuted" if glob["hi"] <= 0 else "inconclusive")
            vb = ("supported" if loc["mean"] <= glob["mean"] / 3 else
                  "refuted" if loc["lo"] > glob["mean"] / 3 else "inconclusive")
            v = st.combine([va, vb])
        rows[fam] = {"alpha_global_d5": glob, "alpha_local_d1": loc, "verdict": v}
        parts.append(v)
    return {"rows": rows, "verdict": _overall(parts)}


def h2(g1: dict, depths: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)) -> dict:
    """H2 depth x gradient at n = 6: the global variance falls with depth
    (Spearman rho < 0, Holm over families) and levels off (the last two depths
    within a factor 2)."""
    rows, pvals = {}, {}
    for fam in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        pts = [g1.get((fam, H2_WIDTH, d)) for d in depths]
        if any(p is None for p in pts):
            rows[fam] = {"verdict": "incomplete"}
            continue
        var = [float(np.mean(p["global"])) for p in pts]
        sp = st.spearman(list(depths), var)
        rows[fam] = {"variance": var, "spearman": sp,
                     "plateau_ratio": float(max(var[-1], var[-2]) / max(min(var[-1], var[-2]), 1e-18))}
        pvals[fam] = sp["p"] / 2 if sp["rho"] < 0 else 1.0
    adj = st.holm(pvals) if pvals else {}
    parts = []
    for fam, row in rows.items():
        if row.get("verdict") == "incomplete":
            parts.append("incomplete")
            continue
        falls = ("supported" if adj[fam] < st.ALPHA else
                 "refuted" if row["spearman"]["rho"] >= 0 else "inconclusive")
        levels = "supported" if row["plateau_ratio"] <= 2.0 else "refuted"
        row["p_holm"] = adj[fam]
        row["verdict"] = st.combine([falls, levels])
        parts.append(row["verdict"])
    return {"rows": rows, "verdict": _overall(parts)}


def _depth_or_width(cells: dict, p: PureProtocol, pairs: list[tuple[tuple, tuple, tuple]]) -> dict:
    """Shared engine of H3 and H4: per item, part a (greater) and part b (not greater)."""
    seeds = p.seeds()
    rows, pvals = {}, {}
    for key, (a_hi, a_lo), (b_hi, b_lo) in pairs:
        pa, pb = paired(cells, a_hi, a_lo, seeds), paired(cells, b_hi, b_lo, seeds)
        if pa is None or pb is None:
            rows[key] = {"verdict": "incomplete"}
            continue
        ca, cb = st.contrast(*pa), st.contrast(*pb)
        rows[key] = {"gain": ca, "beyond": cb}
        pvals[key] = ca["p_greater"]
    adj = st.holm(pvals) if pvals else {}
    parts = []
    for key, row in rows.items():
        if row.get("verdict") == "incomplete":
            parts.append("incomplete")
            continue
        va = st.greater(row["gain"], p.sesoi, adj[key])
        vb = st.not_greater(row["beyond"], p.sesoi)
        row.update(p_holm=adj[key], verdict_gain=va, verdict_beyond=vb, verdict=st.combine([va, vb]))
        parts.append(row["verdict"])
    return {"rows": {f"{k[0]} {k[1]}": v for k, v in rows.items()}, "verdict": _overall(parts)}


def h3(cells: dict, p: PureProtocol) -> dict:
    """H3 depth x performance, per family and width: MCC(d=3) > MCC(d=1) and
    MCC(d=5) - MCC(d=3) <= SESOI."""
    pairs = [((f, f"n={n}"), ((f, n, 3), (f, n, 1)), ((f, n, 5), (f, n, 3)))
             for f in p.families for n in p.widths]
    return _depth_or_width(cells, p, pairs)


def h4(cells: dict, p: PureProtocol) -> dict:
    """H4 width x performance, per family and depth: MCC(n=6) > MCC(n=4) and
    MCC(n=8) - MCC(n=6) <= SESOI."""
    pairs = [((f, f"d={d}"), ((f, 6, d), (f, 4, d)), ((f, 8, d), (f, 6, d)))
             for f in p.families for d in p.depths]
    return _depth_or_width(cells, p, pairs)


def h5(cells: dict, failed: dict, g1: dict, p: PureProtocol) -> dict:
    """H5 ansatz (Class I: A1, A2, A4, A7). (a) every pair equivalent within the
    SESOI at every (n, d); (b) G1 global variance at n = 6, d = 5 of A7 at least
    twice that of each of A1, A2, A4; (c, informed by fair-v1) A1 at d = 1 has
    the most failed seeds of Class I at n = 4."""
    seeds = p.seeds()
    eq_rows, eq_parts = {}, []
    for n in p.widths:
        for d in p.depths:
            for i, a in enumerate(CLASS_I):
                for b in CLASS_I[i + 1:]:
                    pr = paired(cells, (a, n, d), (b, n, d), seeds)
                    key = f"{a}-{b} n={n} d={d}"
                    if pr is None:
                        eq_rows[key] = "incomplete"
                        eq_parts.append("incomplete")
                        continue
                    v = st.equivalent(st.contrast(*pr), p.sesoi)
                    eq_rows[key] = v
                    eq_parts.append(v)
    part_a = _overall(eq_parts)
    var = {f: g1.get((f, H2_WIDTH, 5)) for f in CLASS_I}
    if any(v is None for v in var.values()):
        part_b, ratios = "incomplete", {}
    else:
        m = {f: float(np.mean(v["global"])) for f, v in var.items()}
        ratios = {f: m["A7"] / max(m[f], 1e-18) for f in ("A1", "A2", "A4")}
        part_b = "supported" if all(r >= 2.0 for r in ratios.values()) else "refuted"
    fails = {f: failed.get((f, 4, 1)) for f in CLASS_I}
    if any(v is None for v in fails.values()):
        part_c = "incomplete"
    else:
        part_c = "supported" if fails["A1"] == max(fails.values()) and fails["A1"] > 0 else "refuted"
    return {"equivalence": eq_rows, "variance_ratio_A7_over": ratios, "failed_seeds_n4_d1": fails,
            "verdict_a": part_a, "verdict_b": part_b, "verdict_c_informed": part_c,
            "verdict": st.combine([part_a, part_b]) if "incomplete" not in (part_a, part_b)
            else "incomplete"}


def h6(cells: dict, p: PureProtocol) -> dict:
    """H6 encoding (A2 ansatz, n = 4), per depth: (a) re-upload (A3) beats angle
    (A2); (b) amplitude (A2:amp) beats angle; (c) IQP (A2:iqp) does not beat angle
    by more than the SESOI."""
    seeds, n = p.seeds(), p.encoding_width
    rows, pg = {}, {}
    for d in p.depths:
        for tag in ("A3", "A2:amp", "A2:iqp"):
            pr = paired(cells, (tag, n, d), ("A2", n, d), seeds)
            rows[(tag, d)] = None if pr is None else st.contrast(*pr)
            if pr is not None and tag != "A2:iqp":
                pg[(tag, d)] = rows[(tag, d)]["p_greater"]
    adj = st.holm(pg) if pg else {}
    out, parts = {}, {"a": [], "b": [], "c": []}
    for (tag, d), c in rows.items():
        part = {"A3": "a", "A2:amp": "b", "A2:iqp": "c"}[tag]
        if c is None:
            v = "incomplete"
        elif part == "c":
            v = st.not_greater(c, p.sesoi)
        else:
            v = st.greater(c, p.sesoi, adj[(tag, d)])
        out[f"{tag} vs A2 d={d}"] = {"contrast": c, "verdict": v}
        parts[part].append(v)
    verdicts = {k: _overall(v) for k, v in parts.items()}
    return {"rows": out, "verdict_a": verdicts["a"], "verdict_b": verdicts["b"],
            "verdict_c": verdicts["c"], "verdict": _overall(list(verdicts.values()))}


def h7(noise: list[dict], p: PureProtocol, level: str = "nisq") -> dict:
    """H7 noise (n = 4): across (model, d) cells, the loss of prediction fidelity
    is ranked better by depth than by two-qubit gate count (|rho_depth| > |rho_2q|)."""
    cell = defaultdict(list)
    gates = {}
    for r in noise:
        if r["noise"]["level"] != level or int(r["n"]) != p.noise_width:
            continue
        key = (r["family"], int(r["d"]))
        cell[key].append(1.0 - r["prediction_fidelity"])
        gates[key] = r["two_qubit_gates"]
    if len(cell) < 4:
        return {"verdict": "incomplete", "cells": len(cell)}
    keys = sorted(cell)
    loss = [float(np.mean(cell[k])) for k in keys]
    by_d = st.spearman([k[1] for k in keys], loss)
    by_g = st.spearman([gates[k] for k in keys], loss)
    v = "supported" if abs(by_d["rho"]) > abs(by_g["rho"]) else "refuted"
    return {"by_depth": by_d, "by_two_qubit": by_g, "cells": len(keys), "verdict": v}


def h8(cells: dict, p: PureProtocol, fair_records: list[dict] | None) -> dict:
    """H8 bridge: at n = 4, d = 3, the families A1-A4 are more consistently
    ordered in pure-v1 than in fair-v1 (Kendall W higher) and pure-v1 separates
    them (Friedman p < alpha)."""
    fams, seeds = ("A1", "A2", "A3", "A4"), p.seeds()
    mat = [[cells.get((f, 4, 3), {}).get(s) for f in fams] for s in seeds]
    if any(v is None for row in mat for v in row):
        return {"verdict": "incomplete"}
    pure = st.friedman(np.array(mat))
    if not fair_records:
        return {"pure": pure, "fair": None, "verdict": "incomplete"}
    fcells = per_seed([r for r in fair_records if r.get("n") == 4 and not r.get("noise")])
    fseeds = sorted({s for f in fams for s in fcells.get((f, 4, 3), {})})
    fmat = [[fcells.get((f, 4, 3), {}).get(s) for f in fams] for s in fseeds]
    if not fmat or any(v is None for row in fmat for v in row):
        return {"pure": pure, "fair": None, "verdict": "incomplete"}
    fair = st.friedman(np.array(fmat))
    v = "supported" if (pure["W"] > fair["W"] and pure["p"] < st.ALPHA) else "refuted"
    return {"pure": pure, "fair": fair, "verdict": v}


def _overall(parts: list[str]) -> str:
    if not parts or "incomplete" in parts:
        return "incomplete"
    return st.combine(parts)


# --------------------------------------------------------------------------- #
# References and categories                                                    #
# --------------------------------------------------------------------------- #

def reference_key(tag: str, n: int) -> str:
    """Return the baseline key of the like-for-like reference of one model."""
    return f"logreg_pca{MODELS[tag].n_features(n)}"


def references(baselines: list[dict]) -> dict:
    """Return ``{(n, key): summary}`` of every baseline over seeds."""
    acc: dict = defaultdict(list)
    for r in baselines:
        for k, v in r.items():
            if k.startswith("logreg_"):
                acc[(int(r["n"]), k)].append(v["mcc"])
    return {k: st.summary(v) for k, v in acc.items()}


def categorise(task_data: dict, g1: dict, p: PureProtocol) -> dict:
    """Return ``{(model, n, d): {"category", "mcc", "ref", "failed", "noise_drop", "gradvar"}}``."""
    cells = per_seed(task_data["clean"])
    refs = references(task_data["baselines"])
    drop: dict = defaultdict(list)
    for r in task_data["noise_eval"]:
        if r["noise"]["level"] == "nisq":
            drop[(r["family"], int(r["n"]), int(r["d"]))].append(
                r["clean_test"]["mcc"] - r["test"]["mcc"])
    out = {}
    for key, by_seed in sorted(cells.items(), key=lambda kv: (list(MODELS).index(kv[0][0]), kv[0][1:])):
        tag, n, d = key
        vals = np.array(list(by_seed.values()))
        s = st.summary(vals)
        failed = int(np.sum(vals < p.fail_mcc))
        ref = refs.get((n, reference_key(tag, n)), {}).get("mean")
        g = g1.get((MODELS[tag].ansatz, n, d))
        gradvar = float(np.mean(g["local"])) if g else None
        nd = float(np.mean(drop[key])) if drop.get(key) else None
        if s["mean"] - s["ci95"] <= 0.0:
            cat = "Floor"
        elif (gradvar is not None and gradvar < SHOTS_FLOOR) or failed >= 3:
            cat = "Trainability-limited"
        elif ref is not None and s["mean"] >= ref - p.sesoi and failed <= 1:
            cat = "Noise-limited" if (nd is not None and nd > p.sesoi) else "Performing"
        else:
            cat = "Capacity-limited"
        out[key] = {"category": cat, "mcc": s, "ref": ref, "failed": failed,
                    "noise_drop": nd, "gradvar_local": gradvar}
    return out


def failed_counts(clean: list[dict], p: PureProtocol) -> dict:
    """Return ``{(model, n, d): failed seeds}``."""
    out: dict = defaultdict(int)
    seen: dict = defaultdict(int)
    for r in clean:
        key = (r["family"], int(r["n"]), int(r["d"]))
        seen[key] += 1
        out[key] += int(r["selected"]["test"]["mcc"] < p.fail_mcc)
    return {k: out[k] for k in seen}


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #

def missing_cells(clean: list[dict], p: PureProtocol, task: str) -> list[str]:
    """Return the protocol cells without all seeds."""
    have = defaultdict(set)
    for r in clean:
        have[(r["family"], int(r["n"]), int(r["d"]))].add(int(r["seed"]))
    want = set(p.seeds())
    out = []
    for arm in ("family", "encoding"):
        for cell in p.cells(arm):
            got = len(have[cell] & want)
            if got < p.n_seeds:
                out.append(f"{task} {cell[0]} n={cell[1]} d={cell[2]}: {got}/{p.n_seeds}")
    return out


def _fmt(s: dict | None, digits: int = 3) -> str:
    if not s:
        return "-"
    return f"{s['mean']:+.{digits}f} +- {s['ci95']:.{digits}f}"


def build_report(tasks=TASKS, protocol: PureProtocol = PROTOCOL, root: Path = RESULTS,
                 scorecard: Path = DEFAULT_SCORECARD, fair_archive: Path | None = None) -> tuple[str, dict]:
    """Return the Markdown report and its JSON summary.

    Args:
        tasks: Tasks to report. Hypotheses read :data:`PRIMARY_TASK` only; other
            tasks get their cell table and categories.
        protocol: The protocol (its digest filters the records).
        root: Results folder of the protocol.
        scorecard: Descriptor checkpoint holding the G1 grid.
        fair_archive: Archived fair-v1 checkpoint of the primary task, for H8.

    Returns:
        ``(markdown, summary)``.
    """
    p = protocol
    g1 = load_g1(scorecard)
    lines = [f"# pure-v1 results [{p.digest()}]", "",
             f"Protocol {p.name}: circuit-only training, {p.n_seeds} paired seeds, SESOI "
             f"{p.sesoi} MCC, alpha {st.ALPHA}, Holm within each hypothesis.", ""]
    js: dict = {"protocol": p.name, "digest": p.digest(), "tasks": {}}
    for task in tasks:
        data = load_task(task, p, root)
        miss = missing_cells(data["clean"], p, task)
        cats = categorise(data, g1, p)
        refs = references(data["baselines"])
        lines += [f"## {task}", "",
                  ("every protocol cell complete" if not miss else
                   f"INCOMPLETE: {len(miss)} cells missing seeds (first: {miss[0]})"), "",
                  "| model | n | d | test MCC (95% CI) | ref (logreg same features) | failed | noise drop | category |",
                  "|---|---|---|---|---|---|---|---|"]
        for (tag, n, d), c in cats.items():
            nd = "-" if c["noise_drop"] is None else f"{c['noise_drop']:+.3f}"
            ref = "-" if c["ref"] is None else f"{c['ref']:+.3f}"
            lines.append(f"| {tag} | {n} | {d} | {_fmt(c['mcc'])} | {ref} | {c['failed']} | {nd} | {c['category']} |")
        lines += ["", "References (logistic regression, test MCC):", ""]
        for (n, key), s in sorted(refs.items()):
            lines.append(f"- n={n} {key}: {_fmt(s)}")
        lines.append("")
        js["tasks"][task] = {"missing": miss,
                             "categories": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in cats.items()},
                             "references": {f"{k[0]}|{k[1]}": v for k, v in refs.items()}}
        if task != PRIMARY_TASK:
            continue
        cells = per_seed(data["clean"])
        failed = failed_counts(data["clean"], p)
        fair = load(Path(fair_archive)) if fair_archive else None
        hyp = {"H1": h1(g1), "H2": h2(g1), "H3": h3(cells, p), "H4": h4(cells, p),
               "H5": h5(cells, failed, g1, p), "H6": h6(cells, p),
               "H7": h7(data["noise_eval"], p), "H8": h8(cells, p, fair)}
        lines += ["## Hypotheses (primary task)", "", "| hypothesis | verdict |", "|---|---|"]
        for name, res in hyp.items():
            lines.append(f"| {name} | {res['verdict']} |")
        lines.append("")
        js["hypotheses"] = hyp
    return "\n".join(lines), js


def main(argv: list[str] | None = None) -> None:
    """Write ``RESULTS_pure.md`` and ``results/pure/summary.json``."""
    ap = argparse.ArgumentParser(description="pure-v1 report: hypotheses and categories.")
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    ap.add_argument("--scorecard", default=str(DEFAULT_SCORECARD))
    ap.add_argument("--fair-archive", default=None,
                    help="archived fair-v1 imdb1k.jsonl (for H8); omit to leave H8 incomplete")
    ap.add_argument("--out", default=str(PKG.parent / "RESULTS_pure.md"))
    args = ap.parse_args(argv)
    text, js = build_report(args.tasks, PROTOCOL, RESULTS, Path(args.scorecard),
                            Path(args.fair_archive) if args.fair_archive else None)
    Path(args.out).write_text(text, encoding="utf-8")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(js, indent=1, default=float), encoding="utf-8")
    print(text)
    print(f"\nwrote {args.out} and {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()
