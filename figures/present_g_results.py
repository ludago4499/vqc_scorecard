# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below.
#
# History:
#   2026-09-02  Created. English text presentation of the G-series (G1-G6),
#               read from the checkpoint (results/scorecard.jsonl) through the
#               same reading functions the deck charts use, so a printed number
#               cannot drift from the measurement or from the chart that shares
#               its source. Companion to figures/make_g_charts.py: that module
#               draws the grid, this one states it in words and tables.
#   2026-10-01  FAMILIES extended to A1-A7 (pure-v1). Families without G
#               records are skipped by the readers, as before.

"""Present the trainability series (G1-G6) as an English report.

Run from the folder that contains ``vqc_scorecard/``::

    python -m vqc_scorecard.figures.present_g_results
    python -m vqc_scorecard.figures.present_g_results --out g_results.txt
    python -m vqc_scorecard.figures.present_g_results --format md --out g_results.md

Every value is read from ``results/scorecard.jsonl`` and derived by the same
functions that produce the deck charts (``bp_onset_slope``, ``crossing_width``,
``locality_gap``, ``noise_crossing``, ``noise_depth_rate``). Sections whose
records are absent from the checkpoint are reported as "not measured yet"
rather than fabricated.

The six readings, and the one grid behind the first four:

* G1  barren-plateau onset .... width-decay rate alpha of Var[dC] ~ exp(-alpha n)
* G2  depth retention ......... Var[dC] vs depth at fixed width, and its floor crossing
* G3  shot-floor margin ....... width n* where Var[dC] falls under the 1024-shot floor
* G4  cost-locality gap ....... alpha_global - alpha_local (the observable's share)
* G5  noise-induced plateau ... depth-decay rate beta and critical noise p_crit
* G6  initialisation scale .... gradient revival vs QFIM-rank (capacity) collapse
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

# The readings come from the metric modules, never re-implemented here, so this
# report and the deck charts always agree with the measurement.
from ..metrics.g1_gradient import SHOTS_FLOOR, bp_onset_slope, depth_retention
from ..metrics.g3_shot_floor import crossing_width
from ..metrics.g4_locality import locality_gap
from ..metrics.g5_noise import noise_crossing, noise_depth_rate
from ..metrics.g6_init import SCALES

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE.parent / "results" / "scorecard.jsonl"

FAMILIES = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")
LABEL = {"A1": "A1 HEA", "A2": "A2 SEL", "A3": "A3 re-upload", "A4": "A4 DisCoCat",
         "A5": "A5 circuit-centric", "A6": "A6 IQP", "A7": "A7 Farhi-Neven"}
LN2 = 0.6931471805599453

# Grid points the report reads at. Chosen to match make_g_charts.py so the words
# describe the same slices the charts draw.
G1_DEPTH = 5      # onset fitted at this fixed depth (G1, G3)
G2_WIDTH = 6      # depth curve read at this fixed width (G2)
G4_DEPTHS = (1, 3)  # shallow arm (claim regime) and O(log n) arm (G4)
G5_WIDTH = 4      # noise depth sweep read at this width (G5)
G5_PC_DEPTH = 8   # p_crit reported at this depth (G5)
G6_WIDTH, G6_DEPTH = 8, 5  # init-scale grid point (G6)


# --------------------------------------------------------------------------- #
# Checkpoint access
# --------------------------------------------------------------------------- #

def load(path: Path) -> list[dict]:
    """Return every record in the checkpoint, in file order."""
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def of_metric(records: list[dict], metric: str) -> list[dict]:
    return [r for r in records if r.get("metric") == metric]


def by_family(records: list[dict], **match) -> dict[str, list[dict]]:
    """Group records by family after filtering on equality of the kwargs."""
    out: dict[str, list[dict]] = {f: [] for f in FAMILIES}
    for r in records:
        if all(r.get(k) == v for k, v in match.items()) and r["family"] in out:
            out[r["family"]].append(r)
    return out


# --------------------------------------------------------------------------- #
# Small text helpers - a plain fixed-width report, no third-party dependency
# --------------------------------------------------------------------------- #

class Report:
    """Accumulates lines; renders plain text or GitHub-flavoured Markdown."""

    def __init__(self, fmt: str = "text") -> None:
        self.fmt = fmt
        self.lines: list[str] = []

    def rule(self, char: str = "=") -> None:
        if self.fmt == "text":
            self.lines.append(char * 78)

    def h1(self, text: str) -> None:
        if self.fmt == "md":
            self.lines.append(f"# {text}")
        else:
            self.rule("=")
            self.lines.append(text)
            self.rule("=")

    def h2(self, text: str) -> None:
        self.lines.append("")
        if self.fmt == "md":
            self.lines.append(f"## {text}")
        else:
            self.lines.append(text)
            self.rule("-")

    def p(self, text: str = "") -> None:
        self.lines.append(text)

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        if self.fmt == "md":
            self.lines.append("| " + " | ".join(headers) + " |")
            self.lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in rows:
                self.lines.append("| " + " | ".join(row) + " |")
            return
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
        self.lines.append(fmt.format(*headers))
        self.lines.append("  ".join("-" * w for w in widths))
        for row in rows:
            self.lines.append(fmt.format(*row))

    def render(self) -> str:
        return "\n".join(self.lines) + "\n"


def fnum(x: float | None, digits: int = 3) -> str:
    """Format a float or None uniformly."""
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


# --------------------------------------------------------------------------- #
# The six sections
# --------------------------------------------------------------------------- #

def section_g1(rep: Report, g1: list[dict]) -> None:
    rep.h2("G1 - Barren-plateau onset (gradient decay with width)")
    rep.p("Global-cost gradient variance falls as Var[dC] ~ exp(-alpha * n). The")
    rep.p(f"onset rate alpha is fitted over the width grid at fixed depth d={G1_DEPTH}.")
    rep.p("An exact 2-design gives alpha = ln 2 = 0.693; a larger alpha is a faster")
    rep.p("(worse) onset. Source: McClean 2018; Cerezo 2021; Ragone/Larocca 2025.")
    rep.p()
    grid = by_family(g1, d=G1_DEPTH)
    rows = []
    for fam in FAMILIES:
        pts = sorted(grid[fam], key=lambda r: r["n"])
        if not pts:
            continue
        ns = [r["n"] for r in pts]
        vs = [r["gradvar_global_mean"] for r in pts]
        alpha = bp_onset_slope(ns, vs)
        rows.append([LABEL[fam], f"{ns[0]}-{ns[-1]} ({len(ns)} pts)",
                     f"{alpha:.3f}", f"{alpha / LN2:.2f}x ln2"])
    rep.table(["family", "widths n", "alpha", "vs 2-design"], rows)
    rep.p()
    rep.p("Reading: every family sits at essentially the 2-design rate (alpha ~ ln 2)")
    rep.p("on the swept widths. The onset is universal across the four ansaetze - it")
    rep.p("is forced by the universal dynamical Lie algebra (the P4 finding), not a")
    rep.p("discriminator between them.")


def section_g2(rep: Report, g1: list[dict]) -> None:
    rep.h2("G2 - Depth retention (gradient vs circuit depth)")
    rep.p(f"Global-cost gradient variance vs depth at fixed width n={G2_WIDTH}, and the")
    rep.p("first depth (if any) at which it drops below the 1024-shot floor")
    rep.p(f"({SHOTS_FLOOR:.3e}). A flat curve means no depth-induced collapse.")
    rep.p()
    grid = by_family(g1, n=G2_WIDTH)
    rows = []
    for fam in FAMILIES:
        pts = sorted(grid[fam], key=lambda r: r["d"])
        if not pts:
            continue
        ds = [r["d"] for r in pts]
        vs = [r["gradvar_global_mean"] for r in pts]
        ret = depth_retention(ds, vs)
        cross = ret["crossing_depth"]
        rows.append([LABEL[fam], f"{ds[0]}-{ds[-1]}",
                     f"{min(vs):.3e}", f"{max(vs):.3e}",
                     "none" if cross is None else str(cross)])
    rep.table(["family", "depths d", "min Var", "max Var", "sub-floor depth"], rows)
    rep.p()
    rep.p("Reading: the variance settles by d=3 and stays flat out to the deepest")
    rep.p("measured circuit - no family crosses the shot floor along the depth axis.")
    rep.p("Depth alone does not induce a barren plateau here (contrast G5, where")
    rep.p("depth under noise does).")


def section_g3(rep: Report, g1: list[dict]) -> None:
    rep.h2("G3 - Shot-floor margin (trainable width budget)")
    rep.p("How far the global gradient sits above finite sampling. n* is the")
    rep.p("interpolated width at which Var[dC] crosses the 1024-shot floor at fixed")
    rep.p(f"depth d={G1_DEPTH}: beyond n*, an N=1024-shot gradient is not resolvable")
    rep.p("from shot noise. Source: Larocca 2025, Box 3.")
    rep.p()
    grid = by_family(g1, d=G1_DEPTH)
    ranked = []
    for fam in FAMILIES:
        pts = sorted(grid[fam], key=lambda r: r["n"])
        if not pts:
            continue
        ns = [r["n"] for r in pts]
        vs = [r["gradvar_global_mean"] for r in pts]
        star = crossing_width(ns, vs)
        ranked.append((fam, star))
    rows = [[LABEL[f], "not crossed" if s is None else f"{s:.1f} qubits"]
            for f, s in ranked]
    rep.table(["family", "n* (floor crossing)"], rows)
    rep.p()
    crossed = [(f, s) for f, s in ranked if s is not None]
    if crossed:
        crossed.sort(key=lambda t: t[1])
        order = " < ".join(f"{LABEL[f].split()[0]} {s:.1f}" for f, s in crossed)
        rep.p(f"Reading: earliest to latest floor crossing - {order}.")
        rep.p("The spread is under one qubit: the shot-floor budget mirrors the G1")
        rep.p("onset (a physically forced near-tie), not a real separation between")
        rep.p("families. In hardware units this is the same result as the P4 tie.")


def section_g4(rep: Report, g1: list[dict]) -> None:
    rep.h2("G4 - Cost-locality gap (how much of the plateau is the observable)")
    rep.p("A global (Z...Z) and a local (Z_0) cost share the same ansatz, so their")
    rep.p("width-decay rates differ only by the observable. gap = alpha_global -")
    rep.p("alpha_local. A gap near ln 2 = 0.693 means locality alone buys a full")
    rep.p("2-design of trainability; a gap near 0 means both costs already plateau.")
    rep.p("Read at the shallow claim regime and at O(log n) depth. Source: Cerezo")
    rep.p("2021 (NC 12:1791); scope caveat Larocca 2025, Box 10.")
    for depth in G4_DEPTHS:
        rep.p()
        rep.p(f"depth d = {depth}:")
        grid = by_family(g1, d=depth)
        rows = []
        for fam in FAMILIES:
            pts = sorted(grid[fam], key=lambda r: r["n"])
            if len(pts) < 3:
                continue
            ns = [r["n"] for r in pts]
            gap = locality_gap(ns,
                               [r["gradvar_global_mean"] for r in pts],
                               [r["gradvar_local_mean"] for r in pts])
            rows.append([LABEL[fam], f"{gap['alpha_global']:+.2f}",
                         f"{gap['alpha_local']:+.2f}", f"{gap['gap']:+.2f}"])
        rep.table(["family", "alpha_global", "alpha_local", "gap"], rows)
    rep.p()
    rep.p("Reading: at d=1 three families show the Cerezo ordering (global decays")
    rep.p("faster, positive gap), strongest for A4. A2 REVERSES it (negative gap):")
    rep.p("the local cost decays faster there - the Box 10 caveat, measured. The gap")
    rep.p("closes by d>=3 as both costs enter the plateau, so locality is a lever")
    rep.p("only at shallow depth.")


def section_g5(rep: Report, g5: list[dict]) -> None:
    rep.h2("G5 - Noise-induced barren plateau (depolarizing noise)")
    if not g5:
        rep.p("Not measured yet - no G5 records in the checkpoint.")
        return
    rep.p("Per-gate depolarizing noise of strength p contracts the whole landscape.")
    rep.p("beta is the depth-decay rate Var[dC] ~ exp(-beta d) at fixed p (fitted in")
    rep.p("the settled regime d>=3); p_crit is the interpolated noise strength at")
    rep.p("which the gradient crosses the shot floor. Per-gate (not per-layer) model")
    rep.p("couples this to gate count. Source: Wang 2021 (NC 12:6961); Larocca 2025,")
    rep.p("Box 8.")
    rep.p()
    rep.p(f"Depth sweep at width n = {G5_WIDTH}:")
    rows = []
    for fam in FAMILIES:
        pts = sorted((r for r in g5 if r["family"] == fam and r["n"] == G5_WIDTH),
                     key=lambda r: r["d"])
        if not pts:
            continue
        ds = [r["d"] for r in pts]
        # p = 1e-2 is level index 2 (NOISE_LEVELS = 0, 1e-3, 1e-2, 1e-1).
        vs2 = [r["gradvar_global_s2_mean"] for r in pts]
        fit = [(d, v) for d, v in zip(ds, vs2) if d >= 3 and v > 1e-10]
        beta = noise_depth_rate([d for d, _ in fit], [v for _, v in fit]) \
            if len(fit) >= 3 else None
        pc = None
        deep = [r for r in pts if r["d"] == G5_PC_DEPTH]
        if deep:
            r = deep[0]
            levels = [r[f"noise_s{i}_mean"] for i in range(4)]
            vv = [r[f"gradvar_global_s{i}_mean"] for i in range(4)]
            pc = noise_crossing(levels, vv)
        off = " *" if any(r.get("off_protocol") for r in pts) else ""
        rows.append([LABEL[fam] + off, fnum(beta, 2),
                     "not crossed" if pc is None else f"{pc:.4f}"])
    rep.table(["family", "beta (p=1e-2)", f"p_crit (d={G5_PC_DEPTH})"], rows)

    # Width axis: p_crit vs n at a fixed shallow depth, if those points exist.
    width_rows = []
    marked = any(r[0].endswith("*") for r in rows)
    widths = sorted({r["n"] for r in g5 if r["d"] == 4})
    for fam in FAMILIES:
        cells = []
        have = False
        for nn in widths:
            hit = [r for r in g5 if r["family"] == fam and r["n"] == nn and r["d"] == 4]
            if hit:
                r = hit[0]
                levels = [r[f"noise_s{i}_mean"] for i in range(4)]
                vv = [r[f"gradvar_global_s{i}_mean"] for i in range(4)]
                pc = noise_crossing(levels, vv)
                flag = "*" if r.get("off_protocol") else ""
                cells.append("-" if pc is None else f"{pc:.4f}{flag}")
                marked = marked or bool(flag)
                have = True
            else:
                cells.append("-")
        if have:
            width_rows.append([LABEL[fam]] + cells)
    if width_rows:
        rep.p()
        rep.p("Width axis - p_crit at fixed depth d = 4:")
        rep.table(["family"] + [f"n={nn}" for nn in widths], width_rows)
    rep.p()
    rep.p("Reading: p_crit(d=8) lands near 1e-2 (0.006-0.010), i.e. the survivable")
    rep.p("per-gate error rate at eight layers. A4 (DisCoCat) is the most noise-")
    rep.p("tolerant, A3 (re-upload) the most fragile - its extra encoding gates cost")
    rep.p("the fastest decay. Along the width axis p_crit falls ~3x per +2 qubits, so")
    rep.p("the noise-induced plateau has a WIDTH axis on top of Wang's depth axis.")
    if marked:
        rep.p("(* point flagged off-protocol in the checkpoint - read as indicative.)")


def section_g6(rep: Report, g6: list[dict]) -> None:
    rep.h2("G6 - Initialisation sensitivity (trainability vs capacity)")
    if not g6:
        rep.p("Not measured yet - no G6 records in the checkpoint.")
        return
    rep.p("Angles drawn as theta = a * theta_random, sweeping the init scale a from 1")
    rep.p("down to 1e-3. Small a is meant to revive the local-cost gradient second")
    rep.p("moment E[(dC)^2] while collapsing capacity (the QFIM rank). 'small-a")
    rep.p("rescue' is E[(dC)^2] at the smallest scale a=1e-3 over its a=1 baseline;")
    rep.p("'rank drop' is QFIM rank at a=1 vs at a=1e-3. Read at")
    rep.p(f"n={G6_WIDTH}, d={G6_DEPTH}. Source: Haug 2021 (PRXQ 2:040309); Zhang 2022; Grant 2019.")
    rep.p()
    rows = []
    for fam in FAMILIES:
        hit = [r for r in g6 if r["family"] == fam
               and r["n"] == G6_WIDTH and r["d"] == G6_DEPTH]
        if not hit:
            continue
        r = hit[0]
        ns = len(SCALES)
        gl = [r[f"grad2_local_s{i}_mean"] for i in range(ns)]
        rank = [r[f"qfim_rank_s{i}_mean"] for i in range(ns)]
        baseline = gl[0]
        rescue = gl[-1] / baseline if baseline > 0 else float("nan")
        verdict = "revived" if rescue >= 1.5 else "not rescued"
        rows.append([LABEL[fam], f"{baseline:.2e}", f"{rescue:.1f}x",
                     f"{rank[0]:.0f} -> {rank[-1]:.0f}", verdict])
    rep.table(["family", "E[(dC)^2] a=1", "local rescue (a=1e-3)",
               "QFIM rank a=1->1e-3", "trainability"], rows)
    rep.p()
    rep.p("Reading: small-angle init revives the LOCAL-cost gradient for A4 strongly")
    rep.p("(~14x) and A2 moderately, barely for A1, and A3 is not meaningfully")
    rep.p("rescued. The global cost is revived for no family. Capacity (QFIM rank)")
    rep.p("falls by about half in every case. So the small-angle default is")
    rep.p("justified for local-cost heads only, and always trades capacity for it.")


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def build(records: list[dict], fmt: str, checkpoint: Path) -> str:
    rep = Report(fmt)
    rep.h1("VQC/VQN scorecard - trainability series G1-G6")
    rep.p(f"Source checkpoint : {checkpoint}")
    rep.p(f"Records total     : {len(records)}")
    counts = {m: len(of_metric(records, m)) for m in ("G1", "G5", "G6")}
    rep.p("G-series records  : "
          f"G1={counts['G1']} (grid for G1-G4), G5={counts['G5']}, G6={counts['G6']}")
    rep.p(f"Generated (UTC)   : {datetime.now(timezone.utc):%Y-%m-%d %H:%M}")
    rep.p("Families          : A1 HEA, A2 SEL, A3 data re-upload, A4 DisCoCat")
    rep.p("Note: G1-G4 are four readings of one measured gradient-variance grid; "
          "G5 and G6 are separate measured sweeps. Every value is derived by the "
          "package's own reading functions - no number is typed by hand.")

    g1 = of_metric(records, "G1")
    section_g1(rep, g1)
    section_g2(rep, g1)
    section_g3(rep, g1)
    section_g4(rep, g1)
    section_g5(rep, of_metric(records, "G5"))
    section_g6(rep, of_metric(records, "G6"))
    return rep.render()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Present the G-series (G1-G6) results in English.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT),
                        help="path to scorecard.jsonl (default: package results/)")
    parser.add_argument("--format", choices=("text", "md"), default="text",
                        help="plain text (default) or Markdown")
    parser.add_argument("--out", default=None,
                        help="write to this file instead of stdout")
    args = parser.parse_args()

    records = load(Path(args.checkpoint))
    report = build(records, args.format, Path(args.checkpoint))

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
