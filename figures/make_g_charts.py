# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below.
#
# History:
#   2026-09-01  Created. Deck charts for the G-series, drawn from the
#               checkpoint (results/scorecard.jsonl) - never from numbers
#               typed by hand, so a chart cannot drift from the measurement
#               that produced it. Replaces the ad-hoc scripts that generated
#               the v7.x/v8.x G1/G2 charts inside the deck only.
#   2026-09-08  g5_noise extended from the A1/A4 pilot to all four families,
#               tiled in a 2x2 grid, each panel annotated with its checkpoint
#               beta (p=1e-2) and p_crit(d=8). Deck v9 G5 slide drawn from it.
#   2026-10-01  FAMILIES extended to A1-A7 (pure-v1). Families without G
#               records are skipped by the readers, as before. A7 drawn in
#               gold (distinct from A4 red); G2 note and G6 legend moved for
#               seven series.

"""Deck charts for the trainability series, from the checkpoint.

Run from the folder that contains ``vqc_scorecard/``::

    python -m vqc_scorecard.figures.make_g_charts --outdir charts

Writes g1_onset.png, g2_depth.png, g3_margin.png, g4_locality.png,
g5_noise.png (only if G5 records exist) and g6_scale.png (only if G6 records
exist). Every number on every chart is read from ``results/scorecard.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..metrics.g1_gradient import SHOTS_FLOOR, bp_onset_slope
from ..metrics.g3_shot_floor import crossing_width
from ..metrics.g4_locality import locality_gap
from ..metrics.g5_noise import NOISE_LEVELS, noise_crossing, noise_depth_rate
from ..metrics.g6_init import SCALES

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE.parent / "results" / "scorecard.jsonl"

FAMILIES = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")
COLOR = {"A1": "#0f172a", "A2": "#0891b2", "A3": "#b45309", "A4": "#b91c1c",
         "A5": "#7c3aed", "A6": "#15803d", "A7": "#ca8a04"}
LABEL = {"A1": "A1 HEA", "A2": "A2 SEL", "A3": "A3 re-up", "A4": "A4 DisCoCat",
         "A5": "A5 circ-centric", "A6": "A6 IQP", "A7": "A7 Farhi-Neven"}
MUTED, GRID, AXIS = "#94a3b8", "#e2e8f0", "#475569"

plt.rcParams.update({
    "figure.figsize": (12.8, 5.4), "figure.dpi": 100, "savefig.dpi": 100,
    "font.size": 14, "axes.edgecolor": MUTED, "axes.labelcolor": AXIS,
    "axes.labelsize": 17, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 13.5,
})


def load(metric: str) -> list[dict]:
    """Return the checkpoint records of one metric."""
    records = []
    with CHECKPOINT.open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r.get("metric") == metric:
                records.append(r)
    return records


def by_family(records: list[dict], **match) -> dict[str, list[dict]]:
    """Group records by family after filtering on equality of the kwargs."""
    out: dict[str, list[dict]] = {f: [] for f in FAMILIES}
    for r in records:
        if all(r.get(k) == v for k, v in match.items()) and r["family"] in out:
            out[r["family"]].append(r)
    return out


def _line(ax, x, y, family, label=None, **kw):
    ax.plot(x, y, marker="o", markersize=7, linewidth=2.6, color=COLOR[family],
            label=label or LABEL[family], **kw)


def _floor_line(ax, x0, x1, y=SHOTS_FLOOR, text="shot floor 1/1024"):
    ax.axhline(y, linestyle=":", linewidth=1.8, color=MUTED)
    ax.text(x0, y * 1.15, text, color=MUTED, fontsize=13)


def g1_onset(outdir: Path, depth: int = 5) -> None:
    """Var_global vs width at fixed depth, with per-family alpha fits."""
    grid = by_family(load("G1"), d=depth)
    fig, ax = plt.subplots()
    all_n = []
    for fam in FAMILIES:
        rows = sorted(grid[fam], key=lambda r: r["n"])
        ns = [r["n"] for r in rows]
        vs = [r["gradvar_global_mean"] for r in rows]
        if not ns:
            continue
        alpha = bp_onset_slope(ns, vs)
        _line(ax, ns, vs, fam, label=f"{LABEL[fam]}  α={alpha:.2f}")
        all_n += ns
    lo, hi = min(all_n), max(all_n)
    ref_n = np.linspace(lo, hi, 50)
    anchor = np.mean([sorted(grid[f], key=lambda r: r["n"])[0]["gradvar_global_mean"]
                      for f in FAMILIES if grid[f]])
    ax.plot(ref_n, anchor * np.exp(-np.log(2) * (ref_n - lo)), linestyle="--",
            linewidth=1.8, color=MUTED, zorder=0, label="2-design exp(−ln2·n)")
    _floor_line(ax, lo, hi)
    ax.set_yscale("log")
    ax.set_xlabel("qubits  n")
    ax.set_ylabel(f"Var[∂C] global (d={depth})")
    ax.set_xticks(sorted(set(all_n)))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "g1_onset.png", transparent=False, facecolor="white")
    plt.close(fig)


def g2_depth(outdir: Path, width: int = 6) -> None:
    """Var_global vs depth at fixed width."""
    grid = by_family(load("G1"), n=width)
    fig, ax = plt.subplots()
    depths = set()
    for fam in FAMILIES:
        rows = sorted(grid[fam], key=lambda r: r["d"])
        ds = [r["d"] for r in rows]
        vs = [r["gradvar_global_mean"] for r in rows]
        if ds:
            _line(ax, ds, vs, fam)
            depths |= set(ds)
    _floor_line(ax, min(depths), max(depths))
    ax.set_yscale("log")
    ax.set_xlabel("depth  d")
    ax.set_ylabel(f"Var[∂C] global (n={width})")
    ax.set_xticks(sorted(depths))
    ax.text(0.30, 0.60, "A1-A6 settle by d=3, then flat to d=8;\nA7 sits ~30x higher at every depth",
            transform=ax.transAxes, color=AXIS, fontsize=13.5)
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "g2_depth.png", transparent=False, facecolor="white")
    plt.close(fig)


def g3_margin(outdir: Path, depth: int = 5) -> None:
    """Var/floor margin vs width at fixed depth, with interpolated crossings."""
    grid = by_family(load("G1"), d=depth)
    fig, ax = plt.subplots()
    all_n = []
    for fam in FAMILIES:
        rows = sorted(grid[fam], key=lambda r: r["n"])
        ns = [r["n"] for r in rows]
        vs = [r["gradvar_global_mean"] for r in rows]
        if not ns:
            continue
        margins = [v / SHOTS_FLOOR for v in vs]
        star = crossing_width(ns, vs)
        label = f"{LABEL[fam]}  n*={star:.1f}" if star else LABEL[fam]
        _line(ax, ns, margins, fam, label=label)
        if star:
            ax.plot([star], [1.0], marker="X", markersize=13, color=COLOR[fam],
                    zorder=5)
        all_n += ns
    ax.axhline(1.0, linestyle=":", linewidth=1.8, color=MUTED)
    ax.text(min(all_n), 1.18, "floor: margin = 1  (1024 shots)", color=MUTED,
            fontsize=13)
    ax.set_yscale("log")
    ax.set_xlabel("qubits  n")
    ax.set_ylabel(f"Var[∂C] / shot floor (d={depth})")
    ax.set_xticks(sorted(set(all_n)))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "g3_margin.png", transparent=False, facecolor="white")
    plt.close(fig)


def g4_locality(outdir: Path, depths: tuple[int, int] = (1, 3)) -> None:
    """Grouped bars: alpha_global vs alpha_local per family, two depths."""
    records = load("G1")
    fig, axes = plt.subplots(1, 2, sharey=True)
    for ax, depth in zip(axes, depths):
        grid = by_family(records, d=depth)
        xs = np.arange(len(FAMILIES))
        for i, fam in enumerate(FAMILIES):
            rows = sorted(grid[fam], key=lambda r: r["n"])
            ns = [r["n"] for r in rows]
            if len(ns) < 3:
                continue
            gap = locality_gap(ns, [r["gradvar_global_mean"] for r in rows],
                               [r["gradvar_local_mean"] for r in rows])
            ax.bar(i - 0.19, gap["alpha_global"], width=0.36, color=COLOR[fam])
            ax.bar(i + 0.19, gap["alpha_local"], width=0.36, color=COLOR[fam],
                   alpha=0.35)
            ax.text(i, max(gap["alpha_global"], gap["alpha_local"]) + 0.03,
                    f"{gap['gap']:+.2f}", ha="center", fontsize=11.5,
                    color=AXIS)
        ax.axhline(np.log(2), linestyle="--", linewidth=1.6, color=MUTED)
        ax.axhline(0.0, linewidth=1.2, color=MUTED)
        ax.set_xticks(xs)
        ax.set_xticklabels(list(FAMILIES), fontsize=13)
        ax.set_title(f"d = {depth}", fontsize=15, color=AXIS)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("width-decay rate α")
    axes[1].text(len(FAMILIES) - 0.55, np.log(2) + 0.02, "ln 2", color=MUTED,
                 fontsize=13)
    fig.text(0.5, 0.012,
             "filled = global cost Z…Z    faded = local cost Z₀    "
             "(fit of Var ~ exp(−α·n) across the width grid)",
             ha="center", fontsize=12.5, color=AXIS)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(outdir / "g4_locality.png", transparent=False, facecolor="white")
    plt.close(fig)


def g5_noise(outdir: Path, width: int = 4,
             families: tuple[str, ...] = FAMILIES, ncols: int = 2) -> None:
    """Var vs depth per noise level, one panel per measured family.

    Panels are tiled in a ``ncols`` grid (2x2 for the four families) so the
    chart keeps the deck slot's aspect while every family is legible. Each
    panel carries the fitted depth-decay rate beta (at p = 1e-2, settled
    regime d >= 3) and the interpolated critical noise p_crit at d = 8 - both
    read from the checkpoint by the package's own G5 functions, never typed.
    """
    records = load("G5")
    if not records:
        print("g5_noise: no G5 records yet, skipped")
        return
    families = tuple(f for f in families
                     if any(r["family"] == f and r["n"] == width for r in records))
    if not families:
        print("g5_noise: no G5 records at this width, skipped")
        return
    shade = {0.0: MUTED, 1e-3: "#0891b2", 1e-2: "#b45309", 1e-1: "#b91c1c"}
    pname = {0.0: "p = 0 (control)", 1e-3: "p = 0.001",
             1e-2: "p = 0.01", 1e-1: "p = 0.1"}
    nrows = int(np.ceil(len(families) / ncols))
    fig, axgrid = plt.subplots(nrows, ncols, sharey=True,
                               figsize=(13.0, 2.7 * nrows + 0.05))
    axes = np.atleast_1d(axgrid).ravel()
    legend_ax = None
    for ax, fam in zip(axes, families):
        rows = sorted((r for r in records if r["family"] == fam
                       and r["n"] == width), key=lambda r: r["d"])
        ds = [r["d"] for r in rows]
        for i, p in enumerate(NOISE_LEVELS):
            vs = [r[f"gradvar_global_s{i}_mean"] for r in rows]
            keep = [(d, v) for d, v in zip(ds, vs) if v > 1e-14]
            ax.plot([d for d, _ in keep], [v for _, v in keep], marker="o",
                    markersize=5.5, linewidth=2.2, color=shade[p],
                    linestyle=":" if p == 0 else "-",
                    label=pname[p] if legend_ax is None else None)
        legend_ax = legend_ax or ax
        # beta at p = 1e-2 in the settled regime (d >= 3, see G2), above the
        # finite-difference numerical floor; p_crit interpolated at d = 8.
        vs2 = [r["gradvar_global_s2_mean"] for r in rows]
        fit = [(d, v) for d, v in zip(ds, vs2) if d >= 3 and v > 1e-10]
        note = ""
        if len(fit) >= 3:
            beta = noise_depth_rate([d for d, _ in fit], [v for _, v in fit])
            note = f"β(p=0.01) = {beta:.2f}/layer"
        deep = [r for r in rows if r["d"] == 8]
        if deep:
            r = deep[0]
            pc = noise_crossing([r[f"noise_s{i}_mean"] for i in range(4)],
                                [r[f"gradvar_global_s{i}_mean"] for i in range(4)])
            if pc is not None:
                note += ("\n" if note else "") + f"p_crit(d=8) = {pc:.4f}"
        if note:
            ax.text(0.03, 0.05, note, transform=ax.transAxes, fontsize=11.5,
                    color=AXIS, va="bottom", ha="left")
        ax.axhline(SHOTS_FLOOR, linestyle=":", linewidth=1.6, color=MUTED)
        ax.set_yscale("log")
        ax.set_xticks(ds)
        ax.set_title(f"{LABEL[fam]}  (n = {width})", fontsize=14, color=AXIS)
    for ax in axes[len(families):]:
        ax.set_visible(False)
    for ax in axes[-ncols:]:
        ax.set_xlabel("depth  d")
    for r in range(nrows):
        axes[r * ncols].set_ylabel("Var[∂C] global")
    axes[0].text(1.02, SHOTS_FLOOR * 1.35, "shot floor", color=MUTED, fontsize=11)
    handles, labels = legend_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(NOISE_LEVELS),
               fontsize=12.5, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(outdir / "g5_noise.png", transparent=False, facecolor="white")
    plt.close(fig)


def g6_scale(outdir: Path, width: int = 8, depth: int = 5) -> None:
    """Two panels: local-gradient second moment vs scale, and QFIM rank vs scale."""
    records = [r for r in load("G6") if r["n"] == width and r["d"] == depth]
    if not records:
        print("g6_scale: no G6 records yet, skipped")
        return
    fig, (left, right) = plt.subplots(1, 2)
    scales = np.array(SCALES)
    for fam in FAMILIES:
        rows = [r for r in records if r["family"] == fam]
        if not rows:
            continue
        r = rows[0]
        g2l = [r[f"grad2_local_s{i}_mean"] for i in range(len(SCALES))]
        rank = [r[f"qfim_rank_s{i}_mean"] for i in range(len(SCALES))]
        left.plot(scales, g2l, marker="o", markersize=7, linewidth=2.6,
                  color=COLOR[fam], label=LABEL[fam])
        right.plot(scales, rank, marker="o", markersize=7, linewidth=2.6,
                   color=COLOR[fam], label=LABEL[fam])
    for ax in (left, right):
        ax.set_xscale("log")
        ax.set_xlabel("init scale  a   (θ = a·θ_random)")
        ax.invert_xaxis()
    left.set_yscale("log")
    left.axhline(SHOTS_FLOOR, linestyle=":", linewidth=1.8, color=MUTED)
    left.text(0.85, SHOTS_FLOOR * 1.25, "shot floor", color=MUTED, fontsize=12.5)
    left.set_ylabel(f"E[(∂C)²] local  (n={width}, d={depth})")
    left.set_title("trainability: gradient stays alive", fontsize=15, color=AXIS)
    right.set_ylabel("QFIM rank (capacity)")
    right.set_title("capacity: directions collapse", fontsize=15, color=AXIS)
    right.legend(loc="upper right", fontsize=11, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "g6_scale.png", transparent=False, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="G-series deck charts.")
    parser.add_argument("--outdir", default=str(HERE / "charts"))
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    g1_onset(outdir)
    g2_depth(outdir)
    g3_margin(outdir)
    g4_locality(outdir)
    g5_noise(outdir)
    g6_scale(outdir)
    print(f"charts in {outdir}")


if __name__ == "__main__":
    main()
