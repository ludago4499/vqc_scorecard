# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below.
#
# History:
#   2026-10-01  Created. Reference-normalised heatmaps for the static
#               descriptors P1, P2 and P3 (deck v12.2 slides 18, 20, 22),
#               replacing the 13x10 numeric tables. Every cell is read from
#               ``results/scorecard.jsonl``; nothing is typed by hand.

"""Reference-normalised heatmaps for the static descriptors P1-P3.

Each descriptor is coloured against its own reference, so the colour answers
"how far from the reference" rather than showing a raw number:

* P1: ``KL / floor`` (log scale), where the floor is the sampled Haar KL at the
  same pair budget. Cells within 1.5x of the floor are drawn grey (not
  distinguishable from it), as declared on the P1 definition slide.
* P2: ``1 - Q / Q_Haar``, the relative entanglement gap to the exact Haar
  expectation. Gaps at or below 2% (four times the largest seed spread) are
  drawn grey; values above Haar are clipped to zero.
* P3: ``rank / ceiling``, the fraction of the state-manifold dimension
  ``2^(n+1) - 2`` that the QFIM rank reaches. A filled dot marks rank at the
  ceiling (manifold saturated); a hollow dot marks rank equal to the angle
  count ``p`` below the ceiling (every angle independent).

Run from the folder that contains ``vqc_scorecard/``::

    python -m vqc_scorecard.figures.make_p_heatmaps --outdir charts

Writes ``p1_heatmap.png``, ``p2_heatmap.png`` and ``p3_heatmap.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE.parent / "results" / "scorecard.jsonl"

FAMILIES = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")
COLUMN = {"A1": "A1\nHEA", "A2": "A2\nSEL", "A3": "A3\nre-upload",
          "A4": "A4\nDisCoCat ‡", "A5": "A5\ncirc.", "A6": "A6\nIQP",
          "A7": "A7\nF-N"}
WIDTHS = (2, 4, 6, 8)
DEPTHS = (1, 3, 5)

P1_GREY_RATIO = 1.5
P2_GREY_GAP = 0.02
RANK_TOL = 0.5

INK, MUTED, AT_REF = "#0f172a", "#64748b", "#e5e7eb"
BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
TEAL = LinearSegmentedColormap.from_list(
    "seq_teal", ["#e6f4f7", "#9fd6e2", "#3fa9c2", "#0e7490", "#0b3d4a"])

plt.rcParams.update({
    "font.family": ["Carlito", "Calibri", "DejaVu Sans"],
    "font.size": 15, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "white",
})


def load(metric: str) -> dict[tuple[str, int, int], dict]:
    """Return the checkpoint records of one metric keyed by grid point.

    Args:
        metric: Descriptor tag, for example ``"P1"``.

    Returns:
        Mapping from ``(family, n, d)`` to the record.

    Raises:
        KeyError: If a family, width or depth of the grid has no record.
    """
    out: dict[tuple[str, int, int], dict] = {}
    with CHECKPOINT.open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r.get("metric") == metric:
                out[(r["family"], r["n"], r["d"])] = r
    for f in FAMILIES:
        for n in WIDTHS:
            for d in DEPTHS:
                if (f, n, d) not in out:
                    raise KeyError(f"{metric}: no record for {(f, n, d)}")
    return out


def _grid(records: dict, value) -> np.ndarray:
    """Return a ``(len(WIDTHS) * len(DEPTHS), len(FAMILIES))`` value grid."""
    rows = [(n, d) for n in WIDTHS for d in DEPTHS]
    return np.array([[value(records[(f, n, d)]) for f in FAMILIES]
                     for n, d in rows])


def _row_labels() -> list[str]:
    return [f"n = {n}   d = {d}" if d == DEPTHS[0] else f"d = {d}"
            for n in WIDTHS for d in DEPTHS]


def _draw(
    values: np.ndarray,
    *,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
    grey: np.ndarray,
    records: dict,
    cbar_label: str,
    cbar_ticks: list[float],
    cbar_ticklabels: list[str],
    grey_label: str,
    marks: dict[tuple[int, int], str] | None = None,
    legend_extra: str = "",
) -> plt.Figure:
    """Draw one reference-normalised heatmap in the deck's table footprint.

    Args:
        values: Normalised value per cell, rows ``(n, d)``, columns families.
        cmap: Single-hue sequential colormap.
        norm: Colour normalisation over ``values``.
        grey: Boolean mask of cells drawn as "at reference".
        records: Checkpoint records, used for the off-protocol marker.
        cbar_label: Colour bar title.
        cbar_ticks: Colour bar tick positions in data units.
        cbar_ticklabels: Colour bar tick labels.
        grey_label: Caption of the grey swatch.
        marks: Optional glyph per ``(row, column)`` drawn centred in the cell.
        legend_extra: Optional caption line under the grey swatch.

    Returns:
        The figure, sized to the 5.65 in x 2.54 in table frame at 2x.
    """
    fig = plt.figure(figsize=(11.3, 5.08), dpi=200)
    ax = fig.add_axes([0.135, 0.02, 0.70, 0.83])
    n_rows, n_cols = values.shape
    rows = [(n, d) for n in WIDTHS for d in DEPTHS]
    for i, (n, d) in enumerate(rows):
        for j, f in enumerate(FAMILIES):
            colour = AT_REF if grey[i, j] else cmap(norm(values[i, j]))
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88,
                                       facecolor=colour, edgecolor="none"))
            rgb = np.array(matplotlib.colors.to_rgb(colour))
            dark = rgb @ np.array([0.2126, 0.7152, 0.0722]) < 0.45
            if marks and (i, j) in marks:
                ax.text(j + 0.5, i + 0.52, marks[(i, j)], ha="center",
                        va="center", fontsize=14,
                        color="white" if dark else INK)
            if records[(f, n, d)]["off_protocol"]:
                ax.text(j + 0.93, i + 0.14, "▫", ha="right", va="top",
                        fontsize=12, color="#cbd5e1" if dark else MUTED,
                        family="DejaVu Sans")
    for k in range(1, len(WIDTHS)):
        ax.axhline(k * len(DEPTHS), color="white", lw=5)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels([COLUMN[f] for f in FAMILIES], fontweight="bold",
                       linespacing=1.1)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(_row_labels())
    for lab, (n, d) in zip(ax.get_yticklabels(), rows):
        if d != DEPTHS[0]:
            lab.set_color(MUTED)
    ax.tick_params(length=0, pad=6)
    for side in ax.spines.values():
        side.set_visible(False)

    cax = fig.add_axes([0.875, 0.30, 0.022, 0.58])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.outline.set_visible(False)
    cb.set_ticks(cbar_ticks)
    cb.set_ticklabels(cbar_ticklabels)
    cb.ax.tick_params(length=0, labelsize=13)
    cb.minorticks_off()
    fig.text(0.865, 0.93, cbar_label, ha="left", va="bottom", fontsize=13.5,
             fontweight="bold")
    if grey_label:
        fig.patches.append(plt.Rectangle((0.875, 0.17), 0.022, 0.07,
                                         transform=fig.transFigure,
                                         facecolor=AT_REF, edgecolor="none"))
        fig.text(0.905, 0.205, grey_label, va="center", fontsize=12.5)
    if legend_extra:
        fig.text(0.865, 0.10 if grey_label else 0.25, legend_extra,
                 va="top", fontsize=12.5, linespacing=1.35)
    return fig


def p1_figure() -> plt.Figure:
    """Return the P1 heatmap: KL over the sampled Haar floor, log scale."""
    rec = load("P1")
    ratio = _grid(rec, lambda r: r["kl_nats_mean"] / r["kl_haar_floor"])
    grey = ratio < P1_GREY_RATIO
    top = 10 ** np.ceil(np.log10(ratio.max()))
    norm = LogNorm(vmin=P1_GREY_RATIO, vmax=top)
    marks = {(3, 2): "†"}
    ticks = [t for t in (1.5, 10, 100, 1000, 10000) if t <= top]
    return _draw(ratio, cmap=BLUE, norm=norm, grey=grey, records=rec,
                 cbar_label="KL / floor", cbar_ticks=ticks,
                 cbar_ticklabels=[f"{t:g}x" for t in ticks],
                 grey_label="< 1.5x: at floor", marks=marks)


def p2_figure() -> plt.Figure:
    """Return the P2 heatmap: relative entanglement gap to exact Haar."""
    rec = load("P2")
    gap = _grid(rec, lambda r: 1.0 - r["mw_mean_mean"] / r["mw_haar"])
    gap = np.clip(gap, 0.0, 1.0)
    grey = gap <= P2_GREY_GAP
    norm = Normalize(vmin=0.0, vmax=1.0)
    marks = {(i, j): "Q = 0" for i, j in zip(*np.nonzero(
        _grid(rec, lambda r: r["mw_mean_mean"]) == 0.0))}
    marks[(3, 2)] = "Q = 0 †"
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    return _draw(gap, cmap=BLUE, norm=norm, grey=grey, records=rec,
                 cbar_label="1 - Q / Q_Haar", cbar_ticks=ticks,
                 cbar_ticklabels=["0", "25%", "50%", "75%", "100%"],
                 grey_label="≤ 2%: at Haar", marks=marks,
                 legend_extra="Q = 0: product state")


def p3_figure() -> plt.Figure:
    """Return the P3 heatmap: QFIM rank over the manifold ceiling."""
    rec = load("P3")
    fill = _grid(rec, lambda r: r["qfim_rank_mean"] / r["qfim_rank_ceiling"])
    marks: dict[tuple[int, int], str] = {}
    rows = [(n, d) for n in WIDTHS for d in DEPTHS]
    for i, (n, d) in enumerate(rows):
        for j, f in enumerate(FAMILIES):
            r = rec[(f, n, d)]
            rank, ceil, p = (r["qfim_rank_mean"], r["qfim_rank_ceiling"],
                             r["n_params"])
            if rank >= ceil - RANK_TOL:
                marks[(i, j)] = "●"
            elif rank >= p - RANK_TOL:
                marks[(i, j)] = "○"
    grey = np.zeros_like(fill, dtype=bool)
    norm = Normalize(vmin=0.0, vmax=1.0)
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    return _draw(fill, cmap=TEAL, norm=norm, grey=grey, records=rec,
                 cbar_label="rank / ceiling", cbar_ticks=ticks,
                 cbar_ticklabels=["0", "25%", "50%", "75%", "100%"],
                 grey_label="", marks=marks,
                 legend_extra="●  rank = ceiling\n     (saturated)\n"
                              "○  rank = p < ceiling\n     (no redundancy)")


def main() -> None:
    """Write the three heatmaps to ``--outdir``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=Path("charts"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, build in (("p1", p1_figure), ("p2", p2_figure),
                        ("p3", p3_figure)):
        fig = build()
        fig.savefig(args.outdir / f"{name}_heatmap.png", transparent=True)
        plt.close(fig)


if __name__ == "__main__":
    main()
