"""Deck-styled G5 fine-grid figures, matching vqc_scorecard/figures/make_g_charts.py
house style. Renders at the deck's chart aspect (~2.39:1) so they drop into the
slide-33 image box cleanly."""
import sys, json
sys.path.insert(0, "/mnt/user-data/uploads/QNN")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from vqc_scorecard.metrics.g5_noise import noise_crossing
from vqc_scorecard.metrics.g1_gradient import SHOTS_FLOOR

# ---- house style (from make_g_charts.py) ----
COLOR = {"A1": "#0f172a", "A2": "#0891b2", "A3": "#b45309", "A4": "#b91c1c"}
LABEL = {"A1": "A1 HEA", "A2": "A2 SEL", "A3": "A3 re-up", "A4": "A4 DisCoCat"}
MUTED, GRID, AXIS = "#94a3b8", "#e2e8f0", "#475569"
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 100, "font.size": 14,
    "axes.edgecolor": MUTED, "axes.labelcolor": AXIS, "axes.labelsize": 15,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
FAMS = ["A1", "A2", "A3", "A4"]
DEPTHS = list(range(1, 9))
ANCHOR_LV = [0.0, 1e-3, 1e-2, 1e-1]
NEWK = [0.017783, 0.031623, 0.056234]

# ---- data ----
recs = [json.loads(l) for l in open("/mnt/user-data/uploads/QNN/vqc_scorecard/results/scorecard.jsonl") if l.strip()]
anchor = {}
for r in recs:
    if r["metric"] == "G5" and r["n"] == 4 and r["family"] in FAMS and r["d"] in DEPTHS:
        anchor[(r["family"], r["d"])] = {p: r[f"gradvar_global_s{i}_mean"] for i, p in enumerate(ANCHOR_LV)}
new = json.load(open("/home/claude/g5_new_levels.json"))
newd = {(rec["family"], rec["d"]): {float(p): v["gradvar_mean"] for p, v in rec["levels"].items()} for rec in new}

def curve(fam, d):
    m = dict(anchor[(fam, d)]); m.update(newd[(fam, d)])
    ps = sorted(pp for pp in m if pp > 0)
    return ps, [m[pp] for pp in ps]

# depth ramp: slate light -> navy (on-brand, monochrome sequential)
ramp = LinearSegmentedColormap.from_list("slate", ["#b9c2cf", "#5b6b82", "#0f172a"])
dcol = {d: ramp(0.12 + 0.88 * (d - 1) / (len(DEPTHS) - 1)) for d in DEPTHS}
FLOOR_CLAMP = 3e-9  # plot floor for exact zeros

# ================= FIGURE 1: Var vs p, 2x2, depth-ramped =================
fig, axes = plt.subplots(2, 2, figsize=(13.0, 5.45), sharex=True, sharey=True)
for ax, fam in zip(axes.ravel(), FAMS):
    ax.axvspan(1e-2, 1e-1, color="#f1a15a", alpha=0.10, lw=0, zorder=0)
    for d in DEPTHS:
        ps, vs = curve(fam, d)
        vs = [max(v, FLOOR_CLAMP) for v in vs]
        ax.plot(ps, vs, "-", color=dcol[d], lw=2.0, zorder=2)
        for pp, vv in zip(ps, vs):
            is_new = any(abs(pp - k) < 1e-6 for k in NEWK)
            if is_new:
                ax.plot(pp, vv, "D", color=dcol[d], ms=4.5, mec="white", mew=0.6, zorder=3)
            else:
                ax.plot(pp, vv, "o", color=dcol[d], ms=4.0, zorder=3)
    ax.axhline(SHOTS_FLOOR, ls=":", lw=1.6, color=MUTED, zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"{LABEL[fam]}  (n = 4)", fontsize=13.5, color=AXIS)
    ax.set_xlim(8e-4, 1.3e-1)
axes[0, 0].text(1.05e-2, SHOTS_FLOOR * 1.6, "shot floor", color=MUTED, fontsize=10.5)
axes[1, 1].text(0.97, 0.93, "shaded = the decade\nthe grid skipped", transform=axes[1, 1].transAxes,
                ha="right", va="top", fontsize=10.5, color="#b45309")
for ax in axes[1]:
    ax.set_xlabel("per-gate depolarizing  p")
for ax in axes[:, 0]:
    ax.set_ylabel("Var[∂C] global")
# depth colorbar (right), + marker legend on top
fig.subplots_adjust(left=0.065, right=0.9, top=0.85, bottom=0.115, hspace=0.46, wspace=0.06)
cax = fig.add_axes([0.925, 0.115, 0.015, 0.735])
sm = ScalarMappable(norm=Normalize(0.5, 8.5), cmap=ramp)
cb = fig.colorbar(sm, cax=cax, ticks=DEPTHS)
cb.set_label("circuit depth  d", color=AXIS, fontsize=12.5)
cb.ax.tick_params(labelsize=11, color=MUTED, labelcolor=AXIS)
cb.outline.set_edgecolor(MUTED)
mk = [Line2D([0], [0], marker="o", color=AXIS, ls="none", ms=5, label="original level"),
      Line2D([0], [0], marker="D", color=AXIS, mec="white", ls="none", ms=5, label="new interior level")]
fig.legend(handles=mk, loc="upper center", ncol=2, fontsize=11, bbox_to_anchor=(0.48, 0.995))
fig.savefig("/home/claude/deck_g5_var_vs_p.png", facecolor="white")
plt.close(fig)

# ================= FIGURE 2: p_crit vs depth, dense vs coarse =================
fig2, ax2 = plt.subplots(figsize=(13.0, 5.45))
for fam in FAMS:
    dense, coarse = [], []
    for d in DEPTHS:
        ps, vs = curve(fam, d)
        dense.append(noise_crossing(ps, vs))
        coarse.append(noise_crossing([1e-2, 1e-1], [anchor[(fam, d)][1e-2], anchor[(fam, d)][1e-1]]))
    dd = [d for d, v in zip(DEPTHS, dense) if v]; dv = [v for v in dense if v]
    cd = [d for d, v in zip(DEPTHS, coarse) if v]; cv = [v for v in coarse if v]
    ax2.plot(dd, dv, "-o", color=COLOR[fam], lw=2.6, ms=8, label=f"{LABEL[fam]}  (dense)", zorder=3)
    ax2.plot(cd, cv, "--x", color=COLOR[fam], lw=1.8, ms=8, alpha=0.45,
             label=f"{LABEL[fam]}  (2-pt 0.01/0.1)", zorder=2)
ax2.axhspan(1e-3, 1e-2, color=MUTED, alpha=0.07, lw=0)
ax2.text(8.02, 3.1e-3, "reported NISQ\n2q error band", color=MUTED, fontsize=11, va="center")
ax2.set_yscale("log")
ax2.set_xlabel("circuit depth  d"); ax2.set_ylabel("p_crit   (Var crosses the 1024-shot floor)")
ax2.set_xticks(DEPTHS); ax2.set_xlim(0.7, 9.6)
ax2.legend(loc="upper right", ncol=2, fontsize=11.5)
fig2.subplots_adjust(left=0.075, right=0.985, top=0.95, bottom=0.115)
fig2.savefig("/home/claude/deck_g5_pcrit.png", facecolor="white")
plt.close(fig2)
print("wrote deck_g5_var_vs_p.png, deck_g5_pcrit.png")
from PIL import Image
for f in ["deck_g5_var_vs_p.png", "deck_g5_pcrit.png"]:
    print(f, Image.open("/home/claude/" + f).size)
