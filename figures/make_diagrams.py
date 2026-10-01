# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-09-01  Created.
#   2026-09-22  Added A5 (a5_transfer.png): the bare dressed-VQC circuit of
#               Mari et al. 2020, embedding included.
#   2026-10-01  pure-v1 families: A5 circuit-centric (a5_circuit_centric.png),
#               A6 IQP (a6_iqp.png), A7 Farhi-Neven (a7_farhi_neven.png). The
#               Mari A5 diagram is retired with its family.

"""Render a circuit diagram for every ansatz family, straight from its code.

Each diagram is built by calling the family's own ``apply`` on a PennyLane tape
and drawing it with ``qml.draw_mpl``. Nothing here re-describes a circuit by
hand, so the picture always matches the ansatz the scorecard measures.

Run from the repository root (the folder that contains ``vqc_scorecard/``):

    python -m vqc_scorecard.figures.make_diagrams

Options:

    --n N        qubit count for every diagram        (default 4)
    --d D        circuit depth / repeats              (default 2)
    --dpi DPI    output resolution                    (default 200)
    --outdir P   where to write the PNGs              (default: this folder)

One PNG per family is written: ``a1_hea.png``, ``a2_sel.png``,
``a3_reupload.png``, ``a4_discocat.png``, ``a5_circuit_centric.png``,
``a6_iqp.png``, ``a7_farhi_neven.png``. A depth of 2 is used on purpose: at
``d = 1`` the A3 entangler (a CNOT ring placed *between* re-uploads) has no gap
to sit in, so a single-repeat picture would hide it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # no display in headless / server runs
import matplotlib.pyplot as plt

import pennylane as qml

from vqc_scorecard.ansatze import A1, A2, A3, A4, A5, A6, A7

# family -> (record, output filename, one-line label for the figure title)
FAMILIES = [
    (A1, "a1_hea.png",       "A1 · HEA (Kandala 2017)"),
    (A2, "a2_sel.png",       "A2 · SEL + classical embedding (Schuld 2020)"),
    (A3, "a3_reupload.png",  "A3 · Data re-uploading (Perez-Salinas 2020)"),
    (A4, "a4_discocat.png",  "A4 · DisCoCat sentence circuit (Coecke 2020, pre-contraction)"),
    (A5, "a5_circuit_centric.png", "A5 · Circuit-centric classifier, amplitude encoding (Schuld 2020)"),
    (A6, "a6_iqp.png",       "A6 · IQP feature map + VQC (Havlicek 2019)"),
    (A7, "a7_farhi_neven.png", "A7 · QNN with readout qubit (Farhi-Neven 2018)"),
]

# a fixed, reproducible input for the families that encode data (A2, A3, A5-A7)
_RNG = np.random.default_rng(20260811)


def draw_one(ansatz, n: int, d: int, dpi: int, out: Path, title: str) -> None:
    """Draw one family at (n, d) and save it to ``out``."""
    n_enc = ansatz.n_encoder(n, d)
    x = _RNG.uniform(0.0, 2.0 * np.pi, size=n_enc) if n_enc else None
    # parameter values do not affect the drawn structure; use zeros for clarity
    params = np.zeros(ansatz.n_params(n, d))

    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(p):
        ansatz.apply(p, n, d, x=x)
        return qml.expval(qml.PauliZ(0))

    fig, ax = qml.draw_mpl(circuit, decimals=None, style="default")(params)
    # No in-figure title: the deck slide and the filename already identify the
    # family, and an untitled circuit fills the frame better. `title` is kept in
    # the signature for callers that want to annotate the image themselves.
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render circuit diagrams for A1-A7.")
    ap.add_argument("--n", type=int, default=4, help="qubit count (default 4)")
    ap.add_argument("--d", type=int, default=2, help="depth / repeats (default 2)")
    ap.add_argument("--dpi", type=int, default=200, help="output DPI (default 200)")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent,
                    help="output directory (default: this folder)")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering circuit diagrams at n={args.n}, d={args.d} -> {args.outdir}")
    for ansatz, fname, title in FAMILIES:
        # A4 requires an even qubit count (one qubit per word, noun/verb pairs)
        n = args.n
        if ansatz.tag == "A4" and n % 2 == 1:
            n += 1
            print(f"  (A4 needs even n; using n={n})")
        draw_one(ansatz, n, args.d, args.dpi, args.outdir / fname, title)
    print("done.")


if __name__ == "__main__":
    main()
