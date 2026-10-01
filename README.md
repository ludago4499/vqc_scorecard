# vqc_scorecard

Descriptor battery and task harness for VQC/VQN architectures. Version 3.0.0
(pure-v1: families A1-A7, circuit-only training).

Written in Simplified Technical English. Short sentences. One idea per sentence.

---

## 1. Purpose

The package measures properties of an ansatz **before** the model is trained.
The study then asks one question. Do those properties predict how the model
performs on a task?

There is no 0-100 score in this package. The study is a regression, and a
monotone rescaling adds nothing to a regression. It also destroys information at
the clipped ends. Raw values only.

---

## 2. Layout

Each directory has one job. Each file has one reason to change.

```
vqc_scorecard/
  ansatze/      one file per family. Circuits only.
  probes/       expensive primitives. Physics only.
  metrics/      one file per metric. Definitions and deviations.
  core/         grid, seeding, checkpoint, output. No physics.
  figures/      circuit diagrams, drawn from the ansatze.
  noise/        gate-noise models shared by the task axis.
  tasks/        datasets, features, models, the pure-v1 protocol and its report.
  tests/        checks with known answers.
  cli.py        the descriptor sweep entry point.
```

Data flows from top to bottom. A probe never decides what a number means. A
metric never decides how a sweep runs. The core never knows about circuits.

### Why this layout, and not the old one

The old folder named its files after **when they ran**: `metrics_part1`,
`metrics_part2`, `build_scorecard`, `make_figures`. Stage names always rot,
because stages get added and nothing tells you where a new thing goes.

Worse, one file held four unrelated jobs. `metrics_part1.py` had to change if
you added a family, **or** changed an estimator, **or** changed a scale. Three
reasons to edit one file means you edit it every week and break it.

---

## 3. How to run

Install the dependencies first (pinned in `requirements.txt` at the repo root):

```
pip install -r requirements.txt
```

Run one metric over the declared grid:

```
python -m vqc_scorecard.cli --metrics P2
```

Run more than one metric, for one family, with a time limit:

```
python -m vqc_scorecard.cli --metrics P1 P2 --families A1 --budget 140
```

Check the code before you trust a number:

```
python -m vqc_scorecard.tests
```

The sweep writes one line per finished measurement to
`results/scorecard.jsonl`. Run the command again to continue after a stop.
Finished measurements are skipped.

---

## 4. Metrics available

| Tag | Name | Reference kind | Status |
|---|---|---|---|
| P1 | Expressibility, KL against Haar | sampled | ported |
| P2 | Entangling capability, Meyer-Wallach | **analytic** | new |
| P3 | Effective dimension, QFIM rank | **analytic** | ported |
| P4 | Dynamical Lie algebra dimension | **analytic** | ported |
| P5 | Parameter count | - | ported |
| P6 | Two-qubit gate count | - | ported |
| G1 | Gradient variance (G1 onset, G2 retention, G3 shot margin, G4 locality) | - | **ported; gradient verified vs autodiff, slope vs the 2-design value ln 2** |
| G5 | Noise-induced barren plateau (per-gate depolarizing) | - | **new; p = 0 control tied to G1 in the tests; full grid pending compute** |
| G6 | Initialisation sensitivity (Haug scale sweep + QFIM rank) | **analytic** | **new; replaces the invalid old G6, a = 1 rank tied to P3 in the tests** |

The `reference kind` column matters. P1 needs a **sampled** floor, because its
estimator is biased and the bias must be measured. P2, P3 and P4 need an
**analytic** value, because each has a closed-form reference: the Haar
expectation of the Meyer-Wallach measure for P2, the state-manifold dimension
`2^(n+1)-2` for P3, and the full-algebra dimension `4^n-1` for P4. P5 and P6 are
exact counts with no reference. Forcing one scheme on every metric is what
produced the unsourced constants of the old scorecard. Each metric declares its
own kind.

P3 also reports **redundancy** `R = (M - D_C) / M = 1 - rank/p` in the notation
of Haug's Table I, where `M = p` is the parameter count (P5) and `D_C` is the
effective dimension (the P3 rank at random angles). `R = 0` means every angle is
used, `R -> 1` means most are redundant. Read `R` at `d = 1`: at deeper depth the
rank saturates the manifold while `p` keeps growing, so `R` then also counts
over-parameterisation rather than generator redundancy alone.

P5 and P6 are cost controls, not capabilities. P5 is `n_params`; P6 counts the
two-qubit gates on the circuit tape, so it follows any change to a family's
`apply()` with no separate formula.

### Build recipe (reproduce every number)

    # static descriptors P1-P6
    python -m vqc_scorecard.cli --metrics P1 P2                 # full n x d grid, 5000 draws
    python -m vqc_scorecard.cli --metrics P3 --samples 30       # 30 QFIM points, adds redundancy
    python -m vqc_scorecard.cli --metrics P4 --widths 2 3 4 5 --depths 1 --repeats 1 --samples 1
    python -m vqc_scorecard.cli --metrics P5 P6 --repeats 1 --samples 1

    # trainability grid - one measured grid feeds the G1, G2, G3 and G4 readings
    python -m vqc_scorecard.cli --metrics G1 --widths 2 3 4 5 6 7 8 9 10 --depths 1 2 3 4 5 6 7 8 --samples 200 --repeats 3

    # G6 init-scale sweep (statevector, cheap)
    python -m vqc_scorecard.cli --metrics G6 --widths 4 6 8 --depths 5 --samples 200 --repeats 2

    # G5 noise sweep (density matrix, EXPENSIVE - the checkpoint resumes)
    # pilot arm (measured): A1 + A4 depth axis at n = 4
    python -m vqc_scorecard.cli --metrics G5 --families A1 A4 --widths 4 --depths 1 2 3 4 5 6 7 8 --samples 50 --repeats 2
    # full protocol grid (PENDING COMPUTE): all families, depth axis + width arm
    python -m vqc_scorecard.cli --metrics G5 --widths 4 --depths 1 2 3 4 5 6 7 8 --samples 50 --repeats 2
    python -m vqc_scorecard.cli --metrics G5 --widths 2 4 6 --depths 4 --samples 50 --repeats 2

    # checks
    python -m vqc_scorecard.tests.test_metrics                  # 125/125
    python -m vqc_scorecard.tests.test_tasks                    # 32/32 (skips without torch / the CSV)
    python -m vqc_scorecard.tests.test_pure                     # 57/57
    python -m vqc_scorecard.tests                               # all suites, 214/214

The gradient grid is a single metric, `G1`, that measures the variance of a
cost derivative over random angles at every grid point, for a global and a
local cost. **Four** readings come from that one grid, with no extra sweep:

* **G1** `bp_onset_slope(widths, variances)` - the decay rate `alpha` of
  `Var ~ exp(-alpha n)`. The published reference is the 2-design value
  `ln 2 = 0.693`, which for these families follows from the exact variance
  formula of Ragone et al. 2024 (Larocca et al. 2025, Eq. 12) with a universal
  DLA and a Pauli-string cost. Measured: 0.69-0.72 at `d = 5` for all four.
* **G2** `depth_retention(depths, variances)` - the variance-vs-depth curve
  and the depth at which it falls under the shot floor `1/1024`.
* **G3** `shot_floor_margin` / `crossing_width` (`metrics/g3_shot_floor.py`) -
  how many times over the finite-shot floor (Larocca 2025, Box 3) the
  variance sits at every grid point, and the interpolated width where it
  crosses. Replaces the old single-point G3.
* **G4** `locality_gap` (`metrics/g4_locality.py`) - the width-decay rates of
  the global and the local cost and their gap, read at shallow depth, the
  regime of Cerezo et al. 2021 (the old code compared them at `d = n`, deep in
  the plateau, and found no gap). Scope caveat per Larocca 2025, Box 10.

`G5` and `G6` are separate measured metrics. G5 sweeps per-gate depolarizing
noise on a density matrix (deviation from Wang et al.'s per-layer channel,
declared in the metric record) with the depth axis the theorem names;
`noise_crossing` interpolates `p_crit` and an uncrossed floor is `None`, never
a score. G6 sweeps the initialisation scale `theta = a * theta_random` (Haug
et al. 2021) and reports the gradient **second moment** and the **QFIM rank**
per scale; fixed seeded pre- and post-layers remove the two structural zeros
that invalidated the old G6 (cost eigenstate at `|0...0>`; Clifford-diagonal
observable at `theta = 0`).

Verification in place: the gradient is a central finite difference (the
two-term shift is wrong for A4's `CRZ`), checked against PennyLane autodiff on
all four families; the variance is averaged over several angles; the onset
slope has its published reference (ln 2, above); G5's `p = 0` control must
match G1's statevector value; G6's `a = 1` rank must match P3. All are
permanent tests.

Each run appends to `results/scorecard.jsonl` (the checkpoint) and rewrites the
`.json` and `.csv`. Re-running a metric skips grid points already in the
checkpoint; to recompute one after a definition change, drop its lines from the
`.jsonl` first (they are keyed by `metric, family, n, d, samples`).

---

## 5. Adding a metric

Three steps. Nothing else changes.

1. Write `metrics/pN_name.py`. Copy the shape of `p2_entangling.py`.
2. Build a `Metric` record in it. Declare the source and every deviation.
3. Import the record in `metrics/__init__.py` and add it to `REGISTRY`.

If the metric needs a new primitive, put the primitive in `probes/`. Do not put
it in the metric file. The next metric will need it too.

A metric that is written but not registered cannot run. That is deliberate. It
stops a half-finished metric from reaching a result table by accident.

---

## 6. What changed against the old code

| Defect in `HEA_scorecard/` | Fix here |
|---|---|
| One generator seeded at import. Results depended on call order. | `core/seeding.py` derives every generator from the identity of the work. |
| A1 defined in three files. | One definition in `ansatze/a1_hea.py`. |
| `s_entangling` used the anchor `0.4` at every `n`. | `0.4` is the exact Haar value at `n = 2` only. `probes/haar.expected_meyer_wallach` gives the correct value at every `n`. |
| No library versions recorded. | `core/io.environment()` writes them into every result file. |
| A long run could not stop. | `core/checkpoint.py` resumes. |
| No tests. | 28 checks against closed-form answers. |

---

## 7. Known limits. Read before publishing

### 7.1 P1 is blind to a terminal gate layer

A fixed unitary at the end of a circuit cancels inside the fidelity, because
`|<a|U*U|b>|^2 = |<a|b>|^2`.

This is visible in the results. A2 and A3 give **exactly** the same P1 at
`d = 1`, at every width, while their P2 differs by the full range of the
measure. Do not read the equality as an error.

### 7.2 P2 cannot see multipartite structure

The Meyer-Wallach measure is built from single-qubit reduced states only. A GHZ
state and a W state have different entanglement structure and can give the same
`Q`. P2 is not a complete description of entanglement.

### 7.3 The encoder input is held fixed

A2 and A3 encode data. The sweep holds the input at one fixed vector, so the
metrics describe the ansatz and not the encoder. Use `--embedding random` to
average over inputs instead. The two settings answer different questions. Do not
mix them in one table.

### 7.4 A3 at d = 1 has no entangler

The CNOT ring of A3 is placed **between** re-uploads. At `d = 1` there is no
gap, so there is no entangler, and the circuit is a product state. `P2 = 0`
exactly. This is correct behaviour and not a defect, but it means the `d = 1`
column of A3 is a product-state control and not a re-uploading model.

### 7.5 A4 is not contracted

A real DisCoCat cup needs post-selection. The package queues the `CNOT` and the
`H` of the cup, and no measurement. A4 therefore measures the sentence state
**before** contraction. State this in any write-up.

### 7.6 P3 rank is threshold-defined and derivative-estimated

The QFIM rank counts eigenvalues above a fixed relative threshold, `1e-5`. A
rank is only meaningful next to its threshold, so report both. The state
derivative is a central finite difference with step `1e-3`, not an exact
parameter shift; its `O(step^2)` error is near `1e-6`, well below the threshold,
and is checked against the closed-form single-rotation QFIM in the tests.

### 7.7 P4 is exact but exponential, and does not separate A1 to A4

The DLA dimension is exact and needs no sampling, but the closure is exponential
in `n` whenever the algebra is full, so the exact value is reachable only to
about `n = 5`. Above `DIM_CAP` the value is a lower bound, flagged by
`dla_capped`. More important for reading the result: **all four families in this
package have a universal DLA**, `4^n - 1`, so P4 does not rank them. That is
itself the finding. None of A1 to A4 escapes the barren-plateau regime by its
structure; the single-qubit-complete rotation set plus a connected entangler
lifts every one of them to the full algebra. Discrimination has to come from P1,
P3 and the gradient series, not from P4.

A7 is the exception; see 7.8.

### 7.8 A5-A7: the pure-v1 families (2026-10-01)

The Mari et al. 2020 transfer head (old A5) is retired. Its circuit was fine,
but the model around it trained a 384 -> n classical layer and an n -> 2
classical layer (about 1550 parameters against 12-40 angles), so it broke the
pure-v1 rule that the circuit is the only trainable part. The module and the
fair-v1 protocol are archived unchanged in `old/fair_v1_archive/`.

* **A5 circuit-centric (Schuld et al. 2020).** Amplitude encoding of `2**n`
  features, code blocks of `Rot` and controlled `CRot` (ranges 1 and 3), final
  `Rot` on qubit 0. A2 is PennyLane's simplification of the same paper (fixed
  CNOTs, angle encoding), so A2 vs A5 isolates exactly the two things the paper
  proposes. P6 counts the amplitude encoding as its Mottonen CNOTs (`2**n - 2`
  for `n >= 3`): 14 at `n = 4`, 254 at `n = 8`. That is the honest NISQ price
  of a logarithmic encoding, and the noise model charges it the same way.
* **A6 IQP (Havlicek et al. 2019).** The second-order ZZ feature map applied
  twice, then local `RY RZ` layers and a CZ chain. Interaction graph: the
  nearest-neighbour chain (the paper runs `n = 2`).
* **A7 Farhi-Neven (2018).** Wire 0 is a readout qubit in `|1>`; data on wires
  `1..n-1`; layers of `exp(i t Z_j X_0)` and `exp(i t X_j X_0)`; a fixed
  `RX(pi/2)` maps the paper's `Y` readout onto `Z_0`, so the package's local
  cost is the paper's readout. **Its DLA is `3(n - 1)`**, a direct sum of
  `n - 1` copies of su(2) (tested in closed form). It is the only family whose
  structure rules out the exponential-DLA barren plateau, and by the same token
  the one most exposed to efficient classical simulation (Lie-algebraic
  simulation of small-DLA circuits). Read its trainability together with that.

A1-A6 all close on `su(2**n)` (tested at `n = 2, 3`).

---

## 8. Output files

| File | Content |
|---|---|
| `results/scorecard.jsonl` | One line per finished measurement. The checkpoint. Do not delete during a sweep. |
| `results/scorecard.json` | Records, the environment, and the declared deviations of every metric and family used. |
| `results/scorecard.csv` | The same records, flat, for a correlation matrix. |

---

## 9. Circuit diagrams

Each family has a one-page circuit diagram, drawn from the **same** `apply` the
sweep runs. The picture can never drift from the circuit that produced a number,
because nothing here re-describes a circuit by hand. The renderer lives in
`figures/`.

Regenerate every diagram:

```
python -m vqc_scorecard.figures.make_diagrams
```

This writes `a1_hea.png`, `a2_sel.png`, `a3_reupload.png`, `a4_discocat.png`,
`a5_circuit_centric.png`, `a6_iqp.png` and `a7_farhi_neven.png` into
`vqc_scorecard/figures/`. Defaults are `n = 4`, `d = 2`, 200 dpi. Override
any of them:

```
python -m vqc_scorecard.figures.make_diagrams --n 4 --d 3 --dpi 300 --outdir some/dir
```

Read before trusting a picture:

- `d = 2` is the default on purpose. At `d = 1` the A3 entangler — a CNOT ring
  placed **between** re-uploads — has no gap to sit in, so a single-repeat
  picture would hide it (the same effect as §7.4). Two repeats show it.
- A4 needs an **even** qubit count (one qubit per word, noun/verb pairs). An odd
  `--n` is bumped up by one for that family only.
- Angle values do not change the drawn structure, so they are set to zero and
  hidden (`decimals=None`). The diagrams show gate **structure**, not a trained
  state. A4 is the pre-contraction unitary circuit (§7.5): the cups appear as
  their `CNOT` + `H`, with no measurement.
- Needs `pennylane` and `matplotlib`: `pip install "pennylane==0.42.3" matplotlib`.

These PNGs are embedded in the deck's architecture slides
(Part 1 · Architectures) — rebuilding the diagrams and re-running the deck build
keeps the two in sync.


---

## 10. Task protocol pure-v1 (2026-10-01)

The circuit is the only trainable component. Fixed PCA features of a frozen
encoder feed the family's own encoding; the readout is `sigmoid(a <Z_0> + b)`;
`a` and `b` are the only classical parameters. See `tasks/pure.py` (protocol),
`tasks/model.py` (`MODELS`, `build_pure_model`), `tasks/pure_report.py`
(pre-registered hypotheses H1-H8, decision rules in `tasks/stats.py`, result
categories). The run order is in `RUN_PURE.md` at the repository root.
