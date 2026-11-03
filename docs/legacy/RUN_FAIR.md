# Fair protocol + task noise: run book (v11.1 -> v11.2)

Everything here runs in the **pinned scorecard environment** (PennyLane 0.42.3 /
numpy 2.4.4 / Python 3.11), from the **QNN folder** (the one that contains
`vqc_scorecard\`). Commands are Windows `cmd`.

## What changed, in one paragraph

Every family (A1-A5) is now trained under **one frozen protocol**
(`vqc_scorecard/tasks/fair.py`, `PROTOCOL`, digest printed on every run): the
same frozen MiniLM features, the same dressed interface around the circuit
(trainable `L(384 -> 4)`, `(pi/2) tanh` angles, `<Z_q>` on every qubit,
`L(4 -> 2)`), the same optimiser (Adam 0.05, betas 0.9/0.999, 20 epochs, batch
32), the same small-angle init (`0.01 N(0,1)`), the same class weighting, the
same **10 seeds**, the same **depths {1, 3, 5}** and **n = 4**. 20% of the
training split is held out and the epoch with the best **validation** MCC is the
model (no test-set selection). Two headline results per task: **MCC** and
**macro-F1**. A classical control **C0** (the circuit replaced by a classical
layer) is trained the same way. Nothing can be tuned per family from the command
line, a re-run never duplicates a seed, and the report refuses to rank until
every family x depth x seed is present.

Noise is a separate, reusable package (`vqc_scorecard/noise/`): depolarizing,
bit flip and phase flip, applied after **every gate** with two rates (single-
qubit `p1`, two-qubit `p2 = 10 p1`), at two levels:

| level | p1 (1q gate) | p2 (2q gate) | why |
|---|---|---|---|
| `small` | 1e-4 | 1e-3 | best current devices (~0.1% 2q error) |
| `nisq` | 1e-3 | 1e-2 | textbook NISQ point; inside the G5 critical band |
| `stress` (optional) | 5e-3 | 5e-2 | sensitivity probe, not a hardware claim |

## 0. Once: prerequisites

The frozen feature caches must exist (RUN_COMMANDS.md section 5.0, extraction env):

```bat
dir data\features
:: expect imdb1k__all-MiniLM-L6-v2.npz and cola__all-MiniLM-L6-v2.npz
```

Check the code (about 5 min):

```bat
python -m vqc_scorecard.tests
:: expect: all suites: 194/194 passed  (95 descriptor + 60 A5 + 39 fair/noise; 1 A5 check skipped without torch)
```

## 1. Clean training, the fair comparison (the main result)

One command per task. Each run prints its test MCC / macro-F1 and the selected
epoch. Interrupt at any time; the same command resumes where it stopped.

```bat
python -m vqc_scorecard.tasks.fair train --task imdb1k
python -m vqc_scorecard.tasks.fair train --task cola
```

* 5 families x 3 depths x 10 seeds = 150 quantum runs per task, plus 10 C0 runs.
* Time (measured in the cloud, one CPU): 10-30 s per run -> about 45 min for
  IMDb, about 1 h for CoLA. Your laptop may differ.
* **Do not run two processes on the same task at once** (both would append to
  the same file). The two tasks can run in two terminals in parallel.
* To split a task across sessions, use seed ranges, e.g. `--seeds 0-4` then
  `--seeds 5-9`. Never change anything else.

Check completeness at any time:

```bat
python -m vqc_scorecard.tasks.fair status --task imdb1k
python -m vqc_scorecard.tasks.fair status --task cola
:: every cell must read 10/10
```

## 2. Noise section: the trained models deployed on noisy devices

No retraining: every trained clean model is re-evaluated with each channel at
each level, exact expectation values **and** 1024 shots. Run after step 1 is
complete for the task (a later re-run only adds what is missing).

```bat
python -m vqc_scorecard.tasks.fair noise-eval --task imdb1k
python -m vqc_scorecard.tasks.fair noise-eval --task cola
```

* 150 models x 3 channels x 2 levels = 900 evaluations per task, about 1-2 s
  each -> 20-30 min per task.
* Optional sensitivity probe (adds 450 per task):
  `python -m vqc_scorecard.tasks.fair noise-eval --task imdb1k --levels stress`

## 3. The report and the ranking

```bat
python -m vqc_scorecard.tasks.fair_report --out RESULTS_fair.md
```

Writes `RESULTS_fair.md` and `vqc_scorecard\results\fair\summary.json` (the deck
figures of v11.2 are built from the JSON). It contains, per task: every family x
depth; the headline table at each family's **validation-selected** depth with
95% intervals, next to C0 and two logistic-regression references on the same
splits; the noise tables (MCC, macro-F1, MCC drop, prediction fidelity, flips);
and the **ranking**:

1. **Quality**: per seed, families ranked on the 4 headline numbers (2 tasks x
   MCC, macro-F1); Friedman test; the leader vs everyone by paired Wilcoxon,
   Holm-corrected. Families not significantly worse than the leader are tied.
2. **Noise** (only among the tied): prediction fidelity under NISQ noise,
   same paired test.
3. **Cost** (only if still tied): two-qubit gates, then circuit parameters.

It names the winner and the stage that decided it ("clear winner" only if it
won on quality against every other family), prints each task's ranking alone
(a task stuck at chance, as CoLA may be, only adds noise to the pooled one), and
a weighted sensitivity check. If the data is incomplete it says which cells are
missing and does not rank (`--allow-incomplete` gives a PROVISIONAL look).

## 4. Scaling the winner (after step 3 names it)

Replace `A5` with the winner. Same protocol, only the width changes; C0 is
scaled with it as the classical reference.

```bat
python -m vqc_scorecard.tasks.fair train --task imdb1k --families A5 C0 --width 6
python -m vqc_scorecard.tasks.fair train --task imdb1k --families A5 C0 --width 8
python -m vqc_scorecard.tasks.fair train --task cola   --families A5 C0 --width 6
python -m vqc_scorecard.tasks.fair train --task cola   --families A5 C0 --width 8
python -m vqc_scorecard.tasks.fair noise-eval --task imdb1k --width 6
python -m vqc_scorecard.tasks.fair noise-eval --task imdb1k --width 8
python -m vqc_scorecard.tasks.fair noise-eval --task cola --width 6
python -m vqc_scorecard.tasks.fair noise-eval --task cola --width 8
python -m vqc_scorecard.tasks.fair_report --out RESULTS_fair.md
```

The report adds a scaling table (n = 4, 6, 8 at the validation-selected depth).
About 30 runs per width per task; n = 8 is roughly 2-4x slower per run.
A4 needs even n; A3's protocol width is 4 (other widths are computed and
flagged off-protocol in the descriptor battery, not here).

## 5. Optional: training under noise (expensive)

The noise section above is "trained clean, deployed noisy". Training under
noise answers a different question (can the optimiser adapt to the noise?). It
runs on density matrices, about 2-4 min per run, so do it only for the top two
families of the ranking, at the NISQ level and at their selected depth (read
it from `RESULTS_fair.md`):

```bat
for %C in (depolarizing bitflip phaseflip) do python -m vqc_scorecard.tasks.fair noise-train --task imdb1k --channel %C --level nisq --families A5 A2 --depths 3
for %C in (depolarizing bitflip phaseflip) do python -m vqc_scorecard.tasks.fair noise-train --task cola --channel %C --level nisq --families A5 A2 --depths 3
```

(In a `.bat` file write `%%C`.) 2 families x 10 seeds x 3 channels x 2 tasks =
120 runs, about 5-7 h. The report adds a "trained under noise" table.

## Files

| file | what |
|---|---|
| `vqc_scorecard\results\fair\{task}.jsonl` | clean training records (with the selected parameters) |
| `vqc_scorecard\results\fair\{task}__noise_eval.jsonl` | noise evaluations |
| `vqc_scorecard\results\fair\{task}__noise_train.jsonl` | trained under noise (optional) |
| `vqc_scorecard\results\fair\{task}__baselines.jsonl` | logistic-regression references |
| `RESULTS_fair.md`, `vqc_scorecard\results\fair\summary.json` | report and deck data |

Send me `RESULTS_fair.md` and `summary.json` for v11.2.
