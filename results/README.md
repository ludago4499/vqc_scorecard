# results/

Every number in the decks and in RESULTS_pure.md comes from the files in this
folder. Nothing is typed by hand.

## Layout

| Path | What it holds | Written by |
|---|---|---|
| `scorecard.jsonl` | Descriptor checkpoint P1-P6, G1-G6. One line per (metric, family, n, d, samples). | `python -m vqc_scorecard.cli` |
| `scorecard.json`, `scorecard.csv` | Same data, rewritten after every run. Includes each family's declared deviations. | same |
| `pure/imdb1k.jsonl`, `pure/cola.jsonl` | pure-v1 training runs. One line per (model, n, d, seed). | `python -m vqc_scorecard.tasks.pure train` (see RUN_PURE.md) |
| `pure/*__baselines.jsonl` | Logistic-regression references on the same splits. | same |
| `pure/imdb1k__noise_eval.jsonl` | Clean-trained models evaluated under noise (n = 4). | same |
| `pure/summary.json` | Nested report: cells, categories, references, hypothesis verdicts. | `python -m vqc_scorecard.tasks.pure_report` |
| `pure/cells.csv`, `pure/references.csv`, `pure/tables.json` | Flat copies of summary.json. Start here. | `python -m vqc_scorecard.figures.export_pure_tables` |
| `fair/` | fair-v1 (archived hybrid protocol, digest 1a0592068421). Reference only, not VQC evidence. | fair harness |
| `g5_finegrid/` | G5 dense noise grid and its figures. | `run_finegrid.py` |
| `task_*.jsonl` | Legacy v10 task axis (TF-IDF PCA(4)). Superseded by pure-v1. | legacy harness |

## Columns of `pure/cells.csv`

| Column | Meaning |
|---|---|
| `task` | `imdb1k` (primary) or `cola` (representation floor) |
| `model` | A1-A7, or an encoding-arm variant `A2:amp`, `A2:iqp` |
| `n`, `d` | qubits, circuit depth |
| `mcc_mean`, `mcc_sd`, `mcc_ci95` | test MCC over seeds; `mcc_ci95` is the half-width of the 95% interval |
| `seeds` | completed seeds (10 in the protocol) |
| `ref_logreg_same_features` | logistic regression on the same PCA features |
| `failed_seeds` | seeds with test MCC < 0.2 |
| `noise_drop_nisq` | MCC lost at the nisq noise level (n = 4 only; empty elsewhere) |
| `gradvar_local` | G1 local-cost gradient variance at this (n, d) |
| `category` | Floor, Trainability-limited, Performing, Noise-limited or Capacity-limited (PREREG section 7) |

## Integrity

Each pure-v1 record carries the protocol digest `b2833a59e87d`. The report
refuses mixed digests. CoLA is incomplete at n = 6 and n = 8 (42 cells); the
report lists the missing cells in `summary.json` under `tasks.cola.missing`.
