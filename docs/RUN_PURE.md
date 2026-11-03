# pure-v1: run book (deck v12)

Everything runs in the **pinned scorecard environment** (PennyLane 0.42.3, numpy 2.4.4, Python 3.11), from the **QNN folder** (the one that contains `vqc_scorecard\`). Commands are Windows `cmd`. Every command resumes where it stopped and never duplicates a finished run, so you can interrupt any step and re-issue the same command.

## What changed, in one paragraph

The circuit is now the only trainable component. Each model reads fixed PCA features of the frozen MiniLM vectors (fitted on the training split of each seed), loads them with its own encoding, and reads `sigmoid(a <Z_0> + b)`; `a` and `b` are the only classical parameters. The families are A1 HEA, A2 SEL, A3 re-uploading, A4 DisCoCat block, **A5 circuit-centric (amplitude encoding)**, **A6 IQP feature map + VQC**, **A7 Farhi-Neven**. The Mari A5, the fair-v1 driver and its dressed interface are archived (step 0). Depth and width are experimental factors, not tuned: every (n, d) cell is reported. The hypotheses and their decision rules are in `PREREG_pure_v1.md` and are coded in `vqc_scorecard/tasks/pure_report.py`.

Organisation of the code (unchanged layout, one job per folder):

| Folder / file | Role in pure-v1 |
|----|----|
| `ansatze/a5_circuit_centric.py`, `a6_iqp.py`, `a7_farhi_neven.py` | the three new families (circuit, encoder, DLA generators, deviations) |
| `probes/gates.py` | decomposes multi-wire ops (amplitude encoding) for P6, G5 and task noise |
| `tasks/harness.py` | checkpoints, seeded splits, shared features (moved out of fair.py) |
| `tasks/model.py` | `MODELS` registry and `build_pure_model` |
| `tasks/pure.py` | the protocol and its CLI |
| `tasks/stats.py`, `tasks/pure_report.py` | decision rules, hypotheses H1-H8, result categories |
| `tests/test_pure.py`, `test_tasks.py`, `test_metrics.py` | 214 checks |

## 0. Housekeeping (once, in File Explorer)

The archive copy of every file that pure-v1 replaced is already in `old\fair_v1_archive\vqc_scorecard\` (a full snapshot of package v2.1.0). In the live package those files are now short stubs that raise a clear `ImportError`. Delete the stubs and move the fair-v1 outputs:

1.  Delete `vqc_scorecard\ansatze\a5_transfer.py`
2.  Delete `vqc_scorecard\tasks\fair.py` and `vqc_scorecard\tasks\fair_report.py`
3.  Delete `vqc_scorecard\tests\test_a5.py` and `vqc_scorecard\tests\test_fair.py`
4.  Delete `vqc_scorecard\figures\a5_transfer.png`
5.  Move the folder `vqc_scorecard\results\fair\` to `old\fair_v1_archive\results\fair\`
6.  Move `RUN_FAIR.md`, `RESULTS_fair.md` and `RESULTS_fair2.md` to `old\fair_v1_archive\`
7.  Optional: delete every `__pycache__` folder under `vqc_scorecard\`

`results\scorecard.jsonl` needs no change: it holds A1-A4 descriptors only, and those numbers are still valid (P6 and G5 give identical values for families made of one- and two-qubit gates).

## 1. Check the code (about 10 min)

``` bat
python -m vqc_scorecard.tests
:: expect: all suites: 214/214 passed  (125 descriptor + 32 task machinery + 57 pure-v1)
```

## 2. Freeze the pre-registration (before any protocol run)

``` bat
python -c "from vqc_scorecard.tasks.pure import PROTOCOL; print(PROTOCOL.name, PROTOCOL.digest())"
:: expect: pure-v1 b2833a59e87d
git add -A
git commit -m "pure-v1: freeze pre-registration"
git tag prereg-pure-v1
```

Write the date, the commit hash and the digest into the table at the top of `PREREG_pure_v1.md`, commit again. If the QNN folder is not under git, export `PREREG_pure_v1.md` to PDF and send it to Dr. Ledesma before step 4: a dated third-party copy is the freeze. From here on, a change to `PureProtocol` changes the digest and the report refuses to mix records.

## 3. Descriptors for the new families (A1-A4 are already in the checkpoint)

``` bat
:: static descriptors
python -m vqc_scorecard.cli --metrics P1 P2 --families A5 A6 A7
python -m vqc_scorecard.cli --metrics P3 --families A5 A6 A7 --samples 30
python -m vqc_scorecard.cli --metrics P4 --families A5 A6 A7 --widths 2 3 4 5 --depths 1 --repeats 1 --samples 1
python -m vqc_scorecard.cli --metrics P5 P6 --families A5 A6 A7 --repeats 1 --samples 1

:: trainability grid (H1, H2, H5b and the categories read it)
python -m vqc_scorecard.cli --metrics G1 --families A5 A6 A7 --widths 2 3 4 5 6 7 8 9 10 --depths 1 2 3 4 5 6 7 8 --samples 200 --repeats 3

:: initialisation scale sweep
python -m vqc_scorecard.cli --metrics G6 --families A5 A6 A7 --widths 4 6 8 --depths 5 --samples 200 --repeats 2

:: noise-induced plateau (density matrix, slow; resumes)
python -m vqc_scorecard.cli --metrics G5 --families A5 A6 A7 --widths 4 --depths 1 2 3 4 5 6 7 8 --samples 50 --repeats 2
```

Expected structural results you can check immediately: P4 = 255 for A5 and A6 at n = 4 (full su(16)), P4 = 9 for A7 at n = 4 (3(n-1)); P6 for A5 at n = 4, d = 3 is 26 (12 CRot + 14 Mottonen CNOTs).

## 4. Clean training, IMDb-1k (primary task: every hypothesis reads it)

Run by width so each block finishes in one sitting. Times measured in the cloud on one CPU, one run: about 7-25 s at n = 4, about 70-110 s at n = 8 (A5 is the slowest).

``` bat
python -m vqc_scorecard.tasks.pure status --task imdb1k

:: family arm, 7 models x 3 depths x 10 seeds per width
python -m vqc_scorecard.tasks.pure train --task imdb1k --arm family --widths 4
python -m vqc_scorecard.tasks.pure train --task imdb1k --arm family --widths 6
python -m vqc_scorecard.tasks.pure train --task imdb1k --arm family --widths 8

:: encoding arm (A2 ansatz under amplitude and IQP encodings, n = 4)
python -m vqc_scorecard.tasks.pure train --task imdb1k --arm encoding
```

- About 1 h (n = 4), 2-3 h (n = 6), 5-6 h (n = 8), 15 min (encoding arm).
- Split a long block over sessions with seed slices, e.g. `--seeds 0-4` then `--seeds 5-9`, or by model with `--models A5 A6`.
- The first call of each width also writes the classical references (`results\pure\imdb1k__baselines.jsonl`).

## 5. Clean training, CoLA (reported as the representation floor)

``` bat
python -m vqc_scorecard.tasks.pure train --task cola --arm family --widths 4
python -m vqc_scorecard.tasks.pure train --task cola --arm encoding
:: optional, only if the n = 4 cells are not all Floor:
python -m vqc_scorecard.tasks.pure train --task cola --arm family --widths 6 8
```

## 6. Noise evaluation (n = 4; H7 and the Noise-limited category)

Trained clean models re-evaluated under every channel at the two reported levels; nothing is retrained.

``` bat
python -m vqc_scorecard.tasks.pure noise-eval --task imdb1k --width 4
:: optional sensitivity probe:
python -m vqc_scorecard.tasks.pure noise-eval --task imdb1k --width 4 --levels stress
```

A5 and A6 (and the two encoding-arm models) evaluate one sample at a time on the noisy device (state preparation and IQP phases do not broadcast there), so they take most of this step. Start with `--models A1 A2 A3 A4 A7`, then run the rest.

## 7. Report: verdicts and categories

``` bat
python -m vqc_scorecard.tasks.pure_report --tasks imdb1k cola --fair-archive old\fair_v1_archive\results\fair\imdb1k.jsonl
```

Writes `RESULTS_pure.md` and `vqc_scorecard\results\pure\summary.json`: one table per task with every (model, n, d) cell, its MCC interval, the like-for-like reference, failed seeds, noise drop and **category** (Floor, Trainability-limited, Performing, Noise-limited, Capacity-limited), then the verdict of H1-H8. A hypothesis with a missing cell prints `incomplete`; `status` shows which.

## 8. Figures

``` bat
python -m vqc_scorecard.figures.make_diagrams
python -m vqc_scorecard.figures.make_g_charts --outdir charts
```

## 9. After the runs

Append section 10 of `PREREG_pure_v1.md` (verdict, effect, interval per hypothesis, and any deviation). Send me `RESULTS_pure.md` and `summary.json` and I will fill the results slides of deck v12.
