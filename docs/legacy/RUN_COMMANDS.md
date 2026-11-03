# Task-axis run commands (IMDb-1k, CoLA, CoLA rescue, A5)

> **Superseded for the comparison of families by `RUN_FAIR.md`** (fair protocol, task noise, ranking, scaling). Sections 1-4 are the v10 protocol, kept for the record; section 5.0 (feature extraction) is still required.

Run every line from the **QNN folder** (the one that contains `vqc_scorecard\`). Pinned environment: PennyLane 0.42.3 / numpy 2.4.4 / Python 3.11.

Note on the runner: it *appends* to its JSONL and derives seeds from a fixed base (20260811 + i), so re-running a command re-adds the same seeds as duplicate rows (this is why the previous IMDb file reported "13 seeds" for only 10 distinct ones). Each block below moves the old checkpoint aside first, so every file is a clean regenerate. Delete the `.bak.jsonl` files once you have confirmed the new runs, or restore one to roll back. Commands are Windows `cmd`; in PowerShell use `Move-Item`.

## 1. IMDb-1k (10 seeds, depths 1-5) - clean regenerate, removes the duplicates

``` bat
move vqc_scorecard\results\task_imdb1k.jsonl vqc_scorecard\results\task_imdb1k.bak.jsonl
python -m vqc_scorecard.tasks.run --task imdb1k --families A1 A2 A3 A4 --width 4 --depths 1 2 3 4 5 --seeds 10 --epochs 20 --batch 32 --lr 0.05 --shots-eval 1024
python -m vqc_scorecard.tasks.present_task_results --task imdb1k --format md  --out RESULTS_imdb.md
python -m vqc_scorecard.tasks.present_task_results --task imdb1k --format csv --out RESULTS_imdb.csv
```

## 2. CoLA baseline (unweighted BCE, depths 1-3) - documents the majority-class collapse

``` bat
move vqc_scorecard\results\task_cola.jsonl vqc_scorecard\results\task_cola.bak.jsonl
python -m vqc_scorecard.tasks.run --task cola --families A1 A2 A3 A4 --width 4 --depths 1 2 3 --seeds 10 --epochs 20 --batch 32 --lr 0.05 --shots-eval 1024
python -m vqc_scorecard.tasks.present_task_results --task cola --format md  --out RESULTS_cola.md
python -m vqc_scorecard.tasks.present_task_results --task cola --format csv --out RESULTS_cola.csv
```

## 3. CoLA rescue (class-weighted BCE + longer training)

Writes to a **separate** checkpoint so the rescue never mixes with the baseline (`present_task_results` groups only by family/route/depth). Requires the updated `train.py` and `run.py` (the `--class-weight` flag).

``` bat
python -m vqc_scorecard.tasks.run --task cola --families A1 A2 A3 A4 --width 4 --depths 1 2 3 --seeds 10 --epochs 40 --batch 32 --lr 0.05 --shots-eval 1024 --class-weight balanced --out vqc_scorecard\results\task_cola_rescue.jsonl
python -m vqc_scorecard.tasks.present_task_results --task cola --checkpoint vqc_scorecard\results\task_cola_rescue.jsonl --format md  --out RESULTS_cola_rescue.md
python -m vqc_scorecard.tasks.present_task_results --task cola --checkpoint vqc_scorecard\results\task_cola_rescue.jsonl --format csv --out RESULTS_cola_rescue.csv
```

## 4. (optional) CoLA rescue, stronger optimization - another separate file

``` bat
python -m vqc_scorecard.tasks.run --task cola --families A1 A2 A3 A4 --width 4 --depths 1 2 3 --seeds 10 --epochs 60 --batch 32 --lr 0.1 --shots-eval 1024 --class-weight balanced --out vqc_scorecard\results\task_cola_rescue_lr10.jsonl
python -m vqc_scorecard.tasks.present_task_results --task cola --checkpoint vqc_scorecard\results\task_cola_rescue_lr10.jsonl --format md --out RESULTS_cola_rescue_lr10.md
```

## What to expect

- IMDb: quantum F1 peaks near depth 3 (\~0.68), below the PCA(4) logreg baseline (0.722) and well below full TF-IDF (0.833). Depth 4-5 adds parameters without F1 (redundancy).
- CoLA baseline: A1/A2/A3 collapse to majority prediction (MCC 0.000); the PCA(4) baseline collapses identically, so the bottleneck is the encoding, not the circuits.
- CoLA rescue: class weighting moves the loss-optimal constant from p=0.69 to p=0.50, so the exact collapse is broken. Expect recall/precision to move off (1.0, 0.69); MCC should stay low on the embedding route (PCA(4) has no acceptability signal), and A4 compositional (trainable word boxes, not PCA) is the family most likely to reach a small positive MCC. That contrast is the reportable result.

Send me `RESULTS_cola_rescue.md` (+ .csv) and I will fill the pending rescue slide in v10.

## 5. A5 - hybrid transfer learning, dressed VQC head (Mari et al. 2020)

Two environments. The frozen encoder runs once in `.venv-extract` (torch + transformers); everything else runs in the pinned scorecard environment, which never imports torch. Commands are Windows `cmd` from the QNN folder; inside a `.bat` file write `%%L` / `%%N` instead of `%L` / `%N`.

### 5.0 One time: model files and the feature caches (extraction env)

`models\all-MiniLM-L6-v2\` must hold `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt`.

``` bat
.venv-extract\Scripts\activate
pip install -r vqc_scorecard\requirements-extract.txt
python -m vqc_scorecard.tasks.get_features --task imdb1k
python -m vqc_scorecard.tasks.get_features --task cola
deactivate
```

Writes `data\features\imdb1k__all-MiniLM-L6-v2.npz` (2000 reviews, \~4 min on CPU) and `data\features\cola__all-MiniLM-L6-v2.npz` (1524 sentences, \~15 s).

### 5.1 Checks (pinned env)

``` bat
python -m vqc_scorecard.tests
```

Expect `all suites: 155/155 passed` (95 descriptor + 60 A5; the tiny-BERT extraction check is skipped here because the pinned env has no torch).

### 5.2 Descriptors P1-P6, G1-G6 for A5 (pinned env; appends to results\scorecard.jsonl)

Same recipe as A1-A4 (README, build recipe). A5's protocol grid is n in {4,6,8}, d in {1,3} (deck slide 6); other points are computed and marked off_protocol.

``` bat
python -m vqc_scorecard.cli --metrics P1 P2 --families A5
python -m vqc_scorecard.cli --metrics P3 --samples 30 --families A5
python -m vqc_scorecard.cli --metrics P4 --families A5 --widths 2 3 4 5 --depths 1 --repeats 1 --samples 1
python -m vqc_scorecard.cli --metrics P5 P6 --families A5 --repeats 1 --samples 1
python -m vqc_scorecard.cli --metrics G1 --families A5 --widths 2 3 4 5 6 7 8 9 10 --depths 1 2 3 4 5 6 7 8 --samples 200 --repeats 3
python -m vqc_scorecard.cli --metrics G6 --families A5 --widths 4 6 8 --depths 5 --samples 200 --repeats 2
python -m vqc_scorecard.cli --metrics G5 --families A5 --widths 4 --depths 1 2 3 4 5 6 7 8 --samples 50 --repeats 2
python -m vqc_scorecard.cli --metrics G5 --families A5 --widths 2 4 6 --depths 4 --samples 50 --repeats 2
```

Read P4 against so(2\^n) and P1 against the real-Haar KL (README 7.8).

### 5.3 Task axis (pinned env; A5 writes to its own checkpoints)

(a) A5 under the A1-A4 protocol: the comparable run.

``` bat
python -m vqc_scorecard.tasks.run --task imdb1k --families A5 --features frozen --width 4 --depths 1 2 3 4 5 --seeds 10 --epochs 20 --batch 32 --lr 0.05 --shots-eval 1024 --out vqc_scorecard\results\task_imdb1k_a5.jsonl
```

(b) Pre-layer ablation (fixed PCA(4) instead of the trainable layer), same file:

``` bat
python -m vqc_scorecard.tasks.run --task imdb1k --families A5 --features frozen --prelayer pca --width 4 --depths 1 2 3 4 5 --seeds 10 --epochs 20 --batch 32 --lr 0.05 --shots-eval 1024 --out vqc_scorecard\results\task_imdb1k_a5.jsonl
```

(c) Class II control: A1-A4 on the same frozen features.

``` bat
python -m vqc_scorecard.tasks.run --task imdb1k --families A1 A2 A3 A4 --features frozen --width 4 --depths 1 2 3 4 5 --seeds 10 --epochs 20 --batch 32 --lr 0.05 --shots-eval 1024 --out vqc_scorecard\results\task_imdb1k_frozen.jsonl
```

(d) The paper's own settings (Table 3, ants/bees: q = 6, 30 epochs, batch 4, lr 4e-4 decayed x0.1 every 10 epochs, torch Adam betas). About 4 min per run.

``` bat
python -m vqc_scorecard.tasks.run --task imdb1k --families A5 --features frozen --width 4 --depths 6 --seeds 10 --epochs 30 --batch 4 --lr 0.0004 --adam-beta2 0.999 --lr-step 10 --lr-gamma 0.1 --shots-eval 1024 --out vqc_scorecard\results\task_imdb1k_a5_paper.jsonl
```

(e) Truncation sweep A -\> A' (Fig. 8 of the paper): hidden state 0..6 at d = 3.

``` bat
for %L in (0 1 2 3 4 5 6) do python -m vqc_scorecard.tasks.run --task imdb1k --families A5 --features frozen --layer %L --width 4 --depths 3 --seeds 5 --shots-eval 1024 --out vqc_scorecard\results\task_imdb1k_a5_layers.jsonl
```

(f) Head-size axis of the deck protocol (n_head in {4,6,8}, L_head in {1,3}).

``` bat
for %N in (4 6 8) do python -m vqc_scorecard.tasks.run --task imdb1k --families A5 --features frozen --width %N --depths 1 3 --seeds 10 --shots-eval 1024 --out vqc_scorecard\results\task_imdb1k_a5_heads.jsonl
```

(g) CoLA: the same commands with `--task cola` (MCC is the headline) and `cola` in the output names. The CoLA A5 run is the one that tests the representation-floor reading against a strong pre-trained representation.

Reports (the presenter splits groups by representation, layer and pre-layer):

``` bat
python -m vqc_scorecard.tasks.present_task_results --task imdb1k --checkpoint vqc_scorecard\results\task_imdb1k_a5.jsonl --format md --out RESULTS_imdb_a5.md
python -m vqc_scorecard.tasks.present_task_results --task imdb1k --checkpoint vqc_scorecard\results\task_imdb1k_a5_layers.jsonl --format md --out RESULTS_imdb_a5_layers.md
```

What to expect (single-seed cloud check, IMDb-1k, n = 4, final layer): (a) at d = 3 reached F1 0.754 / MCC 0.589; (d) at d = 6 reached F1 0.788 / MCC 0.580. Both are above the A1-A4 plateau (\~0.68) and the TF-IDF PCA(4) baseline (0.722). The frozen baselines in the same records (logreg on the full encoder vector and on its PCA(4)) are the references A5 must be read against, not TF-IDF.
