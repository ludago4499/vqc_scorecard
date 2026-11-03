# Pre-registration: pure-v1 (VQC/VQN benchmark, circuit-only training)

Status: **DRAFT v1, not frozen.** Freeze = fill the table below and commit before the first protocol run.

| Field | Value |
|---|---|
| Freeze date | (fill) |
| Git commit | (fill) |
| Protocol digest | `b2833a59e87d` (`python -c "from vqc_scorecard.tasks.pure import PROTOCOL; print(PROTOCOL.digest())"`) |
| Package version | vqc_scorecard 3.0.0 |
| Test suite | 214/214 (`python -m vqc_scorecard.tests`) |

## 0. Why a new protocol

fair-v1 (digest `1a0592068421`) gave every family a trainable classical input layer L(384 -> n) and output layer L(n -> 2), about 1550 classical parameters around 12 to 40 circuit angles. Evidence that the dressing carried the learning: the five families tied (Friedman p = 0.64), the classical control C0 (circuit replaced by a classical layer) sat inside their confidence intervals, depth moved MCC by at most 0.03 and width showed no trend. fair-v1 is kept as the archived hybrid reference (`old/fair_v1_archive`), not as VQC evidence. The legacy v10 runs (TF-IDF PCA(4), 2-parameter head) already met the pure criteria; pure-v1 keeps their head and changes the feature source to frozen MiniLM.

## 1. Framing

We do not test for quantum advantage. We characterize how architectural and representational hyperparameters of variational quantum circuits determine performance, trainability and noise robustness when the circuit is the only trainable component. Classical models are reported as reference points, not competitors.

## 2. Inclusion criteria (every model)

1. Trainable parameters only in the circuit, plus the readout scale a and bias b (2 classical parameters).
2. Classical preprocessing fixed and unsupervised: PCA of frozen all-MiniLM-L6-v2 vectors, fitted on the training documents of each seed only.
3. Fixed readout: p(y = 1) = sigmoid(a <Z_0> + b). For A7, <Z_0> is the paper's Y readout through a fixed RX(pi/2).

## 3. Families

| Tag | Model | Encoding | Source |
|---|---|---|---|
| A1 | Hardware-efficient ansatz | single angle RY, PCA(n) | Kandala et al. 2017 |
| A2 | Strongly entangling layers | single angle RY, PCA(n) | Schuld et al. 2020 (PennyLane form) |
| A3 | Data re-uploading | RY re-uploaded per layer, PCA(n) | Perez-Salinas et al. 2020 |
| A4 | DisCoCat word-box block (feature route) | single angle RY, PCA(n) | Coecke 2020; Kartsaklis 2021 |
| A5 | Circuit-centric classifier | amplitude, PCA(2^n) normalised | Schuld, Bocharov, Svore, Wiebe 2020 |
| A6 | IQP feature map + VQC | second-order ZZ map x2, PCA(n) | Havlicek et al. 2019 |
| A7 | QNN with readout qubit | readout qubit prepared in state 1, RY on n-1 data wires, PCA(n-1) | Farhi and Neven 2018 |

Structural facts measured before any task run (analytic, P4): A1 to A6 close on su(2^n); A7 closes on (n-1) copies of su(2), dimension 3(n-1). P6 counts A5's amplitude encoding as its Mottonen CNOTs (2^n - 2 for n >= 3).

## 4. Design

* Family arm: A1 to A7, widths n in {4, 6, 8}, depths d in {1, 3, 5}, 10 paired seeds (63 cells).
* Encoding arm (Class II, A2 ansatz fixed, n = 4): angle (A2), re-upload (A3), amplitude (A2:amp), IQP (A2:iqp), d in {1, 3, 5}.
* Descriptors: P1 to P6, G1 (widths 2 to 10, depths 1 to 8, global and local cost), G5, G6.
* Tasks: IMDb-1k primary (hypotheses read it only); CoLA reported as the representation floor. Binary MNIST control: not implemented in this protocol (open item).
* Training: Adam lr 0.05, betas (0.9, 0.999), 20 epochs, batch 32, class-balanced loss, circuit angles uniform on [0, 2 pi) (the ensemble G1 measures), head (1, 0), 20% of train held out, epoch selected on validation MCC. Depth and width are factors, never tuned.
* Noise: depolarizing, bit flip and phase flip, two rates p2 = 10 p1, small (1e-4/1e-3) and nisq (1e-3/1e-2), at n = 4, trained clean and evaluated noisy.
* References: logistic regression on PCA(n-1), PCA(n), PCA(2^n), the full frozen vector and TF-IDF.

## 5. Hypotheses (stated before testing)

P = prospective. I = informed by data already seen (fair-v1, legacy v10, or the existing A1 to A4 G1 grid). Every test is coded in `vqc_scorecard/tasks/pure_report.py`.

| ID | Prediction | Tag | Decision rule |
|---|---|---|---|
| H1 | A1 to A6: global-cost gradient variance decays exponentially with width at d = 5; local cost at d = 1 decays at most a third as fast. A7: global alpha below ln2/2. | I for A1-A4, P for A5-A7 | alpha per G1 repeat over n = 2..10; A1-A6 supported if alpha CI > 0 and local alpha <= global/3; A7 supported if CI upper < ln2/2 |
| H2 | At n = 6, gradient variance falls with depth and levels off. | I for A1-A4, P for A5-A7 | Spearman(Var, d) < 0, Holm; last two depths within a factor 2 |
| H3 | MCC(d=3) > MCC(d=1) and MCC(d=5) - MCC(d=3) <= SESOI, per family and width. | I | one-sided Wilcoxon, Holm; t-interval vs SESOI |
| H4 | MCC(n=6) > MCC(n=4) and MCC(n=8) - MCC(n=6) <= SESOI, per family and depth. | P | same |
| H5 | (a) Class I families A1, A2, A4, A7 equivalent within SESOI at every (n, d); (b) A7 G1 variance >= 2x A1, A2, A4 at n = 6, d = 5; (c) A1 at d = 1 has the most failed seeds. | a, b P; c I | (a) interval TOST; (b) ratio; (c) count |
| H6 | (a) re-upload > angle; (b) amplitude > angle; (c) IQP not better than angle by more than SESOI. A2 ansatz, n = 4, per depth. | a I; b, c P | (a, b) one-sided Wilcoxon, Holm; (c) interval upper <= SESOI |
| H7 | Loss of noisy prediction fidelity is ranked better by depth than by two-qubit gate count (n = 4, nisq). | I | abs Spearman rho(depth) > abs rho(2q) |
| H8 | At n = 4, d = 3, families A1 to A4 are more consistently ordered in pure-v1 than in fair-v1. | P | Kendall W pure > W fair and Friedman p < 0.05 |
| R | Classical references | none | difference with CI only, no verdict |

## 6. Decision rules

* Unit: per-seed paired difference (same split, minibatch order and initial angles).
* SESOI = 0.03 MCC (fair-v1 depth effects were <= 0.03; seed sd 0.03 to 0.08). alpha = 0.05, Holm within each hypothesis.
* Verdicts: supported, refuted or inconclusive (rules in `vqc_scorecard/tasks/stats.py`). A hypothesis with any incomplete cell is reported as incomplete, never scored.

## 7. Result categories (per task, model, n, d), in this order

1. Floor: MCC 95% interval includes 0.
2. Trainability-limited: G1 local variance below the shot floor 1/1024, or >= 3 of 10 seeds with MCC < 0.2.
3. Performing: mean MCC within SESOI of logistic regression on the same features and <= 1 failed seed; Noise-limited if the nisq MCC drop exceeds SESOI.
4. Capacity-limited: everything else.

## 8. Procedure

1. Freeze this file (table above). 2. Run `RUN_PURE.md`. 3. Append section 10 with every verdict, effect, interval and deviation. 4. Anything decided after data is labelled exploratory.

## 9. Deviation log (before freezing)

* 2026-10-01: a 12-run single-seed timing smoke test (IMDb, n = 4 d = 3 and n = 8 d = 5) ran in the cloud during development. Its numbers were not stored in the project folder and no hypothesis or rule was edited after it. The protocol runs recompute every cell.

## 10. Post-test results

(append after the runs)
