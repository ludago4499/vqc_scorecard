# Implementation notes: from paper to code, A1 to A7

Package `vqc_scorecard` 3.0.0, protocol pure-v1 (digest `b2833a59e87d`).
Written 2026-10-08. Simplified Technical English: short sentences, one idea per sentence.

This file is for a reader who wants to rebuild every circuit of the study from the papers, or check that the code is the circuit the paper describes. For each family it gives:

1. The source and the equations used.
2. How each equation becomes PennyLane gates, with the sign and order conventions.
3. What the task model adds around the circuit.
4. Every difference from the paper, and whether it changes a number.

Every statement in sections 1 to 9 that relates code to a paper has a numerical check in `tests/test_conventions.py`. Run it before you trust this file:

```
python -m vqc_scorecard.tests.test_conventions      # 25/25
python -m vqc_scorecard.tests                       # every suite
```

A check marked `(departure)` passes when a known departure is still exactly as described here. If someone changes the code, the check fails and this file must change with it.

---

## 0. Conventions shared by every family

### 0.1 Gate signs

PennyLane defines rotations with a minus sign and a half angle:

| PennyLane gate | Operator |
|---|---|
| `RX(t)`, `RY(t)`, `RZ(t)` | `exp(-i t P / 2)`, `P = X, Y, Z` |
| `IsingZZ(t)` | `exp(-i t Z Z / 2)` |
| `PauliRot(t, "AB")` | `exp(-i t A B / 2)` |
| `Rot(phi, theta, omega)` | `RZ(omega) RY(theta) RZ(phi)` as an operator, so `RZ(phi)` acts first |

Papers often write `exp(+i theta P)`. To get the paper's operator exactly, the code passes `-2 theta`. A6 and A7 do this. A1 to A5 pass the angle directly. The sign of a free angle changes nothing when angles are drawn uniformly on `[0, 2 pi)` or trained, so A1 to A5 do not convert.

### 0.2 Operator order and time order

A paper writes a product of operators. The rightmost factor acts first. Code queues gates in time order. So `U = A B C` in a paper is queued `C`, then `B`, then `A`. Each family section below states the conversion.

### 0.3 Wires and readout

Wire 0 is the readout for every family. Every model reads `<Z_0>`. For A7 this is the paper's `<Y>` on the readout qubit (section 7.3). A local readout is used for all families because the global cost flattens faster (G4).

### 0.4 The task model (pure-v1)

The circuit is the only trainable component. The model is

```
p(y = 1 | x) = sigmoid(a <Z_0>_theta(x) + b)
```

`a` and `b` are the only classical parameters. Source: `tasks/model.py`, `build_pure_model`.

| Item | Value | Where |
|---|---|---|
| Circuit init | uniform on `[0, 2 pi)` | `tasks/train.py`, `train_one` |
| Head init | `(a, b) = (1, 0)` | same |
| Loss | binary cross-entropy, class-balanced weights `w_c = N / (2 N_c)`, `p` clipped to `[1e-7, 1 - 1e-7]` | `tasks/train.py`, `_bce`, `_balanced_weights` |
| Optimiser | PennyLane `AdamOptimizer`, lr 0.05, betas (0.9, 0.999) | `tasks/pure.py`, `PureProtocol` |
| Epochs, batch | 20, 32 | same |
| Gradients | analytic, `diff_method="backprop"` on `default.qubit` | `build_pure_model` |
| Epoch selection | best validation MCC; 20% of the training split held out | `PureProtocol.select_on`, `val_frac` |
| Seeds | `BASE_SEED + i`, `BASE_SEED = 20260811`, `i < 10` | `core/seeding.py` |

None of the papers use this head or loss. Each family's deviations list says what the paper used instead.

### 0.5 Features

Features are fixed and unsupervised. No gradient reaches them.

1. Frozen `all-MiniLM-L6-v2` vectors, last hidden state, mean pooling (`tasks/get_features.py`, separate environment, `requirements-extract.txt`).
2. PCA fitted on the training documents of each seed only.
3. Rotation encodings (A1 to A4, A6, A7) read **angles**: standardise each component with the training mean and spread, clip at 3 sigma, map `x = (s + 3) / 6 * pi`. So `x` lies on `[0, pi]`, and a mean feature maps to `pi / 2` (`tasks/features.py`, `_to_angles`).
4. The amplitude encoding (A5) reads the standardised scores, unclipped, and normalises each row.

Feature count per model: `n` for A1 to A4 and A6, `2**n` for A5, `n - 1` for A7 (its wire 0 is the readout).

### 0.6 Descriptor path and task path

The descriptor sweep (`cli.py`, metrics P1 to G6) calls `Ansatz.apply(params, n, d, x)` with one fixed `x`. The task model builds its circuit in `tasks/model.py`, `circuit_body`:

| Family | Task path |
|---|---|
| A1, A2, A4 | `RY(x_q)` on every qubit, then `apply(theta, n, d)` with `x=None` |
| A3 | `_reupload_body`, a batched copy of `a3_reupload.apply` (identical state, checked) |
| A5, A6, A7 | the family's own `apply(theta, n, d, x)`; it loads the data itself |

### 0.7 Where the deviations are recorded

Each `Ansatz` record has a `deviations` tuple. `cli.py` writes it into every descriptor result file (`results/scorecard.json`). Result files written before 2026-10-08 hold the shorter lists from before this revision. The circuits did not change, so those numbers stay valid.

---

## 1. A1: hardware-efficient ansatz

**Source.** A. Kandala et al., *Nature* 549, 242 (2017), Eq. (2). File `ansatze/a1_hea.py`.

**Paper.** `d` entanglers `U_ENT` alternate with layers of Euler rotations `U^{q,i}(theta) = Z_{theta1} X_{theta2} Z_{theta3}` on every qubit `q`. The first `Z` of layer 0 is not implemented, because `|0>` is a `Z` eigenstate. The paper gives `p = N(3d + 2)`.

**Code.**

| Paper | Code (time order) |
|---|---|
| Layer 0, `Z X Z` with the first-in-time `Z` dropped | `RX, RZ` on every qubit |
| `U_ENT = exp(-i H_0 tau)`, device drift (cross-resonance) | `CNOT(q, q+1)` for `q = 0 .. n-2` |
| Layer `i >= 1`, `Z X Z` | `RZ, RX, RZ` on every qubit |
| `p = N(3d + 2)` | `n_params = n(3d + 2)` |

**Task path.** The paper's ansatz is a VQE trial state and takes no data. The task model adds `RY(x_q)` before it.

**Deviations.**

| Deviation | Changes a number? |
|---|---|
| CNOT chain instead of the device's cross-resonance drift | Yes, a different entangler. Declared. |
| `RY(x)` encoding added for the task | Task only. Descriptors use the bare ansatz. |
| `<Z_0>` readout and the head, not a Hamiltonian energy | Task only. |
| Angle sign convention | No. |

**Checks.** `test_a1_count_and_dropped_z`.

---

## 2. A2: strongly entangling layers

**Source.** M. Schuld, A. Bocharov, K. M. Svore, N. Wiebe, *Phys. Rev. A* 101, 032308 (2020), as implemented by PennyLane `StronglyEntanglingLayers`. File `ansatze/a2_sel.py`.

**Exact equivalence.** `A2.apply(theta, n, d)` equals `qml.StronglyEntanglingLayers(theta.reshape(d, n, 3), wires=range(n))` with the default ranges and `CNOT`. This is checked at `(n, d) = (2, 2), (4, 3), (5, 5)`. A reader can therefore rebuild A2 with one PennyLane call.

**Code.** Layer `l` (time order): `RZ(phi), RY(theta), RZ(omega)` on every qubit, then `CNOT(q, (q + r_l) mod n)` for `q = 0 .. n-1`, with `r_l = (l mod (n - 1)) + 1`. A ring follows **every** layer, the last one included. `p = 3nd`.

**What PennyLane changes from the paper.** The paper's code block is a layer of general single-qubit gates `G`, then `n / gcd(n, r)` **trainable** controlled gates `C(G)`. Gate `j` has control `j r mod n` and target `(j - 1) r mod n`. The input is **amplitude** encoded. SEL keeps the layer layout, but uses fixed CNOTs with control `q` and target `q + r`, and the study feeds it one angle per qubit. A5 restores the trainable controlled gates and the amplitude encoding (section 5).

**Deviations.**

| Deviation | Changes a number? |
|---|---|
| Fixed CNOTs instead of trainable `C(G)` | Yes. Declared. A5 is the contrast. |
| One `RY(x_q)` per qubit instead of amplitude encoding | Yes. Declared. A5 is the contrast. |
| Ring range grows with the layer (PennyLane default) | Part of the circuit. Declared. |
| Readout and loss (paper: `pi = p(q_0 = 1) + b`, least squares, SGD) | Task only. |

**Checks.** `test_a2_is_pennylane_sel`.

---

## 3. A3: data re-uploading

**Source.** A. Perez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, J. I. Latorre, *Quantum* 4, 226 (2020). File `ansatze/a3_reupload.py`.

**Paper.** Processing layer `L(i) = U(phi_i) U(x)` (Eq. 3), repeated `N` times. `U` is a general SU(2) gate, `U(phi) = e^{i phi_2 Z} e^{i phi_1 Y} e^{i phi_3 Z}` (Eq. 12). Data enter through `U(x)`, up to three components per gate. A weighted form `U(theta_i + w_i o x)` (Eq. 5) also exists. The paper has 1-, 2- and 4-qubit classifiers. The multi-qubit ones apply `CZ` after every layer except the last; with 4 qubits the pairs alternate between (1,2),(3,4) and (2,3),(1,4). The cost is a fidelity (Eq. 7) or a weighted fidelity (Eq. 9) against label states.

**Code.** Repeat `l = 0 .. d-1` (time order): `RY(x_q)` on every qubit, then `RZ, RY, RZ` with free angles, then, if `l < d - 1`, a ring `CNOT(q, (q+1) mod n)`. `p = 3nd`. At `d = 1` there is no entangler (README 7.4).

**Deviations.**

| Deviation | Changes a number? |
|---|---|
| CNOT ring, range 1, between layers; the paper uses `CZ` on alternating pairs | Yes. Declared. |
| One feature per qubit through `RY(x_q)`; the paper uploads `x` through a general SU(2) gate | Yes. Declared. |
| Unweighted layers (Eq. 3); Eq. (5) weights not used | Yes. Declared. |
| `<Z_0>` and the head instead of a fidelity cost | Task only. |
| `permits` marks only `n = 4` as in protocol for the **descriptors**; the pure-v1 task arm runs `n = 4, 6, 8` | No. Scope note. |

The deviation list before 2026-10-08 said "the paper is single-qubit". That was wrong and is corrected.

**Not the A2 entangler.** A3 shares A2's Euler block but not its ring. See section 8.1.

**Checks.** `test_a3_task_body_and_entangler`, `test_a2_a3_entanglers_differ`.

---

## 4. A4: DisCoCat sentence template

**Sources.** B. Coecke et al., arXiv:2005.04147 (2020); D. Kartsaklis et al., *lambeq*, arXiv:2110.04236 (2021). File `ansatze/a4_discocat.py`.

**Code.** One qubit per word, nouns on even wires, verbs on odd wires (`n` must be even).

1. Word states: `RY(theta_q)` on every wire.
2. `d` repeats of a Hadamard layer, then `CRZ(theta)` on `(q, (q + 1) mod n)` for every `q` (a ring with wrap-around).
3. Cups on `(0, 1), (2, 3), ...`: `CNOT(a, b)` then `H(a)`, with **no** post-selection.

`p = n(d + 1)`.

**What a cup is.** A DisCoCat cup is the Bell effect `<Phi+|`. `H(a) CNOT(a, b)` sends `|Phi+>` to `|00>`, so post-selecting `00` after these two gates is the cup. The code queues the gates and does not post-select. The state is the sentence **before** contraction (README 7.5).

**What the IQP layer is not.** In DisCoCat a word box acts only on its own word's wires, and only cups connect words. Here the `CRZ` ring couples different words directly. lambeq's `IQPAnsatz` (source `lambeq/ansatz/circuit.py`) gives a one-qubit box `Rx, Rz, Rx` and a multi-qubit box a `CRz` ladder on `n - 1` adjacent pairs, each followed by a Hadamard layer, with no wrap-around. A4 is therefore an IQP-style **sentence** circuit on a cup template. It is not lambeq's word boxes.

**Readout.** For even `n` every wire is in a cup, so no sentence wire stays open. `<Z_0>` reads a cup wire that a contracted circuit would post-select.

**Task path.** The task queues `RY(x_q)`, then `apply`. The first gate of `apply` is `RY(theta_q)` on the same wire. The two compose to `RY(x_q + theta_q)`, so the word-state angles are trainable offsets on the features. The parse-driven word table (`build_compositional_model`) belongs to the legacy v10 task axis, not to pure-v1.

**Deviations.**

| Deviation | Changes a number? |
|---|---|
| Cups not post-selected | Yes. Declared since v1; state it in any write-up. |
| Fixed alternating noun-verb template, not a parser output | Yes. Declared. |
| IQP ring spans the sentence and wraps around; not lambeq's per-word boxes | Yes. Declared (corrected 2026-10-08). |
| Readout on a cup wire | Task only. Declared. |
| Feature and word-state angles add on the same wire | Task only. Declared. |

**Checks.** `test_a4_cup_and_task_offset`.

---

## 5. A5: circuit-centric classifier

**Source.** M. Schuld, A. Bocharov, K. M. Svore, N. Wiebe, *Phys. Rev. A* 101, 032308 (2020), Secs. II to IV, Fig. 4. File `ansatze/a5_circuit_centric.py`.

**Paper.**

* Amplitude encoding of `x`, padded with non-informative constants to a power of 2, then normalised (Eq. 2).
* Code block `B` (Eq. 9): general single-qubit gates `G(alpha, beta, gamma)` on all `n` qubits, then `n / gcd(n, r)` controlled gates `C(G)`. Gate `j` has target `t_j = (j - 1) r mod n` and control `c_j = j r mod n`.
* Fig. 4 (8 qubits): blocks `B_1` and `B_3` (ranges 1 and 3), 17 single-qubit gates (16 in the blocks plus one final gate) and 16 controlled gates.
* Readout: `pi(x) = p(q_0 = 1) + b` (Eq. 7), least-squares cost, single-sample SGD.

**Code.**

1. `AmplitudeEmbedding(x, normalize=True)`, no padding. PennyLane decomposes it with the Mottonen method; P6 counts `2**n - 2` CNOTs for `n >= 3`.
2. Block `l`: `Rot` on every qubit, then `CRot` with control `q` and target `(q + r_l) mod n` for `q = 0 .. n-1`. `r_l = 1` on even blocks and `3` on odd blocks; `r = 1` whenever `n <= 3`.
3. A final `Rot` on qubit 0.

`p = 6nd + 3`. `Rot` and `CRot` parameterise SU(2) in the `RZ RY RZ` Euler form. The paper's `G(alpha, beta, gamma)` is a different chart on the same group.

**Controlled-gate layout against the paper.** This is the one place where the code is not the paper's rule.

| Width | Paper (per range-`r` block) | Code | Effect |
|---|---|---|---|
| `n = 4, 8` (`gcd(n, r) = 1`) | `n` gates, control = target + `r` | `n` gates, target = control + `r` | Same pairs, same count. Orientation reversed. For `r = 3` the gate order differs. |
| `n = 6`, `r = 3` | 2 gates: `3 -> 0`, `0 -> 3` | 6 gates: every pair `{q, q+3}` twice, both directions | More gates and angles at `n = 6`, `d >= 2`. |

Angle counts at `n = 6`:

| `d` | code `p` | paper rule `p` |
|---|---|---|
| 1 | 39 | 39 |
| 3 | 111 | 99 |
| 5 | 183 | 159 |

`n = 4` and `n = 8` match the paper's count at every depth.

**Deviations.**

| Deviation | Changes a number? |
|---|---|
| Range alternation 1, 3 used at every width (the paper gives Fig. 4 only) | Part of the circuit. Declared. |
| `Rot`/`CRot` chart instead of `G(alpha, beta, gamma)` | No (same group). Declared. |
| No padding constant before normalisation; the vector norm is lost | Yes. Declared. |
| Controlled-gate orientation and order | Yes, in principle. Declared (2026-10-08). |
| `n = 6`, `r = 3`: 6 gates where the paper has 2 | Yes. Declared (2026-10-08). |
| Head and loss (paper: `p(q_0 = 1) + b`, least squares) | Task only. Declared. |

**Checks.** `test_a5_controlled_layout`, and in `test_pure.py`: `test_a5_amplitude_readout_closed_form`, `test_mottonen_cost_is_counted`.

---

## 6. A6: IQP feature map with a variational classifier

**Source.** V. Havlicek et al., *Nature* 567, 209 (2019), the quantum variational classifier. Equation numbers S(k) refer to the supplementary information of arXiv:1804.11326. File `ansatze/a6_iqp.py`.

**Paper.**

```
|Phi(x)> = U_Phi(x) H^n U_Phi(x) H^n |0>^n                          (S26)
U_Phi(x) = exp( i sum_q phi_q(x) Z_q + i sum_(q,r) phi_qr(x) Z_q Z_r )  (S27)
phi_q = x_q,   phi_qr = (pi - x_q)(pi - x_r),   x in (0, 2 pi]
W(theta) = U_loc^(l) U_ent ... U_loc^(2) U_ent U_loc^(1)               (S31)
U_loc^(t) = tensor_m exp(i theta^z Z / 2) exp(i theta^y Y / 2)          (S32)
U_ent = prod_(i,j) in E  CZ(i, j)                                       (S33)
```

The paper measures a parity `f` of all qubits and assigns `sign(<f> + b)`.

**Code.**

| Paper | Code (time order) |
|---|---|
| `H^n` | `Hadamard` on every qubit |
| `exp(i x_q Z_q)` | `RZ(-2 x_q)` |
| `exp(i phi_qr Z_q Z_r)` | `IsingZZ(-2 phi_qr)` on chain edges `(q, q+1)` |
| `(U_Phi H^n)^2` | the block above, twice (`FEATURE_MAP_REPS = 2`) |
| `U_loc` | `RY`, then `RZ` on every qubit |
| `U_ent` | `CZ(q, q+1)` on the chain |
| `W`: `l` entanglers, `l + 1` local layers | `d` entanglers, `d + 1` local layers |
| `theta in R^{2n(l+1)}` | `n_params = 2n(d + 1)` |

The diagonal gates of one block commute, so their order inside the block does not matter. `test_pure.py` builds `(U_Z H^n)^2 |0>` from matrices and compares the state.

**Input range.** The task angles lie on `[0, pi]`, not on the paper's `(0, 2 pi]`. Two effects:

* `phi_q = x_q` covers half the paper's phase range.
* `(pi - x_q)` lies in `[0, pi]`, so the ZZ phase lies in `[0, pi^2]` and never changes sign. In the paper it spans `[-pi^2, pi^2]`.

So A6 in this study is the paper's map restricted to one quadrant of its input domain. A rescale `x -> 2x` inside `embed` would restore the range, but it would change every A6 and A2:iqp result, so it is not done in pure-v1.

**Deviations.**

| Deviation | Changes a number? |
|---|---|
| Chain graph at every `n` (the paper runs `n = 2`) | Part of the circuit. Declared. |
| `<Z_0>` instead of the parity of all qubits | Yes. Declared. |
| Angle sign of `W` (PennyLane `RY`, `RZ`) | No. Declared. |
| Input range `[0, pi]` instead of `(0, 2 pi]` | Yes. Declared (2026-10-08). |
| BCE, Adam, exact expectations, instead of the sigmoid-smoothed binomial error, SPSA, shots and zero-noise extrapolation | Task only. Declared (2026-10-08). |
| Quantum kernel estimation not implemented | Scope. Declared (2026-10-08). |

**Checks.** `test_a6_signs_and_range`, and in `test_pure.py`: `test_a6_feature_map_matches_havlicek`.

---

## 7. A7: Farhi-Neven quantum neural network

**Source.** E. Farhi, H. Neven, arXiv:1802.06002 (2018). File `ansatze/a7_farhi_neven.py`.

### 7.1 Paper

* Input `|z, 1>` with bits `z_i` valued `+1` or `-1` (Eq. 3). The last qubit is the readout.
* Gates `exp(i theta Sigma)`, `Sigma` a Pauli string (Eq. 8).
* Prediction `<z, 1| U^dagger Y_readout U |z, 1>` (Eq. 5).
* Loss `1 - l(z) <Y>` (Eq. 6), with the loss-normalised SGD step of Eq. (32).
* The MNIST circuit uses `Z_j X_readout` and `X_j X_readout` gates only, 3 layers of each, 96 angles on 16 data qubits.

### 7.2 Code

| Paper | Code |
|---|---|
| readout qubit | wire 0, prepared with `PauliX` (state `|1>`, see 7.3) |
| data bits | `RY(x_j)` on wires `1 .. n-1` |
| `exp(i theta Z_j X_0)` | `PauliRot(-2 theta, "ZX", wires=[j, 0])` |
| `exp(i theta X_j X_0)` | `PauliRot(-2 theta, "XX", wires=[j, 0])` |
| one layer | one ZX sweep, then one XX sweep, over all data wires |
| measure `Y` on the readout | `RX(pi/2)` on wire 0, then `<Z_0>`; `RX(pi/2)^dagger Z RX(pi/2) = Y` |

`p = 2(n - 1)d`. At `n = 17`, `d = 3` this is 96, the paper's MNIST count. `RY(0)|0> = |0>` and `RY(pi)|0> = |1>`, so the endpoints of the feature range are the paper's basis states.

### 7.3 The readout state

The paper's `|z, 1>` uses `+1/-1` valued bits. Its "1" is the `Z = +1` state, which is `|0>` in PennyLane. The paper's own algebra needs this: Eq. (10) gives `U_l^dagger Y U_l = cos(pi l / 2) Y + sin(pi l / 2) Z`, and Eq. (11), `<Y> = l(z)`, holds only when `<Z_readout> = +1`.

The code prepares `|1>`. Every gate of A7 commutes with `X_0`: `Z_j X_0`, `X_j X_0`, `RY` on data wires, `RX` on wire 0. So `|1> = X_0 |0>` gives exactly the negative output, `<Y>_package = -<Y>_paper`, at every angle and input. The head scale `a` absorbs the sign. No pure-v1 result depends on it, so it is kept, and it is declared. To build the paper's circuit, remove the `PauliX` on wire 0.

Check: with readout `|0>`, the code's gates reproduce Eq. (20), `<Y> = sin(beta sum_j z_j)`. The package gives the exact negative.

### 7.4 Product form

Write `|1> = (|+> - |->) / sqrt(2)` in the `X_0` basis. A gate `exp(i t P_j X_0)` acts on the data as `exp(+i t P_j)` in the `+` branch and `exp(-i t P_j)` in the `-` branch. Gates on different data wires act on different qubits. Let `u_j^+` and `u_j^-` be the products of wire `j`'s gates in each branch, and `|phi_j> = RY(x_j)|0>`. Then

```
<Y_0> = Im  prod_j  <phi_j| (u_j^+)^dagger u_j^- |phi_j>
```

The output is the imaginary part of a product of `n - 1` single-qubit overlaps. It costs `O(n d)` to compute classically. This is the circuit form of the A7 dynamical Lie algebra, `n - 1` copies of `su(2)` (P4: dimension `3(n - 1)`). The same identity holds for the paper's own MNIST circuit, which uses the same gate set.

### 7.5 Deviations

| Deviation | Changes a number? |
|---|---|
| `RY(x)` angle encoding instead of basis states | Yes. Declared. |
| `n` counts the readout, so A7 reads `n - 1` features | Yes. Declared. |
| One layer = one ZX sweep plus one XX sweep | Part of the circuit. Declared. |
| `Y` readout through `RX(pi/2)` and `Z_0` | No (exact). Declared. |
| Readout `|1>` instead of the paper's `|0>` | Sign of the output only; absorbed by `a`. Declared (2026-10-08). |
| BCE and Adam instead of Eq. (6) and Eq. (32) | Task only. Declared (2026-10-08). |

**Checks.** `test_a7_gate_sign_and_count`, `test_a7_readout_sign`, `test_a7_product_form`, `test_a7_basis_endpoints`, and in `test_pure.py`: `test_a7_layer_and_readout_match_farhi_neven`.

---

## 8. Cross-family facts that affect how results are read

### 8.1 The encoding arm does not hold the entangler fixed for A3

The pre-registration states that the encoding arm holds the A2 ansatz fixed and changes the encoding: `A2` (angle), `A3` (re-upload), `A2:amp`, `A2:iqp`.

* `A2:amp` and `A2:iqp` run the A2 circuit exactly.
* `A3` shares A2's Euler block, but its ring has range 1 and sits only between layers. A2's ring follows every layer, with a range that grows with the layer.

So the contrast angle against re-upload (hypothesis H6a) changes the encoding and the entangler together. At `d = 1`, the two circuits differ even at equal angles, because A3 has no entangler and A2 has one ring. A clean contrast would need an `A2:reupload` model that re-uploads `RY(x)` before each A2 layer. It is not in pure-v1.

### 8.2 A5 at n = 6

At `n = 6` A5 has more controlled gates than the paper's rule (section 5). Width trends of A5 (H4) mix the width with this change of structure between `n = 4, 6, 8`.

### 8.3 A7 is classically simulable

Section 7.4. A7 is the study's structural control for barren plateaus. Its small algebra is also what makes it efficient to simulate. Report the two together.

---

## 9. Library facts the code relies on

| Fact | Where it matters | Checked in |
|---|---|---|
| `RZ(-2x) = exp(i x Z)`, `IsingZZ(-2 phi) = exp(i phi Z Z)` | A6 | `test_conventions` |
| `PauliRot(-2t, P) = exp(i t P)` | A7 | `test_conventions` |
| `StronglyEntanglingLayers` default ranges `(l mod (n-1)) + 1`, CNOT | A2 | `test_conventions` |
| `AmplitudeEmbedding` decomposes to Mottonen, `2**n - 2` CNOTs for `n >= 3` | A5, P6, noise | `test_pure` |
| `default.mixed` mis-shapes batched diagonal gates in PennyLane 0.42 | A6 and A5 under noise: the batch is split with `broadcast_expand` | `tasks/model.py`, `build_pure_model` |

The pinned versions are PennyLane 0.42.3 and numpy 2.4.4 on Python 3.11 (`requirements.txt`). The conventions in this file are PennyLane's. Another library may define the rotation sign or the `Rot` order differently. Rebuild with the operators in the tables, not with gate names.

---

## 10. Reproduce

From the folder that contains `vqc_scorecard/`:

```
pip install -r vqc_scorecard/requirements.txt
python -m vqc_scorecard.tests                       # every suite
python -c "from vqc_scorecard.tasks.pure import PROTOCOL; print(PROTOCOL.digest())"   # b2833a59e87d
```

Then follow `RUN_PURE.md` for the run order. The features need the separate extraction environment in `requirements-extract.txt`.

This revision (2026-10-08) changes documentation and adds `tests/test_conventions.py`. It changes no gate, gate order, angle count, protocol field or result. The protocol digest is unchanged.
