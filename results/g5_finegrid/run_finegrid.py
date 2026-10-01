"""G5 fine-grid: add three log-spaced noise levels in the 0.01->0.1 gap.

Computes only the NEW interior levels (the four original levels are already in
scorecard.jsonl and were shown to reproduce bit-identically). Reuses the
package's own estimator (noisy_gradient_variance), angle-index picker and
per-(family,n,d,repeat) seeding, so every number is comparable to the checkpoint.
"""
import sys, json, time
sys.path.insert(0, "/mnt/user-data/uploads/QNN")
import numpy as np
from vqc_scorecard.ansatze import REGISTRY as ANSATZE
from vqc_scorecard.core.seeding import generator
from vqc_scorecard.metrics.g5_noise import _g5_angle_indices
from vqc_scorecard.probes.noise import noisy_gradient_variance

# quarter-decade log spacing strictly inside (0.01, 0.1)
NEW_LEVELS = tuple(round(10 ** (-2 + k / 4), 6) for k in (1, 2, 3))  # 0.017783, 0.031623, 0.056234
SAMPLES, REPEATS = 50, 2
FAMILIES = ["A1", "A2", "A3", "A4"]
DEPTHS = list(range(1, 9))
WIDTH = 4

def encoder_input(family, n, d, rng):
    size = ANSATZE[family].n_encoder(n, d)
    return None if size == 0 else rng.uniform(0.0, np.pi, size=size)

def measure_point(family, n, d, levels):
    ansatz = ANSATZE[family]
    indices = _g5_angle_indices(ansatz.n_params(n, d))
    per = {i: [] for i in range(len(levels))}
    for repeat in range(REPEATS):
        rng = generator("G5", family, n, d, SAMPLES, repeat)
        source = generator("encoder", family, n, d)  # fixed embedding
        enc = encoder_input(family, n, d, source)
        for i, p in enumerate(levels):
            var, mabs = noisy_gradient_variance(ansatz, n, d, p, SAMPLES, rng, indices, enc)
            per[i].append((var, mabs))
    rows = {}
    for i, p in enumerate(levels):
        vv = [x[0] for x in per[i]]
        rows[p] = {
            "gradvar_mean": round(float(np.mean(vv)), 8),
            "gradvar_std": round(float(np.std(vv, ddof=1)), 8),
            "gradvar_seeds": [round(float(x), 8) for x in vv],
        }
    return rows

print(f"NEW_LEVELS = {NEW_LEVELS}")
out = []
t0 = time.time()
for family in FAMILIES:
    for d in DEPTHS:
        ts = time.time()
        rows = measure_point(family, WIDTH, d, NEW_LEVELS)
        rec = {"family": family, "n": WIDTH, "d": d, "samples": SAMPLES,
               "repeats": REPEATS, "levels": rows}
        out.append(rec)
        vals = " ".join(f"{p}:{rows[p]['gradvar_mean']:.2e}" for p in NEW_LEVELS)
        print(f"{family} n=4 d={d}  {vals}  ({time.time()-ts:.1f}s)", flush=True)

with open("/home/claude/g5_new_levels.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote g5_new_levels.json  total {time.time()-t0:.0f}s")
