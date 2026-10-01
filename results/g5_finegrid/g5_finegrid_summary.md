# G5 fine-grid: the 0.01 -> 0.1 noise gap, filled

n=4, families A1-A4, depths 1-8, samples=50, 2 seeds. Three log-spaced
interior levels added: p = 0.0178, 0.0316, 0.0562 (quarter-decade steps).
Original four levels (0, 0.001, 0.01, 0.1) reproduce the checkpoint
bit-identically (verified), so they are read straight from scorecard.jsonl.

## What the gap actually contains

Per-level global gradient variance, mid-depth d=4 (representative):

| family | 0.01 | 0.0178 | 0.0316 | 0.0562 | 0.1 |
|--------|------|--------|--------|--------|-----|
| A1 | 3.27e-03 | 1.07e-03 | 9.23e-05 | 2.42e-06 | 1.00e-08 |
| A2 | 3.71e-03 | 8.71e-04 | 1.06e-04 | 2.67e-06 | 0.00e+00 |
| A3 | 1.61e-03 | 5.24e-04 | 3.93e-05 | 3.30e-07 | 0.00e+00 |
| A4 | 4.30e-03 | 1.77e-03 | 2.96e-04 | 1.16e-05 | 5.00e-08 |

## Steepness of the wall (local d ln Var / d ln p, d=4)

A single exponential-in-p decay would give a roughly constant slope;
a knee would show a jump. Measured slopes per sub-interval:

| family | 0.01->0.0178 | 0.0178->0.0316 | 0.0316->0.0562 | 0.0562->0.1 |
|--------|--------------|----------------|----------------|-------------|
| A1 | -1.9 | -4.3 | -6.3 | -9.5 |
| A2 | -2.5 | -3.7 | -6.4 | -25.7 |
| A3 | -2.0 | -4.5 | -8.3 | -22.1 |
| A4 | -1.5 | -3.1 | -5.6 | -9.5 |

## p_crit(d): does the denser grid move it?

| family | d | coarse (0.01/0.1) | dense |
|--------|---|-------------------|-------|
| A1 | 1 | 0.0218 | 0.0396 |
| A1 | 2 | 0.0172 | 0.0305 |
| A1 | 3 | 0.0138 | 0.0220 |
| A1 | 4 | 0.0124 | 0.0182 |
| A1 | 5 | 0.0100 | 0.0142 |
| A1 | 6 | 0.0100 | 0.0118 |
| A1 | 7 | 0.0100 | 0.0109 |
| A1 | 8 | 0.0100 | 0.0093 |
| A2 | 1 | 0.0478 | 0.0699 |
| A2 | 2 | 0.0150 | 0.0238 |
| A2 | 3 | 0.0124 | 0.0182 |
| A2 | 4 | 0.0100 | 0.0170 |
| A2 | 5 | 0.0100 | 0.0148 |
| A2 | 6 | 0.0100 | 0.0111 |
| A2 | 7 | 0.0100 | 0.0106 |
| A2 | 8 | 0.0100 | 0.0080 |
| A3 | 1 | 0.0195 | 0.0371 |
| A3 | 2 | 0.0163 | 0.0313 |
| A3 | 3 | 0.0123 | 0.0174 |
| A3 | 4 | 0.0100 | 0.0129 |
| A3 | 5 | 0.0100 | 0.0112 |
| A3 | 6 | 0.0100 | 0.0105 |
| A3 | 7 | 0.0100 | 0.0047 |
| A3 | 8 | 0.0100 | 0.0057 |
| A4 | 1 | 0.0100 | 0.0069 |
| A4 | 2 | 0.0169 | 0.0298 |
| A4 | 3 | 0.0143 | 0.0245 |
| A4 | 4 | 0.0135 | 0.0215 |
| A4 | 5 | 0.0128 | 0.0186 |
| A4 | 6 | 0.0100 | 0.0164 |
| A4 | 7 | 0.0100 | 0.0146 |
| A4 | 8 | 0.0100 | 0.0134 |