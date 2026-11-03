# pure-v1 results [b2833a59e87d]

Protocol pure-v1: circuit-only training, 10 paired seeds, SESOI 0.03 MCC, alpha 0.05, Holm within each hypothesis.

## imdb1k

every protocol cell complete

| model | n | d | test MCC (95% CI) | ref (logreg same features) | failed | noise drop | category |
|---|---|---|---|---|---|---|---|
| A1 | 4 | 1 | +0.061 +- 0.053 | +0.164 | 9 | +0.000 | Trainability-limited |
| A1 | 4 | 3 | +0.141 +- 0.067 | +0.164 | 7 | +0.005 | Trainability-limited |
| A1 | 4 | 5 | +0.145 +- 0.089 | +0.164 | 7 | +0.002 | Trainability-limited |
| A1 | 6 | 1 | +0.097 +- 0.059 | +0.283 | 9 | - | Trainability-limited |
| A1 | 6 | 3 | +0.147 +- 0.058 | +0.283 | 7 | - | Trainability-limited |
| A1 | 6 | 5 | +0.255 +- 0.043 | +0.283 | 3 | - | Trainability-limited |
| A1 | 8 | 1 | +0.094 +- 0.051 | +0.270 | 10 | - | Trainability-limited |
| A1 | 8 | 3 | +0.162 +- 0.042 | +0.270 | 7 | - | Trainability-limited |
| A1 | 8 | 5 | +0.273 +- 0.048 | +0.270 | 1 | - | Performing |
| A2 | 4 | 1 | +0.088 +- 0.069 | +0.164 | 9 | -0.001 | Trainability-limited |
| A2 | 4 | 3 | +0.161 +- 0.058 | +0.164 | 6 | +0.008 | Trainability-limited |
| A2 | 4 | 5 | +0.194 +- 0.076 | +0.164 | 5 | +0.010 | Trainability-limited |
| A2 | 6 | 1 | +0.200 +- 0.055 | +0.283 | 4 | - | Trainability-limited |
| A2 | 6 | 3 | +0.250 +- 0.042 | +0.283 | 2 | - | Capacity-limited |
| A2 | 6 | 5 | +0.257 +- 0.049 | +0.283 | 2 | - | Capacity-limited |
| A2 | 8 | 1 | +0.176 +- 0.071 | +0.270 | 6 | - | Trainability-limited |
| A2 | 8 | 3 | +0.235 +- 0.033 | +0.270 | 2 | - | Capacity-limited |
| A2 | 8 | 5 | +0.262 +- 0.060 | +0.270 | 2 | - | Capacity-limited |
| A3 | 4 | 1 | +0.053 +- 0.039 | +0.164 | 10 | -0.002 | Trainability-limited |
| A3 | 4 | 3 | +0.138 +- 0.059 | +0.164 | 8 | -0.011 | Trainability-limited |
| A3 | 4 | 5 | +0.157 +- 0.072 | +0.164 | 7 | +0.002 | Trainability-limited |
| A3 | 6 | 1 | +0.051 +- 0.045 | +0.283 | 10 | - | Trainability-limited |
| A3 | 6 | 3 | +0.210 +- 0.040 | +0.283 | 5 | - | Trainability-limited |
| A3 | 6 | 5 | +0.220 +- 0.051 | +0.283 | 4 | - | Trainability-limited |
| A3 | 8 | 1 | +0.045 +- 0.037 | +0.270 | 10 | - | Trainability-limited |
| A3 | 8 | 3 | +0.205 +- 0.035 | +0.270 | 4 | - | Trainability-limited |
| A3 | 8 | 5 | +0.220 +- 0.053 | +0.270 | 3 | - | Trainability-limited |
| A4 | 4 | 1 | +0.121 +- 0.064 | +0.164 | 8 | +0.001 | Trainability-limited |
| A4 | 4 | 3 | +0.138 +- 0.063 | +0.164 | 7 | +0.008 | Trainability-limited |
| A4 | 4 | 5 | +0.147 +- 0.058 | +0.164 | 7 | +0.007 | Trainability-limited |
| A4 | 6 | 1 | +0.155 +- 0.059 | +0.283 | 6 | - | Trainability-limited |
| A4 | 6 | 3 | +0.226 +- 0.045 | +0.283 | 2 | - | Capacity-limited |
| A4 | 6 | 5 | +0.246 +- 0.048 | +0.283 | 2 | - | Capacity-limited |
| A4 | 8 | 1 | +0.105 +- 0.034 | +0.270 | 10 | - | Trainability-limited |
| A4 | 8 | 3 | +0.229 +- 0.062 | +0.270 | 4 | - | Trainability-limited |
| A4 | 8 | 5 | +0.252 +- 0.052 | +0.270 | 3 | - | Trainability-limited |
| A5 | 4 | 1 | +0.015 +- 0.060 | +0.331 | 10 | -0.004 | Floor |
| A5 | 4 | 3 | +0.025 +- 0.072 | +0.331 | 10 | -0.005 | Floor |
| A5 | 4 | 5 | +0.030 +- 0.044 | +0.331 | 10 | +0.000 | Floor |
| A5 | 6 | 1 | +0.055 +- 0.042 | +0.477 | 10 | - | Trainability-limited |
| A5 | 6 | 3 | +0.042 +- 0.066 | +0.477 | 9 | - | Floor |
| A5 | 6 | 5 | +0.042 +- 0.049 | +0.477 | 10 | - | Floor |
| A5 | 8 | 1 | +0.070 +- 0.026 | +0.424 | 10 | - | Trainability-limited |
| A5 | 8 | 3 | +0.059 +- 0.052 | +0.424 | 9 | - | Trainability-limited |
| A5 | 8 | 5 | -0.009 +- 0.074 | +0.424 | 10 | - | Floor |
| A6 | 4 | 1 | -0.011 +- 0.037 | +0.164 | 10 | -0.004 | Floor |
| A6 | 4 | 3 | +0.043 +- 0.031 | +0.164 | 10 | -0.008 | Trainability-limited |
| A6 | 4 | 5 | +0.085 +- 0.053 | +0.164 | 10 | -0.003 | Trainability-limited |
| A6 | 6 | 1 | +0.022 +- 0.028 | +0.283 | 10 | - | Floor |
| A6 | 6 | 3 | +0.023 +- 0.056 | +0.283 | 10 | - | Floor |
| A6 | 6 | 5 | +0.042 +- 0.048 | +0.283 | 10 | - | Floor |
| A6 | 8 | 1 | +0.015 +- 0.058 | +0.270 | 10 | - | Floor |
| A6 | 8 | 3 | +0.051 +- 0.047 | +0.270 | 10 | - | Trainability-limited |
| A6 | 8 | 5 | +0.086 +- 0.045 | +0.270 | 10 | - | Trainability-limited |
| A7 | 4 | 1 | +0.123 +- 0.055 | +0.115 | 9 | +0.018 | Trainability-limited |
| A7 | 4 | 3 | +0.096 +- 0.059 | +0.115 | 9 | +0.004 | Trainability-limited |
| A7 | 4 | 5 | +0.077 +- 0.039 | +0.115 | 10 | +0.020 | Trainability-limited |
| A7 | 6 | 1 | +0.213 +- 0.052 | +0.251 | 4 | - | Trainability-limited |
| A7 | 6 | 3 | +0.195 +- 0.056 | +0.251 | 4 | - | Trainability-limited |
| A7 | 6 | 5 | +0.182 +- 0.064 | +0.251 | 5 | - | Trainability-limited |
| A7 | 8 | 1 | +0.235 +- 0.052 | +0.275 | 4 | - | Trainability-limited |
| A7 | 8 | 3 | +0.195 +- 0.072 | +0.275 | 4 | - | Trainability-limited |
| A7 | 8 | 5 | +0.159 +- 0.054 | +0.275 | 7 | - | Trainability-limited |
| A2:amp | 4 | 1 | -0.033 +- 0.030 | +0.331 | 10 | - | Floor |
| A2:amp | 4 | 3 | +0.005 +- 0.064 | +0.331 | 10 | - | Floor |
| A2:amp | 4 | 5 | +0.006 +- 0.060 | +0.331 | 10 | - | Floor |
| A2:iqp | 4 | 1 | +0.007 +- 0.055 | +0.164 | 10 | - | Floor |
| A2:iqp | 4 | 3 | +0.038 +- 0.062 | +0.164 | 10 | - | Floor |
| A2:iqp | 4 | 5 | +0.074 +- 0.055 | +0.164 | 9 | - | Trainability-limited |

References (logistic regression, test MCC):

- n=4 logreg_frozen: +0.476 +- 0.022
- n=4 logreg_pca16: +0.331 +- 0.035
- n=4 logreg_pca3: +0.115 +- 0.026
- n=4 logreg_pca4: +0.164 +- 0.042
- n=4 logreg_tfidf: +0.631 +- 0.018
- n=6 logreg_frozen: +0.474 +- 0.033
- n=6 logreg_pca5: +0.251 +- 0.062
- n=6 logreg_pca6: +0.283 +- 0.039
- n=6 logreg_pca64: +0.477 +- 0.030
- n=6 logreg_tfidf: +0.633 +- 0.028
- n=8 logreg_frozen: +0.474 +- 0.033
- n=8 logreg_pca256: +0.424 +- 0.033
- n=8 logreg_pca7: +0.275 +- 0.039
- n=8 logreg_pca8: +0.270 +- 0.047
- n=8 logreg_tfidf: +0.633 +- 0.028

## Hypotheses (primary task)

| hypothesis | verdict |
|---|---|
| H1 | refuted |
| H2 | refuted |
| H3 | refuted |
| H4 | refuted |
| H5 | refuted |
| H6 | refuted |
| H7 | refuted |
| H8 | incomplete |

## cola

INCOMPLETE: 42 cells missing seeds (first: cola A1 n=6 d=1: 0/10)

| model | n | d | test MCC (95% CI) | ref (logreg same features) | failed | noise drop | category |
|---|---|---|---|---|---|---|---|
| A1 | 4 | 1 | -0.008 +- 0.025 | +0.000 | 10 | - | Floor |
| A1 | 4 | 3 | +0.018 +- 0.020 | +0.000 | 10 | - | Floor |
| A1 | 4 | 5 | -0.002 +- 0.023 | +0.000 | 10 | - | Floor |
| A2 | 4 | 1 | -0.013 +- 0.034 | +0.000 | 10 | - | Floor |
| A2 | 4 | 3 | -0.011 +- 0.024 | +0.000 | 10 | - | Floor |
| A2 | 4 | 5 | +0.019 +- 0.024 | +0.000 | 10 | - | Floor |
| A3 | 4 | 1 | +0.009 +- 0.019 | +0.000 | 10 | - | Floor |
| A3 | 4 | 3 | +0.021 +- 0.026 | +0.000 | 10 | - | Floor |
| A3 | 4 | 5 | -0.006 +- 0.025 | +0.000 | 10 | - | Floor |
| A4 | 4 | 1 | -0.011 +- 0.038 | +0.000 | 10 | - | Floor |
| A4 | 4 | 3 | -0.007 +- 0.032 | +0.000 | 10 | - | Floor |
| A4 | 4 | 5 | +0.015 +- 0.025 | +0.000 | 10 | - | Floor |
| A5 | 4 | 1 | +0.008 +- 0.028 | +0.015 | 10 | - | Floor |
| A5 | 4 | 3 | +0.005 +- 0.023 | +0.015 | 10 | - | Floor |
| A5 | 4 | 5 | +0.006 +- 0.040 | +0.015 | 10 | - | Floor |
| A6 | 4 | 1 | +0.030 +- 0.034 | +0.000 | 10 | - | Floor |
| A6 | 4 | 3 | +0.021 +- 0.027 | +0.000 | 10 | - | Floor |
| A6 | 4 | 5 | +0.014 +- 0.029 | +0.000 | 10 | - | Floor |
| A7 | 4 | 1 | +0.015 +- 0.025 | +0.000 | 10 | - | Floor |
| A7 | 4 | 3 | +0.023 +- 0.023 | +0.000 | 10 | - | Trainability-limited |
| A7 | 4 | 5 | +0.017 +- 0.030 | +0.000 | 10 | - | Floor |
| A2:amp | 4 | 1 | -0.027 +- 0.034 | +0.015 | 10 | - | Floor |
| A2:amp | 4 | 3 | -0.012 +- 0.018 | +0.015 | 10 | - | Floor |
| A2:amp | 4 | 5 | -0.006 +- 0.039 | +0.015 | 10 | - | Floor |
| A2:iqp | 4 | 1 | +0.029 +- 0.034 | +0.000 | 10 | - | Floor |
| A2:iqp | 4 | 3 | +0.022 +- 0.025 | +0.000 | 10 | - | Floor |
| A2:iqp | 4 | 5 | +0.008 +- 0.042 | +0.000 | 10 | - | Floor |

References (logistic regression, test MCC):

- n=4 logreg_frozen: +0.034 +- 0.020
- n=4 logreg_pca16: +0.015 +- 0.039
- n=4 logreg_pca3: +0.000 +- 0.000
- n=4 logreg_pca4: +0.000 +- 0.000
- n=4 logreg_tfidf: +0.088 +- 0.018
