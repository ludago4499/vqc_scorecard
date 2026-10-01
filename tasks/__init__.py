# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# Any modification of this file must keep this notice, and must record the
# change in the module history below.
#
# History:
#   2026-09-09  Created. Task axis (supervised NLP classification) for the
#               A1-A4 families, built on the same Ansatz interface the
#               descriptor scorecard uses. Two routes: embedding (A1/A2/A3, and
#               A4 as a feature block) and compositional (A4, parse-driven word
#               boxes, sentence-level tasks only). Generalizes the old
#               old/HEA_scorecard/task_sentiment.py from A1-only to all four.
#   2026-09-22  A5 (hybrid transfer learning, Mari et al. 2020): the dressed
#               route, the frozen-encoder representation and get_features.
#   2026-09-23  The fair protocol (fair) and its report (fair_report).
#   2026-10-01  pure-v1. fair, fair_report and the Mari A5 are archived
#               (old/fair_v1_archive). New: harness (checkpoints, splits,
#               shared features), pure (the protocol), pure_report
#               (pre-registered hypotheses and result categories), stats.

"""Task axis: train the families as variational classifiers and score them.

Modules, in the order the data flows:

* :mod:`data`     - load and split IMDb-1k (movie_reviews CSV) and CoLA.
* :mod:`get_features` - one-shot frozen-encoder extraction (separate torch
  environment; writes the ``.npz`` cache the other modules read).
* :mod:`features` - TF-IDF -> PCA or frozen encoder -> PCA views (angles,
  standardised scores, raw), and the word-id vocabulary (compositional route).
* :mod:`harness`  - keyed checkpoints, the seeded train / validation / test
  split, and the shared frozen-feature store with per-seed PCA views.
* :mod:`model`    - the QNodes: the pure-v1 models (A1-A7 and the encoding-arm
  variants), the legacy feature route (A1-A4), the compositional A4 builder.
* :mod:`metrics`  - accuracy, precision, recall, F1, macro-F1, MCC.
* :mod:`baseline` - logistic-regression references (pure NumPy).
* :mod:`train`    - the Adam training loop and the per-run record.
* :mod:`pure`     - the pre-registered pure-v1 protocol: circuit-only training.
* :mod:`stats`    - paired contrasts and the frozen decision rules.
* :mod:`pure_report` - hypothesis verdicts, references and result categories.
* :mod:`run`, :mod:`present_task_results` - the legacy v10 runner and table.

Every value written to a task checkpoint is produced here; nothing is typed by
hand, the same rule the G-series reporting follows.
"""
