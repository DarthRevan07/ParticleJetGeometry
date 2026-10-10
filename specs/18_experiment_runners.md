# 18 · Experiment runners for Q1–Q3 and the results table

**Status:** to write · **Location:** `experiments/` scripts plus `results/`

## Common contract
- Every runner takes `--dataset {top,qg_pythia}`, `--train-size`, `--seeds`, and `--config configs/tiers.yaml`.
- Every runner appends one JSON line per (model, train_size, seed) to `results/results.jsonl` with these keys: model, dataset, train_size, seed, acc, auc, rej@30, rej@50, wall_seconds, n_params, git_hash, timestamp.
- `scripts/make_tables.py` aggregates mean ± std into Markdown and LaTeX tables, and writes the figures listed in the proposal.

## Runners
1. **`q1_ladder.py`:** classical + tree, EFP + linear, PFN, EFN, Rips-PH + tree, DTM-PH + tree, SCNN. Import ParticleNet results via spec 16.
   - Tiers: dev (30k) → main (300k, 5 seeds); SCNN at 100k.
   - Also produce the data-efficiency curve at 30k, 100k and 300k, plus the 1.2M anchors.
2. **`q2_attribution.py`** (quark/gluon Pythia, 300k):
   - (a) AUC of each feature set within multiplicity bins (5 quantile bins, test set).
   - (b) Stacking gain: tree on [EFP + multiplicity] vs [EFP + multiplicity + PH], with bootstrap CI on the AUC difference over test jets (paired bootstrap).
   - (c) H0-only vs H1-only vs both.
   - (d) Rips on (dy, dphi) vs (pT, dy, dphi).
3. **`q3_irc.py`:** spec 15.

## Gate (18 October)
The PFN anchor at full 1.2M training must reach AUC within 0.002 of 0.9819 on the full test set. If it does not, stop and debug the data pipeline (specs 01–03) before any topology runs.

## Definition of done for the whole project
- `pytest -q` shows no failures and no remaining xfails, except stretch modules that were explicitly deferred.
- `results/results.jsonl` holds every number in the report.
- `make_tables.py` regenerates every table and figure from it.
