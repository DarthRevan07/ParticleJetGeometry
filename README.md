# JetTopoBench

A benchmark-first study of what persistent homology and simplicial message passing add to jet tagging. It is the 2026 rebuild of the 2024 `GENIE_Simpl` project (kept unchanged in `../GENIE_Simpl` for reference).

**Start here:** `specs/00_overview.md`. It has the status of every module, the rules for coding agents, the compute tiers, and the list of 2024 bugs fixed by this port.

```bash
pip install -e ".[dev,physics,viz]"
pytest -q          # on a fresh port: 64 passed, 37 xfailed (stubs), 0 failed
```

## Layout

```
configs/tiers.yaml        training-subset sizes, seeds, where each tier runs
scripts/                  convert_top.py (h5 -> npy), make_splits.py (nested subsets)
specs/NN_*.md             one objective per file: API, math, acceptance tests, verify-first notes
src/jettopo/
  data/                   top_tagging, qg, preprocess (dy, dphi, z), splits
  features/               classical observables, EFPs
  topology/               Rips complexes + Hodge Laplacians, Rips persistence, DTM persistence (stub),
                          vectorizations (stub), simplex features, block-diagonal batching
  models/                 SCNN (ported, fixed), GNN + jumping knowledge (ported),
                          PFN/EFN (stub), energy-weighted simplicial MP (stub)
  training/               metrics (AUC, 1/eps_B), trainer with checkpoint/resume for Colab
  experiments/            IRC perturbations (stub), anomaly scoring (stub)
  viz/, utils/            plots, rich console, seeding
tests/test_NN_*.py        one file per spec; stubs show as XFAIL until implemented
```

## Compute plan

Train on 30k / 100k / 300k nested subsets of the official split, and always test on the full official test set. ParticleNet runs on a Colab T4 via the official ParT repository (spec 16). Full 1.2M-jet runs are used only for the anchor checks. The budget is roughly 20–25 Colab compute units in total.
