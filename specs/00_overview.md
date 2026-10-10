# 00 · Overview and rules for coding agents

JetTopoBench is the 2026 port of the 2024 `GENIE_Simpl/Non_local_Jet_Classification_Tanmay_Bakshi` code. It supports the research proposal *What Does Topology See in a Jet?* (questions Q1–Q5). Each `specs/NN_*.md` file describes one modular objective: what to build, its public API, and the acceptance tests that decide when it is done.

## Status of every module

| Spec | Module | Status | Tests |
| --- | --- | --- | --- |
| 01 | `data/top_tagging.py` | Ported, fixed | `test_01_top_tagging.py` (pass) |
| 02 | `data/preprocess.py` | Ported, fixed | `test_02_preprocess.py` (pass) |
| 03 | `data/splits.py` | New, implemented | `test_03_splits.py` (pass) |
| 04 | `data/qg.py` | New, implemented, untested on real data | none offline (needs download) |
| 05 | `features/classical.py` | Implemented except `nsubjettiness` (stub) | `test_05_classical.py` |
| 06 | `features/efp.py` | Thin EnergyFlow wrapper | `test_06_efp.py` (pass when energyflow installed) |
| 07 | `topology/complexes.py`, `simplex_features.py` | Ported, fixed | `test_07_complexes.py` (pass) |
| 08 | `topology/persistence.py` | Ported, fixed | `test_08_persistence.py` (pass) |
| 09 | `topology/dtm.py` | **Stub** (core of Q3) | `test_09_dtm.py` (xfail) |
| 10 | `topology/vectorize.py` | **Stub** | `test_10_vectorize.py` (xfail) |
| 11 | `models/deepsets.py`, `models/gnn.py`, `models/jumping_knowledge.py` | Deep Sets **stub**; GNN, JK ported | `test_11_deepsets_gnn.py` |
| 12 | `training/metrics.py`, `training/trainer.py` | New, implemented | `test_12_metrics_trainer.py` (pass) |
| 13 | `models/scnn.py`, `topology/batching.py` | Ported, fixed, vectorized | `test_13_scnn.py` (pass) |
| 14 | `models/simplicial_mp.py` | **Stub** (Q4) | `test_14_q4_probe.py` (xfail) |
| 15 | `experiments/irc.py` | **Stub** (Q3) | `test_15_irc.py` (xfail) |
| 16 | ParticleNet via weaver / ParT repo | External; config only | manual smoke test |
| 17 | `experiments/anomaly.py` | **Stub** (Q5) | `test_17_anomaly.py` (xfail) |
| 18 | Experiment runners Q1–Q3 | To write | output-schema checks |

## Rules for any agent working here

1. **Verify the tests before implementing.** The stub tests (`xfail`) were written from the math in each spec but have not been run against a reference implementation. Before writing any code for a stub:
   - Read the spec and its test file side by side.
   - For each test, check by hand (or with a 10-line throwaway script) that the expected values follow from the spec's definitions.
   - If a test is wrong, fix the test, and record the reason in the spec's "Changelog" section. A test that contradicts its spec is a bug in the test, not permission to change the definition.
   - Never loosen a tolerance or delete an assertion just to make an implementation pass.
2. Run `pytest -q` before and after every change. Ported tests must stay green. A stub test that hits `NotImplementedError` is reported as XFAIL by `tests/conftest.py`; once the module is implemented, it must PASS.
3. Keep public function names and signatures from the stub files. Add new helpers freely.
4. Geometry is always (dy, dphi) relative to the jet axis, in dimensionless angular units. Never build complexes or distances on momentum components (that was the 2024 bug).
5. Every random operation takes a `seed`. Every reported number is mean ± std over seeds, evaluated on the **full** official test split.
6. Do not commit data, checkpoints, or credentials. The 2024 notebook leaked a Plotly API key, so treat secrets carefully.
7. Before any long run (> 10 minutes or any Colab job), time one epoch or 1,000 jets and log the projection.

## Data and compute tiers

Defined in `configs/tiers.yaml`. Training uses nested, stratified subsets of the official training split (30k ⊂ 100k ⊂ 300k). The full 1.2M split is used only for the anchor runs (ParticleNet, PFN; one seed). Every evaluation uses the full official test set.

## Setup

```bash
pip install -e ".[dev,physics,viz]"     # add [graphs] for torch_geometric, [trees] for xgboost
pytest -q                                # expect: all pass + xfails, nothing failing
python scripts/convert_top.py --raw data/raw --out data/top --download
python scripts/make_splits.py --data data/top --config configs/tiers.yaml
```

## Bugs fixed from 2024 (for the report's post-mortem)

- The dataset was top vs QCD (Zenodo 2603256), but the code and notes called it quark/gluon.
- The complexes were built on (p_y in GeV, phi in rad) with a Rips cut of 25. They now use (dy, dphi) with cuts of order 0.1–0.8.
- Simplex inputs were `torch.rand`. They are now physical features (spec 07).
- The training loop referenced undefined `j` and `labels_grnd`, had a stray `input()`, fed 2 logits to BCE, and effectively ran with batch size 1. It is replaced by `training/trainer.py` and block-diagonal batching.
- `build_laplacians` crashed on jets without edges, and errors were silently swallowed by try/except. Jets without edges are now handled explicitly.
- A field named `jet_pt` held per-particle pT. It is renamed `part_pt`.
- The multiparameter KDE was fitted on all columns but scored on `[:, :3]`. It now scores the same columns it was fitted on.
- Per-jet `.pt` graph files (1.1 GB) and dense Laplacian dumps (8 GB) are replaced by on-the-fly construction.
