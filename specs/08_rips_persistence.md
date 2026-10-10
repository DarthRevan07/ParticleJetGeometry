# 08 · Vietoris–Rips persistence of jets (ladder rung v; Q2)

**Module:** `topology/persistence.py` · **Status:** ported, fixed · **Tests:** `tests/test_08_persistence.py`

## Objective
Compute H0 and H1 persistence diagrams of the unweighted constituent cloud in (dy, dphi). This is the method of Thomas, Demers, Krishnaswamy and Rieck (2022), which the proposal re-examines.

## API
- `rips_persistence(points, max_edge_length=inf, homology_dims=(0, 1), inf_value=None)` returns `{dim: [k, 2]}`. Zero-length bars are dropped.
- `bottleneck(d1, d2)`.
- `multiparameter_rips_codensity(...)` is exploratory only (multipers).

## Acceptance tests (also used as evidence in the report)
| ID | Check |
| --- | --- |
| T8.1 | H0 deaths on a line |
| T8.2 | A circle has one long H1 bar |
| T8.3 | inf replacement |
| T8.4 | **Symmetry lemma:** invariant under E(2) motions and permutations (bottleneck 0) |
| T8.5 | Stability: d_B ≤ 2δ when points move by ≤ δ |
| T8.6 | **Claim A, collinear part:** an exactly collinear duplicate changes nothing |
| T8.7 | **Claim A, infrared part:** an arbitrarily soft far particle changes H0 by about its distance, so the method is not IR safe |

## Batch extraction (agent to add: `topology/extract.py`)
- `extract_rips_features(p4, mask, n_jobs)` computes diagrams per jet in parallel with joblib and vectorizes them (spec 10). Cache to `data/top/rips_{split}_{size}.npz`.
- Before running on 300k jets, time 1,000 jets and record the projection. Expect about 1–5 ms per jet for 2-D Rips with ≤ 100 points. If slower, use `gudhi.AlphaComplex` for H1 and say so in the report: alpha and Rips filtrations differ, but both give valid diagrams.
