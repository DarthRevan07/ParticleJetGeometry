# 05 · Classical observables (ladder rung i)

**Module:** `src/jettopo/features/classical.py` · **Status:** mass, multiplicity, girth and pTD implemented; **`nsubjettiness` is a stub** · **Tests:** `tests/test_05_classical.py`

## Objective
Build the cheapest physics baseline: a gradient-boosted tree on {mass, multiplicity, girth, pTD, τ21, τ32}.

## To implement: `nsubjettiness(p4, mask, n_max=3, beta=1.0, R=0.8)`
- τ_N = Σ_i pT_i · min_k ΔR_{i,k}^β / (Σ_i pT_i · R^β), with axes from exclusive-kT clustering into N subjets.
- Recommended route: the scikit-hep `fastjet` package (`pip install fastjet`). Use `ClusterSequence(..., JetDefinition(kt_algorithm, R))`, then `exclusive_jets(N)` for the axes. Compute τ_N yourself in numpy from those axes, with ΔR measured in (y, φ).
- Use R = 0.8 for top tagging and R = 0.4 for quark/gluon. Make it an argument.
- A jet with fewer than N constituents has τ_N = 0.
- Return float[N_jets, n_max]. Ratios τ21 and τ32 are computed by the caller with a safe division.

## Verify before implementing
- T5.4 asserts that τ_N is non-increasing in N. With exclusive-kT axes this holds in practice but is not a theorem. If you find a counterexample on the toy jet, relax T5.4 to the toy fixture only and record why here.

## Acceptance tests
| ID | Check |
| --- | --- |
| T5.1 | Multiplicity |
| T5.2 | Single-particle girth = 0 and pTD = 1 |
| T5.3 | Feature table is finite |
| T5.4 | τ shape and monotonicity (stub) |
| T5.5 | Single particle gives τ1 = 0 (stub) |
| Real data | AUC of the tree on {mass, τ32} at 300k jets should be in the 0.97–0.98 range on top tagging. The literature's N-subjettiness DNN gets 0.981 on 1.2M jets. Record the value. |
