# 06 · Energy Flow Polynomials

**Module:** `src/jettopo/features/efp.py` · **Status:** thin EnergyFlow wrapper, tested with energyflow 1.4.0 · **Tests:** `tests/test_06_efp.py`

## Objective
EFPs serve two purposes:

1. Ladder rung (ii): a linear model (logistic regression or Fisher LDA) on all EFPs with degree d ≤ 5 and β = 1.
2. Regression targets for the Q4 expressivity probe (spec 14): the named graphs in `PROBE_GRAPHS`, namely dumbbell, open path, triangle and square.

## API
- `efp_set(p4, mask, dmax=5, beta=1.0)` returns float[N, n_efps]. EnergyFlow uses `coords="ptyphim"`, and the input is (pT, y, φ) per real constituent.
- `efp_graph(p4, mask, graph)` returns one EFP.

## Acceptance tests
| ID | Check |
| --- | --- |
| T6.1 | Shape and finiteness |
| T6.2 | Dumbbell = Σ_ij z_i z_j θ_ij with θ measured in (dy, dphi), matching a manual numpy sum |
| T6.3 | Adding a zero-energy particle leaves the triangle EFP unchanged |

## Notes for the agent
- Cost: on 300k jets, d ≤ 5 is 102 EFPs (connected + disconnected; check the count EnergyFlow reports and record it). Use `n_jobs=-1` and cache the result to `data/top/efp_d5_{split}.npy`.
- Standardize EFPs (fit on the training subset only) before the linear model, because their scales span orders of magnitude. A log transform of positive EFPs is acceptable; document whichever you use.
