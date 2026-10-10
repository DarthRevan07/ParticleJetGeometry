# 09 · IRC-safe persistence via the distance-to-measure (DTM) filtration (Q3, Claim B)

**Module:** `topology/dtm.py` · **Status:** STUB · **Tests:** `tests/test_09_dtm.py` (xfail until implemented)

## Objective
Implement a topological jet descriptor with a proven continuity bound in the energy mover's distance. By the Komiske–Metodiev–Thaler definition (JHEP 2020, Sec. 2.2), such a descriptor is IRC safe.

## Mathematics
- The jet measure is μ = Σ z_i δ_{x_i} on the box [−R, R]², with x_i = (dy_i, dphi_i) and z_i = pT_i / Σ pT. Points outside the box are clipped to its boundary; record how often this happens.
- The DTM with mass parameter m ∈ (0, 1] is d_{μ,m}(x)² = (1/m) ∫₀^m δ_{μ,t}(x)² dt, where δ_{μ,t}(x) = inf{r : μ(B(x, r)) ≥ t}.
- **Algorithm for a weighted point set at one query x:**
  1. Sort points by distance r_(1) ≤ r_(2) ≤ … with weights w_(1), w_(2), ….
  2. Accumulate weights until the cumulative weight reaches m; the last point contributes only the remainder.
  3. d² = (1/m) Σ_j w'_(j) r_(j)², where w' are the truncated weights summing to m.
- Vectorize over grid points. For 1,600 queries × ≤ 100 points, a dense distance matrix with `np.argsort` along the point axis is fine.
- Persistence: build a gudhi `CubicalComplex(top_dimensional_cells=values.reshape(n, n))` from the DTM values on the n × n grid of cell centres, then compute sublevel-set persistence for dims 0 and 1. Replace the essential H0 death with the grid maximum.
- **Guarantee** (Chazal, Cohen-Steiner, Mérigot 2011, FoCM; restated in Anai et al. 2019, Prop. 4.2):
  - ‖d_μ − d_ν‖_∞ ≤ m^{−1/2} W_2(μ, ν).
  - On a fixed grid, sublevel stability gives d_B ≤ the same bound.
  - On the box, W_2 ≤ √(D W_1) with D = 2√2 R.

## API (keep the stub signatures)
- `make_grid(radius, n)` returns n² cell centres in row-major order.
- `dtm_values(points, weights, queries, m)` normalizes the weights internally.
- `dtm_cubical_persistence(points, weights, m, radius, n_grid, homology_dims)`.

## Verify the tests before implementing (required)
Each test's expected value follows from the definitions above. Check each by hand before coding:
- T9.2: one point carries all the mass, so δ_{μ,t}(x) = |x − p| for every t.
- T9.3: for m ≤ ½ only the nearest point is used.
- T9.4: for m = 1, d² is the weighted mean squared distance.
- T9.6: moving each point by ≤ δ gives W_2 ≤ δ.
- T9.7: adding mass ε at distance ≤ D gives W_2 ≤ √ε·D.
- T9.8: splitting a point in place leaves μ unchanged.
- T9.9: sublevel stability.
- T9.10: the ring's DTM rises toward the centre.
- T9.11: the grid is symmetric under a quarter turn.

If any reasoning step fails, fix the test, explain why in the Changelog below, and keep the bound itself.

## Acceptance tests
| ID | Check |
| --- | --- |
| T9.1 | Grid |
| T9.2 | Single point |
| T9.3 | Two points, small m |
| T9.4 | m = 1 |
| T9.5 | Weight normalization |
| T9.6 | W_2 stability (3 values of m) |
| T9.7 | IR bound (ε = 1e-2, 1e-4, 1e-6) |
| T9.8 | Exact collinear split |
| T9.9 | Cubical diagrams finite and IR-stable |
| T9.10 | Ring loop |
| T9.11 | Quarter-turn invariance |

## Performance target
Under 2 ms per jet at n = 40 on a laptop core. Time 1,000 jets before any full run, and add `extract_dtm_features` beside the Rips extractor (spec 08).

## Changelog
- (empty)
