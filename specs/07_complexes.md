# 07 · Rips complexes, boundary matrices, Hodge Laplacians, simplex features

**Modules:** `topology/complexes.py`, `topology/simplex_features.py` · **Status:** ported, fixed · **Tests:** `tests/test_07_complexes.py`

## Objective
Build the clique (Vietoris–Rips) complex of a jet at a fixed scale r in the (dy, dphi) plane, up to triangles. Also provide signed boundary matrices B_1, B_2, the Hodge Laplacians L_0, L_1, L_2, and physical features on every simplex.

## Definitions
- Orientation: simplices are oriented by increasing vertex index. Constituents are stored in pT-descending order, so the orientation is canonical and the network is permutation invariant up to pT ties.
- Boundary convention: ∂[v0, v1] = v1 − v0 (column of B_1 = (−1, +1)), and in general B_d[face_i, σ] = (−1)^i, where face_i omits the i-th sorted vertex.
- Laplacians: L_0 = B_1 B_1ᵀ; L_1 = B_1ᵀ B_1 + B_2 B_2ᵀ; L_2 = B_2ᵀ B_2.
- `normalize_laplacian`: 2L/λ_max − I, so the spectrum lies in [−1, 1]. A zero Laplacian maps to −I.
- Features:
  - nodes: 5 E(2)-invariant constituent features (log pT, log E, log z, log E/E_jet, dR)
  - edges: [length, log z_i z_j]
  - triangles: [perimeter, area, log z_i z_j z_k]

## Acceptance tests
| ID | Check |
| --- | --- |
| T7.1 | Simplex counts |
| T7.2 | B_1 B_2 = 0 |
| T7.3 | L_0 is the graph Laplacian |
| T7.4 | dim ker L_1 = β_1 on a 4-cycle |
| T7.5 | A single-constituent jet is valid |
| T7.6 | Normalized spectrum is in [−1, 1] |
| T7.7 | Edgeless Laplacian handling |
| T7.8 | Orientation convention |
| T7.9 | Feature shapes and values |
| T7.10 | scipy → torch sparse round trip |

## Open question for the agent (do not change behavior without recording it)
- **Scale r.** The default is `max_edge_length=0.2` for top jets (R = 0.8). Before Q4/Q1-simplicial runs, measure the mean number of edges and triangles per jet at r ∈ {0.05, 0.1, 0.2, 0.3} on 1,000 jets. Pick the largest r with fewer than 2,000 triangles per jet on average, and record the table here.
- **Orientation equivariance.** Edge features are orientation-invariant scalars, but L_1 has orientation-dependent signs. This is consistent because orientation is canonical by pT order. State this in the report rather than claiming orientation equivariance.
