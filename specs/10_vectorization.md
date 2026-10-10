# 10 · Persistence vectorizations

**Module:** `topology/vectorize.py` · **Status:** STUB · **Tests:** `tests/test_10_vectorize.py`

## Objective
Turn diagrams into fixed-length vectors for gradient-boosted trees and MLPs. Landscapes are the main choice because they are 1-Lipschitz in bottleneck distance, so the IRC guarantee of spec 09 survives vectorization. Betti curves and images are ablations.

## Definitions
- **Landscape.** For bars (b, d), the tent is f(t) = max(0, min(t − b, d − t)). λ_k(t) is the k-th largest tent value at t (0 if fewer than k bars). Sample on `np.linspace(t_min, t_max, n_bins)` and return λ_1…λ_K concatenated.
- **Betti curve.** β(t) = #{bars : b ≤ t < d} on the same grid.
- **Persistence image.** Map to (birth, persistence), weight each point by persistence / max(extent persistence), place a Gaussian with σ, and integrate over a resolution² grid on `extent`. Pixel-centre evaluation times pixel area is acceptable; document which you use.
- **Summary statistics.** [n_bars, Σ persistence, max persistence, persistence entropy], where entropy = −Σ p_i log p_i with p_i = pers_i / Σ pers (0 for an empty diagram).

## Verify the tests before implementing
- T10.2: the tent of [0.2, 0.6].
- T10.3: Lipschitz in bottleneck. It holds for landscapes with matched or unmatched bars; check that gudhi's bottleneck treats diagonal projections the same way.
- T10.5: deliberately demonstrates that Betti curves are unstable.
- T10.7: hand computation.

## Acceptance tests
| ID | Check |
| --- | --- |
| T10.1 | Empty landscape |
| T10.2 | Single-bar tent |
| T10.3 | 1-Lipschitz |
| T10.4 | Betti counts |
| T10.5 | Betti instability |
| T10.6 | Image shape |
| T10.7 | Summary statistics |

## Defaults
Use t ∈ [0, 0.8] for top tagging (R = 0.8) and [0, 0.4] for quark/gluon. The DTM's values live on the same distance scale.
