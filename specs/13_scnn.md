# 13 · Simplicial CNN classifier (ladder rung vi), repaired 2024 model

**Modules:** `models/scnn.py`, `topology/batching.py` · **Status:** ported, fixed, vectorized · **Tests:** `tests/test_13_scnn.py`

## What changed from 2024
- Physical simplex features replace `torch.rand`.
- Complexes are built on (dy, dphi).
- Block-diagonal batching replaces batch size 1 and the 8 GB of precomputed dense tensors.
- One logit replaces two logits fed to BCE.
- Chebyshev assembly is vectorized and checked against the 2024 loop (T13.1).

## Architecture
- For each dimension d ∈ {0, 1, 2}: a stack of `SimplicialConvolution(K=3)` on the normalized L_d, with LeakyReLU, then mean pooling per jet.
- Concatenate the three pooled vectors, then an MLP gives 1 logit.
- A dimension a jet lacks pools to zeros.

## Acceptance tests
| ID | Check |
| --- | --- |
| T13.1 | Vectorized Chebyshev equals the 2024 loop |
| T13.2 | Coboundary equals dense |
| T13.3 | scatter_mean |
| T13.4 | Batched forward equals one-by-one |
| T13.5 | Edgeless and triangle-free jets |
| T13.6 | Gradients reach every dimension |
| T13.7 | Shapes |

## Agent tasks before real runs
1. Write `topology/dataset.py`: a torch `Dataset` wrapping (p4, mask, label, indices) that builds `ComplexItem`s on the fly with `jet_to_complex_item`, plus a `DataLoader` using `collate_complexes`. Use `num_workers > 0`; complex construction is CPU-bound.
2. Time 1,000 jets of construction plus one forward/backward. If construction dominates, cache items with `pickle` per chunk of 10k jets for the 100k subset only.
3. Standardize simplex features using training-subset statistics.
4. Train at 100k with 3 seeds (tier "simplicial"). Report it next to PFN and ParticleNet at the same training size, not against full-data literature numbers.

## Known limitation to state in the report
This model has no messages between dimensions. The Q4 model (spec 14) adds node ↔ edge ↔ triangle messages.
