# 11 · Deep Sets (PFN, EFN) and kNN GNN baselines (ladder rung iii)

**Modules:** `models/deepsets.py` (STUB), `models/gnn.py` (ported), `models/jumping_knowledge.py` (ported) · **Tests:** `tests/test_11_deepsets_gnn.py`

## Objective
Provide PyTorch PFN and EFN that train with `training/trainer.py`, reproducing Komiske, Metodiev and Thaler (2019).

## To implement
- **PFN(in_dim=7, phi_sizes=(100, 100, 128), f_sizes=(100, 100, 100))**
  - Φ is a per-particle MLP with ReLU. Sum Φ over real constituents (mask), then F is an MLP ending in 1 logit.
  - Padded rows must not affect the output, whatever their values are (multiply by the mask after Φ).
- **EFN(phi_sizes, f_sizes)**
  - Φ acts on the coordinates (dy, dphi) only; the latent sum is Σ_i z_i Φ(x_i).
  - IRC safe by construction: zero-z particles contribute 0, and a split at the same position is linear in z.
- **Inputs from the trainer.** PFN gets `(features, mask)`; EFN gets `(z, coords, mask)`. Build tensors from `preprocess.constituent_features` and `relative_coords`. Standardize features using the training-subset statistics only.

## Verify the tests before implementing
- T11.4: permutation invariance follows from the sum.
- T11.5: padding values are ignored because of the mask.
- T11.6: the EFN IR and collinear checks are exact (up to float error) by linearity in z.

## Acceptance tests
| ID | Check |
| --- | --- |
| T11.1–2 | kNN edges (pass) |
| T11.3 | Jumping-knowledge shapes (pass) |
| T11.4–5 | PFN invariances (stub) |
| T11.6 | EFN IRC safety (stub) |

## Real-data target
At full 1.2M, PFN AUC is about 0.982 on top tagging (published 0.9819). This is the anchor run in spec 18 and the gate on 18 October. At 300k, record the value; expect it to be slightly lower.
