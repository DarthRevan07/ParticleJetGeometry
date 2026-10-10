# 14 · Q4: can simplicial message passing represent 3-point energy correlators?

**Module:** `models/simplicial_mp.py` · **Status:** STUB (stretch goal) · **Tests:** `tests/test_14_q4_probe.py`

## Hypothesis
Energy-weighted pairwise message passing at depth 1 cannot fit the triangle EFP (Σ z_i z_j z_k θ_ij θ_jk θ_ki) to R² > 0.95. Energy-weighted simplicial message passing on the Rips 2-complex can. Motivation: Konar, Ngairangbam and Spannowsky, JHEP 2024 (arXiv:2309.17351), show that EFN/EMPN only reach N = 2^k-point correlators.

## Models (parameter counts matched within 10%)
- **EnergyWeightedPairMP**
  - Node states h_i = MLP(node features).
  - One round of messages along edges: m_i = Σ_j z_j ψ(h_i, h_j, θ_ij).
  - Readout Σ_i z_i ρ(m_i).
- **EnergyWeightedSimplicialMP**
  - Same node update, plus triangle states t_σ = φ(θ_ij, θ_jk, θ_ki) weighted by z_i z_j z_k, passed to edges through |B_2| and to nodes through |B_1|.
  - Readout over nodes and triangles, both energy-weighted.
- **EFN baseline** from spec 11.
- **IR safety.** Every message and readout is multiplied by the relevant z product, so zero-energy simplices contribute nothing (T14.3). The graph must also be IRC-invariant; Rips at fixed r satisfies this only for collinear splits. Note this limitation.

## Probe protocol (agent writes `experiments/q4_probe.py`)
1. Use 200k top-tagging jets (no labels needed). Targets are `efp_graph` for the dumbbell, open path, triangle and square; log-transform and standardize them.
2. Train each model as a regressor (MSE) with layers ∈ {1, 2}, 3 seeds each, and report test R² per target.
3. Expected table: the triangle row separates the models; the dumbbell row does not.

## Verify the tests before implementing
- T14.3: with z = 0 on the new node, all its messages and readout terms vanish. This holds only if no unweighted term (for example, a degree normalization) touches the node. Avoid mean-aggregation for that reason.

## Acceptance tests
| ID | Check |
| --- | --- |
| T14.1 | Shapes |
| T14.2 | Parameter match |
| T14.3 | Zero-energy node invariance |
