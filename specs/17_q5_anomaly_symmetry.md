# 17 · Q5 (stretch): is anomaly detection symmetric between QCD and top?

**Module:** `experiments/anomaly.py` · **Status:** STUB · **Tests:** `tests/test_17_anomaly.py`

## Background
Finke et al., JHEP 2021 (arXiv:2104.09051): an autoencoder trained on QCD flags tops, but one trained on tops fails to flag QCD.

## Functions
- `fit_density_score(train_features, method="gmm", seed)`:
  - Standardize the features.
  - Fit `sklearn.mixture.GaussianMixture` (default 8 components, full covariance), or a `kde` / `iforest` alternative.
  - Return a callable giving the negative log-likelihood (higher = more anomalous).
- `symmetry_gap(a, b)` = |a − b|.

## Experiment (agent writes `experiments/q5_anomaly.py`)
- Feature sets:
  - DTM-PH landscapes
  - Rips-PH landscapes
  - EFPs with d ≤ 4 (after PCA to 20 dimensions)
  - a small particle-cloud autoencoder's reconstruction error
- Train on QCD only and score QCD + top test jets: AUC(QCD → top). Train on top only and score: AUC(top → QCD).
- Report both AUCs and the gap, with 3 seeds and 100k training jets.

## Verify the tests before implementing
- T17.2: shifted Gaussian blobs at distance 3σ in 4-D. AUC > 0.95 is easily achievable with a GMM; confirm on a throwaway run.
