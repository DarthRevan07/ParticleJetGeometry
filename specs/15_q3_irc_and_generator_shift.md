# 15 · Q3: IRC stress tests and the Pythia → Herwig shift

**Module:** `experiments/irc.py` · **Status:** STUB · **Tests:** `tests/test_15_irc.py`

## Functions to implement
- `add_soft_emission(p4, mask, energy_fraction, dy, dphi)`: add a massless particle with pT = fraction × jet pT at (y_jet + dy, φ_jet + dphi), in the first free padding slot. Do not mutate the inputs.
- `collinear_split(p4, mask, index, fraction, angle=0)`:
  - With angle = 0, the daughters are fraction × p and (1 − fraction) × p, so four-momentum is exactly conserved.
  - With angle > 0, place massless daughters at ± angle/2 about the parent's (y, φ) along a fixed direction (for example, along φ). Rescale so the total pT equals the parent's pT, and document it.
- `emd(...)`: the energy mover's distance between normalized pT measures with β = 1 and radius R. Use `energyflow.emd.emd(ev0, ev1, R=R, norm=True)`, or POT if unavailable. Check which convention EnergyFlow uses for `norm` and record it.

## Experiments (agent writes `experiments/q3_irc.py`)
1. **Response curves.** Take 2,000 test jets and ε ∈ {1e-1, …, 1e-6}.
   - For each jet, apply (a) a soft emission at a random angle in [0.3, 0.7] and (b) a collinear split of the hardest particle, with angle ∈ {0, 1e-3, 1e-2}.
   - Plot median |Δ classifier score| and median bottleneck change versus ε for:
     - Rips-PH + tree
     - DTM-PH + tree (m ∈ {0.01, 0.03, 0.1})
     - PFN
     - EFN
   - Expected: DTM and EFN go to 0 as ε → 0; Rips and PFN do not.
   - Overlay the theoretical bound m^{−1/2} √(2R·W_1) using the measured EMD.
2. **Generator shift.**
   - Train on Pythia quark/gluon (300k) and test on both Pythia and Herwig (200k each) for the same models.
   - Report ΔAUC = AUC_Pythia − AUC_Herwig, mean ± std over 5 seeds.
   - Hypothesis: ΔAUC is smaller for the IRC-safe models.

## Verify the tests before implementing
- T15.1: rapidity of the new particle relative to the jet. The jet axis moves slightly after the addition, so the test compares against the pre-addition axis. Confirm the tolerance 1e-6 is achievable for fraction 1e-3, or loosen it to about 1e-3 × 0.5 and record why.
- T15.4: the EMD of a 1e-4 emission at distance ~0.7 with R = 0.8 should be about 1e-4 × 0.7/0.8. Check that EnergyFlow's normalization agrees.
