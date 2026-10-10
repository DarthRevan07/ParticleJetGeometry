# 04 · Quark/gluon datasets (Pythia8 and Herwig7.1)

**Module:** `src/jettopo/data/qg.py` · **Status:** implemented, not yet run on real data · **Tests:** none offline

## Objective
Load the EnergyFlow quark/gluon jets for Q2 and for the Pythia → Herwig shift test in Q3.

- Pythia8 is Zenodo 3164691; Herwig7.1 is Zenodo 3066475.
- Each has 20 files × 100k jets: R = 0.4, pT 500–550 GeV, |y| < 1.7, with PID.

Output follows the package convention, plus `pid`. Label 1 = quark.

## Verify before use (agent)
1. Download 1 file per generator with `energyflow.qg_jets.load(num_data=100000, generator=...)`.
2. Check that the column order is (pT, y, φ, pid), and record the energyflow version in this spec.
3. Check that the label balance is 50/50 per file.
4. Check that `load_qg(split="test")` returns exactly rows 1.8M–2.0M (files 18–19), and that it loads only the files needed. If energyflow loads everything up to `num_data`, document the RAM cost (2M × 100 × 4 × 4 B ≈ 3.2 GB float32) and the workaround.
5. Write a smoke test `tests/test_04_qg.py` marked `needs_data` covering items 2–4.

## Notes
- Use no PID features in the topological models. PID is IRC-unsafe information and would confound Q2/Q3. The PID-using baselines are cited from the literature, not rerun.
