# 02 · Preprocessing: jet-relative kinematics

**Module:** `src/jettopo/data/preprocess.py` · **Status:** ported, fixed · **Tests:** `tests/test_02_preprocess.py`

## Objective
Give every constituent its coordinates relative to the jet, so that all later geometry (complexes, persistence, kNN graphs) is physical and dimensionless.

## Definitions
- Rapidity is y = ½ ln((E + pz)/(E − pz)). Pseudorapidity is η = asinh(pz/pT).
- The jet axis is the E-scheme four-vector sum of the real constituents.
- dy = y − y_jet. dphi = φ − φ_jet, wrapped to (−π, π]. dR = √(dy² + dphi²).
- z_i = pT_i / Σ pT, so Σ z = 1 per jet.
- `constituent_features` returns the float32 stack `[dy, dphi, log pT, log E, log z, log(E/E_jet), dR]`. Masked rows are 0.

## Acceptance tests
| ID | Check |
| --- | --- |
| T2.1 | y, η, φ, pT of a massless particle |
| T2.2 | dphi wraps across ±π |
| T2.3 | z sums to 1; masked entries are 0 |
| T2.4 | dy and dphi are invariant under a longitudinal boost (the symmetry lemma for the report) |
| T2.5 | Jet mass of two massless particles: m² = 2 pT1 pT2 (cosh Δy − cos Δφ) |
| T2.6 | Feature stack shape, finiteness, masked zeros |
| T2.7 | (pT, y, φ) → p4 round trip |

## Notes for the agent
- Use rapidity (not η) for all geometry. Keep the η option only to match ParticleNet/weaver inputs.
- Do not standardize features here. Standardization statistics belong to the training subset (spec 12) to avoid test leakage.
