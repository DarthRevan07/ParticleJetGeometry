# 01 · Top-tagging data I/O

**Module:** `src/jettopo/data/top_tagging.py` · **Status:** ported, fixed · **Tests:** `tests/test_01_top_tagging.py`

## Objective
Convert the official Kasieczka et al. HDF5 files (Zenodo 2603256; train ≈ 1.2M, val ≈ 400k, test ≈ 400k jets) into memory-mappable arrays:

- `p4`: float32[N, P, 4] = (E, px, py, pz)
- `mask`: bool[N, P], true iff E > 0
- `label`: int8[N], 1 = top

Default P = 100.

## API
- `frame_to_arrays(df, max_particles)` converts one DataFrame chunk to the dict above.
- `convert_split(h5_path, out_dir, split, max_particles=100, chunk_size=100_000, limit=None)` writes `{split}_{p4,mask,label}.npy`.
- `load_split(data_dir, split, indices=None, mmap=True)` loads them back.
- `extract_coords_awkward(df)` is the legacy awkward/vector view, for plotting only.

## Acceptance tests
| ID | Check |
| --- | --- |
| T1.1 | Shapes and dtypes |
| T1.2 | Column order is (E, px, py, pz) |
| T1.3 | Mask comes from E > 0; padded entries are exactly 0 |
| T1.4 | All-zero energies give empty jets |
| T1.5 | Missing columns raise KeyError |
| T1.6 | HDF5 → npy round trip, including subset loading |

## Real-data checks (agent runs once after download, records results in this file)
- Row counts per split match the Zenodo description. Record the exact numbers here.
- Label balance is about 50/50 in every split.
- Fraction of jets with more than 100 constituents. ParticleNet reports that truncating at 100 is harmless; confirm it is < 1%.
- Conversion time and peak RAM for `train`. The target is < 6 GB RAM so it runs on Colab and on a laptop.

## Changelog
- 2026-10: ported. The `.npy` header bug (a numpy int in the shape) is fixed with an `int()` cast.
