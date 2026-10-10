# 03 · Splits and nested training subsets

**Module:** `src/jettopo/data/splits.py`, `scripts/make_splits.py` · **Status:** implemented · **Tests:** `tests/test_03_splits.py`

## Objective
Allow fair comparison on a laptop and Colab budget:

- Train on nested, class-stratified subsets of the official training split (30k ⊂ 100k ⊂ 300k).
- Validate on 100k of the official validation split.
- Always test on the full official test split.

Quark/gluon has no official split. Use files 0–15 for training, 16–17 for validation and 18–19 for testing, as ParT does.

## API
- `nested_subsets(labels, sizes, seed, stratify=True)` returns `{size: sorted indices}`.
- `save_subsets(path, subsets, meta)` and `load_subsets(path)`.
- `qg_split_indices(split)`.

## Acceptance tests
| ID | Check |
| --- | --- |
| T3.1 | Sizes are exact; indices are unique, sorted and in range |
| T3.2 | Subsets are nested |
| T3.3 | Class counts are within 2 jets of the proportional share |
| T3.4 | Same seed gives the same subset; a different seed gives a different one |
| T3.5 | Oversized request raises |
| T3.6 | Save/load round trip |
| T3.7 | QG file split is disjoint and complete (1.6M / 200k / 200k) |

## Rule
The subset files (`subsets_train.npz` + `.json`) are created once and committed as small artifacts. Every experiment loads them; none re-samples.
