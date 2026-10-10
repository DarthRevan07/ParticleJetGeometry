"""specs/03_splits.md: T3.x"""
import numpy as np
import pytest

from jettopo.data.splits import load_subsets, nested_subsets, qg_split_indices, save_subsets

LABELS = np.r_[np.ones(3000, int), np.zeros(7000, int)][np.random.default_rng(1).permutation(10000)]


def test_T3_1_sizes_sorted_unique_in_range():
    s = nested_subsets(LABELS, [100, 1000, 5000], seed=0)
    for k, v in s.items():
        assert len(v) == k and len(np.unique(v)) == k
        assert np.all(np.diff(v) > 0) and v.min() >= 0 and v.max() < len(LABELS)


def test_T3_2_nested():
    s = nested_subsets(LABELS, [37, 100, 1000, 5000], seed=3)
    assert set(s[37]) <= set(s[100]) <= set(s[1000]) <= set(s[5000])


def test_T3_3_stratified_within_one_jet_per_class():
    for n, v in nested_subsets(LABELS, [101, 999, 4321], seed=0).items():
        assert abs(LABELS[v].sum() - 0.3 * n) <= 2


def test_T3_4_deterministic_and_seed_dependent():
    a = nested_subsets(LABELS, [500], seed=0)[500]
    assert np.array_equal(a, nested_subsets(LABELS, [500], seed=0)[500])
    assert not np.array_equal(a, nested_subsets(LABELS, [500], seed=1)[500])


def test_T3_5_too_large_raises():
    with pytest.raises(ValueError):
        nested_subsets(LABELS, [20000])


def test_T3_6_save_load(tmp_path):
    s = nested_subsets(LABELS, [10, 20])
    save_subsets(tmp_path / "s.npz", s, {"seed": 0})
    t = load_subsets(tmp_path / "s.npz")
    assert all(np.array_equal(s[k], t[k]) for k in s)


def test_T3_7_qg_file_split_disjoint_and_complete():
    parts = [qg_split_indices(s) for s in ("train", "val", "test")]
    assert [len(p) for p in parts] == [1_600_000, 200_000, 200_000]
    assert len(np.unique(np.concatenate(parts))) == 2_000_000
