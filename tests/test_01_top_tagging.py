"""specs/01_data_top_tagging.md: T1.x"""
import numpy as np
import pandas as pd
import pytest

from jettopo.data.top_tagging import convert_split, frame_to_arrays, load_split


def test_T1_1_shapes_and_dtypes(toy_frame):
    out = frame_to_arrays(toy_frame, max_particles=4)
    assert out["p4"].shape == (2, 4, 4) and out["p4"].dtype == np.float32
    assert out["mask"].shape == (2, 4) and out["mask"].dtype == np.bool_
    assert out["label"].tolist() == [1, 0]


def test_T1_2_order_is_E_px_py_pz(toy_frame):
    out = frame_to_arrays(toy_frame, max_particles=2)
    np.testing.assert_allclose(out["p4"][0, 0], [7, 1, 3, 5])


def test_T1_3_mask_from_energy_and_padding_zeroed(toy_frame):
    out = frame_to_arrays(toy_frame, max_particles=4)
    assert out["mask"].tolist() == [[True, True, False, False], [True, False, False, False]]
    assert np.all(out["p4"][~out["mask"]] == 0)


def test_T1_4_all_zero_energy_gives_empty_jets(toy_frame):
    toy_frame[["E_0", "E_1"]] = 0
    assert frame_to_arrays(toy_frame)["mask"].sum() == 0


def test_T1_5_missing_columns_raise():
    with pytest.raises(KeyError):
        frame_to_arrays(pd.DataFrame({"AX_0": [1.0], "E_0": [1.0]}))


def test_T1_6_convert_roundtrip(tmp_path, toy_frame):
    pytest.importorskip("tables")
    h5 = tmp_path / "test.h5"
    toy_frame.astype("float64").assign(is_signal_new=[1, 0]).to_hdf(h5, key="table", format="table")
    convert_split(h5, tmp_path / "out", "test", max_particles=3, chunk_size=1)
    d = load_split(tmp_path / "out", "test")
    assert d["p4"].shape == (2, 3, 4) and d["label"].tolist() == [1, 0]
    sub = load_split(tmp_path / "out", "test", indices=np.array([1]))
    assert sub["label"].tolist() == [0]
