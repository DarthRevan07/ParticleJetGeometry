"""Write the nested, stratified training subsets and the validation subset.

    python scripts/make_splits.py --data data/top --config configs/tiers.yaml
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import yaml

from jettopo.data.splits import nested_subsets, save_subsets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/top")
    ap.add_argument("--config", default="configs/tiers.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())["top_tagging"]
    data = pathlib.Path(args.data)
    seed = cfg["subsets_seed"]
    train_labels = np.load(data / "train_label.npy", mmap_mode="r")
    subsets = nested_subsets(np.asarray(train_labels), cfg["train_sizes"], seed=seed)
    save_subsets(data / "subsets_train.npz", subsets, {"split": "train", "seed": seed, "sizes": cfg["train_sizes"]})
    val_labels = np.load(data / "val_label.npy", mmap_mode="r")
    val = nested_subsets(np.asarray(val_labels), [cfg["val_size"]], seed=seed)
    save_subsets(data / "subsets_val.npz", val, {"split": "val", "seed": seed, "sizes": [cfg["val_size"]]})
    for k, v in subsets.items():
        print(f"train n={k}: top fraction {np.asarray(train_labels)[v].mean():.4f}")


if __name__ == "__main__":
    main()
