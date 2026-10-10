"""Download (optional) and convert the top-tagging .h5 files to .npy arrays.

Port of the 2024 ``datasets.py __main__`` block, without the 100k/30k/10k truncation.

    python scripts/convert_top.py --raw data/raw --out data/top --download
"""
from __future__ import annotations

import argparse
import logging
import pathlib

from jettopo.data.top_tagging import URLS, convert_split, download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="folder holding train.h5 / val.h5 / test.h5")
    ap.add_argument("--out", default="data/top")
    ap.add_argument("--max-particles", type=int, default=100)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N jets (for smoke tests)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = pathlib.Path(args.raw)
    for split in args.splits:
        h5 = raw / f"{split}.h5"
        if args.download and not h5.exists():
            download(URLS[split], h5)
        convert_split(h5, args.out, split, max_particles=args.max_particles, limit=args.limit)


if __name__ == "__main__":
    main()
