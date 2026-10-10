# 16 · ParticleNet (ladder rung iv) through the official ParT repository

**Status:** external; no code in this package · **Test:** manual smoke run

## Decision
Do not re-implement ParticleNet. Use `jet-universe/particle_transformer`, which ships training scripts for TopLandscape (the same Kasieczka dataset) and QuarkGluon, with ParT, ParticleNet (PN), PFN and P-CNN. The 2024 vendored `weaver-benchmark` configs use the deprecated awkward-0 API (`awkward.JaggedArray`) and are kept only for reference.

## Steps (Colab T4)
1. `pip install 'weaver-core>=0.4'`, then clone `https://github.com/jet-universe/particle_transformer`.
2. `./get_datasets.py TopLandscape -d /content/drive/MyDrive/jetdata`. This downloads the repo's parquet version of the same dataset.
3. Smoke test: run `./train_TopLandscape.sh PN kin --num-epochs 1` on a 10k-jet subset. To build the subset, write the parquet rows for the indices in `subsets_train.npz` with pyarrow, or use weaver's `--data-fraction` on a single file. Time it.
4. Main tier: PN at 300k, 5 seeds, 20 epochs. Use **the same 300k indices** as spec 03 by exporting them to parquet. Evaluate on the full test file.
5. Anchor: PN on the full train set, 1 seed, 20 epochs, with `--batch-size 384` and the default one-cycle schedule.
6. Convert weaver's prediction ROOT/parquet output into `results/results.jsonl` (spec 12 schema) with a small adapter script, `scripts/import_weaver_predictions.py`.

## Verify before relying on it
- Confirm the TopLandscape parquet test file has the same jets as the official `test.h5`. Compare jet counts and the first 10 jets' summed four-momenta against `data/top/test_p4.npy`. If the order differs, match on four-momentum.
- Record the weaver-core version, the git commit of the ParT repo, and the exact command lines in this spec.

## Budget
About 1 hour per 300k run on a T4, and 3–4 hours for the full anchor. Measure in step 3 and update `configs/tiers.yaml`.
