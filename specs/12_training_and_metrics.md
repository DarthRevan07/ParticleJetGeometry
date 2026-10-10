# 12 · Training loop and metrics

**Modules:** `training/metrics.py`, `training/trainer.py`, `utils/seed.py` · **Status:** implemented · **Tests:** `tests/test_12_metrics_trainer.py`

## Metrics (literature conventions)
- **acc** at a probability threshold of 0.5, and **auc**.
- **rej@30 / rej@50:** 1/ε_B at signal efficiency 0.3 / 0.5, with ε_B linearly interpolated on the ROC curve.
- **bootstrap_ci:** percentile CI over test jets.
- **aggregate_seeds:** mean ± std (ddof = 1) over seeds.

## Trainer
- `fit(model, train_loader, val_loader, TrainConfig)` uses AdamW, a cosine LR schedule, BCE-with-logits on one logit, gradient clipping, and keeps the best epoch by validation AUC.
- With `ckpt_dir` set, it saves `last.pt` every epoch and resumes from it automatically. Point `ckpt_dir` at Google Drive on Colab.
- `predict(model, loader)` returns (labels, probabilities).

## Acceptance tests
| ID | Check |
| --- | --- |
| T12.1 | Perfect and random classifiers |
| T12.2 | Rejection on known uniform distributions |
| T12.3 | Bootstrap CI brackets the AUC |
| T12.4 | Seed aggregation |
| T12.5 | Learns separable data |
| T12.6 | Resume continues the epoch count |

## Agent tasks
- Add a `results` writer: one JSON line per (model, dataset, train_size, seed) with all metrics, wall time, parameter count and git hash, appended to `results/results.jsonl`. Spec 18 depends on this schema.
- Add optional AMP (`torch.autocast`) for CUDA only. Test that a CPU run ignores it.
