"""One training loop for every torch model in the package.

Replaces the 2024 ``scnn/model.py::main`` loop (undefined ``j`` / ``labels_grnd``,
stray ``input()``, BCE on two logits, no seeds, no checkpoints). Designed for Colab:
it checkpoints every epoch and resumes from the last checkpoint after a disconnect.

Batches are ``(inputs, y)``. ``inputs`` may be a tensor, tuple, dict, or an object
with a ``.to(device)`` method (e.g. ComplexBatch); the model returns one logit per
example. Spec: specs/12_training_and_metrics.md
"""
from __future__ import annotations

import json
import logging
import pathlib
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from .metrics import benchmark_metrics

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    monitor: str = "auc"
    device: str = "auto"
    ckpt_dir: str | None = None
    grad_clip: float | None = 5.0
    amp: bool = False


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _move(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {k: _move(v, device) for k, v in inputs.items()}
    if isinstance(inputs, (tuple, list)):
        return type(inputs)(_move(v, device) for v in inputs)
    return inputs


def _call(model, inputs):
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


@torch.no_grad()
def predict(model: nn.Module, loader, device: str | torch.device = "auto") -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, probabilities) over a loader."""
    dev = _device(device) if isinstance(device, str) else device
    model.eval().to(dev)
    ys, ps = [], []
    for inputs, y in loader:
        logits = _call(model, _move(inputs, dev))
        ps.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(np.asarray(y.cpu() if torch.is_tensor(y) else y))
    return np.concatenate(ys), np.concatenate(ps)


def fit(model: nn.Module, train_loader, val_loader, cfg: TrainConfig) -> dict:
    """Train with AdamW + cosine LR, keep the best epoch by validation ``cfg.monitor``.

    Returns {"best_epoch", "best_val", "history"} and leaves the best weights loaded.
    """
    dev = _device(cfg.device)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    loss_fn = nn.BCEWithLogitsLoss()
    ckpt = pathlib.Path(cfg.ckpt_dir) if cfg.ckpt_dir else None
    start, best, best_epoch, history = 0, -np.inf, -1, []
    if ckpt and (ckpt / "last.pt").exists():
        state = torch.load(ckpt / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start, best, best_epoch, history = state["epoch"] + 1, state["best"], state["best_epoch"], state["history"]
        log.info("resumed from epoch %d", start)
    for epoch in range(start, cfg.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for inputs, y in train_loader:
            inputs, y = _move(inputs, dev), y.to(dev).float()
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(_call(model, inputs), y)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        sched.step()
        yv, pv = predict(model, val_loader, dev)
        m = benchmark_metrics(yv, pv)
        rec = {"epoch": epoch, "train_loss": tot / max(n, 1), "seconds": time.time() - t0, **{f"val_{k}": v for k, v in m.items()}}
        history.append(rec)
        log.info(json.dumps(rec))
        if m[cfg.monitor] > best:
            best, best_epoch = m[cfg.monitor], epoch
            if ckpt:
                ckpt.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt / "best.pt")
            else:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ckpt:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
                        "best": best, "best_epoch": best_epoch, "history": history, "cfg": asdict(cfg)}, ckpt / "last.pt")
    if ckpt and (ckpt / "best.pt").exists():
        model.load_state_dict(torch.load(ckpt / "best.pt", map_location=dev))
    elif not ckpt and best_epoch >= 0:
        model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_val": best, "history": history}
