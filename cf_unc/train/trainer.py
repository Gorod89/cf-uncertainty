"""Generic minibatch training loop with early stopping, shared by all torch models."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from typing import Callable, Mapping

import numpy as np
import torch
from torch import nn

Batch = dict[str, torch.Tensor]
LossFn = Callable[[nn.Module, Batch], torch.Tensor]


@dataclass
class TrainConfig:
    max_epochs: int = 40
    batch_size: int = 1024
    val_batch_size: int = 8192
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    patience: int = 6  # epochs without validation improvement before stopping
    min_delta: float = 1e-4
    lr_factor: float = 0.5  # ReduceLROnPlateau
    lr_patience: int = 2
    min_lr: float = 1e-5
    ema_decay: float = 0.0  # > 0: keep an exponential moving average of the weights and return it
    max_seconds: float | None = None

    @classmethod
    def from_any(cls, cfg: "TrainConfig | Mapping | None") -> "TrainConfig":
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        values = dict(cfg)
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown training options {unknown}; known: {sorted(cls.__dataclass_fields__)}")
        return cls(**values)


def _to_tensors(arrays: Mapping[str, np.ndarray | torch.Tensor], device: torch.device) -> Batch:
    out = {}
    for key, value in arrays.items():
        if value is None:
            continue
        t = torch.as_tensor(value)
        if t.is_floating_point():
            t = t.to(torch.float32)
        out[key] = t.to(device)
    sizes = {int(t.shape[0]) for t in out.values()}
    if len(sizes) != 1:
        raise ValueError(f"arrays have different lengths: {sizes}")
    return out


@torch.no_grad()
def evaluate_loss(module: nn.Module, loss_fn: LossFn, data: Batch, batch_size: int) -> float:
    """Sample-weighted mean of ``loss_fn`` over ``data`` with the module in eval mode."""
    was_training = module.training
    module.eval()
    n = next(iter(data.values())).shape[0]
    total = 0.0
    for start in range(0, n, batch_size):
        batch = {k: v[start : start + batch_size] for k, v in data.items()}
        m = next(iter(batch.values())).shape[0]
        total += float(loss_fn(module, batch)) * m
    module.train(was_training)
    return total / max(n, 1)


def fit_module(
    module: nn.Module,
    loss_fn: LossFn,
    train_arrays: Mapping[str, np.ndarray | torch.Tensor],
    val_arrays: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    cfg: TrainConfig | Mapping | None = None,
    *,
    seed: int = 0,
    device: str | torch.device = "cpu",
    verbose: bool = False,
    val_loss_fn: LossFn | None = None,
) -> dict:
    """Minimise ``loss_fn(module, batch)`` (a scalar mean over the batch).

    Early stopping and the returned weights use the validation loss
    (``val_loss_fn`` if given, else ``loss_fn``); without validation data the
    training loss of the epoch is used. The best weights are restored in place.
    """
    cfg = TrainConfig.from_any(cfg)
    device = torch.device(device)
    module.to(device)
    train = _to_tensors(train_arrays, device)
    val = _to_tensors(val_arrays, device) if val_arrays is not None else None
    val_loss_fn = val_loss_fn or loss_fn
    n = next(iter(train.values())).shape[0]
    if n == 0:
        raise ValueError("empty training set")

    params = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr
    )
    shuffle_rng = torch.Generator(device="cpu").manual_seed(int(seed))
    ema = copy.deepcopy(module).requires_grad_(False) if cfg.ema_decay > 0 else None

    history: list[dict] = []
    best_val, best_epoch, best_state = math.inf, -1, None
    started = time.perf_counter()
    for epoch in range(cfg.max_epochs):
        module.train()
        perm = torch.randperm(n, generator=shuffle_rng).to(device)
        running, seen = 0.0, 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            batch = {k: v[idx] for k, v in train.items()}
            loss = loss_fn(module, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            if ema is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema.parameters(), module.parameters()):
                        p_ema.lerp_(p, 1.0 - cfg.ema_decay)
                    for b_ema, b in zip(ema.buffers(), module.buffers()):
                        b_ema.copy_(b)
            running += float(loss.detach()) * idx.shape[0]
            seen += idx.shape[0]
        train_loss = running / seen
        scored = ema if ema is not None else module
        val_loss = evaluate_loss(scored, val_loss_fn, val, cfg.val_batch_size) if val is not None else train_loss
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss, "lr": lr})
        if verbose:
            print(f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}  lr {lr:.2e}")
        if val_loss < best_val - cfg.min_delta:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(scored.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            break
        if cfg.max_seconds is not None and time.perf_counter() - started > cfg.max_seconds:
            break

    if best_state is not None:
        module.load_state_dict(best_state)
    module.eval()
    return {
        "epochs": len(history),
        "best_epoch": best_epoch,
        "best_val": best_val,
        "seconds": time.perf_counter() - started,
        "n_train": int(n),
        "config": asdict(cfg),
        "history": history,
    }
