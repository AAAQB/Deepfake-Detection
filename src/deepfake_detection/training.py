from __future__ import annotations

import json
import random
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import ModelConfig, ProjectConfig
from .data import create_loaders
from .legacy_models import build_legacy_model
from .models import build_model, checkpoint_metadata


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, labels in tqdm(loader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
    }


def save_checkpoint(path: Path, model, optimizer, epoch, metrics, mode, config):
    payload = checkpoint_metadata(mode, config.data.classes, config.model)
    payload.update(
        {
            "epoch": epoch,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _safe_load_checkpoint(path: str | Path, device: torch.device) -> dict:
    numpy_multiarray = import_module("numpy.core.multiarray")
    numpy_dtype_types = {
        type(np.dtype(np.int64)),
        type(np.dtype(np.float64)),
        type(np.dtype(np.float32)),
        type(np.dtype(np.bool_)),
    }
    safe_globals = [
        (np.ndarray, "numpy.ndarray"),
        (np.dtype, "numpy.dtype"),
        (
            numpy_multiarray._reconstruct,
            "numpy.core.multiarray._reconstruct",
        ),
        *numpy_dtype_types,
    ]
    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a dictionary")
    return checkpoint


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = _safe_load_checkpoint(path, device)
    required = {"mode", "classes", "model_config", "model_state_dict"}
    missing = required - set(checkpoint)
    if not missing:
        model_config = ModelConfig(**checkpoint["model_config"])
        model = build_model(checkpoint["mode"], len(checkpoint["classes"]), model_config)
    elif "model_state_dict" in checkpoint:
        model, mode = build_legacy_model(checkpoint["model_state_dict"])
        checkpoint = {
            **checkpoint,
            "mode": mode,
            "classes": ["real", "filter", "deepfake"],
            "checkpoint_format": (
                "legacy_timm_efficientnet_b4"
                if mode == "frame"
                else "legacy_timm_efficientnet_b4_lstm"
            ),
        }
    else:
        raise ValueError(f"checkpoint is missing required fields: {sorted(missing)}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def train(config: ProjectConfig, mode: str, device_name: str | None = None) -> Path:
    seed_everything(config.training.seed)
    device = resolve_device(device_name)
    loaders = create_loaders(config, mode)
    model = build_model(mode, len(config.data.classes), config.model).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.training.epochs
    )
    run_name = f"{mode}_{config.model.kind}"
    checkpoint_dir = Path(config.training.checkpoint_root) / run_name
    writer = SummaryWriter(Path(config.training.log_root) / run_name)
    best_score = -1.0
    stale_epochs = 0
    history = []
    try:
        for epoch in range(config.training.epochs):
            train_metrics = _run_epoch(model, loaders["train"], criterion, device, optimizer)
            val_metrics = _run_epoch(model, loaders["val"], criterion, device)
            scheduler.step()
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            for split, metrics in (("train", train_metrics), ("val", val_metrics)):
                for name, value in metrics.items():
                    writer.add_scalar(f"{split}/{name}", value, epoch)
            is_best = val_metrics["macro_f1"] > best_score
            save_checkpoint(
                checkpoint_dir / "last.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                mode,
                config,
            )
            if is_best:
                best_score = val_metrics["macro_f1"]
                stale_epochs = 0
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    val_metrics,
                    mode,
                    config,
                )
            else:
                stale_epochs += 1
            print(
                f"epoch={epoch + 1} train_loss={train_metrics['loss']:.4f} "
                f"val_f1={val_metrics['macro_f1']:.4f}"
            )
            if stale_epochs >= config.training.early_stopping_patience:
                break
    finally:
        writer.close()
    (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_path = checkpoint_dir / "best.pt"
    model, _ = load_checkpoint(best_path, device)
    test_metrics = _run_epoch(model, loaders["test"], criterion, device)
    (checkpoint_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps({"held_out_test": test_metrics}, indent=2))
    return best_path


def evaluate_checkpoint(
    config: ProjectConfig, checkpoint_path: str | Path, device_name: str | None = None
) -> dict:
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    loaders = create_loaders(config, checkpoint["mode"])
    criterion = nn.CrossEntropyLoss()
    return _run_epoch(model, loaders["test"], criterion, device)
