from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    raw_root: str = "data_raw"
    processed_root: str = "data_processed"
    classes: list[str] = field(default_factory=lambda: ["real", "filter", "deepfake"])
    image_size: int = 224
    frame_interval: int = 5
    fallback_full_frame: bool = True


@dataclass
class ModelConfig:
    kind: str = "efficientnet_b4"
    pretrained: bool = True
    dropout: float = 0.4
    hidden_size: int = 512
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    bidirectional: bool = True


@dataclass
class TrainingConfig:
    seed: int = 42
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 32
    temporal_batch_size: int = 4
    sequence_length: int = 16
    sequence_stride: int = 4
    workers: int = 2
    epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    early_stopping_patience: int = 8
    checkpoint_root: str = "checkpoints"
    log_root: str = "logs"


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        fractions = (
            self.training.train_fraction,
            self.training.val_fraction,
            self.training.test_fraction,
        )
        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError("train/val/test fractions must sum to 1")
        if len(self.data.classes) < 2:
            raise ValueError("at least two classes are required")
        if self.model.kind not in {"efficientnet_b4", "tiny"}:
            raise ValueError("model.kind must be efficientnet_b4 or tiny")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ProjectConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = ProjectConfig(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )
    config.validate()
    return config

