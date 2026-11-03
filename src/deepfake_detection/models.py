from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn
from torchvision import models

from .config import ModelConfig


class FrameEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_config = config
        if config.kind == "tiny":
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.output_dim = 32
        else:
            weights = models.EfficientNet_B4_Weights.DEFAULT if config.pretrained else None
            backbone = models.efficientnet_b4(weights=weights)
            self.features = nn.Sequential(backbone.features, backbone.avgpool)
            self.output_dim = backbone.classifier[1].in_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(inputs), 1)

    def gradcam_target_layer(self) -> nn.Module:
        if self.model_config.kind == "tiny":
            return self.features[4]
        return self.features[0][-1]


class FrameClassifier(nn.Module):
    def __init__(self, num_classes: int, config: ModelConfig):
        super().__init__()
        self.model_config = config
        self.encoder = FrameEncoder(config)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.encoder.output_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(inputs))


class TemporalClassifier(nn.Module):
    def __init__(self, num_classes: int, config: ModelConfig):
        super().__init__()
        self.model_config = config
        self.encoder = FrameEncoder(config)
        self.lstm = nn.LSTM(
            input_size=self.encoder.output_dim,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.lstm_dropout if config.lstm_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )
        output_dim = config.hidden_size * (2 if config.bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Dropout(config.dropout),
            nn.Linear(output_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = inputs.shape
        encoded = self.encoder(inputs.reshape(batch * time, channels, height, width))
        sequence = encoded.reshape(batch, time, -1)
        output, _ = self.lstm(sequence)
        return self.classifier(output.mean(dim=1))


def build_model(mode: str, num_classes: int, config: ModelConfig) -> nn.Module:
    if mode == "frame":
        return FrameClassifier(num_classes, config)
    if mode == "temporal":
        return TemporalClassifier(num_classes, config)
    raise ValueError(f"unsupported model mode: {mode}")


def checkpoint_metadata(mode: str, classes: list[str], config: ModelConfig) -> dict:
    return {"mode": mode, "classes": classes, "model_config": asdict(config)}
