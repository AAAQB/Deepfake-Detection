from __future__ import annotations

import torch
from torch import nn


class LegacyFrameClassifier(nn.Module):
    """Architecture used by the imported original-branch frame checkpoint."""

    def __init__(self, num_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise RuntimeError(
                "The imported legacy checkpoints require the 'timm' package."
            ) from error
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=0
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.num_features, num_classes),
        )

    @property
    def encoder(self) -> LegacyFrameClassifier:
        """Expose the interface expected by the shared Grad-CAM implementation."""
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(inputs))

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

    def gradcam_target_layer(self) -> nn.Module:
        return self.backbone.blocks[-1]


class LegacyTemporalClassifier(nn.Module):
    """Architecture used by the imported original-branch temporal checkpoint."""

    def __init__(
        self,
        num_classes: int = 3,
        hidden_size: int = 512,
        num_layers: int = 4,
        dropout: float = 0.4,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.cnn = LegacyFrameClassifier(num_classes=num_classes, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=self.cnn.backbone.num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Dropout(dropout),
            nn.Linear(output_size, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = inputs.shape
        flat = inputs.reshape(batch * time, channels, height, width)
        features = self.cnn.extract_features(flat).reshape(batch, time, -1)
        output, _ = self.lstm(features)
        return self.classifier(output.mean(dim=1))


def build_legacy_model(state_dict: dict[str, torch.Tensor]) -> tuple[nn.Module, str]:
    keys = set(state_dict)
    if "head.1.weight" in keys and any(key.startswith("backbone.") for key in keys):
        return LegacyFrameClassifier(), "frame"
    if "cnn.head.1.weight" in keys and any(key.startswith("lstm.") for key in keys):
        return LegacyTemporalClassifier(), "temporal"
    raise ValueError("checkpoint does not match a supported legacy architecture")

