import numpy as np
import pytest
import torch

from deepfake_detection.config import ModelConfig
from deepfake_detection.inference import fuse_probabilities
from deepfake_detection.models import FrameClassifier, TemporalClassifier


def test_tiny_frame_forward():
    config = ModelConfig(kind="tiny", pretrained=False)
    model = FrameClassifier(3, config)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 3)


def test_tiny_temporal_forward():
    config = ModelConfig(
        kind="tiny", pretrained=False, hidden_size=16, lstm_layers=1, bidirectional=False
    )
    model = TemporalClassifier(3, config)
    assert model(torch.randn(2, 4, 3, 64, 64)).shape == (2, 3)


def test_probability_fusion():
    frame = np.array([0.7, 0.2, 0.1])
    temporal = np.array([0.1, 0.3, 0.6])
    fused = fuse_probabilities(frame, temporal, frame_weight=0.25)
    np.testing.assert_allclose(fused, np.array([0.25, 0.275, 0.475]))


def test_probability_fusion_rejects_invalid_weight():
    with pytest.raises(ValueError, match="frame_weight"):
        fuse_probabilities(np.ones(3), np.ones(3), frame_weight=1.1)
