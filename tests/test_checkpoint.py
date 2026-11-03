from pathlib import Path

import torch

from deepfake_detection.config import ModelConfig
from deepfake_detection.models import FrameClassifier, checkpoint_metadata
from deepfake_detection.training import load_checkpoint


def test_checkpoint_load_is_strict(tmp_path: Path):
    config = ModelConfig(kind="tiny", pretrained=False)
    model = FrameClassifier(3, config)
    payload = checkpoint_metadata("frame", ["real", "filter", "deepfake"], config)
    payload["model_state_dict"] = model.state_dict()
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    loaded, metadata = load_checkpoint(path, torch.device("cpu"))
    assert metadata["classes"][2] == "deepfake"
    assert loaded(torch.randn(1, 3, 64, 64)).shape == (1, 3)


def test_legacy_checkpoint_detection_rejects_unknown_state(tmp_path: Path):
    path = tmp_path / "unknown.pt"
    torch.save({"model_state_dict": {"unknown.weight": torch.zeros(1)}}, path)
    try:
        load_checkpoint(path, torch.device("cpu"))
    except ValueError as error:
        assert "supported legacy architecture" in str(error)
    else:
        raise AssertionError("unknown legacy checkpoint was accepted")
