from pathlib import Path

from deepfake_detection.config import load_config


def test_default_config_loads():
    config = load_config(Path(__file__).parents[1] / "configs/default.yaml")
    assert config.data.classes == ["real", "filter", "deepfake"]
    assert config.training.train_fraction == 0.70

