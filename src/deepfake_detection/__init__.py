"""Deepfake detection training and inference package."""

from .config import ProjectConfig, load_config
from .models import FrameClassifier, TemporalClassifier

__all__ = ["FrameClassifier", "ProjectConfig", "TemporalClassifier", "load_config"]
__version__ = "1.0.0"

