from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .config import ProjectConfig
from .data import create_loaders
from .training import load_checkpoint, resolve_device


def robustness_report(
    config: ProjectConfig, checkpoint_path: str, device_name: str | None = None
) -> dict[str, dict[str, float]]:
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint["mode"] != "frame":
        raise ValueError("robustness evaluation currently supports frame checkpoints")
    loader = create_loaders(config, "frame")["test"]
    corruptions: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "clean": lambda x: x,
        "noise": lambda x: x + torch.randn_like(x) * 0.08,
        "blur": lambda x: torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1),
        "dark": lambda x: x * 0.7,
    }
    report = {}
    for name, corruption in corruptions.items():
        truth, predicted = [], []
        with torch.no_grad():
            for inputs, labels in loader:
                logits = model(corruption(inputs.to(device)))
                truth.extend(labels.tolist())
                predicted.extend(logits.argmax(dim=1).cpu().tolist())
        report[name] = {
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(
                f1_score(truth, predicted, average="macro", zero_division=0)
            ),
        }
    return report


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, inputs: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        logits = self.model(inputs)
        target = int(logits.argmax(dim=1).item()) if class_index is None else class_index
        self.model.zero_grad(set_to_none=True)
        logits[:, target].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))[0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.detach().cpu().numpy()

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def save_gradcam(
    checkpoint_path: str,
    image_path: str,
    output_path: str,
    device_name: str | None = None,
) -> None:
    from .data import image_to_tensor

    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint["mode"] != "frame":
        raise ValueError("Grad-CAM requires a frame checkpoint")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    resized = cv2.resize(image, (224, 224))
    tensor = image_to_tensor(resized).unsqueeze(0).to(device)
    target_layer = model.encoder.gradcam_target_layer()
    cam = GradCAM(model, target_layer)
    try:
        heat = cam.generate(tensor)
    finally:
        cam.close()
    heat = cv2.resize(heat, (resized.shape[1], resized.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(resized, 0.6, heat, 0.4, 0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, overlay)
