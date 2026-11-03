from __future__ import annotations

from collections import deque
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch

from .data import image_to_tensor
from .preprocessing import FaceDetector
from .training import load_checkpoint, resolve_device


def _detector_backend(checkpoint: dict) -> str:
    if str(checkpoint.get("checkpoint_format", "")).startswith("legacy_"):
        return "mediapipe"
    return "opencv"


def _face_tensor(image: np.ndarray, box: tuple[int, int, int, int]) -> torch.Tensor:
    x, y, width, height = box
    face = cv2.resize(image[y : y + height, x : x + width], (224, 224))
    return image_to_tensor(face)


def _probabilities(model, inputs: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        return torch.softmax(model(inputs), dim=1).cpu().numpy()[0]


def fuse_probabilities(
    frame_probabilities: np.ndarray,
    temporal_probabilities: np.ndarray,
    frame_weight: float = 0.4,
) -> np.ndarray:
    if not 0.0 <= frame_weight <= 1.0:
        raise ValueError("frame_weight must be between 0 and 1")
    if frame_probabilities.shape != temporal_probabilities.shape:
        raise ValueError("frame and temporal probability shapes must match")
    return (
        frame_weight * frame_probabilities
        + (1.0 - frame_weight) * temporal_probabilities
    )


def _prediction_result(classes: list[str], probabilities: np.ndarray) -> dict:
    index = int(np.argmax(probabilities))
    return {
        "class": classes[index],
        "confidence": float(probabilities[index]),
        "probabilities": {
            label: float(value)
            for label, value in zip(classes, probabilities, strict=True)
        },
    }


class FramePredictor:
    """Load one frame checkpoint and reuse it for in-memory predictions."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device_name: str | None = None,
        fallback_full_frame: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = resolve_device(device_name)
        self.model, self.checkpoint = load_checkpoint(self.checkpoint_path, self.device)
        if self.checkpoint["mode"] != "frame":
            raise ValueError("FramePredictor requires a frame checkpoint")
        self.classes = list(self.checkpoint["classes"])
        self.detector = FaceDetector(
            fallback_full_frame=fallback_full_frame,
            backend=_detector_backend(self.checkpoint),
        )
        self._lock = Lock()

    @property
    def metadata(self) -> dict:
        return {
            "mode": self.checkpoint["mode"],
            "classes": self.classes,
            "device": str(self.device),
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_format": self.checkpoint.get("checkpoint_format", "current"),
            "detector": self.detector.backend,
            "input_size": 224,
        }

    def predict(self, image: np.ndarray) -> list[dict]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected a BGR image with three channels")
        results = []
        with self._lock:
            boxes = self.detector.detect(image)
            for box in boxes:
                tensor = _face_tensor(image, box).unsqueeze(0).to(self.device)
                result = _prediction_result(
                    self.classes,
                    _probabilities(self.model, tensor),
                )
                x, y, width, height = box
                results.append(
                    {
                        **result,
                        "box": {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                        },
                    }
                )
        return results

    def close(self) -> None:
        self.detector.close()

    def __enter__(self) -> FramePredictor:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def _annotate(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    result: dict,
) -> None:
    x, y, width, height = box
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"{result['class']} {result['confidence']:.1%}",
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def predict_image(
    checkpoint_path: str,
    image_path: str,
    output_path: str,
    device_name: str | None = None,
) -> list[dict]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    with FramePredictor(checkpoint_path, device_name) as predictor:
        results = predictor.predict(image)
    for result in results:
        box = result.pop("box")
        _annotate(
            image,
            (box["x"], box["y"], box["width"], box["height"]),
            result,
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(output_path, image):
        raise OSError(f"failed to write output image: {output_path}")
    return results


def predict_video(
    checkpoint_path: str,
    video_path: str,
    output_path: str,
    device_name: str | None = None,
    sequence_length: int = 16,
) -> dict:
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"cannot create output video: {output_path}")
    sequence: deque[torch.Tensor] = deque(maxlen=sequence_length)
    prediction_counts = {label: 0 for label in checkpoint["classes"]}
    processed_frames = 0
    predicted_frames = 0
    with FaceDetector(
        fallback_full_frame=False,
        backend=_detector_backend(checkpoint),
    ) as detector:
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                boxes = detector.detect(frame)
                if boxes:
                    box = boxes[0]
                    tensor = _face_tensor(frame, box)
                    if checkpoint["mode"] == "frame":
                        model_input = tensor.unsqueeze(0).to(device)
                    else:
                        sequence.append(tensor)
                        if len(sequence) < sequence_length:
                            writer.write(frame)
                            processed_frames += 1
                            continue
                        model_input = torch.stack(list(sequence)).unsqueeze(0).to(device)
                    result = _prediction_result(
                        checkpoint["classes"],
                        _probabilities(model, model_input),
                    )
                    prediction_counts[result["class"]] += 1
                    predicted_frames += 1
                    _annotate(frame, box, result)
                writer.write(frame)
                processed_frames += 1
        finally:
            capture.release()
            writer.release()
    return {
        "mode": checkpoint["mode"],
        "processed_frames": processed_frames,
        "predicted_frames": predicted_frames,
        "prediction_counts": prediction_counts,
        "output": output_path,
    }


def predict_ensemble_video(
    frame_checkpoint_path: str,
    temporal_checkpoint_path: str,
    video_path: str,
    output_path: str,
    device_name: str | None = None,
    sequence_length: int = 16,
    frame_weight: float = 0.4,
) -> dict:
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if not 0.0 <= frame_weight <= 1.0:
        raise ValueError("frame_weight must be between 0 and 1")
    device = resolve_device(device_name)
    frame_model, frame_checkpoint = load_checkpoint(frame_checkpoint_path, device)
    temporal_model, temporal_checkpoint = load_checkpoint(
        temporal_checkpoint_path, device
    )
    if frame_checkpoint["mode"] != "frame":
        raise ValueError("the first ensemble checkpoint must be a frame model")
    if temporal_checkpoint["mode"] != "temporal":
        raise ValueError("the second ensemble checkpoint must be a temporal model")
    if frame_checkpoint["classes"] != temporal_checkpoint["classes"]:
        raise ValueError("ensemble checkpoints use different class orders")
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"cannot create output video: {output_path}")
    sequence: deque[torch.Tensor] = deque(maxlen=sequence_length)
    prediction_counts = {label: 0 for label in frame_checkpoint["classes"]}
    processed_frames = 0
    predicted_frames = 0
    backend = (
        "mediapipe"
        if "legacy_" in str(frame_checkpoint.get("checkpoint_format", ""))
        or "legacy_" in str(temporal_checkpoint.get("checkpoint_format", ""))
        else "opencv"
    )
    with FaceDetector(fallback_full_frame=False, backend=backend) as detector:
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                boxes = detector.detect(frame)
                if boxes:
                    box = boxes[0]
                    tensor = _face_tensor(frame, box)
                    sequence.append(tensor)
                    frame_probabilities = _probabilities(
                        frame_model,
                        tensor.unsqueeze(0).to(device),
                    )
                    probabilities = frame_probabilities
                    if len(sequence) == sequence_length:
                        temporal_probabilities = _probabilities(
                            temporal_model,
                            torch.stack(list(sequence)).unsqueeze(0).to(device),
                        )
                        probabilities = fuse_probabilities(
                            frame_probabilities,
                            temporal_probabilities,
                            frame_weight,
                        )
                    result = _prediction_result(
                        frame_checkpoint["classes"],
                        probabilities,
                    )
                    prediction_counts[result["class"]] += 1
                    predicted_frames += 1
                    _annotate(frame, box, result)
                writer.write(frame)
                processed_frames += 1
        finally:
            capture.release()
            writer.release()
    return {
        "mode": "ensemble",
        "processed_frames": processed_frames,
        "predicted_frames": predicted_frames,
        "prediction_counts": prediction_counts,
        "output": output_path,
    }
