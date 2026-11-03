from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DataConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass(frozen=True)
class ManifestRow:
    path: str
    label: str
    modality: str
    source_id: str
    frame_index: int


class FaceDetector:
    """Face detector with MediaPipe support and an OpenCV fallback."""

    def __init__(
        self,
        fallback_full_frame: bool = True,
        backend: str = "opencv",
    ):
        if backend not in {"opencv", "mediapipe", "auto"}:
            raise ValueError("backend must be opencv, mediapipe, or auto")
        self.fallback_full_frame = fallback_full_frame
        self.backend = backend
        self._mediapipe_detector = None
        if backend in {"mediapipe", "auto"}:
            try:
                import mediapipe as mp

                self._mediapipe_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.5,
                )
                self.backend = "mediapipe"
            except (AttributeError, ImportError):
                if backend == "mediapipe":
                    raise RuntimeError(
                        "MediaPipe face detection is required for this operation. "
                        "Install the project with: pip install -e '.[legacy]'"
                    ) from None
                self.backend = "opencv"
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise RuntimeError(f"failed to load OpenCV face cascade: {cascade_path}")

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self.backend == "mediapipe":
            result = self._detect_mediapipe(image)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            boxes = self.cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            result = [(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes]
        if not result and self.fallback_full_frame:
            height, width = image.shape[:2]
            result = [(0, 0, width, height)]
        return sorted(result, key=lambda box: box[2] * box[3], reverse=True)

    def _detect_mediapipe(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self._mediapipe_detector.process(rgb).detections
        result = []
        for detection in detections or []:
            box = detection.location_data.relative_bounding_box
            x1 = max(0, int(box.xmin * width))
            y1 = max(0, int(box.ymin * height))
            x2 = min(width, int((box.xmin + box.width) * width))
            y2 = min(height, int((box.ymin + box.height) * height))
            if x2 > x1 and y2 > y1:
                result.append((x1, y1, x2 - x1, y2 - y1))
        return result

    def close(self) -> None:
        if self._mediapipe_detector is not None:
            self._mediapipe_detector.close()

    def __enter__(self) -> FaceDetector:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def _save_face(image: np.ndarray, box: tuple[int, int, int, int], output: Path, size: int):
    x, y, width, height = box
    face = image[y : y + height, x : x + width]
    if face.size == 0:
        raise ValueError("empty face crop")
    face = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, face.astype(np.uint8))


def _process_image(
    source: Path, destination: Path, label: str, detector: FaceDetector, config: DataConfig
) -> list[ManifestRow]:
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"cannot read image: {source}")
    source_id = f"image:{label}:{source.stem}"
    rows = []
    for face_index, box in enumerate(detector.detect(image)):
        output = destination / "image" / label / source.stem / f"{face_index:03d}.npy"
        _save_face(image, box, output, config.image_size)
        rows.append(ManifestRow(str(output), label, "image", source_id, 0))
    return rows


def _process_video(
    source: Path, destination: Path, label: str, detector: FaceDetector, config: DataConfig
) -> list[ManifestRow]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {source}")
    source_id = f"video:{label}:{source.stem}"
    rows: list[ManifestRow] = []
    frame_index = 0
    saved_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % config.frame_interval == 0:
                boxes = detector.detect(frame)
                if boxes:
                    output = (
                        destination
                        / "video"
                        / label
                        / source.stem
                        / f"{saved_index:06d}.npy"
                    )
                    _save_face(frame, boxes[0], output, config.image_size)
                    rows.append(ManifestRow(str(output), label, "video", source_id, frame_index))
                    saved_index += 1
            frame_index += 1
    finally:
        capture.release()
    return rows


def preprocess_dataset(config: DataConfig) -> Path:
    raw_root = Path(config.raw_root)
    output_root = Path(config.processed_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"raw dataset not found: {raw_root}")
    detector = FaceDetector(config.fallback_full_frame)
    rows: list[ManifestRow] = []
    failures: list[str] = []
    for modality, extensions in (("image", IMAGE_EXTENSIONS), ("video", VIDEO_EXTENSIONS)):
        for label in config.classes:
            class_root = raw_root / modality / label
            if not class_root.exists():
                continue
            sources = sorted(
                path for path in class_root.iterdir() if path.suffix.lower() in extensions
            )
            for source in sources:
                try:
                    if modality == "image":
                        rows.extend(_process_image(source, output_root, label, detector, config))
                    else:
                        rows.extend(_process_video(source, output_root, label, detector, config))
                except (OSError, ValueError) as error:
                    failures.append(f"{source}: {error}")
    if not rows:
        raise RuntimeError("preprocessing produced no samples")
    manifest = output_root / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ManifestRow.__annotations__))
        writer.writeheader()
        writer.writerows(row.__dict__ for row in rows)
    if failures:
        (output_root / "preprocessing_failures.txt").write_text(
            "\n".join(failures) + "\n", encoding="utf-8"
        )
    return manifest
