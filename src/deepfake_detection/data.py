from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import ProjectConfig

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Sample:
    path: str
    label: str
    modality: str
    source_id: str
    frame_index: int
    split: str = ""


def read_manifest(path: str | Path) -> list[Sample]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            Sample(
                path=row["path"],
                label=row["label"],
                modality=row["modality"],
                source_id=row["source_id"],
                frame_index=int(row["frame_index"]),
                split=row.get("split", ""),
            )
            for row in csv.DictReader(handle)
        ]


def _split_groups(
    groups_by_class: dict[str, list[str]], val_fraction: float, test_fraction: float, seed: int
) -> dict[str, str]:
    assignments = {}
    for class_index, (label, groups) in enumerate(sorted(groups_by_class.items())):
        shuffled = sorted(groups)
        random.Random(seed + class_index).shuffle(shuffled)
        count = len(shuffled)
        test_count = max(1, round(count * test_fraction)) if count >= 3 and test_fraction else 0
        val_count = max(1, round(count * val_fraction)) if count >= 3 and val_fraction else 0
        while test_count + val_count >= count:
            if test_count >= val_count and test_count:
                test_count -= 1
            elif val_count:
                val_count -= 1
        for group in shuffled[:test_count]:
            assignments[group] = "test"
        for group in shuffled[test_count : test_count + val_count]:
            assignments[group] = "val"
        for group in shuffled[test_count + val_count :]:
            assignments[group] = "train"
        if not any(assignments[group] == "train" for group in shuffled):
            raise ValueError(f"class {label} has no training group")
    return assignments


def create_split_manifest(config: ProjectConfig, force: bool = False) -> Path:
    root = Path(config.data.processed_root)
    source_path = root / "manifest.csv"
    split_path = root / "manifest_split.csv"
    if split_path.exists() and not force:
        return split_path
    samples = read_manifest(source_path)
    groups_by_class: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        if sample.source_id not in groups_by_class[sample.label]:
            groups_by_class[sample.label].append(sample.source_id)
    missing = set(config.data.classes) - set(groups_by_class)
    if missing:
        raise ValueError(f"missing classes in processed data: {sorted(missing)}")
    assignments = _split_groups(
        groups_by_class,
        config.training.val_fraction,
        config.training.test_fraction,
        config.training.seed,
    )
    with split_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["path", "label", "modality", "source_id", "frame_index", "split"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = sample.__dict__.copy()
            row["split"] = assignments[sample.source_id]
            writer.writerow(row)
    return split_path


def _augment(image: np.ndarray) -> np.ndarray:
    result = image.copy()
    if random.random() < 0.5:
        result = cv2.flip(result, 1)
    if random.random() < 0.3:
        alpha = random.uniform(0.8, 1.2)
        result = np.clip(result.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    if random.random() < 0.2:
        result = cv2.GaussianBlur(result, (3, 3), 0)
    if random.random() < 0.2:
        quality = random.randint(45, 95)
        ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return result


def image_to_tensor(image: np.ndarray, augment: bool = False) -> torch.Tensor:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC BGR image, got shape {image.shape}")
    if augment:
        image = _augment(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(normalized, (2, 0, 1))).float()


class FrameDataset(Dataset):
    def __init__(self, samples: list[Sample], classes: list[str], training: bool):
        self.samples = samples
        self.class_to_index = {label: index for index, label in enumerate(classes)}
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = np.load(sample.path, allow_pickle=False)
        return image_to_tensor(image, self.training), self.class_to_index[sample.label]


class TemporalDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        classes: list[str],
        sequence_length: int,
        stride: int,
        training: bool,
    ):
        self.class_to_index = {label: index for index, label in enumerate(classes)}
        self.sequence_length = sequence_length
        self.training = training
        grouped: dict[str, list[Sample]] = defaultdict(list)
        for sample in samples:
            if sample.modality == "video":
                grouped[sample.source_id].append(sample)
        self.clips: list[tuple[list[Sample], int]] = []
        for group in grouped.values():
            ordered = sorted(group, key=lambda item: item.frame_index)
            for start in range(0, len(ordered) - sequence_length + 1, stride):
                clip = ordered[start : start + sequence_length]
                self.clips.append((clip, self.class_to_index[clip[0].label]))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int):
        clip, label = self.clips[index]
        frames = [
            image_to_tensor(np.load(sample.path, allow_pickle=False), self.training)
            for sample in clip
        ]
        return torch.stack(frames), label


def create_loaders(config: ProjectConfig, mode: str) -> dict[str, DataLoader]:
    split_manifest = create_split_manifest(config)
    samples = read_manifest(split_manifest)
    loaders = {}
    for split in ("train", "val", "test"):
        selected = [sample for sample in samples if sample.split == split]
        if mode == "frame":
            dataset: Dataset = FrameDataset(
                selected, config.data.classes, training=split == "train"
            )
            batch_size = config.training.batch_size
        else:
            dataset = TemporalDataset(
                selected,
                config.data.classes,
                config.training.sequence_length,
                config.training.sequence_stride,
                training=split == "train",
            )
            batch_size = config.training.temporal_batch_size
        if len(dataset) == 0:
            raise ValueError(f"{mode} {split} split contains no usable samples")
        loaders[split] = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=split == "train",
            num_workers=config.training.workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders

