import csv
from pathlib import Path

import numpy as np

from deepfake_detection.config import ProjectConfig
from deepfake_detection.data import create_split_manifest, read_manifest


def test_split_is_source_grouped_and_has_all_splits(tmp_path: Path):
    config = ProjectConfig()
    config.data.processed_root = str(tmp_path)
    manifest = tmp_path / "manifest.csv"
    fields = ["path", "label", "modality", "source_id", "frame_index"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label in config.data.classes:
            for group in range(10):
                for frame in range(2):
                    path = tmp_path / f"{label}_{group}_{frame}.npy"
                    np.save(path, np.zeros((8, 8, 3), dtype=np.uint8))
                    writer.writerow(
                        {
                            "path": path,
                            "label": label,
                            "modality": "video",
                            "source_id": f"{label}:{group}",
                            "frame_index": frame,
                        }
                    )
    samples = read_manifest(create_split_manifest(config))
    group_splits = {}
    for sample in samples:
        group_splits.setdefault(sample.source_id, set()).add(sample.split)
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert {sample.split for sample in samples} == {"train", "val", "test"}

