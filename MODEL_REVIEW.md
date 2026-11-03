# Imported Model Review

## Inventory

Two model checkpoints were imported locally from
`OneDrive_1_6-20-2026.zip`:

| Model | Local path | Size | SHA-256 |
|---|---|---:|---|
| Frame | `checkpoints/frame_deepfake_v1/best_model.pt` | 211,768,249 bytes | `96ccce87a61b726740d317a086b11198d044abc1f29f8895eff3b74d6ce15e0f` |
| Temporal | `checkpoints/temporal_deepfake_v1/best_model.pt` | 551,958,607 bytes | `07972af133a44c5d39c5685ee160bb77cc6e29b3d23ead1a31b63112bcd04dd3` |

The binary checkpoints are ignored by Git. Their tracked
`MODEL_INFO.example.json` files preserve identity and reported metadata.

## Architecture compatibility

These checkpoints use the original branch architecture rather than the new
repository's default torchvision architecture.

Frame checkpoint:

```text
timm EfficientNet-B4
→ 1,792 features
→ Dropout(0.4)
→ Linear(1792, 3)
```

Temporal checkpoint:

```text
16 face frames
→ shared timm EfficientNet-B4
→ four-layer bidirectional LSTM, hidden size 512
→ temporal mean pooling
→ LayerNorm(1024)
→ Dropout(0.4)
→ Linear(1024, 3)
```

The repository now detects this legacy state-dictionary format, reconstructs
the exact architecture, and performs strict weight loading. Current-format
checkpoints continue to use their stored architecture metadata.

## Verification performed

- ZIP contents and file hashes verified.
- Restricted `weights_only=True` checkpoint deserialization succeeded.
- Frame state dictionary loaded strictly.
- Temporal state dictionary loaded strictly.
- Frame forward pass produced finite output with shape `(1, 3)`.
- Temporal forward pass produced finite output with shape `(1, 3)`.
- Frame CLI inference produced an annotated JPEG.
- Frame Grad-CAM produced a valid heatmap JPEG.
- MediaPipe face extraction matched the original `model_selection=1` setup.
- Frame video inference produced an annotated MP4.
- Temporal video inference produced an annotated MP4 after filling its
  16-frame buffer.
- Frame and temporal soft-voting ensemble inference produced an annotated MP4.
- Full lint suite passed.
- Full unit test suite passed.

## Stored metrics

The frame checkpoint reports:

- accuracy: 90.12%;
- macro-F1: 89.08%;
- weighted-F1: 90.04%;
- AUC: 97.53%.

The temporal checkpoint reports:

- accuracy: 87.52%;
- macro-F1: 87.61%;
- weighted-F1: 87.48%;
- AUC: 95.10%.

These values are embedded in the checkpoints. The original manifests, exact
split files, and complete training provenance were not included, so these
metrics cannot currently be reproduced or independently confirmed.

## Local sample-corpus results

The imported models were also evaluated without training on the small,
branch-provided sample corpus.

Frame checkpoint, 60 labeled images:

- accuracy: 35.0%;
- macro-F1: 20.3%;
- confusion matrix: `[[1, 19, 0], [0, 20, 0], [3, 17, 0]]`.

Temporal checkpoint, 15 labeled videos:

- accuracy: 46.7%;
- macro-F1: 35.6%;
- confusion matrix: `[[5, 0, 0], [2, 0, 3], [3, 0, 2]]`.

These tests use the original MediaPipe face detector configuration. The sample
set and temporal sampling process may still differ from the original training
distribution, so this is not a definitive benchmark. It does show that the
high stored metrics should not be treated as evidence of generalization.

## Advantages

- Both files are complete, executable checkpoints rather than placeholder
  state dictionaries.
- The models use a strong EfficientNet-B4 visual backbone.
- The temporal checkpoint models relationships across multiple frames.
- Both checkpoints include optimizer, scheduler, epoch, and metric data.
- The class order is recoverable as `real`, `filter`, `deepfake`.
- Exact architecture loading is now supported without weakening strict loading
  for new checkpoints.
- Restricted checkpoint deserialization reduces arbitrary-pickle risk.
- Git ignores the large binaries while preserving tracked examples and hashes.

## Disadvantages and risks

- The original training data manifest and split are missing.
- The reported metrics cannot be independently reproduced.
- Sample-corpus performance is near chance and indicates poor transfer or a
  preprocessing/domain mismatch.
- The frame model strongly collapsed toward the `filter` class on the included
  image sample.
- The temporal model never predicted `filter` on the included video sample.
- The temporal checkpoint is large and computationally expensive.
- Legacy models add a dependency on `timm`.
- Neither checkpoint contains explicit architecture metadata; compatibility is
  inferred from state-dictionary keys.
- The original checkpoints store NumPy metric objects, requiring a narrow
  deserialization allowlist.
- There is no calibration analysis, external-dataset validation, or threshold
  study.
- There is no evidence that identities were separated between training and
  testing.

## Recommendation

Treat these files as legacy reference models and convenient functional
examples, not as production-ready detectors.

Before deployment:

1. evaluate both models on a representative, identity-separated held-out
   dataset;
2. reproduce preprocessing as closely as possible;
3. inspect class balance and confusion matrices;
4. compare against newly trained current-format checkpoints;
5. retain the legacy model only if it outperforms the reproducible model on
   external data.
