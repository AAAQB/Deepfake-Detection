# Local model checkpoints

Checkpoint weights in this directory are intentionally ignored by Git because
they are large binary artifacts. The two local files imported from
`OneDrive_1_6-20-2026.zip` should exist at:

```text
checkpoints/frame_deepfake_v1/best_model.pt
checkpoints/temporal_deepfake_v1/best_model.pt
```

The tracked `MODEL_INFO.example.json` files document the expected filename,
hash, size, architecture, and metrics reported inside each checkpoint.

These are legacy checkpoints from the original branch implementation. They use
`timm` EfficientNet-B4, while newly trained checkpoints use the repository's
current self-describing format. The loader supports both formats with strict
state-dictionary validation.

The checkpoint files are user-provided binary artifacts. Their stored metrics
are reported by the checkpoint and have not been independently reproduced.

## Local compatibility results

Both models load with exact state-dictionary matching and produce finite
three-class outputs.

On the included branch-provided sample corpus:

- the frame checkpoint scored 35.0% accuracy and 20.3% macro-F1 on 60 images;
- the temporal checkpoint scored 46.7% accuracy and 35.6% macro-F1 on 15 videos.

These checks use the original MediaPipe face-box configuration but still rely
on a small sample corpus and may not reproduce the original data distribution.
They prove that the files execute, not that the stored 87–90% metrics
generalize. Do not use the imported checkpoints as production models without
evaluation on a representative held-out dataset.
