# Source Branch Review

The consolidated project was built after reviewing every branch in
`AAAQB/deepfake_detection`.

## `main`

Useful material: unified preprocessing/training concept, group-level video
splitting, checkpoint management, fusion, evaluation, Grad-CAM, and robustness
ideas.

Rejected defects: broken CLI imports, incompatible training and inference model
definitions, whole-corpus evaluation leakage, silent lightweight model fallback,
non-runnable robustness code, commented-out Grad-CAM utilities, and duplicated
training modules.

## `sxy`

Useful material: the most complete coherent training loop, augmentations,
real-time inference, export flow, temporal smoothing, and model interfaces.

Rejected defects: frame-level rather than source-level splitting, temporal
validation reused as test data, permissive checkpoint loading, and import-time
MediaPipe state.

## `syr`

Useful material: original EfficientNet-B4, CNN-LSTM, normalization, face
extraction, and basic augmentation foundation.

Rejected defects: incomplete project surface and no training/evaluation
orchestration.

## `dataset_sample`

Useful material: sample image/video corpus and dataset documentation.

Normalization applied during consolidation: the sample video folder named
`filtered` is exposed as class `filter`, matching the model class contract.

