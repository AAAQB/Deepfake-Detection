# Deepfake Detection — Train-Ready Project

This repository is a consolidated, corrected version of the useful work found
across the branches of the original `AAAQB/deepfake_detection` project.

It provides a complete pipeline for:

- preparing image and video datasets;
- detecting and extracting faces;
- creating leakage-resistant train, validation, and test splits;
- training a frame-level EfficientNet-B4 classifier;
- training a temporal EfficientNet-B4 + bidirectional LSTM classifier;
- evaluating trained checkpoints on a held-out test split;
- running frame-model inference on an image;
- producing Grad-CAM visualizations;
- measuring basic corruption robustness;
- saving strict, self-describing PyTorch checkpoints.

No trained deepfake-detection weights are included. The repository is ready to
train, but you must provide data and run training yourself.

Local installations may also contain two ignored legacy checkpoints imported
from the original project:

```text
checkpoints/frame_deepfake_v1/best_model.pt
checkpoints/temporal_deepfake_v1/best_model.pt
```

They are not uploaded to Git. See [checkpoints/README.md](checkpoints/README.md)
and the tracked model information examples for hashes and reported metrics.
Both execute successfully, but their accuracy on the included sample corpus is
poor; consult the checkpoint README before relying on them.

See [MODEL_REVIEW.md](MODEL_REVIEW.md) for the complete compatibility,
performance, advantages, and limitations review.

## Classification task

Every model predicts one of three classes:

| Index | Class | Meaning |
|---:|---|---|
| 0 | `real` | An unmanipulated face |
| 1 | `filter` | A face modified by a beauty or AR filter |
| 2 | `deepfake` | An AI-generated or identity-manipulated face |

The class order is stored inside every checkpoint. Checkpoint loading is strict,
so a checkpoint cannot silently load into a different architecture.

## Current capabilities and limitations

| Capability | Frame model | Temporal model |
|---|:---:|:---:|
| Training | Yes | Yes |
| Held-out evaluation | Yes | Yes |
| Image inference | Yes | No |
| Grad-CAM | Yes | No |
| Robustness report | Yes | No |
| Video-file inference | Yes | Yes |
| Webcam inference | Yes, through the local Next.js dev UI | Not currently implemented |
| Frame + temporal video ensemble | Yes | Yes |

Frame and temporal models are trained independently. At video inference time,
their class probabilities can be combined through weighted soft voting.

## How the complete pipeline works

```text
Raw images and videos
        │
        ▼
OpenCV face detection
        │
        ▼
224×224 BGR face crops saved as uint8 .npy files
        │
        ▼
manifest.csv records:
path, label, modality, source_id, frame_index
        │
        ▼
Source-grouped 70% / 15% / 15% split
        │
        ├──────────────► FrameDataset ─────► EfficientNet-B4
        │
        └──────────────► TemporalDataset ──► EfficientNet-B4 + BiLSTM
                                                │
                                                ▼
                                best.pt, last.pt, history.json
                                                │
                                                ▼
                                      Held-out test evaluation
```

The important detail is that splitting happens by `source_id`, not by
individual frame. All frames extracted from one source video remain in exactly
one split. This prevents nearly identical adjacent frames from appearing in
both training and testing data.

## Model architectures

### Frame classifier

Input shape:

```text
(batch, 3, 224, 224)
```

Architecture:

```text
224×224 RGB face
    │
    ▼
ImageNet-pretrained EfficientNet-B4
    │
    ▼
Global average pooling
    │
    ▼
1,792-dimensional feature vector
    │
    ▼
Dropout (0.4)
    │
    ▼
Linear(1792, 3)
    │
    ▼
Three class logits
```

Training applies augmentation to face crops:

- random horizontal flip;
- random brightness adjustment;
- occasional Gaussian blur;
- occasional JPEG recompression.

### Temporal classifier

Input shape with the default configuration:

```text
(batch, 16, 3, 224, 224)
```

Architecture:

```text
16 sequential face frames
    │
    ▼
Shared EfficientNet-B4 encoder applied to each frame
    │
    ▼
16 × 1,792-dimensional feature sequence
    │
    ▼
Two-layer bidirectional LSTM
hidden size: 512 per direction
    │
    ▼
Mean pooling across the 16 time steps
    │
    ▼
LayerNorm(1024)
    │
    ▼
Dropout (0.4)
    │
    ▼
Linear(1024, 3)
    │
    ▼
Three class logits
```

The temporal model only uses samples originating from videos. A video must
produce at least `sequence_length` processed frames before it can contribute a
temporal clip.

### Tiny test architecture

The configuration also accepts `model.kind: tiny`. This is a small convolutional
network used for fast development and automated tests. It is not the recommended
architecture for real training.

## Repository structure

```text
.
├── configs/
│   └── default.yaml              # Dataset, model, and training configuration
├── sample_data/                  # Local-only sample corpus (gitignored)
├── src/deepfake_detection/
│   ├── analysis.py               # Grad-CAM and robustness evaluation
│   ├── cli.py                    # deepfake-detect command
│   ├── config.py                 # Validated YAML configuration
│   ├── data.py                   # Splitting, datasets, transforms, DataLoaders
│   ├── inference.py              # Frame checkpoint image inference
│   ├── models.py                 # Frame and temporal architectures
│   ├── preprocessing.py          # Face detection and crop extraction
│   ├── training.py               # Training, evaluation, checkpoints
│   └── web_api.py                # Local API for webcam-frame inference
├── web/                          # Next.js webcam development UI
├── scripts/dev-webcam.sh         # Starts the API and UI together
├── tests/                        # Unit and smoke tests
├── .github/workflows/ci.yml      # GitHub Actions checks
├── BRANCH_REVIEW.md              # Review of all original branches
├── dataset_info.md               # Original full-dataset reference
└── pyproject.toml                # Package metadata and dependencies
```

Generated directories are ignored by Git:

```text
data_raw/
data_processed/
checkpoints/
logs/
outputs/
```

## Webcam development UI

The local GUI opens the laptop webcam in the browser, captures a JPEG snapshot,
and sends it to the Python API on the same machine. The API keeps the frame
model loaded between requests, detects the largest face, and returns the
three-class probabilities.

This UI uses the frame checkpoint only. The temporal checkpoint requires a
rolling sequence of video frames and is not part of the current webcam path.

Install the Python API and legacy-checkpoint dependencies:

```bash
source .venv/bin/activate
pip install -e ".[web,legacy]"
```

Install the Next.js dependencies:

```bash
cd web
npm install
cd ..
```

Make sure this ignored local checkpoint exists:

```text
checkpoints/frame_deepfake_v1/best_model.pt
```

Start both development servers:

```bash
./scripts/dev-webcam.sh
```

Then open [http://localhost:3000](http://localhost:3000), select **Start
camera**, allow webcam access, and select **Analyze frame**. **Auto scan**
captures a frame about every 1.8 seconds.

The defaults are:

```text
Next.js UI: http://localhost:3000
Python API: http://127.0.0.1:8000
```

To select a different frame checkpoint or device:

```bash
DEEPFAKE_CHECKPOINT=/absolute/path/to/frame.pt \
DEEPFAKE_DEVICE=cpu \
./scripts/dev-webcam.sh
```

Frames are sent only to the local Python service. Do not expose this development
API directly to the public internet; it has no authentication or production
rate limiting.

## Requirements

- Python 3.10, 3.11, or 3.12
- Enough disk space for the raw and processed datasets
- A CUDA-capable GPU is strongly recommended for EfficientNet-B4 training
- CPU training is supported but will be slow, especially for the temporal model

The code automatically selects CUDA when it is available; otherwise it uses
CPU. You can explicitly select a device with `--device cpu` or
`--device cuda`.

## Installation

Clone the private repository:

```bash
git clone git@github.com:Lachi-Amine/deepfake_detection_ready.git
cd deepfake_detection_ready
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade packaging tools and install the project:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

This installs the `deepfake-detect` command.

To use the imported legacy checkpoints with the same MediaPipe face detector
used during their original preprocessing:

```bash
pip install -e ".[dev,legacy]"
```

For an NVIDIA GPU, install the appropriate CUDA-enabled PyTorch build from the
official PyTorch instructions before installing this project. Confirm the
environment:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
deepfake-detect --help
```

## Dataset layout

Place the raw dataset in this exact structure:

```text
data_raw/
├── image/
│   ├── real/
│   │   ├── real_001.jpg
│   │   └── ...
│   ├── filter/
│   │   ├── filter_001.jpg
│   │   └── ...
│   └── deepfake/
│       ├── fake_001.jpg
│       └── ...
└── video/
    ├── real/
    │   ├── real_001.mp4
    │   └── ...
    ├── filter/
    │   ├── filter_001.mp4
    │   └── ...
    └── deepfake/
        ├── fake_001.mp4
        └── ...
```

Supported image extensions:

```text
.jpg .jpeg .png .bmp .webp
```

Supported video extensions:

```text
.mp4 .mov .avi .mkv
```

Do not use `filtered` as a class directory. The expected class name is
`filter`.

### Using the sample data

The development corpus lives under `sample_data/` and is **not committed to
Git** (it is listed in `.gitignore`): it is mostly video, roughly 42 MB in
total. A fresh clone therefore does not contain it. If you have the folder
locally, it is useful for checking that preprocessing and DataLoaders work, but
it is too small to demonstrate production-quality accuracy.

Copy it into the expected location:

```bash
cp -R sample_data data_raw
```

On Windows PowerShell:

```powershell
Copy-Item -Recurse sample_data data_raw
```

## Configuration

The default configuration is [configs/default.yaml](configs/default.yaml):

```yaml
data:
  raw_root: data_raw
  processed_root: data_processed
  classes: [real, filter, deepfake]
  image_size: 224
  frame_interval: 5
  fallback_full_frame: true

model:
  kind: efficientnet_b4
  pretrained: true
  dropout: 0.4
  hidden_size: 512
  lstm_layers: 2
  lstm_dropout: 0.3
  bidirectional: true

training:
  seed: 42
  train_fraction: 0.70
  val_fraction: 0.15
  test_fraction: 0.15
  batch_size: 32
  temporal_batch_size: 4
  sequence_length: 16
  sequence_stride: 4
  workers: 2
  epochs: 40
  learning_rate: 0.0001
  weight_decay: 0.0001
  label_smoothing: 0.1
  early_stopping_patience: 8
  checkpoint_root: checkpoints
  log_root: logs
```

Important settings:

| Setting | Effect |
|---|---|
| `frame_interval` | Processes one video frame every N source frames |
| `fallback_full_frame` | Uses the whole image when no face is detected |
| `pretrained` | Downloads ImageNet EfficientNet-B4 weights on first use |
| `batch_size` | Frame-model training batch size |
| `temporal_batch_size` | Temporal-model batch size |
| `sequence_length` | Number of frames in each temporal clip |
| `sequence_stride` | Offset between the starts of neighboring temporal clips |
| `workers` | DataLoader worker process count |
| `epochs` | Maximum number of epochs |
| `early_stopping_patience` | Stops after this many non-improving validation epochs |

The split fractions must add up to exactly `1.0`.

For a machine with limited GPU memory, start with:

```yaml
training:
  batch_size: 8
  temporal_batch_size: 1
```

For offline use where ImageNet weights cannot be downloaded:

```yaml
model:
  pretrained: false
```

## Step 1: Preprocess the dataset

Run:

```bash
deepfake-detect --config configs/default.yaml preprocess
```

For each raw image:

1. OpenCV attempts to detect faces.
2. Every detected face is resized to `224×224`.
3. Each crop is saved as an unnormalized BGR `uint8` NumPy array.

For each raw video:

1. Every `frame_interval`-th frame is inspected.
2. The largest detected face is selected.
3. The crop is resized and saved.

With `fallback_full_frame: true`, the whole image or video frame is used when
the face detector finds no face. This helps pipeline testing but can reduce
model quality if backgrounds correlate with labels. For a strict face-only
dataset, set it to `false`.

Preprocessing creates:

```text
data_processed/
├── image/
│   ├── real/<source-name>/*.npy
│   ├── filter/<source-name>/*.npy
│   └── deepfake/<source-name>/*.npy
├── video/
│   ├── real/<source-name>/*.npy
│   ├── filter/<source-name>/*.npy
│   └── deepfake/<source-name>/*.npy
├── manifest.csv
└── preprocessing_failures.txt     # Created only if some files failed
```

`manifest.csv` records the processed path, class, modality, source identity,
and original video frame index.

If you change the raw dataset, remove `data_processed/` and preprocess again.
The current preprocessor does not incrementally remove crops belonging to
deleted source files.

## Step 2: Create the data split

Run:

```bash
deepfake-detect --config configs/default.yaml split
```

This creates:

```text
data_processed/manifest_split.csv
```

The default split is:

- 70% training sources;
- 15% validation sources;
- 15% held-out test sources.

Splits are stratified independently by class at the source-group level. The
same seed produces the same split.

Training automatically creates the split if it does not exist, so this command
is optional. Running it explicitly lets you inspect the split before training.

To regenerate it after changing split settings:

```bash
deepfake-detect --config configs/default.yaml split --force
```

Do not use `--force` after training if you want evaluation to remain comparable
with the original training run.

## Step 3A: Train the frame model

Run on the automatically selected device:

```bash
deepfake-detect --config configs/default.yaml train-frame
```

Explicit CUDA:

```bash
deepfake-detect --config configs/default.yaml train-frame --device cuda
```

Explicit CPU:

```bash
deepfake-detect --config configs/default.yaml train-frame --device cpu
```

Training uses:

- AdamW optimizer;
- learning rate `1e-4`;
- weight decay `1e-4`;
- cross-entropy with label smoothing `0.1`;
- cosine annealing learning-rate schedule;
- gradient clipping at norm `1.0`;
- validation macro-F1 for checkpoint selection;
- early stopping after 8 non-improving validation epochs.

The frame model uses both processed images and individual processed video
frames.

Outputs:

```text
checkpoints/frame_efficientnet_b4/
├── best.pt              # Highest validation macro-F1
├── last.pt              # Most recent completed epoch
├── history.json         # Per-epoch training and validation metrics
└── test_metrics.json    # Final metrics on the held-out test split

logs/frame_efficientnet_b4/
└── events.out.tfevents.* # TensorBoard event file
```

At the end of training, the code reloads `best.pt` and evaluates it once on the
held-out test split.

## Step 3B: Train the temporal model

Run:

```bash
deepfake-detect --config configs/default.yaml train-temporal
```

Or explicitly select CUDA:

```bash
deepfake-detect --config configs/default.yaml train-temporal --device cuda
```

Outputs:

```text
checkpoints/temporal_efficientnet_b4/
├── best.pt
├── last.pt
├── history.json
└── test_metrics.json

logs/temporal_efficientnet_b4/
└── events.out.tfevents.*
```

Temporal training needs enough videos in every class and split. Each selected
video must also produce at least 16 processed frames with the default
configuration. If a split contains no usable clips, training stops with a clear
error instead of silently reusing validation data as test data.

EfficientNet-B4 runs once per frame in every sequence, so temporal training uses
substantially more memory and compute than frame training.

## Monitor training with TensorBoard

In another terminal with the same virtual environment:

```bash
tensorboard --logdir logs
```

Then open:

```text
http://localhost:6006
```

Logged values include:

- training loss;
- training accuracy;
- validation loss;
- validation accuracy;
- validation macro-F1.

## Checkpoint format

Each `.pt` checkpoint contains:

```text
mode                    frame or temporal
classes                 ordered class names
model_config            exact architecture settings
epoch                   saved epoch
metrics                 validation metrics
model_state_dict        model parameters
optimizer_state_dict    optimizer state
```

The loader reconstructs the model from checkpoint metadata and calls
`load_state_dict(..., strict=True)`. It will fail if weights and architecture do
not match.

Only load checkpoints from trusted sources. PyTorch checkpoint files are not a
safe format for untrusted downloads.

### Imported legacy checkpoint format

The imported frame and temporal models predate the current metadata format.
They use the original branch architecture:

- `timm` EfficientNet-B4 frame model;
- `timm` EfficientNet-B4 plus a four-layer bidirectional LSTM temporal model.

The loader identifies these models from their state-dictionary keys, rebuilds
the exact legacy architecture, and still loads weights with `strict=True`.
Checkpoint deserialization uses PyTorch's restricted `weights_only=True` mode
with only the NumPy types required by the stored metrics allowlisted.
Legacy inference automatically selects MediaPipe face detection with
`model_selection=1`, matching the original preprocessing code.

Test the local imported frame checkpoint:

```bash
deepfake-detect --config configs/default.yaml predict-image \
  checkpoints/frame_deepfake_v1/best_model.pt \
  sample_data/image/real/real11.jpg \
  --output outputs/legacy-frame-example.jpg \
  --device cpu
```

The temporal checkpoint can be evaluated when a compatible processed dataset
and split manifest are available:

```bash
deepfake-detect --config configs/default.yaml evaluate \
  checkpoints/temporal_deepfake_v1/best_model.pt
```

Run frame-model video inference:

```bash
deepfake-detect --config configs/default.yaml predict-video \
  checkpoints/frame_deepfake_v1/best_model.pt \
  path/to/video.mp4 \
  --output outputs/frame-video.mp4
```

Run temporal-model video inference:

```bash
deepfake-detect --config configs/default.yaml predict-video \
  checkpoints/temporal_deepfake_v1/best_model.pt \
  path/to/video.mp4 \
  --output outputs/temporal-video.mp4 \
  --sequence-length 16
```

Run weighted frame + temporal ensemble inference:

```bash
deepfake-detect --config configs/default.yaml predict-ensemble-video \
  checkpoints/frame_deepfake_v1/best_model.pt \
  checkpoints/temporal_deepfake_v1/best_model.pt \
  path/to/video.mp4 \
  --output outputs/ensemble-video.mp4 \
  --sequence-length 16 \
  --frame-weight 0.4
```

`--frame-weight 0.4` assigns 40% of the fused probability to the frame model
and 60% to the temporal model once the temporal buffer is full.

## Evaluate a trained checkpoint

Frame checkpoint:

```bash
deepfake-detect --config configs/default.yaml evaluate \
  checkpoints/frame_efficientnet_b4/best.pt
```

Temporal checkpoint:

```bash
deepfake-detect --config configs/default.yaml evaluate \
  checkpoints/temporal_efficientnet_b4/best.pt
```

The command reads the checkpoint mode and evaluates the correct architecture
using only `manifest_split.csv` rows marked `test`.

Example output structure:

```json
{
  "loss": 0.42,
  "accuracy": 0.86,
  "macro_f1": 0.84
}
```

Do not treat these example numbers as expected performance. Actual performance
depends on dataset quality, manipulation diversity, subject independence, and
class balance.

## Run inference on an image

Image inference currently supports the frame checkpoint only:

```bash
deepfake-detect --config configs/default.yaml predict-image \
  checkpoints/frame_efficientnet_b4/best.pt \
  path/to/input.jpg \
  --output outputs/prediction.jpg
```

The command:

1. loads the frame checkpoint;
2. detects faces in the input image;
3. runs classification on each face;
4. prints class probabilities as JSON;
5. writes an annotated output image.

Example JSON shape:

```json
[
  {
    "class": "deepfake",
    "confidence": 0.91,
    "probabilities": {
      "real": 0.04,
      "filter": 0.05,
      "deepfake": 0.91
    }
  }
]
```

## Generate Grad-CAM

Grad-CAM currently supports frame checkpoints:

```bash
deepfake-detect --config configs/default.yaml gradcam \
  checkpoints/frame_efficientnet_b4/best.pt \
  path/to/input.jpg \
  --output outputs/gradcam.jpg
```

The output overlays a heatmap showing which final EfficientNet feature regions
most influenced the predicted class. Grad-CAM is an interpretability aid, not
proof that a prediction is correct.

## Run the robustness report

Run:

```bash
deepfake-detect --config configs/default.yaml robustness \
  checkpoints/frame_efficientnet_b4/best.pt
```

The command evaluates the frame model’s held-out test split under:

- clean input;
- additive noise;
- average blur;
- reduced brightness.

It prints accuracy and macro-F1 for each condition. These are simple diagnostic
corruptions, not a complete adversarial or cross-dataset robustness study.

## Testing and quality checks

Run:

```bash
ruff check .
pytest -q
```

The tests verify:

- configuration loading;
- source-grouped split behavior;
- frame-model output shape;
- temporal-model output shape;
- strict checkpoint reconstruction and loading.

The test suite does not train EfficientNet-B4. GitHub Actions runs lint and the
test suite on every push and pull request.

## Recommended training workflow

For an actual experiment:

1. Verify every raw source has the correct label.
2. Ensure subjects or identities do not occur across multiple source files in
   different splits. The code groups by source file, not by human identity.
3. Preprocess the complete dataset.
4. Inspect `manifest.csv` for class counts and failed files.
5. Create and preserve `manifest_split.csv`.
6. Train the frame model first.
7. Review validation and held-out test metrics.
8. Inspect incorrect predictions and Grad-CAM outputs.
9. Train the temporal model if the video dataset has sufficient sequences.
10. Compare frame and temporal results on the same held-out sources.
11. Validate on a separate external dataset before claiming generalization.

## Common problems

### `raw dataset not found`

Confirm that `data_raw/` exists or update:

```yaml
data:
  raw_root: /absolute/path/to/dataset
```

### `missing classes in processed data`

At least one processed source is required for every configured class:

```text
real
filter
deepfake
```

Check directory spelling and preprocessing failures.

### `temporal ... split contains no usable samples`

Possible causes:

- too few videos in one class;
- too few extracted frames per video;
- `frame_interval` is too large;
- `sequence_length` is too large;
- a validation or test split has no qualifying video.

Possible configuration changes:

```yaml
data:
  frame_interval: 2

training:
  sequence_length: 8
  sequence_stride: 2
```

Regenerate processed data and the split after changing these values.

### CUDA out-of-memory

Reduce batch sizes:

```yaml
training:
  batch_size: 8
  temporal_batch_size: 1
```

The temporal model is typically the first one to exceed GPU memory.

### ImageNet weights cannot download

Set:

```yaml
model:
  pretrained: false
```

Training from random initialization generally requires more data and more
epochs.

### Evaluation does not match an earlier run

Make sure:

- the same `manifest_split.csv` still exists;
- the dataset has not been reprocessed differently;
- the same YAML configuration is used;
- the checkpoint belongs to this repository’s strict format.

## What this repository does not claim

- It does not include trained deepfake-detection weights.
- It does not guarantee accuracy on unseen manipulation methods.
- It does not perform identity-level splitting automatically.
- It does not currently provide webcam inference.
- It does not replace external-dataset validation.
- The local `sample_data/` corpus (gitignored, not in the repository) is for
  pipeline testing, not final research results.

## Source branch review

See [BRANCH_REVIEW.md](BRANCH_REVIEW.md) for details about what was retained or
rejected from the original `main`, `sxy`, `syr`, and `dataset_sample` branches.
