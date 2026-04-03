# miles-holodec

Deep learning pipeline for 3D particle detection in digital inline holograms. Given a raw hologram image, the model predicts a **particle mask** and **depth (z) offset** for each pixel, enabling reconstruction of full 3D particle positions (x, y, z, d) from a single 2D image.

---

## Table of Contents
1. [Overview](#overview)
2. [Setup](#setup)
3. [Data](#data)
4. [Pipeline: Start to Finish](#pipeline-start-to-finish)
   - [Step 1: Train](#step-1-train)
   - [Step 2: Inference](#step-2-inference)
   - [Step 3: Evaluate](#step-3-evaluate)
   - [Step 4: Visualize](#step-4-visualize)
5. [Running as a Full Pipeline Job](#running-as-a-full-pipeline-job)
6. [Model Architecture](#model-architecture)
7. [Key Config Options](#key-config-options)
8. [Key Files](#key-files)

---

## Overview

A hologram encodes 3D particle positions as 2D diffraction patterns. Classical reconstruction propagates the wavefield to each depth plane separately (via FFT-based Fresnel propagation), then searches for particles in focus — requiring ~1000 propagations per hologram.

This pipeline trains a fully convolutional segmentation model to:
- **Predict which pixels contain particles** (mask channel, sigmoid output)
- **Predict each particle's z-distance from the propagation plane** (depth channel, regression)

Inference passes the full hologram image directly to the model with no tiling, sweeping a small number of z-center positions and using the depth output to refine the absolute z within each window.

### Two modes

| Mode | Input channels | z evaluations per hologram | Script |
|------|---------------|---------------------------|--------|
| Single-plane | 1 (amplitude) | ~1000 | `trainer_unet_planer.py` |
| Multi-plane | 2×lookahead (amp+phase per plane) | ~200 | `trainer_unet.py` |

With `lookahead=5`, the multi-plane model sees 5 propagation planes simultaneously and regresses depth within that window — requiring only ~200 forward passes per hologram instead of ~1000.

---

## Setup

### Conda environment

```bash
module load conda
conda create -n holodec python=3.10 -y
conda activate holodec
pip install torch torchvision segmentation-models-pytorch
pip install xarray netcdf4 scipy pandas tqdm pyyaml matplotlib
pip install -e /path/to/miles-holodec
```

On Derecho/Casper (NCAR), use the pre-built environment:
```bash
conda activate /glade/work/schreck/conda-envs/holodec
```

---

## Data

Synthetic hologram datasets (NCAR Casper/Derecho):
```
/glade/p/cisl/aiml/ai4ess_hackathon/holodec/
  synthetic_holograms_500particle_gamma_4872x3248_training.nc
  synthetic_holograms_500particle_gamma_4872x3248_validation.nc
  synthetic_holograms_500particle_gamma_4872x3248_test.nc
```

Each NetCDF file contains:
- `image(hologram_number, x, y)` — raw hologram intensity, shape `(N_holograms, 4872, 3248)`
- `x, y, z` — particle positions in **micrometers**, origin at image center (x,y) / sensor (z)
- `d` — particle diameter in **micrometers**
- `hid` — hologram ID (1-indexed), links particle records to holograms
- Attributes: `Nx=4872`, `Ny=3248`, `dx/dy=2.96e-6 m/pixel`, `lambda`, `zMin/zMax`

The depth range is approximately `zMin ≈ 14,000 μm` to `zMax ≈ 158,000 μm`. With `n_bins=1000`, each z-bin spans ~144 μm.

> **Note:** Despite what some older docs say, x/y/z/d are all stored in **micrometers** in the NetCDF file — no unit conversion needed when loading.

---

## Pipeline: Start to Finish

### Step 1: Train

#### Multi-plane (primary)

```bash
qsub scripts/submit_multiplane.sh
# or interactively:
python applications/trainer_unet.py -c config/multiplane.yml
```

Config: `config/multiplane.yml`  
Checkpoint saved to: `save_loc` in the config (default: `/glade/derecho/scratch/schreck/holodec/multiplane/`)  
Speed: ~2–3 min/epoch on A100 with AMP enabled.

Key settings:
```yaml
model:
    name: "fpn"
    encoder_name: "vgg11"
    encoder_weights: "imagenet"
    in_channels: 10     # 2 * lookahead
    classes: 2          # ch0: sigmoid mask, ch1: depth regression (μm)
    activation: null

loss:
    training_loss_mask: "focal-tyversky"
    training_loss_depth: "intersectedmae"   # only where mask overlaps truth
    training_loss_depth_mult: 0.03

trainer:
    amp: true           # mixed precision — ~2x speedup on A100
    stopping_patience: 4
```

#### Single-plane (baseline)

```bash
qsub scripts/submit_single_plane.sh
# or interactively:
python applications/trainer_unet_planer.py -c config/single_plane_validate.yml
```

Config: `config/single_plane_validate.yml`  
Checkpoint saved to: `/glade/derecho/scratch/schreck/holodec/single_plane_validate/`

---

### Step 2: Inference

Inference sweeps z-center positions across the depth range, propagates the full hologram to `lookahead` planes at each position, runs the model on the full image (no tiling), and extracts particles via connected-component labeling.

#### Multi-plane

```bash
python applications/inference_multiplane.py \
    -c config/multiplane.yml \
    -k /path/to/checkpoint.pt \
    -o /path/to/output_dir \
    --z-stride 5 \
    --halo 64 \
    --threshold 0.5 \
    --device cuda
```

| Flag | Default | Description |
|------|---------|-------------|
| `-c` | required | Model config YAML |
| `-k` | required | Checkpoint `.pt` file |
| `-o` | required | Output directory |
| `--z-stride` | lookahead | Step between z-center positions |
| `--halo` | 64 | Reflect-pad pixels to reduce edge artifacts |
| `--threshold` | 0.5 | Mask probability cutoff |
| `--cluster` | off | Apply 3D NMS after detection (see Step 3) |
| `--xy-radius` | 50 μm | NMS suppression radius in x,y |
| `--z-radius` | 200 μm | NMS suppression radius in z |
| `--max-holograms` | all | Limit to first N holograms |
| `--h-start` | 0 | Starting hologram index |

Output: `detections.csv` with columns `h_idx, x_pix, y_pix, x_um, y_um, z_um, d_pix, d_um, mask_score`.

#### Single-plane

```bash
python applications/inference_planar.py \
    -c config/single_plane_validate.yml \
    -k /path/to/checkpoint.pt \
    -o /path/to/output_dir \
    --z-stride 1 \
    --halo 64 \
    --threshold 0.5 \
    --device cuda
```

Same output format as multi-plane. `z_um = z_centers[z_idx]` (no depth regression in single-plane mode).

---

### Step 3: Evaluate

Matches detections against ground truth using the **Hungarian algorithm** (optimal 1:1 bipartite matching) within anisotropic distance tolerances, then reports detection metrics and position RMSEs.

#### Why NMS first?

The same particle may appear in multiple z-windows (especially at window boundaries), producing duplicate detections at similar (x, y, z). Pass `--cluster` to collapse these with 3D NMS before matching. NMS sorts by `mask_score` descending and suppresses detections within an ellipsoid of radii `(xy_radius, xy_radius, z_radius)`.

```bash
python applications/evaluate_detections.py \
    -d /path/to/detections.csv \
    -n /path/to/validation.nc \
    -o /path/to/output_dir \
    --cluster \
    --xy-radius 50 \
    --z-radius 200 \
    --xy-tol 50 \
    --z-tol 360
```

| Flag | Default | Description |
|------|---------|-------------|
| `-d` | required | `detections.csv` from inference |
| `-n` | required | Ground-truth NetCDF file |
| `-o` | required | Output directory |
| `--xy-tol` | 50 μm | Match tolerance in x,y |
| `--z-tol` | 360 μm | Match tolerance in z (~2.5 z-bins) |
| `--cluster` | off | Apply NMS before matching |
| `--xy-radius` | 50 μm | NMS x,y suppression radius |
| `--z-radius` | 200 μm | NMS z suppression radius |

**Outputs:**
- `matched_pairs.csv` — one row per particle (true + predicted coords, matched flag)
- `metrics_per_hologram.csv` — per-hologram TP/FP/FN and all metrics
- `metrics_summary.csv` — aggregate metrics across all holograms

**Metrics:**

| Metric | Formula | Description |
|--------|---------|-------------|
| POD | TP / (TP + FN) | Probability of Detection |
| FAR | FP / (TP + FP) | False Alarm Ratio |
| F1 | 2TP / (2TP + FP + FN) | Harmonic mean of precision/recall |
| CSI | TP / (TP + FP + FN) | Critical Success Index (threat score) |
| RMSE_x/y/z/d | — | Position/diameter error for matched pairs (μm) |

A detection is a **true positive** if it falls within `xy_tol` μm in x,y **and** `z_tol` μm in z of a ground-truth particle. Each ground-truth particle can only be matched once (Hungarian assignment).

**Current results — VGG11/FPN multiplane, validation set (10 holograms, 500 particles each):**

| Metric | Value |
|--------|-------|
| POD | 0.936 |
| FAR | 0.927 |
| F1 | 0.135 |
| CSI | 0.073 |
| RMSE_x | 3.5 μm |
| RMSE_y | 3.8 μm |
| RMSE_z | 208 μm |
| RMSE_d | 9.0 μm |

High FAR indicates the model is still generating false positives from diffraction ring artifacts — expected to improve with more training epochs.

---

### Step 4: Visualize

```bash
python applications/visualize_detections.py \
    -d /path/to/detections.csv \
    -n /path/to/validation.nc \
    -o /path/to/plots \
    --h-idx 0 \
    --z-tol 360 \
    --xy-tol 50 \
    --threshold 0.5 \
    --training-log /path/to/training_log.csv
```

Generates per-hologram diagnostic plots:

| File | Description |
|------|-------------|
| `hologram_raw.png` | Raw hologram intensity |
| `detections_xy.png` | x-y map colored by z depth: predicted vs truth |
| `detections_3d.png` | 3D scatter: predicted (blue) vs ground truth (red) |
| `z_distribution.png` | Histogram of z positions: predicted vs true |
| `d_distribution.png` | Histogram of particle diameters: predicted vs true |
| `match_summary.png` | TP/FP/FN counts, precision/recall/F1, z accuracy scatter |
| `training_curves.png` | Loss curves (if `--training-log` provided) |

---

## Running as a Full Pipeline Job

The pipeline scripts chain training → inference → evaluation in a single PBS submission. Use these when starting a fresh training run and you want metrics automatically at the end.

```bash
# Multi-plane: train, infer on full validation set, evaluate
qsub scripts/pipeline_multiplane.sh

# Single-plane: same
qsub scripts/pipeline_single_plane.sh
```

To chain inference+evaluation onto an **already-running** training job (e.g., job ID `3019076`):

```bash
qsub -W depend=afterok:3019076 scripts/submit_infer_eval_multiplane.sh
qsub -W depend=afterok:3018965 scripts/submit_infer_eval_single_plane.sh
```

---

## Model Architecture

Both modes use [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch).

### Multi-plane

```
Input:  (B, 10, H, W)   ← 5 amplitude + 5 phase channels
  FPN encoder: VGG11 (ImageNet pretrained), 11M parameters
  FPN decoder: pyramid feature aggregation
  Output: (B, 2, H, W)
    ch0: Sigmoid(x)  → particle mask probability [0, 1]
    ch1: Identity(x) → z offset from z_center (μm, unbounded)
```

**How lookahead works** (lookahead=5):
- Planes: 2 back + center + 2 forward = 5 total
- Input channels: 5 amplitude + 5 phase = 10
- Depth label: `z_particle − z_centers[z_idx]` in μm
- Absolute z at inference: `z_abs = z_centers[z_idx] + model_depth_output`

**intersectedMAE loss**: depth loss is computed only where predicted mask and true mask intersect — prevents large gradients from background regions where depth is undefined.

### Single-plane

```
Input:  (B, 1, H, W)   ← amplitude only
  FPN encoder: VGG11
  Output: (B, 1, H, W)
    ch0: Sigmoid(x) → mask probability
```

### Best hyperparameters (Matt Hayman, Nov 2024 hyperopt)

- `lr = 4.52e-4`, `weight_decay = 0`
- `scheduler: ReduceLROnPlateau(factor=0.1, patience=5)`
- Loss: `focal-tversky (mask) + 0.03 × intersectedMAE (depth)`
- Architecture: FPN + VGG11 (ImageNet pretrained)

A trained checkpoint (FPN + ResNet50, `in_channels=10`, `classes=2`) is at:
`results/checkpoint.pt`

---

## Key Config Options

| Option | Default | Description |
|--------|---------|-------------|
| `training_data.lookahead` | 5 | z-planes per sample; `in_channels = 2×lookahead` |
| `training_data.n_bins` | 1000 | z-discretization; dz ≈ 144 μm |
| `training_data.device` | `"cuda"` | Wave propagation device; **requires `thread_workers: 0`** |
| `trainer.batches_per_epoch` | 500 | Batches per epoch |
| `trainer.amp` | true | Mixed precision (~2x speedup on A100, negligible quality loss) |
| `loss.training_loss_depth_mult` | 0.03 | Weight of depth loss relative to mask loss |
| `trainer.stopping_patience` | 4 | Early stop after N epochs without improvement |

---

## Key Files

```
applications/
  trainer_unet.py               multi-plane training (mask + depth)
  trainer_unet_planer.py        single-plane training (mask only)
  inference_multiplane.py       full-image multi-plane inference
  inference_planar.py           full-image single-plane inference
  evaluate_detections.py        3D NMS + Hungarian matching + POD/FAR/F1/CSI/RMSE
  visualize_detections.py       diagnostic plots vs ground truth

holodec/
  propagation.py                WavePropagator — FFT Fresnel wave propagation
  planer_datasets.py            LoadMultiplaneHolograms, LoadHolograms
  unet.py                       SegmentationModel, PlanerSegmentationModel
  losses.py                     focal-tversky, dice, intersectedMAE/MSE
  trainer.py                    Trainer.fit() — multi-plane training loop
  planer_trainer.py             Trainer.fit() — single-plane training loop

config/
  multiplane.yml                multi-plane training config (VGG11/FPN)
  single_plane_validate.yml     single-plane training config
  inference_multiplane_matt.yml inference config for Matt's ResNet50 checkpoint

scripts/
  pipeline_multiplane.sh        PBS: train → infer → evaluate (multi-plane)
  pipeline_single_plane.sh      PBS: train → infer → evaluate (single-plane)
  submit_multiplane.sh          PBS: training only (multi-plane)
  submit_single_plane.sh        PBS: training only (single-plane)
  submit_infer_eval_multiplane.sh  PBS: infer + evaluate (multi-plane)
  submit_infer_eval_single_plane.sh PBS: infer + evaluate (single-plane)

results/
  checkpoint.pt                 Matt's ResNet50 checkpoint (in_channels=10, classes=2)
  training_log.csv              Training history for that run
```
