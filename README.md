# miles-holodec

Deep learning pipeline for 3D particle detection in digital inline holograms. Given a raw hologram image, the model predicts a **particle mask** and **depth (z) offset** for each pixel, enabling reconstruction of full 3D particle positions (x, y, z, d) from a single 2D image.

## Table of Contents
1. [Overview](#overview)
2. [Setup](#setup)
3. [Data](#data)
4. [Single-Plane Pipeline](#single-plane-pipeline)
5. [Multi-Plane Pipeline](#multi-plane-pipeline)
6. [Inference](#inference)
7. [Visualization](#visualization)
8. [Model Architecture](#model-architecture)
9. [Key Config Options](#key-config-options)
10. [Key Files](#key-files)

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

### 1. Conda environment

```bash
module load conda
conda create -n holodec python=3.10 -y
conda activate holodec
pip install torch torchvision segmentation-models-pytorch
pip install xarray netcdf4 scipy pandas tqdm pyyaml matplotlib
pip install -e /path/to/miles-holodec   # editable install so PBS jobs use repo code
```

On Derecho/Casper (NCAR), use the pre-built environment:
```bash
conda activate /glade/work/schreck/conda-envs/holodec
```

### 2. Clone

```bash
git clone git@github.com:NCAR/miles-holodec.git
cd miles-holodec
pip install -e .
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

Each file contains multiple holograms. The `500particle` in the filename indicates **~500 particles per hologram**, distributed throughout the 3D sample volume.

Each NetCDF file contains:
- `image(hologram_number, x, y)` — raw hologram intensity, shape `(N_holograms, 4872, 3248)`
- `x, y, z` — particle positions in **meters** (multiply by 1e6 for micrometers)
- `d` — particle diameter in **micrometers**
- `hid` — hologram ID (**1-indexed**), links particle records to holograms
- Attributes: `Nx=4872`, `Ny=3248`, `dx/dy` (pixel pitch in m), `lambda` (wavelength), `zMin/zMax`

The depth range is approximately `zMin ≈ 14,000 μm` to `zMax ≈ 158,000 μm` (~144 mm). With `n_bins=1000`, each z-bin spans ~144 μm.

---

## Single-Plane Pipeline

A simpler baseline: propagate to one z-plane, predict a binary particle mask (no depth regression).

### Config: `config/single_plane_validate.yml`

Key settings:
```yaml
data:
    file_path: "...training.nc"
    n_bins: 1000       # z-planes to sweep
    lookahead: 0       # 0 = single plane
    tile_size: 512
    step_size: 128
    device: "cuda"     # GPU wave propagation (~5x faster than CPU)

model:
    name: "fpn"
    encoder_name: "vgg11"
    encoder_weights: "imagenet"
    in_channels: 1     # amplitude only
    classes: 1         # mask only
    activation: "sigmoid"
```

### Train

```bash
qsub scripts/submit_single_plane.sh
# or interactively:
python applications/trainer_unet_planer.py -c config/single_plane_validate.yml
```

Checkpoints are saved to `save_loc` after each epoch.

---

## Multi-Plane Pipeline

The primary science target. The model receives amplitude **and phase** from `lookahead` propagation planes centered on a z-position, and predicts both a binary mask and a continuous depth offset.

### How lookahead works

With `lookahead=5`:
- Planes backward: `floor((5-1)/2) = 2`
- Planes forward:  `ceil((5-1)/2) = 2`
- Total planes: **5** (2 back + center + 2 forward)
- Input channels: 5 amplitude + 5 phase = **10 channels**
- Depth label: `z_particle - z_centers[z_idx]` in **micrometers**

Absolute z at inference: `z_abs = z_centers[z_idx] + model_depth_output`

### Config: `config/multiplane.yml`

Key settings:
```yaml
training_data:
    type: "multiplane"
    file_path: "...training.nc"
    n_bins: 1000
    lookahead: 5
    tile_size: 512
    step_size: 128
    device: "cuda"     # GPU propagation — must set thread_workers: 0

model:
    name: "fpn"
    encoder_name: "vgg11"
    encoder_weights: "imagenet"
    in_channels: 10    # 2 * lookahead
    classes: 2         # ch0: sigmoid mask, ch1: depth regression
    activation: null

loss:
    training_loss_mask: "focal-tyversky"
    training_loss_depth: "intersectedmae"
    training_loss_depth_mult: 0.03
```

> **`intersectedmae`**: depth loss is computed only where predicted mask and true mask
> intersect — prevents large gradients from empty background regions where depth is undefined.

### Train

```bash
qsub scripts/submit_multiplane.sh
# or interactively:
python applications/trainer_unet.py -c config/multiplane.yml
```

Speed on A100: ~2.5 sec/batch → ~22 min/epoch (500 batches).

> **Important:** `thread_workers: 0` is required when `device: "cuda"` for wave
> propagation. CUDA tensors cannot be shared across DataLoader worker processes.

### Best hyperparameters (Matt Hayman, Nov 2024 hyperopt)

- `lr = 4.52e-4`, `weight_decay = 0`
- `scheduler: ReduceLROnPlateau(factor=0.1, patience=5)`
- Loss: `focal-tversky (mask) + 0.03 × intersectedMAE (depth)`
- Architecture: FPN + VGG11 (ImageNet pretrained), 11M parameters

Matt's trained checkpoint (FPN + ResNet50, `in_channels=10`, `classes=2`) is at:
`results/checkpoint.pt`

---

## Inference

`applications/inference_multiplane.py` runs the model on full holograms — no tiling required since the model is fully convolutional.

### Algorithm

For each hologram, sweep z-center positions at stride = `lookahead` (non-overlapping windows):

1. Propagate the full `4872×3248` hologram to `lookahead` planes → complex field `(n_planes, Nx, Ny)`
2. Compute amplitude + phase, normalize → `(2*lookahead, Nx, Ny)` float tensor
3. Reflect-pad by `halo` pixels (reduces edge artifacts from limited receptive field)
4. Zero-pad to next multiple of 32 (required by ResNet/VGG encoder downsampling)
5. Single forward pass → `(2, Nx, Ny)` mask probability + depth prediction
6. Crop back to hologram size, threshold mask
7. Connected-component labeling → one detection per component
8. `z_abs = z_centers[z_idx] + depth_pred[centroid_x, centroid_y]`

With `lookahead=5` and stride=5: **~200 z evaluations per hologram** on a single A100.

### Run (single hologram test)

```bash
python applications/inference_multiplane.py \
    -c config/inference_multiplane_matt.yml \
    -k results/checkpoint.pt \
    -o /path/to/output \
    --max-holograms 1 \
    --z-stride 5 \
    --halo 64 \
    --threshold 0.5 \
    --device cuda
```

### Run (full validation set)

```bash
qsub scripts/submit_inference.sh
```

### Output: `detections.csv`

| Column | Description |
|--------|-------------|
| `h_idx` | Hologram index (0-indexed) |
| `x_pix`, `y_pix` | Centroid in pixel coordinates |
| `x_um`, `y_um` | Centroid in micrometers |
| `z_um` | Absolute depth = z_center + predicted offset |
| `d_pix` | Diameter in pixels (bounding-box extent) |
| `d_um` | Diameter in micrometers |
| `mask_score` | Sigmoid mask probability at centroid [0–1] |

---

## Visualization

`applications/visualize_detections.py` compares model output to ground truth and generates standard diagnostic plots.

```bash
python applications/visualize_detections.py \
    -d /path/to/detections.csv \
    -n /path/to/synthetic_holograms_500particle_gamma_4872x3248_validation.nc \
    -o /path/to/plots \
    --h-idx 0 \
    --z-tol 360 \
    --xy-tol 50 \
    --threshold 0.5 \
    --training-log /path/to/training_log.csv
```

### Plots generated

| File | Description |
|------|-------------|
| `hologram_raw.png` | Raw hologram intensity |
| `detections_xy.png` | x-y map colored by z depth: predicted vs truth |
| `detections_3d.png` | 3D scatter: predicted (blue) vs ground truth (red) |
| `z_distribution.png` | Histogram of z positions: predicted vs true |
| `d_distribution.png` | Histogram of particle diameters: predicted vs true |
| `match_summary.png` | TP/FP/FN counts, precision/recall/F1, z accuracy scatter |
| `training_curves.png` | Loss over epochs (optional) |

**Match tolerance:** a prediction is a TP if within `xy_tol` μm in x,y **and** `z_tol` μm in z of any ground-truth particle. Default: `xy_tol=50 μm`, `z_tol=360 μm` (~2.5 z-bins).

---

## Model Architecture

Both modes use [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch).

### Multi-plane (production)

```
Input:  (B, 10, H, W)   ← 5 amplitude + 5 phase channels
  FPN encoder: VGG11 or ResNet50 (ImageNet pretrained)
  FPN decoder: pyramid feature aggregation
  Segmentation head: (B, 2, H, W)
    Channel 0: Sigmoid(x)  → particle mask probability [0, 1]
    Channel 1: Identity(x) → z offset regression (micrometers, unbounded)
```

### Single-plane (baseline)

```
Input:  (B, 1, H, W)   ← amplitude only
  FPN encoder: VGG11
  Segmentation head: (B, 1, H, W)
    Channel 0: Sigmoid(x) → mask probability
```

---

## Key Config Options

| Option | Default | Description |
|--------|---------|-------------|
| `training_data.lookahead` | 5 | z-planes per sample; `in_channels = 2×lookahead` |
| `training_data.n_bins` | 1000 | z-discretization; dz ≈ 144 μm for this dataset |
| `training_data.device` | `"cuda"` | Wave propagation device; requires `thread_workers: 0` |
| `trainer.batches_per_epoch` | 500 | Batches per epoch (random crops make dataset effectively infinite) |
| `loss.training_loss_depth_mult` | 0.03 | Weight of depth loss relative to mask loss |
| `trainer.stopping_patience` | 4 | Early stop after N epochs without improvement |
| `trainer.amp` | false | Mixed precision; set true to roughly halve memory/time |

---

## Key Files

```
applications/
  trainer_unet.py               multi-plane training (mask + depth)
  trainer_unet_planer.py        single-plane training (mask only)
  inference_multiplane.py       full-image inference, no tiling
  visualize_detections.py       diagnostic plots vs ground truth
  inference_planar.py           legacy tiled single-plane inference

holodec/
  propagation.py                WavePropagator — FFT Fresnel wave propagation
  planer_datasets.py            LoadMultiplaneHolograms, LoadHolograms
  unet.py                       SegmentationModel, PlanerSegmentationModel
  losses.py                     focal-tversky, dice, intersectedMAE/MSE
  trainer.py                    Trainer.fit() — multi-plane training loop
  planer_trainer.py             Trainer.fit() — single-plane training loop

config/
  multiplane.yml                multi-plane training (production)
  single_plane_validate.yml     single-plane training (baseline)
  inference_multiplane_matt.yml inference config for Matt's checkpoint

scripts/
  submit_multiplane.sh          PBS: multi-plane training, 12h, 1×A100
  submit_single_plane.sh        PBS: single-plane training
  submit_inference_test.sh      PBS: inference test (1 hologram)

results/
  checkpoint.pt                 Matt's best checkpoint
                                (FPN+ResNet50, in_channels=10, classes=2, epoch 3)
  training_log.csv              Training history for that run
```
