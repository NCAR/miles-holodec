"""
Full-image multi-plane hologram inference.

For each hologram, sweeps z-center positions (default stride = lookahead),
propagates the full hologram to `lookahead` planes at each position, optionally
pads with a reflective halo to reduce boundary artifacts, runs the model on the
full image (no tiling), then extracts particle coordinates via connected-component
labeling.

For SegmentationModel (classes=2): absolute z is reconstructed as
    z_abs_um = z_centers[z_idx] + pred_depth[centroid_x, centroid_y]
For PlanerSegmentationModel (classes=1): z_abs_um = z_centers[z_idx].

Output CSV columns:
    h_idx, x_pix, y_pix, x_um, y_um, z_um, d_pix, d_um, mask_score

Usage:
    python inference_multiplane.py \\
        -c config/multiplane.yml \\
        -k /path/to/checkpoint.pt \\
        -o /path/to/output_dir \\
        [--data-path /override/data.nc] \\
        [--halo 64] [--z-stride 5] [--threshold 0.5] \\
        [--max-holograms 2] [--h-start 0]
"""

import argparse
import logging
import math
import os
import warnings

import numpy as np
import pandas as pd
import scipy.ndimage
import torch
import tqdm
import yaml

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from holodec.propagation import WavePropagator
from holodec.seed import seed_everything
from holodec.unet import PlanerSegmentationModel, SegmentationModel

logger = logging.getLogger(__name__)


def _load_model(conf, checkpoint_path, device):
    """
    Load the right model class from checkpoint.
    SegmentationModel (classes=2) already applies sigmoid+identity in forward.
    PlanerSegmentationModel (classes=1) returns raw logits — we apply sigmoid
    explicitly during inference.
    """
    n_classes = conf["model"].get("classes", 1)
    if n_classes == 2:
        model = SegmentationModel(conf).to(device)
    else:
        model = PlanerSegmentationModel(conf).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    # strip DDP / FSDP wrapper prefix if present
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded {'2-channel (mask+depth)' if n_classes == 2 else '1-channel (mask)'} "
                f"model from {checkpoint_path}")
    return model, n_classes


def _build_input(field, in_channels):
    """
    Build the normalized input tensor from a propagated complex field.

    field: (n_planes, Nx, Ny) complex tensor (on any device)
    in_channels: the model's expected input channel count

    Returns a CPU float32 tensor of shape (in_channels, Nx, Ny).
    """
    if in_channels == 1:
        # amplitude only (e.g. Matt's single-channel model)
        return (torch.abs(field[0:1]).float() / 255.0).cpu()
    elif in_channels == 2:
        # amplitude + phase for the centre plane only
        ampl  = torch.abs(field[0:1]).float()  / 255.0
        phase = torch.angle(field[0:1]).float() / math.pi
        return torch.cat([ampl, phase], dim=0).cpu()
    else:
        # multiplane: all amplitudes then all phases
        ampl  = torch.abs(field).float()  / 255.0
        phase = torch.angle(field).float() / math.pi
        return torch.cat([ampl, phase], dim=0).cpu()


def run_inference(
    conf,
    checkpoint_path,
    output_dir,
    threshold=0.5,
    device_str="cuda",
    halo=64,
    z_stride=None,
    max_holograms=None,
    h_start=0,
):
    seed_everything(conf.get("seed", 1000))
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # ── model ──────────────────────────────────────────────────────────────────
    model, n_classes = _load_model(conf, checkpoint_path, device)
    in_channels = conf["model"].get("in_channels", 1)

    # ── data / propagation settings ────────────────────────────────────────────
    infer_conf = conf.get("inference", {})
    data_conf  = conf.get("training_data", conf.get("data", {}))

    data_path = infer_conf.get("data_path") or data_conf.get("file_path") or data_conf.get("data_path")
    n_bins    = data_conf.get("n_bins", 1000)
    lookahead = data_conf.get("lookahead", 1)
    tile_size = data_conf.get("tile_size", 512)
    step_size = data_conf.get("step_size", 128)
    prop_device = infer_conf.get("prop_device", device_str if torch.cuda.is_available() else "cpu")

    propagator = WavePropagator(
        data_path,
        n_bins=n_bins,
        tile_size=tile_size,
        step_size=step_size,
        device=prop_device,
    )

    # ── z-sweep parameters ─────────────────────────────────────────────────────
    if lookahead > 1:
        z_bck = int(math.floor((lookahead - 1) / 2))
        z_fwd = int(math.ceil((lookahead - 1) / 2)) + 1
    else:
        z_bck, z_fwd = 0, 1

    if z_stride is None:
        z_stride = max(1, lookahead)

    z_indices = list(range(z_bck, n_bins - z_fwd, z_stride))
    logger.info(
        f"Z sweep: {len(z_indices)} positions, stride={z_stride}, "
        f"lookahead={lookahead} planes ({z_bck} back / {z_fwd - 1} forward)"
    )
    logger.info(f"Halo: {halo}px  |  threshold: {threshold}  |  device: {device}")

    # ── hologram loop ──────────────────────────────────────────────────────────
    n_holograms = len(propagator.h_ds.hologram_number)
    h_end       = min(h_start + max_holograms, n_holograms) if max_holograms else n_holograms
    h_indices   = list(range(h_start, h_end))
    logger.info(f"Processing hologram indices {h_start} – {h_end - 1}")

    all_detections = []
    sigmoid = torch.nn.Sigmoid()

    for h_idx in tqdm.tqdm(h_indices, desc="Holograms"):
        image = propagator.h_ds["image"].isel(hologram_number=h_idx).values.astype(float)
        image_tensor = torch.from_numpy(image).to(prop_device)

        n_detections_this_holo = 0

        for z_idx in tqdm.tqdm(z_indices, desc=f"  h={h_idx} z-sweep", leave=False):

            # propagate full hologram to lookahead planes
            if lookahead > 1:
                z_slc = propagator.z_centers[z_idx - z_bck : z_idx + z_fwd] * 1e-6
            else:
                z_slc = propagator.z_centers[z_idx : z_idx + 1] * 1e-6

            z_tensor = torch.tensor(
                z_slc[:, np.newaxis, np.newaxis], dtype=torch.float32
            ).to(prop_device)

            with torch.no_grad():
                field   = propagator.torch_holo_set(image_tensor, z_tensor)
                in_ch   = _build_input(field, in_channels).to(device)  # (C, Nx, Ny)

                # Pad: reflect halo to reduce edge artifacts, then align to
                # multiples of 32 (required by ResNet/VGG encoders with 5
                # downsampling stages).
                _, H, W = in_ch.shape
                x = in_ch.unsqueeze(0)
                if halo > 0:
                    x = torch.nn.functional.pad(
                        x, (halo, halo, halo, halo), mode="reflect"
                    )
                pH, pW = x.shape[-2], x.shape[-1]
                align = 32
                tH = math.ceil(pH / align) * align
                tW = math.ceil(pW / align) * align
                pad_b = tH - pH  # extra bottom
                pad_r = tW - pW  # extra right
                if pad_b > 0 or pad_r > 0:
                    x = torch.nn.functional.pad(
                        x, (0, pad_r, 0, pad_b), mode="replicate"
                    )

                pred = model(x)  # (1, classes, tH, tW)

                # crop back to original hologram size
                pred = pred[:, :, halo:halo + H, halo:halo + W]

                # mask probability
                if n_classes == 1:
                    # PlanerSegmentationModel returns raw logits
                    mask_prob = sigmoid(pred[0, 0]).cpu().numpy()
                else:
                    # SegmentationModel already applied sigmoid to ch0
                    mask_prob = pred[0, 0].cpu().numpy()

                depth_pred = pred[0, 1].cpu().numpy() if n_classes == 2 else None

            # ── connected-component extraction ────────────────────────────────
            binary_mask = (mask_prob > threshold).astype(np.uint8)
            if binary_mask.sum() == 0:
                continue

            labeled, n_labels = scipy.ndimage.label(binary_mask)
            if n_labels == 0:
                continue

            objects = scipy.ndimage.find_objects(labeled)
            z_center_um = float(propagator.z_centers[z_idx])

            for obj in objects:
                if obj is None:
                    continue
                x0, x1 = obj[0].start, obj[0].stop
                y0, y1 = obj[1].start, obj[1].stop
                xind = min((x0 + x1) // 2, propagator.Nx - 1)
                yind = min((y0 + y1) // 2, propagator.Ny - 1)
                d_pix = max(x1 - x0, y1 - y0)

                x_um = float(propagator.x_arr[xind]) * 1e6
                y_um = float(propagator.y_arr[yind]) * 1e6
                d_um = float(d_pix * propagator.dx * 1e6)

                z_um = (z_center_um + float(depth_pred[xind, yind])
                        if depth_pred is not None else z_center_um)

                all_detections.append({
                    "h_idx":      h_idx,
                    "x_pix":      int(xind),
                    "y_pix":      int(yind),
                    "x_um":       round(x_um, 3),
                    "y_um":       round(y_um, 3),
                    "z_um":       round(z_um, 3),
                    "d_pix":      int(d_pix),
                    "d_um":       round(d_um, 3),
                    "mask_score": round(float(mask_prob[xind, yind]), 4),
                })
                n_detections_this_holo += 1

        logger.info(f"  h={h_idx}: {n_detections_this_holo} raw detections")

    out_path = os.path.join(output_dir, "detections.csv")
    df = pd.DataFrame(all_detections)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} total detections → {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full-image multiplane hologram inference (no tiling)"
    )
    parser.add_argument("-c",  dest="model_config",  required=True,
                        help="Path to model config YAML")
    parser.add_argument("-k",  dest="checkpoint",    required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("-o",  dest="output_dir",    required=True,
                        help="Directory to write detections.csv")
    parser.add_argument("--data-path",     default=None,
                        help="Override data file path from config")
    parser.add_argument("--threshold",     type=float, default=0.5,
                        help="Mask probability threshold (default: 0.5)")
    parser.add_argument("--device",        default="cuda",
                        help="Inference device: cuda or cpu (default: cuda)")
    parser.add_argument("--halo",          type=int, default=64,
                        help="Reflective halo padding in pixels (default: 64)")
    parser.add_argument("--z-stride",      type=int, default=None,
                        help="Z sweep stride (default: lookahead)")
    parser.add_argument("--max-holograms", type=int, default=None,
                        help="Max number of holograms to process")
    parser.add_argument("--h-start",       type=int, default=0,
                        help="Starting hologram index (default: 0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    with open(args.model_config) as f:
        conf = yaml.safe_load(f)

    if args.data_path:
        for key in ("training_data", "data"):
            if key in conf:
                conf[key]["file_path"] = args.data_path
                conf[key]["data_path"] = args.data_path

    run_inference(
        conf,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        threshold=args.threshold,
        device_str=args.device,
        halo=args.halo,
        z_stride=args.z_stride,
        max_holograms=args.max_holograms,
        h_start=args.h_start,
    )
