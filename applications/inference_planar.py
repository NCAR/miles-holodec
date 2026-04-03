"""
Full-image single-plane hologram inference.

Sweeps all z-center positions, propagates the hologram to a single plane,
runs the model, and extracts particle coordinates via connected-component
labeling. Output format matches inference_multiplane.py and is compatible
with evaluate_detections.py.

For single-plane models (classes=1): z_um = z_centers[z_idx] (no depth offset).

Output CSV columns:
    h_idx, x_pix, y_pix, x_um, y_um, z_um, d_pix, d_um, mask_score

Usage:
    python inference_planar.py \\
        -c config/single_plane_validate.yml \\
        -k /path/to/checkpoint.pt \\
        -o /path/to/output_dir \\
        [--halo 64] [--z-stride 1] [--threshold 0.5] \\
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
import torch.nn.functional as F
import tqdm
import yaml

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from holodec.propagation import WavePropagator
from holodec.seed import seed_everything
from holodec.unet import PlanerSegmentationModel

logger = logging.getLogger(__name__)


def run_inference(
    conf,
    checkpoint_path,
    output_dir,
    threshold=0.5,
    device_str="cuda",
    halo=64,
    z_stride=1,
    max_holograms=None,
    h_start=0,
):
    seed_everything(conf.get("seed", 1000))
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # ── model ──────────────────────────────────────────────────────────────────
    model = PlanerSegmentationModel(conf).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded single-plane model from {checkpoint_path}")

    # ── data / propagation ─────────────────────────────────────────────────────
    data_conf = conf.get("data", {})
    data_path = data_conf.get("data_path") or data_conf.get("file_path")
    n_bins    = data_conf.get("n_bins", 1000)
    tile_size = data_conf.get("tile_size", 512)
    step_size = data_conf.get("step_size", 128)
    prop_device = device_str if torch.cuda.is_available() else "cpu"

    propagator = WavePropagator(
        data_path,
        n_bins=n_bins,
        tile_size=tile_size,
        step_size=step_size,
        device=prop_device,
    )

    z_indices = list(range(0, n_bins, z_stride))
    logger.info(f"Z sweep: {len(z_indices)} positions, stride={z_stride}")
    logger.info(f"Halo: {halo}px  |  threshold: {threshold}  |  device: {device}")

    # ── hologram loop ──────────────────────────────────────────────────────────
    n_holograms = len(propagator.h_ds.hologram_number)
    h_end       = min(h_start + max_holograms, n_holograms) if max_holograms else n_holograms
    h_indices   = list(range(h_start, h_end))

    all_detections = []
    sigmoid = torch.nn.Sigmoid()

    for h_idx in tqdm.tqdm(h_indices, desc="Holograms"):
        image = propagator.h_ds["image"].isel(hologram_number=h_idx).values.astype(float)
        image_tensor = torch.from_numpy(image).to(prop_device)

        n_detections_this_holo = 0

        for z_idx in tqdm.tqdm(z_indices, desc=f"  h={h_idx} z-sweep", leave=False):
            z_m = propagator.z_centers[z_idx] * 1e-6
            z_tensor = torch.tensor(
                [[[z_m]]], dtype=torch.float32
            ).to(prop_device)

            with torch.no_grad():
                field = propagator.torch_holo_set(image_tensor, z_tensor)
                # amplitude only, normalized
                ampl = (torch.abs(field[0:1]).float() / 255.0).to(device)

                # pad: reflect halo, then align to multiple of 32
                _, H, W = ampl.shape
                x = ampl.unsqueeze(0)
                if halo > 0:
                    x = F.pad(x, (halo, halo, halo, halo), mode="reflect")
                pH, pW = x.shape[-2], x.shape[-1]
                tH = math.ceil(pH / 32) * 32
                tW = math.ceil(pW / 32) * 32
                if tH > pH or tW > pW:
                    x = F.pad(x, (0, tW - pW, 0, tH - pH), mode="replicate")

                pred = model(x)           # (1, 1, tH, tW) — raw logits
                pred = pred[:, :, halo:halo + H, halo:halo + W]
                mask_prob = sigmoid(pred[0, 0]).cpu().numpy()

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

                all_detections.append({
                    "h_idx":      h_idx,
                    "x_pix":      int(xind),
                    "y_pix":      int(yind),
                    "x_um":       round(x_um, 3),
                    "y_um":       round(y_um, 3),
                    "z_um":       round(z_center_um, 3),
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
        description="Full-image single-plane hologram inference"
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
    parser.add_argument("--z-stride",      type=int, default=1,
                        help="Z sweep stride (default: 1, all planes)")
    parser.add_argument("--max-holograms", type=int, default=None,
                        help="Max number of holograms to process")
    parser.add_argument("--h-start",       type=int, default=0,
                        help="Starting hologram index (default: 0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    with open(args.model_config) as f:
        conf = yaml.safe_load(f)

    if args.data_path:
        conf["data"]["data_path"] = args.data_path
        conf["data"]["file_path"] = args.data_path

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
