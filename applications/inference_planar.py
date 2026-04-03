"""
Planar hologram inference script.

Runs a trained segmentation model over all (hologram, z-plane) pairs in a
hologram dataset, then extracts particle coordinates using connected-component
labeling (Jeff Boothe's approach, ported from the Lightning predict_step).

Output columns: h_idx, x_pix, y_pix, z_idx, d_pix
"""

import argparse
import logging
import os
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.ndimage
import torch
import tqdm
import yaml

from holodec.planer_datasets import LoadHolograms
from holodec.planer_transforms import LoadTransformations
from holodec.seed import seed_everything
from holodec.unet import PlanerSegmentationModel as SegmentationModel

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


logger = logging.getLogger(__name__)


def extract_coordinates(pred_mask, target_mask, h_idx, z_idx):
    """
    Given a binary predicted mask and the ground-truth mask for one z-plane,
    extract (h_idx, x, y, z_idx, d) for each connected component.

    Returns:
        pred_coords  : list of [h_idx, x_pix, y_pix, z_idx, d_pix]
        true_coords  : list of [h_idx, x_pix, y_pix, z_idx, d_pix]
    """
    def _label_to_coords(binary_mask):
        coords = []
        if binary_mask.sum() == 0:
            return coords
        labeled, _ = scipy.ndimage.label(binary_mask)
        objects = scipy.ndimage.find_objects(labeled)
        for particle in objects:
            if particle is None:
                continue
            xind = (particle[0].stop + particle[0].start) // 2
            yind = (particle[1].stop + particle[1].start) // 2
            dind = max(
                abs(particle[0].stop - particle[0].start),
                abs(particle[1].stop - particle[1].start),
            )
            coords.append([h_idx, xind, yind, z_idx, dind])
        return coords

    pred_coords = _label_to_coords(pred_mask)
    true_coords = _label_to_coords(target_mask)
    return pred_coords, true_coords


def run_inference(conf, checkpoint_path, output_dir, threshold=0.5, device_str="cuda"):
    seed = conf.get("seed", 1000)
    seed_everything(seed)

    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(checkpoint_path, os.path.join(output_dir, "checkpoint_used.pt"))

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Load transforms (inference-mode)
    infer_transforms = LoadTransformations(conf["transforms"].get("inference", conf["transforms"]["validation"]))

    # Load dataset
    data_conf = conf["data"]
    dataset = LoadHolograms(
        data_conf.get("infer_data_path", data_conf["data_path"]),
        n_bins=data_conf["n_bins"],
        shuffle=False,
        device=data_conf.get("device", "cpu"),
        transform=infer_transforms,
        lookahead=data_conf.get("lookahead", 0),
        tile_size=data_conf["tile_size"],
        step_size=data_conf["step_size"],
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=conf.get("inference", {}).get("batch_size", 1),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # Load model
    model = SegmentationModel(conf).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    # strip DDP wrapper prefix if present
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    all_pred = []
    all_true = []

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Inference"):
            images, masks, h_idxs, z_idxs = batch

            preds = model(images.to(device).float())

            for i in range(len(h_idxs)):
                pred_binary = (preds[i, 0].cpu().numpy() > threshold).astype(np.uint8)
                true_binary = (masks[i].cpu().numpy() > 0).astype(np.uint8)

                h_idx = int(h_idxs[i])
                z_idx = int(z_idxs[i])

                pred_coords, true_coords = extract_coordinates(
                    pred_binary, true_binary, h_idx, z_idx
                )
                all_pred.extend(pred_coords)
                all_true.extend(true_coords)

    cols = ["h_idx", "x_pix", "y_pix", "z_idx", "d_pix"]
    pd.DataFrame(all_pred, columns=cols).to_csv(
        os.path.join(output_dir, "predicted_coordinates.csv"), index=False
    )
    pd.DataFrame(all_true, columns=cols).to_csv(
        os.path.join(output_dir, "true_coordinates.csv"), index=False
    )
    logger.info(
        f"Saved {len(all_pred)} predicted and {len(all_true)} true particle coordinates to {output_dir}"
    )


if __name__ == "__main__":
    description = "Run planar hologram inference and extract particle coordinates"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-c", dest="model_config", type=str, required=True,
                        help="Path to model config yaml")
    parser.add_argument("-k", dest="checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("-o", dest="output_dir", type=str, required=True,
                        help="Directory to write coordinate CSVs")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binary mask threshold (default: 0.5)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Override data path in config (optional)")
    args = parser.parse_args()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root.addHandler(ch)

    with open(args.model_config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    if args.data_path:
        conf["data"]["infer_data_path"] = args.data_path

    run_inference(
        conf,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        threshold=args.threshold,
        device_str=args.device,
    )
