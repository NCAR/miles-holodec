"""
Visualize holodec inference results.

Produces a standard set of diagnostic plots comparing predicted particle
detections (from inference_multiplane.py) against ground-truth coordinates
from the original hologram NC file.

Plots generated (saved as PNG to output_dir):
  1. hologram_raw.png        — raw hologram intensity image
  2. detections_xy.png       — x-y particle map, colored by depth (z_um)
  3. detections_3d.png       — 3D scatter: predicted vs ground truth
  4. z_distribution.png      — histogram of predicted vs true z positions
  5. d_distribution.png      — histogram of predicted vs true diameters
  6. match_summary.png       — TP / FP / FN breakdown and precision/recall

Usage:
    python visualize_detections.py \\
        -d /path/to/detections.csv \\
        -n /path/to/holograms.nc \\
        -o /path/to/output_dir \\
        [--h-idx 0] [--z-tol 360] [--xy-tol 50] [--threshold 0.5]
"""

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


# ── styling ────────────────────────────────────────────────────────────────────
PRED_COLOR  = "#2196F3"   # blue
TRUE_COLOR  = "#F44336"   # red
TP_COLOR    = "#4CAF50"   # green
FP_COLOR    = "#FF9800"   # orange
FN_COLOR    = "#9C27B0"   # purple


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_truth(nc_path, h_idx):
    """Return DataFrame of ground-truth particles for one hologram."""
    ds = xr.open_dataset(nc_path)
    hid = h_idx + 1  # NC file uses 1-indexed hid
    mask = ds["hid"].values == hid
    truth = pd.DataFrame({
        "x_um": ds["x"].values[mask] * 1e6,
        "y_um": ds["y"].values[mask] * 1e6,
        "z_um": ds["z"].values[mask] * 1e6,
        "d_um": ds["d"].values[mask],
    })
    ds.close()
    return truth


def _load_hologram(nc_path, h_idx):
    """Return (Nx, Ny) float array of hologram intensity."""
    ds = xr.open_dataset(nc_path)
    img = ds["image"].isel(hologram_number=h_idx).values.astype(float)
    ds.close()
    return img


def _match(pred_df, truth_df, z_tol, xy_tol):
    """
    Match predicted detections to ground-truth particles using a KD-tree on
    (x_um, y_um, z_um) with anisotropic tolerances.

    Returns (tp_pred, tp_true, fp, fn) DataFrames.
    """
    if len(pred_df) == 0 or len(truth_df) == 0:
        return (pd.DataFrame(), pd.DataFrame(),
                pred_df.copy(), truth_df.copy())

    # Scale coordinates so tolerances are isotropic
    scale = np.array([1.0 / xy_tol, 1.0 / xy_tol, 1.0 / z_tol])

    p_pts = pred_df[["x_um", "y_um", "z_um"]].values * scale
    t_pts = truth_df[["x_um", "y_um", "z_um"]].values * scale

    tree = cKDTree(t_pts)
    dists, idxs = tree.query(p_pts, k=1, workers=-1)

    matched = dists <= 1.0   # within tolerance sphere

    tp_pred_idx  = np.where(matched)[0]
    tp_true_idx  = np.unique(idxs[matched])
    fp_idx       = np.where(~matched)[0]
    fn_idx       = np.setdiff1d(np.arange(len(truth_df)), tp_true_idx)

    return (
        pred_df.iloc[tp_pred_idx].copy(),
        truth_df.iloc[tp_true_idx].copy(),
        pred_df.iloc[fp_idx].copy(),
        truth_df.iloc[fn_idx].copy(),
    )


# ── individual plots ───────────────────────────────────────────────────────────

def plot_hologram_raw(img, h_idx, out_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img, cmap="gray", aspect="auto", origin="lower",
              vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
    ax.set_title(f"Raw hologram — h_idx={h_idx}", fontsize=13)
    ax.set_xlabel("y pixel"); ax.set_ylabel("x pixel")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_detections_xy(pred_df, truth_df, h_idx, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, df, label, cmap in [
        (axes[0], pred_df,  "Predicted", "viridis"),
        (axes[1], truth_df, "Ground truth", "viridis"),
    ]:
        if len(df):
            sc = ax.scatter(df["y_um"], df["x_um"], c=df["z_um"],
                            cmap=cmap, s=8, alpha=0.6)
            plt.colorbar(sc, ax=ax, label="z (μm)")
        ax.set_title(f"{label} — {len(df)} particles", fontsize=12)
        ax.set_xlabel("y (μm)"); ax.set_ylabel("x (μm)")
        ax.invert_yaxis()

    fig.suptitle(f"Particle x-y map — h_idx={h_idx}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_detections_3d(pred_df, truth_df, h_idx, out_path):
    fig = plt.figure(figsize=(12, 5))

    for i, (df, label, color) in enumerate([
        (pred_df,  "Predicted",    PRED_COLOR),
        (truth_df, "Ground truth", TRUE_COLOR),
    ]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        if len(df):
            ax.scatter(df["x_um"], df["y_um"], df["z_um"],
                       c=color, s=4, alpha=0.5)
        ax.set_xlabel("x (μm)", labelpad=2)
        ax.set_ylabel("y (μm)", labelpad=2)
        ax.set_zlabel("z (μm)", labelpad=2)
        ax.set_title(f"{label}\n({len(df)} particles)", fontsize=11)

    fig.suptitle(f"3D particle positions — h_idx={h_idx}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_z_distribution(pred_df, truth_df, h_idx, out_path):
    fig, ax = plt.subplots(figsize=(9, 4))

    bins = np.linspace(
        min(pred_df["z_um"].min() if len(pred_df) else 0,
            truth_df["z_um"].min() if len(truth_df) else 0),
        max(pred_df["z_um"].max() if len(pred_df) else 1,
            truth_df["z_um"].max() if len(truth_df) else 1),
        80,
    )

    if len(truth_df):
        ax.hist(truth_df["z_um"], bins=bins, alpha=0.6,
                color=TRUE_COLOR, label=f"Ground truth (n={len(truth_df)})")
    if len(pred_df):
        ax.hist(pred_df["z_um"], bins=bins, alpha=0.6,
                color=PRED_COLOR, label=f"Predicted (n={len(pred_df)})")

    ax.set_xlabel("z (μm)"); ax.set_ylabel("count")
    ax.set_title(f"z distribution — h_idx={h_idx}", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_d_distribution(pred_df, truth_df, h_idx, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))

    all_d = pd.concat([pred_df["d_um"] if len(pred_df) else pd.Series(dtype=float),
                       truth_df["d_um"] if len(truth_df) else pd.Series(dtype=float)])
    if len(all_d) == 0:
        plt.close(fig); return
    bins = np.linspace(0, np.percentile(all_d, 98), 50)

    if len(truth_df):
        ax.hist(truth_df["d_um"], bins=bins, alpha=0.6,
                color=TRUE_COLOR, label=f"Ground truth (n={len(truth_df)})")
    if len(pred_df):
        ax.hist(pred_df["d_um"], bins=bins, alpha=0.6,
                color=PRED_COLOR, label=f"Predicted (n={len(pred_df)})")

    ax.set_xlabel("diameter (μm)"); ax.set_ylabel("count")
    ax.set_title(f"Particle diameter — h_idx={h_idx}", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_match_summary(tp_pred, tp_true, fp, fn, h_idx, z_tol, xy_tol, out_path):
    n_tp = len(tp_pred)
    n_fp = len(fp)
    n_fn = len(fn)
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    recall    = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # ── left: bar chart ─────────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0])
    bars = ax_bar.bar(["TP", "FP", "FN"],
                      [n_tp, n_fp, n_fn],
                      color=[TP_COLOR, FP_COLOR, FN_COLOR])
    for bar, val in zip(bars, [n_tp, n_fp, n_fn]):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom", fontsize=11)
    ax_bar.set_ylabel("count")
    ax_bar.set_title(
        f"Detections\nP={precision:.2f}  R={recall:.2f}  F1={f1:.2f}",
        fontsize=11)
    ax_bar.set_ylim(0, max(n_tp, n_fp, n_fn) * 1.15 + 1)

    # ── middle: TP+FP on x-y ────────────────────────────────────────────────
    ax_xy = fig.add_subplot(gs[1])
    if len(tp_pred):
        ax_xy.scatter(tp_pred["y_um"], tp_pred["x_um"],
                      c=TP_COLOR, s=5, alpha=0.5, label=f"TP ({n_tp})")
    if len(fp):
        ax_xy.scatter(fp["y_um"], fp["x_um"],
                      c=FP_COLOR, s=5, alpha=0.5, label=f"FP ({n_fp})")
    if len(fn):
        ax_xy.scatter(fn["y_um"], fn["x_um"],
                      c=FN_COLOR, s=5, alpha=0.5, marker="x", label=f"FN ({n_fn})")
    ax_xy.set_xlabel("y (μm)"); ax_xy.set_ylabel("x (μm)")
    ax_xy.invert_yaxis()
    ax_xy.legend(fontsize=8, markerscale=2)
    ax_xy.set_title("Spatial match map", fontsize=11)

    # ── right: z comparison for matched pairs ───────────────────────────────
    ax_z = fig.add_subplot(gs[2])
    if len(tp_pred) and len(tp_true):
        # align by nearest match order (simple reindex for paired comparison)
        ax_z.scatter(tp_true["z_um"].values, tp_pred["z_um"].values,
                     c=TP_COLOR, s=8, alpha=0.5)
        z_lim = [min(tp_true["z_um"].min(), tp_pred["z_um"].min()),
                 max(tp_true["z_um"].max(), tp_pred["z_um"].max())]
        ax_z.plot(z_lim, z_lim, "k--", lw=1, alpha=0.4, label="1:1")
        ax_z.set_xlabel("true z (μm)"); ax_z.set_ylabel("pred z (μm)")
        ax_z.set_title("z accuracy (matched pairs)", fontsize=11)
        ax_z.legend(fontsize=8)

    tol_str = f"tol: xy={xy_tol}μm, z={z_tol}μm"
    fig.suptitle(f"Match summary — h_idx={h_idx}  [{tol_str}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_training_curves(log_path, out_path):
    """Plot training/validation loss curves from training_log.csv."""
    if not os.path.isfile(log_path):
        logger.warning(f"No training log at {log_path}")
        return

    df = pd.read_csv(log_path)
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    if not loss_cols:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    styles = {"train": dict(ls="-"), "valid": dict(ls="--")}
    for col in loss_cols:
        prefix = "train" if col.startswith("train") else "valid"
        ax.plot(df["epoch"], df[col],
                label=col.replace("_", " "),
                **styles.get(prefix, {}))

    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Training curves", fontsize=12)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def run_visualizer(
    detections_csv,
    nc_path,
    output_dir,
    h_idx=0,
    z_tol=360,
    xy_tol=50,
    threshold=0.5,
    training_log=None,
):
    os.makedirs(output_dir, exist_ok=True)

    # load predictions for this hologram
    pred_all = pd.read_csv(detections_csv)
    pred_df  = pred_all[
        (pred_all["h_idx"] == h_idx) &
        (pred_all["mask_score"] >= threshold)
    ].reset_index(drop=True)
    logger.info(f"Loaded {len(pred_df)} predictions (score ≥ {threshold}) "
                f"for h_idx={h_idx}")

    # load ground truth
    truth_df = _load_truth(nc_path, h_idx)
    logger.info(f"Loaded {len(truth_df)} ground-truth particles for h_idx={h_idx}")

    # match
    tp_pred, tp_true, fp, fn = _match(pred_df, truth_df, z_tol, xy_tol)
    n_tp, n_fp, n_fn = len(tp_pred), len(fp), len(fn)
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    recall    = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    logger.info(f"Match: TP={n_tp}  FP={n_fp}  FN={n_fn}  "
                f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")

    # hologram image
    img = _load_hologram(nc_path, h_idx)
    plot_hologram_raw(img, h_idx,
                      os.path.join(output_dir, "hologram_raw.png"))

    # x-y spatial map
    plot_detections_xy(pred_df, truth_df, h_idx,
                       os.path.join(output_dir, "detections_xy.png"))

    # 3D scatter
    plot_detections_3d(pred_df, truth_df, h_idx,
                       os.path.join(output_dir, "detections_3d.png"))

    # distributions
    plot_z_distribution(pred_df, truth_df, h_idx,
                        os.path.join(output_dir, "z_distribution.png"))
    plot_d_distribution(pred_df, truth_df, h_idx,
                        os.path.join(output_dir, "d_distribution.png"))

    # match summary
    plot_match_summary(tp_pred, tp_true, fp, fn, h_idx, z_tol, xy_tol,
                       os.path.join(output_dir, "match_summary.png"))

    # training curves (optional)
    if training_log:
        plot_training_curves(training_log,
                             os.path.join(output_dir, "training_curves.png"))

    # print summary table
    print(f"\n{'='*50}")
    print(f"  h_idx={h_idx}  |  threshold={threshold}")
    print(f"  Predicted:    {len(pred_df):>6}")
    print(f"  Ground truth: {len(truth_df):>6}")
    print(f"  TP: {n_tp:>5}   FP: {n_fp:>5}   FN: {n_fn:>5}")
    print(f"  Precision: {precision:.3f}   Recall: {recall:.3f}   F1: {f1:.3f}")
    print(f"  (match tol: xy={xy_tol}μm, z={z_tol}μm)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize multiplane holodec inference results"
    )
    parser.add_argument("-d", dest="detections_csv", required=True,
                        help="Path to detections.csv from inference_multiplane.py")
    parser.add_argument("-n", dest="nc_path", required=True,
                        help="Path to hologram NC file (for ground truth + raw image)")
    parser.add_argument("-o", dest="output_dir", required=True,
                        help="Directory to write PNG plots")
    parser.add_argument("--h-idx",    type=int,   default=0,
                        help="Hologram index to visualize (default: 0)")
    parser.add_argument("--z-tol",    type=float, default=360,
                        help="z match tolerance in μm (default: 360, ~2.5 z-bins)")
    parser.add_argument("--xy-tol",   type=float, default=50,
                        help="x/y match tolerance in μm (default: 50)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Min mask_score to include (default: 0.5)")
    parser.add_argument("--training-log", default=None,
                        help="Path to training_log.csv for loss curves (optional)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    run_visualizer(
        detections_csv=args.detections_csv,
        nc_path=args.nc_path,
        output_dir=args.output_dir,
        h_idx=args.h_idx,
        z_tol=args.z_tol,
        xy_tol=args.xy_tol,
        threshold=args.threshold,
        training_log=args.training_log,
    )
