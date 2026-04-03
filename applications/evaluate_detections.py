"""
Evaluate multiplane hologram detections against ground truth.

Loads a detections CSV (from inference_multiplane.py, optionally already NMS-clustered)
and a ground-truth NetCDF file, then performs optimal bipartite matching per hologram
(Hungarian algorithm) within anisotropic distance tolerances.

Metrics reported (per hologram and overall):
  POD  = TP / (TP + FN)         — Probability of Detection
  FAR  = FP / (TP + FP)         — False Alarm Ratio
  F1   = 2TP / (2TP + FP + FN)  — F1 score
  CSI  = TP / (TP + FP + FN)    — Critical Success Index (threat score)
  RMSE_x, RMSE_y, RMSE_z, RMSE_d — for matched (TP) pairs, in micrometers

Usage:
    python evaluate_detections.py \\
        -d /path/to/detections.csv \\
        -n /path/to/validation.nc \\
        -o /path/to/output_dir \\
        [--xy-tol 50] [--z-tol 360] \\
        [--cluster] [--xy-radius 50] [--z-radius 200] \\
        [--max-holograms 10] [--h-start 0]
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")


# ── matching ──────────────────────────────────────────────────────────────────

def _nms_3d(df: pd.DataFrame, xy_radius: float = 50.0, z_radius: float = 200.0) -> pd.DataFrame:
    """3D NMS: collapse duplicates before matching. See inference_multiplane.py."""
    if len(df) == 0:
        return df
    df = df.sort_values("mask_score", ascending=False).reset_index(drop=True)
    keep = np.ones(len(df), dtype=bool)
    x = df["x_um"].values
    y = df["y_um"].values
    z = df["z_um"].values
    for i in range(len(df) - 1):
        if not keep[i]:
            continue
        dx = (x[i + 1:] - x[i]) / xy_radius
        dy = (y[i + 1:] - y[i]) / xy_radius
        dz = (z[i + 1:] - z[i]) / z_radius
        suppress = dx ** 2 + dy ** 2 + dz ** 2 <= 1.0
        keep[i + 1:][suppress] = False
    return df[keep].reset_index(drop=True)


def match_hologram(pred_df: pd.DataFrame, truth_df: pd.DataFrame,
                   xy_tol: float, z_tol: float):
    """
    Optimal 1:1 bipartite matching between predicted and true particles.

    Uses the Hungarian algorithm on a cost matrix of anisotropic scaled distances.
    A match is valid only if the normalised distance <= 1 (i.e. within the
    (xy_tol, xy_tol, z_tol) ellipsoid).

    Returns a list of dicts with one row per true particle:
        matched (bool), and if matched: pred coords + dist_um
    """
    n_pred = len(pred_df)
    n_true = len(truth_df)

    pairs = []

    if n_pred == 0 or n_true == 0:
        for _, t in truth_df.iterrows():
            pairs.append(dict(
                x_t=t["x_um"], y_t=t["y_um"], z_t=t["z_um"], d_t=t["d_um"],
                x_p=np.nan, y_p=np.nan, z_p=np.nan, d_p=np.nan,
                matched=False, dist_um=np.nan,
            ))
        # unmatched predictions become FP rows later
        fp_rows = [dict(
            x_t=np.nan, y_t=np.nan, z_t=np.nan, d_t=np.nan,
            x_p=r["x_um"], y_p=r["y_um"], z_p=r["z_um"], d_p=r["d_um"],
            matched=False, dist_um=np.nan,
        ) for _, r in pred_df.iterrows()]
        return pairs + fp_rows

    # build cost matrix: normalised anisotropic distance
    px = pred_df["x_um"].values[:, None]   # (n_pred, 1)
    py = pred_df["y_um"].values[:, None]
    pz = pred_df["z_um"].values[:, None]

    tx = truth_df["x_um"].values[None, :]  # (1, n_true)
    ty = truth_df["y_um"].values[None, :]
    tz = truth_df["z_um"].values[None, :]

    dist_norm = np.sqrt(
        ((px - tx) / xy_tol) ** 2 +
        ((py - ty) / xy_tol) ** 2 +
        ((pz - tz) / z_tol) ** 2
    )  # (n_pred, n_true)

    # for Hungarian: make it square by padding with a large cost
    BIG = 1e9
    n = max(n_pred, n_true)
    cost = np.full((n, n), BIG)
    cost[:n_pred, :n_true] = dist_norm

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pred = set()
    matched_true = set()
    true_to_pred = {}

    for r, c in zip(row_ind, col_ind):
        if r < n_pred and c < n_true and dist_norm[r, c] <= 1.0:
            true_to_pred[c] = r
            matched_pred.add(r)
            matched_true.add(c)

    # build output rows
    truth_rows = []
    for ci, (_, t) in enumerate(truth_df.iterrows()):
        if ci in true_to_pred:
            pi = true_to_pred[ci]
            p = pred_df.iloc[pi]
            d_raw = np.sqrt((t["x_um"] - p["x_um"]) ** 2 +
                            (t["y_um"] - p["y_um"]) ** 2 +
                            (t["z_um"] - p["z_um"]) ** 2)
            truth_rows.append(dict(
                x_t=t["x_um"], y_t=t["y_um"], z_t=t["z_um"], d_t=t["d_um"],
                x_p=p["x_um"], y_p=p["y_um"], z_p=p["z_um"], d_p=p["d_um"],
                matched=True, dist_um=d_raw,
            ))
        else:
            truth_rows.append(dict(
                x_t=t["x_um"], y_t=t["y_um"], z_t=t["z_um"], d_t=t["d_um"],
                x_p=np.nan, y_p=np.nan, z_p=np.nan, d_p=np.nan,
                matched=False, dist_um=np.nan,
            ))

    # unmatched predictions = FP rows
    fp_rows = []
    for pi, (_, p) in enumerate(pred_df.iterrows()):
        if pi not in matched_pred:
            fp_rows.append(dict(
                x_t=np.nan, y_t=np.nan, z_t=np.nan, d_t=np.nan,
                x_p=p["x_um"], y_p=p["y_um"], z_p=p["z_um"], d_p=p["d_um"],
                matched=False, dist_um=np.nan,
            ))

    return truth_rows + fp_rows


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pairs_df: pd.DataFrame) -> dict:
    """
    Compute POD, FAR, F1, CSI and RMSE from a matched-pairs dataframe.

    Rows where both x_t and x_p are defined and matched=True → TP
    Rows where x_t is defined and matched=False → FN
    Rows where x_t is NaN and matched=False → FP
    """
    tp_mask = pairs_df["matched"] & pairs_df["x_t"].notna()
    fn_mask = ~pairs_df["matched"] & pairs_df["x_t"].notna()
    fp_mask = ~pairs_df["matched"] & pairs_df["x_t"].isna()

    TP = int(tp_mask.sum())
    FN = int(fn_mask.sum())
    FP = int(fp_mask.sum())

    denom_pod = TP + FN
    denom_far = TP + FP
    denom_f1  = 2 * TP + FP + FN
    denom_csi = TP + FP + FN

    pod = TP / denom_pod if denom_pod > 0 else np.nan
    far = FP / denom_far if denom_far > 0 else np.nan
    f1  = 2 * TP / denom_f1 if denom_f1 > 0 else np.nan
    csi = TP / denom_csi if denom_csi > 0 else np.nan

    tp_rows = pairs_df[tp_mask]
    def rmse(col_t, col_p):
        if len(tp_rows) == 0:
            return np.nan
        return float(np.sqrt(np.mean((tp_rows[col_t] - tp_rows[col_p]) ** 2)))

    return dict(
        TP=TP, FN=FN, FP=FP,
        POD=pod, FAR=far, F1=f1, CSI=csi,
        RMSE_x=rmse("x_t", "x_p"),
        RMSE_y=rmse("y_t", "y_p"),
        RMSE_z=rmse("z_t", "z_p"),
        RMSE_d=rmse("d_t", "d_p"),
    )


# ── ground truth loader ───────────────────────────────────────────────────────

def load_ground_truth(nc_path: str, h_idx: int, h_start: int = 0) -> pd.DataFrame:
    """
    Extract per-hologram ground truth from the NetCDF file.

    The NC uses 1-indexed hid; h_idx is 0-indexed (h_start offset applied).
    Returns DataFrame with columns: x_um, y_um, z_um, d_um.
    """
    ds = xr.open_dataset(nc_path)
    hid_target = h_idx + h_start + 1  # 1-indexed in file

    hid = ds["hid"].values
    mask = hid == hid_target

    # All coordinates are stored in microns in the NetCDF file
    x_um = ds["x"].values[mask].astype(float)
    y_um = ds["y"].values[mask].astype(float)
    z_um = ds["z"].values[mask].astype(float)
    d_um = ds["d"].values[mask].astype(float)

    ds.close()
    return pd.DataFrame({"x_um": x_um, "y_um": y_um, "z_um": z_um, "d_um": d_um})


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # load detections
    det = pd.read_csv(args.detections)
    print(f"Loaded {len(det)} detections from {args.detections}")

    # optional NMS clustering
    if args.cluster:
        n_raw = len(det)
        groups = []
        for h_idx, grp in det.groupby("h_idx"):
            groups.append(_nms_3d(grp, xy_radius=args.xy_radius, z_radius=args.z_radius))
        det = pd.concat(groups).reset_index(drop=True)
        print(f"NMS: {n_raw} raw → {len(det)} after clustering "
              f"(xy_r={args.xy_radius}μm, z_r={args.z_radius}μm)")

    # hologram range
    h_indices = sorted(det["h_idx"].unique())
    if args.max_holograms is not None:
        h_indices = h_indices[:args.max_holograms]

    all_pairs = []
    per_holo_metrics = []

    for h_idx in h_indices:
        pred_h = det[det["h_idx"] == h_idx].reset_index(drop=True)
        truth_h = load_ground_truth(args.nc_file, h_idx, h_start=args.h_start)

        pairs = match_hologram(pred_h, truth_h, xy_tol=args.xy_tol, z_tol=args.z_tol)
        pairs_df = pd.DataFrame(pairs)
        pairs_df.insert(0, "h_idx", h_idx)
        all_pairs.append(pairs_df)

        m = compute_metrics(pairs_df)
        m["h_idx"] = h_idx
        m["n_pred"] = len(pred_h)
        m["n_true"] = len(truth_h)
        per_holo_metrics.append(m)

        print(f"  h={h_idx:3d}: n_true={len(truth_h):4d}  n_pred={len(pred_h):5d}  "
              f"TP={m['TP']:4d}  FP={m['FP']:5d}  FN={m['FN']:4d}  "
              f"POD={m['POD']:.3f}  FAR={m['FAR']:.3f}  F1={m['F1']:.3f}  CSI={m['CSI']:.3f}  "
              f"RMSE_z={m['RMSE_z']:.1f}μm")

    # save matched pairs
    pairs_all = pd.concat(all_pairs).reset_index(drop=True)
    pairs_path = os.path.join(args.output_dir, "matched_pairs.csv")
    pairs_all.to_csv(pairs_path, index=False)
    print(f"\nSaved matched pairs → {pairs_path}")

    # per-hologram metrics
    metrics_df = pd.DataFrame(per_holo_metrics)
    metrics_path = os.path.join(args.output_dir, "metrics_per_hologram.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # overall aggregate
    overall = compute_metrics(pairs_all)
    summary_path = os.path.join(args.output_dir, "metrics_summary.csv")
    pd.DataFrame([overall]).to_csv(summary_path, index=False)

    print("\n=== OVERALL METRICS ===")
    print(f"  N holograms : {len(h_indices)}")
    print(f"  True particles: {metrics_df['n_true'].sum()}")
    print(f"  Detections    : {metrics_df['n_pred'].sum()}")
    print(f"  TP={overall['TP']}  FP={overall['FP']}  FN={overall['FN']}")
    print(f"  POD  = {overall['POD']:.4f}")
    print(f"  FAR  = {overall['FAR']:.4f}")
    print(f"  F1   = {overall['F1']:.4f}")
    print(f"  CSI  = {overall['CSI']:.4f}")
    print(f"  RMSE_x = {overall['RMSE_x']:.2f} μm")
    print(f"  RMSE_y = {overall['RMSE_y']:.2f} μm")
    print(f"  RMSE_z = {overall['RMSE_z']:.2f} μm")
    print(f"  RMSE_d = {overall['RMSE_d']:.2f} μm")
    print(f"\nSaved metrics → {metrics_path}, {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate multiplane hologram detections against ground truth"
    )
    parser.add_argument("-d", "--detections", required=True,
                        help="Path to detections.csv from inference_multiplane.py")
    parser.add_argument("-n", "--nc-file", required=True,
                        help="Path to ground-truth NetCDF file")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Directory to write output CSVs")
    parser.add_argument("--xy-tol",    type=float, default=50.0,
                        help="Matching tolerance in x,y (μm, default: 50)")
    parser.add_argument("--z-tol",     type=float, default=360.0,
                        help="Matching tolerance in z (μm, default: 360 = 2.5 bins)")
    parser.add_argument("--cluster",   action="store_true",
                        help="Apply 3D NMS before matching")
    parser.add_argument("--xy-radius", type=float, default=50.0,
                        help="NMS suppression radius in x,y (μm, default: 50)")
    parser.add_argument("--z-radius",  type=float, default=200.0,
                        help="NMS suppression radius in z (μm, default: 200)")
    parser.add_argument("--max-holograms", type=int, default=None,
                        help="Limit evaluation to first N holograms")
    parser.add_argument("--h-start",   type=int, default=0,
                        help="Hologram index offset (passed to ground-truth loader)")
    args = parser.parse_args()
    main(args)
