"""
validacao_hardware.py

Validates the firmware's acceleration-magnitude threshold logic (~6 m/s^2,
per MASELLO et al., 2025; BRUHWILER et al., 2022) against two combined
public driving-behavior datasets, replacing the now-unavailable UAH-DriveSet.

Sources (see Seção 3.3 / Part B.5 of the technical spec):
  - Yuksel, A. S. (2021). Driving Behavior Dataset. Mendeley Data, V3.
    DOI: 10.17632/jj3tw8kj6h.3
  - Ferreira Jr., J. et al. (2017). Driver behavior profiling: An
    investigation with different smartphone sensors and machine learning.
    PLoS ONE, 12(4), e0174959.

This script is a DISCRIMINATION validation, not a continuous-stream
DETECTION validation: the input is already pre-segmented into labeled
event windows (statistical features per window), not a raw continuous
signal. See docstring notes inline for why this reframing is a closer
match to what the ESP32 firmware itself produces (summarized events,
never a raw stream).
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/combined_normalized_driver_conduct.csv"
OUT_PATH = "data/processed/driver_conduct_harmonized.csv"
METRICS_PATH = "data/processed/driver_conduct_metrics.csv"

G_TO_MPS2 = 9.80665
HARSH_THRESHOLD_MPS2 = 6.0  # literature threshold (Masello et al., 2025; Bruhwiler et al., 2022)

# ---------------------------------------------------------------------------
# Step 1 — Load
# ---------------------------------------------------------------------------
df = pd.read_csv(RAW_PATH)

RAW_ACC_COLS = [c for c in df.columns
                if c.startswith("Acc") and not c.endswith("_z")]

# ---------------------------------------------------------------------------
# Step 2 — Empirically verify the physical unit scale of each source
# (do NOT trust either source's documentation at face value — see Part B.5).
# Proxy for a near-steady/idle-like window: lowest decile of resultant
# acceleration variance magnitude within each source.
# ---------------------------------------------------------------------------
acc_var_cols = ["AccVarX", "AccVarY", "AccVarZ"]
df["_AccVarMag"] = np.sqrt((df[acc_var_cols] ** 2).sum(axis=1))
df["_AccMagMean_raw"] = np.sqrt(
    df["AccMeanX"] ** 2 + df["AccMeanY"] ** 2 + df["AccMeanZ"] ** 2
)

print("=== Step 2: empirical unit check per source ===")
unit_scale = {}
for src, g in df.groupby("SourceDataset"):
    low_var_thresh = g["_AccVarMag"].quantile(0.10)
    steady = g[g["_AccVarMag"] <= low_var_thresh]
    steady_mag = steady["_AccMagMean_raw"].mean()
    # gravity is ~1 in g-scale, ~9.8 in m/s^2-scale — pick whichever is closer
    scale = "g" if abs(steady_mag - 1.0) < abs(steady_mag - G_TO_MPS2) else "m/s^2"
    unit_scale[src] = scale
    print(f"{src}: steady-window |Acc| = {steady_mag:.3f}  -> inferred unit = {scale}")

# ---------------------------------------------------------------------------
# Step 3 — Harmonize units: convert every g-scale source's raw Acc columns to m/s^2
# ---------------------------------------------------------------------------
for src, scale in unit_scale.items():
    if scale == "g":
        mask = df["SourceDataset"] == src
        df.loc[mask, RAW_ACC_COLS] = df.loc[mask, RAW_ACC_COLS] * G_TO_MPS2

print("\n=== Step 3: post-harmonization sanity check (should all read ~9.8) ===")
df["_AccMagMean_mps2"] = np.sqrt(
    df["AccMeanX"] ** 2 + df["AccMeanY"] ** 2 + df["AccMeanZ"] ** 2
)
print(df.groupby("SourceDataset")["_AccMagMean_mps2"].mean())

# ---------------------------------------------------------------------------
# Step 4 — Peak *dynamic* acceleration proxy (gravity/baseline-subtracted)
#
# The firmware never thresholds raw magnitude — Seção 2.8 subtracts a
# calibrated gravity/baseline component first, then thresholds the
# resulting *effective* acceleration. We don't have raw per-sample
# vectors here (only per-window statistics), so the closest available
# proxy is the peak deviation of Max/Min from the window Mean on each
# axis (the window Mean approximates the roughly-constant gravity +
# steady-cruise baseline the firmware calibrates away), combined into a
# resultant magnitude across the three axes.
# ---------------------------------------------------------------------------
for axis in ["X", "Y", "Z"]:
    up = (df[f"AccMax{axis}"] - df[f"AccMean{axis}"]).abs()
    down = (df[f"AccMean{axis}"] - df[f"AccMin{axis}"]).abs()
    df[f"_PeakDev{axis}"] = np.maximum(up, down)

df["PeakDynamicAccel_mps2"] = np.sqrt(
    df["_PeakDevX"] ** 2 + df["_PeakDevY"] ** 2 + df["_PeakDevZ"] ** 2
)

# ---------------------------------------------------------------------------
# Step 5 — Harmonize label taxonomy across the two sources into shared
# categories used by the firmware's own event classes.
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "Sudden Acceleration": ("ACCELERATION", "harsh"),
    "Aggressive acceleration": ("ACCELERATION", "harsh"),
    "Sudden Break": ("BRAKING", "harsh"),
    "Aggressive braking": ("BRAKING", "harsh"),
    "Sudden Left Turn": ("TURN", "harsh"),
    "Sudden Right Turn": ("TURN", "harsh"),
    "Aggressive left turn": ("TURN", "harsh"),
    "Aggressive right turn": ("TURN", "harsh"),
    "Aggressive left lane change": ("LANE_CHANGE", "excluded"),
    "Aggressive right lane change": ("LANE_CHANGE", "excluded"),
    "Non-aggressive event": ("NON_AGGRESSIVE", "baseline"),
}
df["EventCategory"] = df["EventLabel"].map(lambda x: LABEL_MAP[x][0])
df["ValidationRole"] = df["EventLabel"].map(lambda x: LABEL_MAP[x][1])

print("\n=== Step 5: harmonized category counts (by validation role) ===")
print(df.groupby(["ValidationRole", "EventCategory"]).size())

# ---------------------------------------------------------------------------
# Step 6 — Apply the ~6 m/s^2 threshold and score discrimination.
# Ground truth: 'harsh' maneuvers (accel/brake/turn) = positive;
# 'Non-aggressive event' = negative. Lane-change is excluded — a single
# accelerometer+GPS node without steering data is not architecturally
# capable of detecting a lane change, so it is out of scope by design,
# not a silently dropped row.
# ---------------------------------------------------------------------------
evalset = df[df["ValidationRole"] != "excluded"].copy()
evalset["y_true"] = (evalset["ValidationRole"] == "harsh").astype(int)
evalset["y_pred"] = (evalset["PeakDynamicAccel_mps2"] > HARSH_THRESHOLD_MPS2).astype(int)

def confusion(sub):
    tp = ((sub.y_true == 1) & (sub.y_pred == 1)).sum()
    fn = ((sub.y_true == 1) & (sub.y_pred == 0)).sum()
    tn = ((sub.y_true == 0) & (sub.y_pred == 0)).sum()
    fp = ((sub.y_true == 0) & (sub.y_pred == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return dict(n=len(sub), tp=tp, fn=fn, tn=tn, fp=fp, precision=precision, recall=recall, f1=f1)

print(f"\n=== Step 6: threshold = {HARSH_THRESHOLD_MPS2} m/s^2 on PeakDynamicAccel ===")
print("\n-- Pooled (both sources) --")
print(confusion(evalset))

# ---------------------------------------------------------------------------
# Step 6b — Collect the same metrics into a flat table so the dashboard can
# load them directly instead of recomputing (per-source AND per-category,
# never pooled into one figure — see Part B.5, sources are ~95/5 imbalanced).
# ---------------------------------------------------------------------------
metrics_rows = [dict(recorte="pooled", categoria="todas", **confusion(evalset))]

print("\n-- Per source (imbalance: report separately, do not pool blindly) --")
for src, sub in evalset.groupby("SourceDataset"):
    print(src)
    print(" ", confusion(sub))
    metrics_rows.append(dict(recorte=src, categoria="todas", **confusion(sub)))

print("\n-- Per harmonized category (recall within 'harsh' classes only) --")
for cat, sub in evalset[evalset.ValidationRole == "harsh"].groupby("EventCategory"):
    recall = (sub.y_pred == 1).mean()
    print(f"  {cat}: n={len(sub)}  recall={recall:.3f}")
    metrics_rows.append(dict(recorte="pooled", categoria=cat, n=len(sub), recall=recall))
    for src, sub_src in sub.groupby("SourceDataset"):
        recall_src = (sub_src.y_pred == 1).mean()
        metrics_rows.append(dict(recorte=src, categoria=cat, n=len(sub_src), recall=recall_src))

pd.DataFrame(metrics_rows).to_csv(METRICS_PATH, index=False)
print(f"\nMetrics table written to {METRICS_PATH}.")

# ---------------------------------------------------------------------------
# Step 7 — Persist the harmonized dataset for reuse (e.g., by the dashboard).
# Keeps both the pre-harmonization (_AccMagMean_raw) and post-harmonization
# magnitude so a consumer can render a before/after unit-harmonization audit
# view without re-deriving the g-vs-m/s^2 detection logic itself.
# ---------------------------------------------------------------------------
keep_cols = (
    ["SourceDataset", "GroupID", "EventLabel", "EventCategory", "ValidationRole",
     "WindowIndex"]
    + RAW_ACC_COLS
    + ["PeakDynamicAccel_mps2", "_AccMagMean_raw", "_AccMagMean_mps2"]
)
out = df[keep_cols].rename(columns={
    "_AccMagMean_raw": "AccMagMean_raw",
    "_AccMagMean_mps2": "AccMagMean_mps2",
})
out["HarshPredicted_6mps2"] = (out["PeakDynamicAccel_mps2"] > HARSH_THRESHOLD_MPS2).astype(int)
out.to_csv(OUT_PATH, index=False)
print(f"\nHarmonized dataset written to {OUT_PATH} ({len(out)} rows, {len(out.columns)} columns).")
