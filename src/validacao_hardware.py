"""
validacao_hardware.py

Validates the firmware's acceleration-magnitude threshold logic (~6 m/s^2,
per MASELLO et al., 2025; BRUHWILER et al., 2022) against a public
driving-behavior dataset, replacing the now-unavailable UAH-DriveSet.

Source (see Seção 3.3 / Part B.5 of the technical spec):
  - Ferreira Jr., J.; Carvalho, E.; Ferreira, B. V.; Souza, C. de;
    Suhara, Y.; Pentland, A.; Pessin, G. (2017). Driver behavior
    profiling: An investigation with different smartphone sensors and
    machine learning. PLoS ONE, 12(4), e0174959.
    Dataset: github.com/jair-jr/driverBehaviorDataset

A single-source dataset was chosen over combining it with a second
Kaggle/Mendeley source (Yuksel, 2021) after an initial two-source run
surfaced two problems specific to that combination: (1) the two sources
turned out to be in different physical units (g vs m/s^2), requiring a
harmonization step verified only empirically, since neither source's own
documentation could be trusted at face value; and (2) the Yuksel source's
"Sudden X" labels appear to be assigned per recording session rather than
per window (near-flat peak-deviation statistics across ~250 consecutive
windows under one label), which is a weaker ground truth than Ferreira
Jr.'s labels — timestamped by researchers watching a reference video of
the driver deliberately performing each maneuver. Ferreira Jr. alone also
has the one property the validation actually needs most: a genuine
"Non-aggressive event" baseline class, enabling real positive/negative
discrimination rather than only within-"harsh" category comparison.

Trade-off accepted explicitly: n=55 windows total (as few as 3 per
class) is a small-sample feasibility check, not a statistically powered
validation — report it that way, in the same spirit as the bench
tests/road tests in the author's own vehicle (Seção 3.3).

Note: this source has NO GPS or speed data (confirmed against the
dataset repository and the source paper) — only accelerometer, linear
acceleration, gyroscope, and magnetometer from the recording smartphone.
The vehicle's speedometer appears only in the reference video used for
manual labeling, never as a machine-readable field. Speed-linked
validation (e.g., the "excesso de velocidade" criterion) remains out of
scope regardless of source choice.
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/combined_normalized_driver_conduct.csv"
OUT_PATH = "data/processed/driver_conduct_harmonized.csv"
METRICS_PATH = "data/processed/driver_conduct_metrics.csv"
CONFOUND_PATH = "data/processed/driver_conduct_confound.csv"
SOURCE_NAME = "jair_jr_driverBehaviorDataset_2016"

HARSH_THRESHOLD_MPS2 = 6.0  # literature threshold (Masello et al., 2025; Bruhwiler et al., 2022)

# ---------------------------------------------------------------------------
# Step 1 — Load and restrict to the single chosen source.
# ---------------------------------------------------------------------------
df_all = pd.read_csv(RAW_PATH)
df = df_all[df_all["SourceDataset"] == SOURCE_NAME].copy()
print(f"=== Step 1: {len(df)} windows loaded from {SOURCE_NAME} "
      f"(out of {len(df_all)} in the combined raw file) ===")

RAW_ACC_COLS = [c for c in df.columns
                if c.startswith("Acc") and not c.endswith("_z")]

# ---------------------------------------------------------------------------
# Step 2 — Empirical unit check (still worth doing even for one source —
# never assume documented units are correct without checking).
# ---------------------------------------------------------------------------
acc_var_cols = ["AccVarX", "AccVarY", "AccVarZ"]
df["_AccVarMag"] = np.sqrt((df[acc_var_cols] ** 2).sum(axis=1))
df["_AccMagMean_raw"] = np.sqrt(
    df["AccMeanX"] ** 2 + df["AccMeanY"] ** 2 + df["AccMeanZ"] ** 2
)
low_var_thresh = df["_AccVarMag"].quantile(0.10)
steady_mag = df.loc[df["_AccVarMag"] <= low_var_thresh, "_AccMagMean_raw"].mean()
print(f"\n=== Step 2: empirical unit check ===")
print(f"Steady-window |Acc| = {steady_mag:.3f} "
      f"(~9.8 confirms m/s^2 as documented — no conversion needed)")

# ---------------------------------------------------------------------------
# Step 3 — Peak *dynamic* acceleration proxy (gravity/baseline-subtracted).
# The firmware subtracts a calibrated gravity/baseline component before
# thresholding (Seção 2.8). We only have per-window statistics, not raw
# per-sample vectors, so the closest available proxy is the peak deviation
# of Max/Min from the window Mean on each axis (Mean approximates the
# roughly-constant gravity + steady-driving baseline), combined into a
# resultant magnitude across axes.
# ---------------------------------------------------------------------------
for axis in ["X", "Y", "Z"]:
    up = (df[f"AccMax{axis}"] - df[f"AccMean{axis}"]).abs()
    down = (df[f"AccMean{axis}"] - df[f"AccMin{axis}"]).abs()
    df[f"_PeakDev{axis}"] = np.maximum(up, down)

df["PeakDynamicAccel_mps2"] = np.sqrt(
    df["_PeakDevX"] ** 2 + df["_PeakDevY"] ** 2 + df["_PeakDevZ"] ** 2
)

# ---------------------------------------------------------------------------
# Step 4 — Harmonize labels into the firmware's own event categories.
# This source's taxonomy is already internally consistent (all seven
# labels come from one study/protocol), so this is a straight mapping,
# not a cross-source reconciliation.
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "Aggressive acceleration": ("ACCELERATION", "harsh"),
    "Aggressive braking": ("BRAKING", "harsh"),
    "Aggressive left turn": ("TURN", "harsh"),
    "Aggressive right turn": ("TURN", "harsh"),
    "Aggressive left lane change": ("LANE_CHANGE", "excluded"),
    "Aggressive right lane change": ("LANE_CHANGE", "excluded"),
    "Non-aggressive event": ("NON_AGGRESSIVE", "baseline"),
}
df["EventCategory"] = df["EventLabel"].map(lambda x: LABEL_MAP[x][0])
df["ValidationRole"] = df["EventLabel"].map(lambda x: LABEL_MAP[x][1])

print("\n=== Step 4: category counts (by validation role) ===")
print(df.groupby(["ValidationRole", "EventCategory"]).size())
print("\nLane-change is excluded from discrimination scoring below — a single "
      "accelerometer+GPS node without steering data cannot detect a lane "
      "change; this is an architectural scope limit, not a dropped row.")

# ---------------------------------------------------------------------------
# Step 5 — Apply the ~6 m/s^2 threshold and score discrimination.
# Ground truth: 'harsh' maneuvers (accel/brake/turn) = positive;
# 'Non-aggressive event' = negative.
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

print(f"\n=== Step 5: threshold = {HARSH_THRESHOLD_MPS2} m/s^2 on PeakDynamicAccel ===")
print("n =", len(evalset), "— small-sample feasibility check, report accordingly")
print(confusion(evalset))

metrics_rows = [dict(categoria="todas", **confusion(evalset))]

print("\n-- Per harmonized category (recall within 'harsh' classes; n as low as 6, "
      "read as directional, not statistically powered) --")
for cat, sub in evalset[evalset.ValidationRole == "harsh"].groupby("EventCategory"):
    recall = (sub.y_pred == 1).mean()
    print(f"  {cat}: n={len(sub)}  recall={recall:.3f}")
    metrics_rows.append(dict(categoria=cat, n=len(sub), recall=recall))

pd.DataFrame(metrics_rows).to_csv(METRICS_PATH, index=False)
print(f"\nMetrics table written to {METRICS_PATH}.")

# ---------------------------------------------------------------------------
# Step 5b — Driver/session (GroupID) confound: record, per harmonized
# category, which recording session(s) it comes from. Persisted (not just
# printed) so the dashboard can show the confound with real counts instead
# of a static caption — see Part B.5: "not every GroupID session contains
# every event category ... do not present per-driver conclusions".
# ---------------------------------------------------------------------------
confound = df.groupby(["EventCategory", "GroupID"]).size().reset_index(name="n")
confound["n_sessions_com_categoria"] = confound.groupby("EventCategory")["GroupID"].transform("nunique")
confound.to_csv(CONFOUND_PATH, index=False)
print(f"\n-- Per driver/session (GroupID) — descriptive only, not enough n for per-group metrics --")
print(df.groupby("GroupID")["EventCategory"].value_counts())
print(f"Confound table written to {CONFOUND_PATH}.")

# ---------------------------------------------------------------------------
# Step 6 — Persist the harmonized single-source dataset.
# ---------------------------------------------------------------------------
keep_cols = (
    ["SourceDataset", "GroupID", "EventLabel", "EventCategory", "ValidationRole",
     "WindowIndex"]
    + RAW_ACC_COLS
    + ["PeakDynamicAccel_mps2", "_AccMagMean_raw"]
)
out = df[keep_cols].rename(columns={"_AccMagMean_raw": "AccMagMean_mps2"})
out["HarshPredicted_6mps2"] = (out["PeakDynamicAccel_mps2"] > HARSH_THRESHOLD_MPS2).astype(int)
out.to_csv(OUT_PATH, index=False)
print(f"\nHarmonized dataset written to {OUT_PATH} ({len(out)} rows, {len(out.columns)} columns).")
