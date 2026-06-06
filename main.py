"""
NIFTY IV Surface Reconstruction — PURE STRIKE-LINEAR ENGINE
===========================================================
Author: Nishant Sou

Quantitative Strategy:
1. 100% weight on Cross-Sectional Linear Interpolation (Strike space).
2. Linear extrapolation for deep OTM/ITM wings to capture volatility smirks.
3. Zero temporal blending (time-axis is only used if a cross-section is entirely empty).
   This ensures the model reacts instantly to underlying spot movements without lag.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import warnings

# Suppress mathematically expected warnings from scipy interpolators
warnings.filterwarnings('ignore')

# ── 1. CONFIGURATION & DATA LOAD ─────────────────────────────────────────────
DATASET_PATH    = "dataset.csv" 
FILLED_PATH     = "filled_dataset.csv"
SUBMISSION_PATH = "submission.csv"
SEPARATOR       = "||"

print("Loading data...")
df = pd.read_csv(DATASET_PATH)
df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df = df.sort_values("datetime").reset_index(drop=True)

# Parse Call (CE) and Put (PE) strikes automatically
ce_cols = sorted([c for c in df.columns if c.endswith("CE")], key=lambda x: int(x[12:17]))
pe_cols = sorted([c for c in df.columns if c.endswith("PE")], key=lambda x: int(x[12:17]))
iv_cols = ce_cols + pe_cols

ce_strikes = np.array([int(c[12:17]) for c in ce_cols])
pe_strikes = np.array([int(c[12:17]) for c in pe_cols])

# ── 2. EMERGENCY TIME-AXIS FALLBACK ──────────────────────────────────────────
# We only use this if an entire row is completely empty, ensuring no null outputs.
print("Computing emergency time-axis fallback...")
df_time_filled = df[iv_cols].interpolate(method="linear", limit_direction="both")

# ── 3. PURE STRIKE-AXIS INTERPOLATION ────────────────────────────────────────
print("Computing pure strike-axis interpolation (100% weight)...")
df_filled = df.copy()

for col_group, strikes in [(ce_cols, ce_strikes), (pe_cols, pe_strikes)]:
    for idx, row in df.iterrows():
        obs_mask = ~row[col_group].isna().values.astype(bool)
        
        # If less than 2 points exist, a line cannot be drawn. Route to emergency fallback.
        if obs_mask.sum() < 2:
            for j, col in enumerate(col_group):
                if pd.isna(row[col]):
                    df_filled.at[idx, col] = max(df_time_filled.at[idx, col], 0.005)
            continue

        obs_k  = strikes[obs_mask]
        obs_iv = row[col_group].values.astype(float)[obs_mask]

        # Sort strikes (required by scipy interpolator for stability)
        sort_idx = np.argsort(obs_k)
        obs_k, obs_iv = obs_k[sort_idx], obs_iv[sort_idx]

        # Build the pure linear interpolator with linear extrapolation for the wings
        f_linear = interp1d(
            obs_k, obs_iv, 
            kind="linear", 
            fill_value="extrapolate", 
            assume_sorted=True
        )

        for j, col in enumerate(col_group):
            if pd.isna(row[col]):
                target_k = strikes[j]
                
                # Predict purely based on cross-sectional strike geometry
                strike_pred = float(f_linear(target_k))
                
                # Hard floor to prevent linear extrapolation from predicting negative IV (Arbitrage check)
                df_filled.at[idx, col] = max(strike_pred, 0.005)

# ── 4. EXPORT SUBMISSION ─────────────────────────────────────────────────────
print("Generating complete surface and formatting submission...")
df_filled['datetime'] = df_filled['datetime'].dt.strftime('%d-%m-%Y %H:%M')
df_filled.to_csv(FILLED_PATH, index=False)

original = pd.read_csv(DATASET_PATH)
filled   = pd.read_csv(FILLED_PATH)
feature_cols = [c for c in original.columns if c != "datetime"]

rows = []
for col in feature_cols:
    was_missing = original[col].isna()
    for i in original.index[was_missing]:
        dt  = original.loc[i, "datetime"]
        uid = f"{dt}{SEPARATOR}{col}"
        val = filled.loc[i, col]
        rows.append({"id": uid, "value": val})

solution = pd.DataFrame(rows, columns=["id", "value"]).sort_values("id").reset_index(drop=True)
solution.to_csv(SUBMISSION_PATH, index=False)

print(f" Submission file generated successfully → {SUBMISSION_PATH} ({len(solution)} rows)")
//end of main.py