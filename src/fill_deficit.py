import pandas as pd
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "merged_cleaned_dataset.csv"
OUT_PATH  = BASE_DIR.parent / "data" / "merged_cleaned_dataset_filled.csv"



"""
This script processes the merged economic dataset and ensures that the
deficit variable is properly aligned with monthly financial data. It converts
incomplete fiscal data into **smooth, gap-free monthly signals**, enabling
consistent feature engineering and model training.

Workflow:
1. Read `merged_cleaned_dataset.csv` from `/data`.
2. Convert the date column into proper datetime format.
3. For each country:
       • Interpolate missing values (linear method) to preserve
         monthly economic continuity.
       • Fill remaining gaps using country-level median values to
         avoid bias or extreme jumps.
4. Remove rows where ALL three indicators are missing.
5. Sort data by country and date (descending) to prepare rolling windows.
6. Save final output to `merged_cleaned_dataset_filled.csv`.

This file finalizes the **clean, monthly-aligned input dataset** used for:
- economic shock creation,
- rolling-window feature engineering,
- GMM regime detection,
- and crisis probability estimation.
"""

print("Reading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Ensure datetime format
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Target columns
cols = ["bond_yield_change", "cds_change", "deficit_change"]

# 1. Interpolate — keeps monthly economic smoothness
df[cols] = (
    df.groupby("country")[cols]
      .transform(lambda group: group.interpolate(method="linear", limit_direction="both"))
)

# 2. Median fill — avoids overfitting / wild values
df[cols] = (
    df.groupby("country")[cols]
      .transform(lambda x: x.fillna(x.median()))
)

# 3. Drop fully empty rows
df = df.dropna(subset=cols, how="all").reset_index(drop=True)

# 4. Sort for rolling windows later
df = df.sort_values(by=["country", "date"], ascending=[True, False]).reset_index(drop=True)

# 5. Save result
df.to_csv(OUT_PATH, index=False)
print(f"✔ Saved cleaned dataset to: {OUT_PATH}")
print("Final shape:", df.shape)

# OPTIONAL: diagnostic print for report
print("\nRemaining NaNs per column:")
print(df[cols].isna().sum())
