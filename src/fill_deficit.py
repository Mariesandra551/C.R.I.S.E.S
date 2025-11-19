import pandas as pd
from pathlib import Path



"""
fill_deficit.py

This script loads the merged dataset produced by `model.py` and
ensures that annual deficit values are properly aligned with monthly data.

Purpose:
• Deficit values are reported yearly, while bond & CDS values are monthly.
• For each year, the deficit value should fill all months in that year.
  → Forward-fill and backward-fill are applied within each country and year.

Process:
1. Load `merged_cleaned_dataset.csv`.
2. Extract the year from the `date` column.
3. Group by (country, year) and forward-fill / backward-fill deficit values.
4. Preserve original monthly bond and CDS data.
5. Save updated CSV (overwrites existing file).

This file ensures that deficit data is consistently usable for machine
learning and statistical models that require aligned time-series inputs.
"""

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "merged_cleaned_dataset.csv"
OUT_PATH  = BASE_DIR.parent / "data" / "merged_cleaned_dataset_filled.csv"

print("Reading:", DATA_PATH)

# Read
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# ---- FILL DEFICIT CHANGE WITHIN SAME YEAR AND COUNTRY ----
df["year"] = df["date"].dt.year  # temporary
df = df.sort_values(["country", "year", "date"])

df["deficit_change"] = (
    df.groupby(["country", "year"])["deficit_change"]
      .transform(lambda x: x.ffill().bfill())      # works with missing in middle
)

# Drop the temp year column
df = df.drop(columns=["year"])

# Final save
df.to_csv(OUT_PATH, index=False)

print("\nSUCCESS — Saved deficit-filled dataset to:")
print(OUT_PATH)
print(df.head())
