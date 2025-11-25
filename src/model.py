import pandas as pd
from pathlib import Path



# ------------------------------
# 1. Paths
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "Final Greek Crisis Data.xlsx"
OUT_PATH = BASE_DIR.parent / "data" / "merged_cleaned_dataset.csv"



"""
This script reads multiple sheets from the Excel file
`Final Greek Crisis Data.xlsx`, each sheet representing one country.
It restructures the data into a machine-learning–ready format and merges all
countries into a single CSV output (`merged_cleaned_dataset.csv`).

Workflow:
1. Read all sheets from the Excel file.
2. Standardize column names and datetime formats.
3. Reshape each sheet into tidy form:
       | date | bond_yield_change | cds_change | deficit_change | country |
4. Merge all sheets using outer join to keep all available data.
5. Remove rows where ALL three values are missing.
6. Sort results by country (A–Z) and date in descending order.
7. Save final output to `data/merged_cleaned_dataset.csv`.

This file forms the core dataset used for time-series analysis,
feature engineering, and machine learning model development for
crisis prediction.
"""

print("Reading:", DATA_PATH)

# ------------------------------
# 2. Function to process ONE sheet
# ------------------------------
def process_sheet(country_name, df):
    df.columns = df.columns.str.strip().str.lower()

    expected = ["date1", "bond_yield_change", "date2", "cds_change", "date3", "deficit_change"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        print(f"⚠ Missing columns in {country_name}: {missing}")
        return None

    # Convert date columns
    for col in ["date1", "date2", "date3"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%d")

    # Create tidy format
    df_bond = df[["date1", "bond_yield_change"]].rename(columns={"date1": "date"})
    df_cds = df[["date2", "cds_change"]].rename(columns={"date2": "date"})
    df_def = df[["date3", "deficit_change"]].rename(columns={"date3": "date"})

    merged = df_bond.merge(df_cds, on="date", how="outer")
    merged = merged.merge(df_def, on="date", how="outer")
    merged = merged.sort_values("date", ascending=False)
    merged["country"] = country_name

    return merged

# ------------------------------
# 3. Process all sheets
# ------------------------------
excel_data = pd.read_excel(DATA_PATH, sheet_name=None)
dfs = []

for name, sheet in excel_data.items():
    print(f"\nProcessing sheet: {name}")
    cleaned = process_sheet(name, sheet)
    if cleaned is not None:
        dfs.append(cleaned)

# ------------------------------
# 4. Final dataset + filter + sort
# ------------------------------
if dfs:
    df_final = pd.concat(dfs, ignore_index=True)

    # Keep rows with at least ONE valid value
    df_final = df_final.dropna(subset=["bond_yield_change", "cds_change", "deficit_change"], how="all")

    # Correct sorting by country + descending date
    df_final = df_final.sort_values(["country", "date"], ascending=[True, False])

    df_final.to_csv(OUT_PATH, index=False)
    print("\nSUCCESS — Saved to:", OUT_PATH)
    print("Final shape:", df_final.shape)
    print(df_final.head())
else:
    print("\n⚠ No valid data found in any sheet.")
