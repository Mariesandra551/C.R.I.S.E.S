import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

print("Loading model and preprocessing tools...")
DATA_DIR = "../data"
MODEL_PATH = os.path.join(DATA_DIR, "crisis_model.pkl")
IMPUTER_PATH = os.path.join(DATA_DIR, "imputer.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
DATASET_PATH = os.path.join(DATA_DIR, "merged_cleaned_dataset.csv")

# Load model and preprocessing tools
model = joblib.load(MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)
scaler = joblib.load(SCALER_PATH)
df = pd.read_csv(DATASET_PATH)

# Identify key columns
change_col = [c for c in df.columns if "change" in c.lower()][0]
country_col = next((c for c in df.columns if "country" in c.lower()), None)
year_col = next((c for c in df.columns if "year" in c.lower()), None)

# Drop unwanted columns
df = df.loc[:, ~df.columns.str.contains("Unnamed", case=False)]
X = df.select_dtypes(include=["float64", "int64"]).drop(columns=["crisis_label"], errors="ignore")

# Align with imputer
expected_cols = getattr(imputer, "feature_names_in_", X.columns)
X = X[[c for c in expected_cols if c in X.columns]]

# Impute + scale
X_imputed = pd.DataFrame(imputer.transform(X), columns=X.columns)
X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=X.columns)

# Predict probabilities
df["crisis_prob"] = model.predict_proba(X_scaled)[:, 1]

# Region mapping (expand as needed)
region_map = {
    "Greece": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Turkey": "Europe/Asia",
    "Egypt": "Africa",
    "India": "Asia",
    "Indonesia": "Asia",
    "South Africa": "Africa",
}

if country_col:
    df[country_col] = df[country_col].str.strip().str.title()
    df["region"] = df[country_col].map(region_map).fillna("Other")
else:
    df["region"] = "Unknown"

# Label creation
if country_col and year_col:
    df["label"] = df[country_col].astype(str) + " (" + df[year_col].astype(str) + ")"
elif country_col:
    df["label"] = df[country_col].astype(str)
else:
    df["label"] = df.index.astype(str)

# Sort logic (handles missing year)
if year_col and year_col in df.columns:
    sort_cols = ["crisis_prob", year_col]
    ascending = [False, False]
else:
    sort_cols = ["crisis_prob"]
    ascending = [False]

top10 = df.sort_values(by=sort_cols, ascending=ascending).head(10)

print("\nTop 10 Highest-Risk Observations:")
cols_to_show = [c for c in [country_col, year_col, change_col, "crisis_prob", "region"] if c]
print(top10[cols_to_show])

# Automatic region color palette
unique_regions = sorted(top10["region"].unique())
color_palette = plt.cm.tab10.colors  # 10 distinct colors
color_map = {region: color_palette[i % len(color_palette)] for i, region in enumerate(unique_regions)}
top10["color"] = top10["region"].map(color_map)

# Save CSV
CSV_OUT = os.path.join(DATA_DIR, "top10_crisis_probs.csv")
top10[cols_to_show].to_csv(CSV_OUT, index=False)

# Plot visualization
plt.figure(figsize=(10,6))
bars = plt.barh(top10["label"], top10["crisis_prob"], color=top10["color"])
plt.gca().invert_yaxis()

# Legend
handles = [plt.Rectangle((0,0),1,1, color=color_map[r], label=r) for r in unique_regions]
plt.legend(handles=handles, title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.title("Top 10 Predicted Crisis Probabilities by Region")
plt.xlabel("Predicted Crisis Probability")
plt.ylabel("Country (Year)" if year_col else "Country")
plt.tight_layout()

PNG_OUT = os.path.join(DATA_DIR, "top10_crisis_probs.png")
plt.savefig(PNG_OUT, dpi=150, bbox_inches="tight")

print(f"\nSaved visualization to {PNG_OUT}")
print(f"Saved labeled data to {CSV_OUT}")
