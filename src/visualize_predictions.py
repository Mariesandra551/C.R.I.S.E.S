import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------
# visualize_predictions.py
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "gmm_predictions.csv"

print(f"Loading predictions from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded predictions: {df.shape}")

# ---------------------------------------------------------
# 1. GREECE YEARLY TREND (Improved readability)
# ---------------------------------------------------------
if "country" in df.columns and "date" in df.columns:
    greek = df[df["country"] == "Greece"].copy()
    greek["year"] = pd.to_datetime(greek["date"]).dt.year
    yearly = greek.groupby("year")["crisis_prob"].mean().reset_index()

    plt.figure(figsize=(12, 5))
    plt.plot(yearly["year"], yearly["crisis_prob"], marker="o")
    plt.title("Greece Crisis Probability (Yearly Average)")
    plt.xlabel("Year")
    plt.ylabel("Crisis Probability")
    plt.grid(True)
    plt.xticks(yearly["year"], rotation=45)

    output_greece = BASE_DIR.parent / "data" / "greece_gmm_trend_yearly.png"
    plt.savefig(output_greece, dpi=300, bbox_inches="tight")
    print(f"✔ Saved Yearly Greece Trend → {output_greece}")

# ---------------------------------------------------------
# 2. CONTAGION MATRIX (Country Correlation)
# ---------------------------------------------------------
print("Computing contagion matrix...")
df["year"] = pd.to_datetime(df["date"]).dt.year
country_year = df.groupby(["country", "year"])["crisis_prob"].mean().unstack()

corr_matrix = country_year.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="magma", annot=False)
plt.title("Contagion Risk Between Countries (Correlation of Crisis Probability)")
plt.tight_layout()

output_contagion = BASE_DIR.parent / "data" / "contagion_matrix.png"
plt.savefig(output_contagion, dpi=300)
print(f"✔ Saved Contagion Matrix → {output_contagion}")

# ---------------------------------------------------------
# 3. REGIME CLUSTERS VISUALIZATION
# ---------------------------------------------------------
if "regime" in df.columns:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="bond_dev",
        y="cds_dev",
        hue="regime",
        palette="viridis",
        alpha=0.6,
        legend="brief"
    )
    plt.title("GMM Regimes: Economic States Detected")
    plt.xlabel("Bond Shock (Dev)")
    plt.ylabel("CDS Shock (Dev)")
    plt.tight_layout()

    output_regimes = BASE_DIR.parent / "data" / "gmm_regimes.png"
    plt.savefig(output_regimes, dpi=300)
    print(f"✔ Saved Regime Plot → {output_regimes}")
else:
    print("⚠ 'regime' column not found → Cannot plot regime clusters")

print("\nAll visualizations done.")

# ---------------------------------------------------------
# 4. ECONOMIC VALIDATION AGAINST REAL EVENTS
# ---------------------------------------------------------
print("\n=== MODEL VALIDATION AGAINST REAL EVENTS ===")
validation_events = [
    ("Greece", 2009), ("Greece", 2010), ("Greece", 2011), ("Greece", 2012),
    ("USA", 2008), ("USA", 2020),
    ("Germany", 2008), ("Germany", 2011),
    ("Italy", 2011), ("Italy", 2012),
    ("Spain", 2010), ("Spain", 2011), ("Spain", 2012),
    ("UK", 2008), ("UK", 2020),
    ("France", 2008), ("France", 2011),
]

for country, year in validation_events:
    prob = df[(df["country"] == country) & (df["year"] == year)]["crisis_prob"].mean()
    print(f" {country} {year}:  Crisis prob = {0 if pd.isna(prob) else round(prob, 3)}")

# ---------------------------------------------------------
# 5. GMM CLUSTER CENTROIDS (INTERPRETABILITY)
# ---------------------------------------------------------
print("\n=== GMM CLUSTER CENTROIDS ===")

gmm = joblib.load(BASE_DIR.parent / "data" / "gmm_model.pkl")
scaler = joblib.load(BASE_DIR.parent / "data" / "scaler.pkl")

features = ["bond_dev", "cds_dev", "deficit_dev"]
print("Columns:", features)

centroids_scaled = gmm.means_
centroids = scaler.inverse_transform(centroids_scaled)

for i, center in enumerate(centroids):
    center_dict = dict(zip(features, center))
    print(f"\nRegime {i}:")
    print(f"  {center_dict}")

print("\n=== REGIME INTERPRETATION ===")

def interpret_regime(center):
    shocks = np.array(list(center.values()))
    if shocks.max() < 0.5:
        return "NORMAL ECONOMIC STATE"
    elif shocks.max() < 1.5:
        return "STRESS REGIME"
    else:
        return "CRISIS REGIME"

for i, center in enumerate(centroids):
    print(f"Regime {i}: {interpret_regime(dict(zip(features, center)))}")

# ---------------------------------------------------------
# 6. REGIME TIMELINE ACROSS COUNTRIES (NEW FIX)
# ---------------------------------------------------------
print("\nGenerating regime timeline across countries...")

df["date"] = pd.to_datetime(df["date"])  # FIX: ensure datetime type

plt.figure(figsize=(12, 6))
for country in df["country"].unique():
    subset = df[df["country"] == country].sort_values("date")
    plt.plot(subset["date"], subset["regime"], label=country, alpha=0.4)

plt.title("Regime Timeline Across Countries (Economic Cycle Evolution)")
plt.xlabel("Date")
plt.ylabel("Regime Index")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

output_timeline = BASE_DIR.parent / "data" / "regime_timeline.png"
plt.savefig(output_timeline, dpi=300)
plt.close()

print(f"✔ Saved Regime Timeline → {output_timeline}")

# -----------------------------------------------
# 7. FEATURE IMPORTANCE (LOAD FROM SAVED CSV)
# -----------------------------------------------
feature_path = BASE_DIR.parent / "data" / "feature_importance.csv"

if feature_path.exists():
    importance_df = pd.read_csv(feature_path)
    print("\n=== FEATURE IMPORTANCE (FROM MODEL) ===")
    print(importance_df)
else:
    print("\n⚠ No saved feature Importance found. Run train_model.py first.")
