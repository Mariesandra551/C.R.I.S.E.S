import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns



"""
Generates visual outputs and economic diagnostics for the early-warning crisis model.
This script validates model behavior, checks contagion risks, and produces clean visuals
for dashboards and academic reports.

Outputs generated in `/data`:
    • greece_gmm_trend_yearly.png
    • contagion_matrix.png
    • gmm_regimes.png
    • regime_timeline.png
    • feature_importance.csv (loaded if available)
"""

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "gmm_predictions.csv"
print(f"Loading predictions from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded predictions: {df.shape}")

# ------------------------------
# 1. Greece yearly crisis trend
# ------------------------------
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
    print(f"✔ Saved: {output_greece}")

# ------------------------------
# 2. Contagion Matrix (correlation)
# ------------------------------
print("Computing contagion matrix...")
df["year"] = pd.to_datetime(df["date"]).dt.year
country_year = df.groupby(["country", "year"])["crisis_prob"].mean().unstack()
corr_matrix = country_year.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="magma", annot=False)
plt.title("Contagion Risk Between Countries")
plt.tight_layout()
output_contagion = BASE_DIR.parent / "data" / "contagion_matrix.png"
plt.savefig(output_contagion, dpi=300)
print(f"✔ Saved: {output_contagion}")

# ------------------------------
# 3. Regime clusters
# ------------------------------
if "regime" in df.columns:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x="bond_dev", y="cds_dev", hue="regime",
                    palette="viridis", alpha=0.6)
    plt.title("GMM Regimes – Economic States Detected")
    plt.xlabel("Bond Shock (Dev)")
    plt.ylabel("CDS Shock (Dev)")
    plt.tight_layout()
    output_regimes = BASE_DIR.parent / "data" / "gmm_regimes.png"
    plt.savefig(output_regimes, dpi=300)
    print(f"✔ Saved: {output_regimes}")

# ------------------------------
# 4. Regime timeline
# ------------------------------
print("\nGenerating regime timeline...")
df["date"] = pd.to_datetime(df["date"])
plt.figure(figsize=(12, 6))
for country in df["country"].unique():
    subset = df[df["country"] == country].sort_values("date")
    plt.plot(subset["date"], subset["regime"], label=country, alpha=0.4)

plt.title("Regime Timeline Across Countries")
plt.xlabel("Date")
plt.ylabel("Regime Index")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
output_timeline = BASE_DIR.parent / "data" / "regime_timeline.png"
plt.savefig(output_timeline, dpi=300)
plt.close()
print(f"✔ Saved: {output_timeline}")

# ------------------------------
# 5. Load feature importance (from train_model.py)
# ------------------------------
feature_path = BASE_DIR.parent / "data" / "feature_importance.csv"
if feature_path.exists():
    importance_df = pd.read_csv(feature_path)
    print("\n=== FEATURE IMPORTANCE (Next-Month Prediction) ===")
    print(importance_df)
else:
    print("\n⚠ No feature importance found. Run train_model.py first.")

# ------------------------------
# 6. Regime Transition Accuracy Plot
# ------------------------------
if "regime_next" in df.columns:
    df["correct_pred"] = (df["regime"] == df["regime_next"]).astype(int)

    trans_accuracy = (
        df.groupby("country")["correct_pred"]
          .mean()
          .reset_index()
          .sort_values("correct_pred", ascending=False)
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(data=trans_accuracy, x="country", y="correct_pred")
    plt.title("Model Ability to Predict Next-Month Regime")
    plt.xlabel("Country")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)

    output_acc = BASE_DIR.parent / "data" / "regime_transition_accuracy.png"
    plt.savefig(output_acc, dpi=300, bbox_inches="tight")
    print(f"✔ Saved: {output_acc}")