import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

"""
train_model.py
-------------------------------------------
Unsupervised crisis detection using GMM + shock features.
Now includes regime classifier validation (Random Forest).
-------------------------------------------
"""

# ------------------------------
# 1. Load dataset
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "merged_cleaned_dataset_filled.csv"

print(" Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f" Dataset loaded: {df.shape}")

# ------------------------------
# 2. Detect feature names automatically
# ------------------------------
bond_col = next((c for c in df.columns if "bond" in c.lower() or "yield" in c.lower()), None)
cds_col = next((c for c in df.columns if "cds" in c.lower()), None)
deficit_col = next((c for c in df.columns if "deficit" in c.lower()), None)

print("\nBond statistics:")
print(df[bond_col].describe())

print("\nCountry averages (mean):")
print(df.groupby("country")[[bond_col, cds_col, deficit_col]].mean())

print("\nCountry volatility (std):")
print(df.groupby("country")[[bond_col, cds_col, deficit_col]].std())

# ------------------------------
# 3. Economic shock features
# ------------------------------
K = 12  # rolling window (1 year of monthly data)

df["bond_roll"] = df[bond_col].rolling(K).mean()
df["cds_roll"] = df[cds_col].rolling(K).mean()
df["deficit_roll"] = df[deficit_col].rolling(K).mean()

df["bond_std"] = df[bond_col].rolling(K).std()
df["cds_std"] = df[cds_col].rolling(K).std()
df["deficit_std"] = df[deficit_col].rolling(K).std()

df["bond_dev"] = (df[bond_col] - df["bond_roll"]) / df["bond_std"]
df["cds_dev"] = (df[cds_col] - df["cds_roll"]) / df["cds_std"]
df["deficit_dev"] = (df[deficit_col] - df["deficit_roll"]) / df["deficit_std"]

df = df.dropna(subset=["bond_dev", "cds_dev", "deficit_dev"])
df = df.reset_index(drop=True)

# ------------------------------
# 4. Select features
# ------------------------------
features = ["bond_dev", "cds_dev", "deficit_dev"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 5. Fit GMM – unsupervised regime detection
# ------------------------------
gmm = GaussianMixture(
    n_components=3,
    random_state=42,
    covariance_type="full"
)
gmm.fit(X_scaled)

df["crisis_prob"] = gmm.predict_proba(X_scaled)[:, 2]
df["regime"] = gmm.predict(X_scaled)   # REQUIRED FOR CLASSIFIER

print("\nRegime counts:")
print(df["regime"].value_counts())

# ------------------------------
# 6. Train regime classifier
# ------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, df["regime"], test_size=0.2, random_state=42
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n=== REGIME CLASSIFIER VALIDATION ===")
print(classification_report(y_test, y_pred))

# ------------------------------
# 7. Save outputs
# ------------------------------
joblib.dump(gmm, BASE_DIR.parent / "data" / "gmm_model.pkl")
joblib.dump(scaler, BASE_DIR.parent / "data" / "scaler.pkl")
df.to_csv(BASE_DIR.parent / "data" / "gmm_predictions.csv", index=False)

print(f"\n✔ Saved model and outputs to {BASE_DIR.parent / 'data'}")

# ------------------------------
# 8. Plot Greece yearly trend
# ------------------------------
if "country" in df.columns:
    greek = df[df["country"] == "Greece"].copy()
    greek["year"] = pd.to_datetime(greek["date"]).dt.year
    yearly = greek.groupby("year")["crisis_prob"].mean().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(yearly["year"], yearly["crisis_prob"], marker="o")
    plt.title("Greece Crisis Probability (Yearly Average)")
    plt.xlabel("Year")
    plt.ylabel("Crisis Probability")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    output_path = BASE_DIR.parent / "data" / "greece_gmm_trend.png"
    plt.savefig(output_path)
    print(f"✔ Greece yearly trend saved to {output_path}")

# ------------------------------
# 9. Print validation summary
# ------------------------------
print("\n--- Average Crisis Probability by Country ---")
print(df.groupby("country")["crisis_prob"].mean().sort_values(ascending=False))

print("\n--- Top High-Risk Observations ---")
print(df.sort_values("crisis_prob", ascending=False).head(10)[["country", "date", "crisis_prob"]])

print("\n--- Greece Sample ---")
print(df[df['country'] == "Greece"][["date", "crisis_prob"]].head(20))

# ------------------------------------------------------
# 10. PREDICT REGIME ONE MONTH AHEAD (FORECAST SIGNAL)
# ------------------------------------------------------
df_sorted = df.sort_values(["country", "date"]).copy()
df_sorted["regime_next"] = df_sorted.groupby("country")["regime"].shift(-1)

# remove last rows (no future regime to compare with)
df_future = df_sorted.dropna(subset=["regime_next"]).copy()

X_future = df_future[features]
y_future = df_future["regime_next"]

# scale again for next-period prediction
X_future_scaled = scaler.transform(X_future)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_future_scaled, y_future, test_size=0.2, random_state=42
)

clf_future = RandomForestClassifier(random_state=42)
clf_future.fit(X_train_f, y_train_f)
y_pred_f = clf_future.predict(X_test_f)

print("\n=== CAN THE MODEL SEE A CRISIS BEFORE IT HAPPENS? ===")
print(classification_report(y_test_f, y_pred_f))

# ---------------------------------------------
# 10. FEATURE IMPORTANCE – Which shocks drive future crises?
# ---------------------------------------------
importances = clf_future.feature_importances_
feature_importance = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n=== FEATURE IMPORTANCE (Next-Month Regime Prediction) ===")
print(feature_importance)

# Save to CSV for visualization/use in dashboard
feature_importance.to_csv(BASE_DIR.parent / "data" / "feature_importance.csv", index=False)
print(f"✔ Saved feature importance to: {BASE_DIR.parent / 'data' / 'feature_importance.csv'}")

# ---------------------------------------------
# 11. CRISIS “HEAT ALERT” SYSTEM (Binary Crisis Signal)
# ---------------------------------------------
df_sorted = df_sorted.copy()
df_sorted["crisis_alert"] = (df_sorted["regime_next"] == 2).astype(int)

print("\n=== SAMPLE CRISIS ALERT SIGNAL ===")
print(df_sorted[["country", "date", "regime", "regime_next", "crisis_prob", "crisis_alert"]].head(15))

# Save for dashboard / visualization
df_sorted.to_csv(BASE_DIR.parent / "data" / "gmm_predictions_with_alert.csv", index=False)
print(f"✔ Saved alerts to: {BASE_DIR.parent / 'data' / 'gmm_predictions_with_alert.csv'}")

# ---------------------------------------------
# 12. VISUALIZE FEATURE IMPORTANCE
# ---------------------------------------------
plt.figure(figsize=(8,4))
plt.bar(feature_importance["feature"], feature_importance["importance"])
plt.title("Feature Importance – What Drives a Crisis?")
plt.xlabel("Economic Shock Variables")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(BASE_DIR.parent / "data" / "feature_importance_plot.png")
print(f"✔ Saved feature importance plot → {BASE_DIR.parent / 'data' / 'feature_importance_plot.png'}")
plt.close()

# ---------------------------------------------
# 13. CRISIS HEAT ALERT TIMELINE PER COUNTRY
# ---------------------------------------------
plt.figure(figsize=(14,6))
for country in df_sorted['country'].unique():
    subset = df_sorted[df_sorted["country"] == country]
    plt.plot(subset['date'], subset['crisis_alert'], label=country, alpha=0.4)

plt.title("Crisis Heat Alert Timeline (1 = Next-Month Crisis Expected)")
plt.xlabel("Date")
plt.ylabel("Crisis Alert")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(BASE_DIR.parent / "data" / "crisis_alert_timeline.png")
print(f"✔ Saved timeline → {BASE_DIR.parent / 'data' / 'crisis_alert_timeline.png'}")
plt.close()

# --------------------------------------------------
# 14. TRAFFIC-LIGHT ALERT SYSTEM (VISUAL RISK LEVEL)
# --------------------------------------------------
def risk_level(prob):
    if prob < 0.2:
        return "GREEN"
    elif prob < 0.5:
        return "YELLOW"
    else:
        return "RED"

df_sorted["risk_level"] = df_sorted["crisis_prob"].apply(risk_level)

print("\n=== TRAFFIC LIGHT SAMPLE ===")
print(df_sorted[["country", "date", "crisis_prob", "risk_level"]].head(10))

df_sorted.to_csv(BASE_DIR.parent / "data" / "gmm_predictions_dashboard.csv", index=False)
print(f"✔ Dashboard-ready data saved → {BASE_DIR.parent / 'data' / 'gmm_predictions_dashboard.csv'}")
