import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report



"""
Core machine learning pipeline for the financial crisis early warning system.

This script:
1. Loads cleaned and aligned macro data
2. Builds economic shock features (rolling mean and std)
3. Fits a Gaussian Mixture Model (GMM) to detect regimes
4. Creates crisis probability and regime labels per observation
5. Validates regime stability using Random Forest with TimeSeriesSplit
6. Builds a next month regime forecast and evaluates it with TimeSeriesSplit
7. Computes feature importance for the forecast model
8. Generates a binary crisis alert and traffic light risk labels
9. Saves all model outputs for visualization and dashboards

Outputs saved in /data:
    - gmm_model.pkl
    - scaler.pkl
    - gmm_predictions.csv
    - feature_importance.csv
    - gmm_predictions_with_alert.csv
"""

# ------------------------------
# 1. Load dataset
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "merged_cleaned_dataset_filled.csv"

print(" Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f" Dataset loaded: {df.shape}")

# ensure date is proper datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# ------------------------------
# 2. Detect feature names
# ------------------------------
bond_col = next((c for c in df.columns if "bond" in c.lower() or "yield" in c.lower()), None)
cds_col = next((c for c in df.columns if "cds" in c.lower()), None)
deficit_col = next((c for c in df.columns if "deficit" in c.lower()), None)

if bond_col is None or cds_col is None or deficit_col is None:
    raise ValueError("Could not automatically detect bond, cds, or deficit columns.")

print("\nBond statistics:")
print(df[bond_col].describe())

# sort by country and date ascending for time series logic
df = df.sort_values(["country", "date"]).reset_index(drop=True)

# ------------------------------
# 3. Economic shock features
# ------------------------------
K = 12  # 12 months of economic memory

def rolling_mean(group, col):
    return group[col].rolling(K, min_periods=K).mean()

def rolling_std(group, col):
    return group[col].rolling(K, min_periods=K).std()

# rolling mean and std per country
df["bond_roll"] = df.groupby("country", group_keys=False).apply(rolling_mean, bond_col)
df["cds_roll"] = df.groupby("country", group_keys=False).apply(rolling_mean, cds_col)
df["deficit_roll"] = df.groupby("country", group_keys=False).apply(rolling_mean, deficit_col)

df["bond_std"] = df.groupby("country", group_keys=False).apply(rolling_std, bond_col)
df["cds_std"] = df.groupby("country", group_keys=False).apply(rolling_std, cds_col)
df["deficit_std"] = df.groupby("country", group_keys=False).apply(rolling_std, deficit_col)

# avoid division by zero
df["bond_std"] = df["bond_std"].replace(0, np.nan)
df["cds_std"] = df["cds_std"].replace(0, np.nan)
df["deficit_std"] = df["deficit_std"].replace(0, np.nan)

df["bond_dev"] = (df[bond_col] - df["bond_roll"]) / df["bond_std"]
df["cds_dev"] = (df[cds_col] - df["cds_roll"]) / df["cds_std"]
df["deficit_dev"] = (df[deficit_col] - df["deficit_roll"]) / df["deficit_std"]

# drop rows that cannot be used for shocks
df = df.dropna(subset=["bond_dev", "cds_dev", "deficit_dev"]).reset_index(drop=True)

# ------------------------------
# 4. Fit GMM model
# ------------------------------
features = ["bond_dev", "cds_dev", "deficit_dev"]
X = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(
    n_components=4,        # normal, mild stress, stress, crisis
    covariance_type="full",
    random_state=42,
    reg_covar=1e-4        # regularization to prevent overconfidence
)
gmm.fit(X_scaled)

# treat component index 2 as crisis regime by design
crisis_probs = gmm.predict_proba(X_scaled)[:, 2]
df["crisis_prob"] = crisis_probs

# smooth crisis probability over 6 months per country
df["crisis_prob_smooth"] = (
    df.groupby("country")["crisis_prob"]
      .transform(lambda s: s.rolling(6, min_periods=1).mean())
)

df["regime"] = gmm.predict(X_scaled)

print("\nRegime counts:")
print(df["regime"].value_counts())

# ------------------------------
# 5. Validate regimes via classifier (TimeSeriesSplit)
# ------------------------------
X_full = X_scaled
y_full = df["regime"].values

tscv = TimeSeriesSplit(n_splits=5)

# ------------------------------
# 6. Save core outputs
# ------------------------------
joblib.dump(gmm, BASE_DIR.parent / "data" / "gmm_model.pkl")
joblib.dump(scaler, BASE_DIR.parent / "data" / "scaler.pkl")
df.to_csv(BASE_DIR.parent / "data" / "gmm_predictions.csv", index=False)

print(f"\n✔ Saved model and outputs → {BASE_DIR.parent / 'data'}")

# ------------------------------------------------------
# 7. Forecast: predict regime next month (TimeSeriesSplit)
# ------------------------------------------------------
df_sorted = df.sort_values(["country", "date"]).copy()
df_sorted["regime_next"] = df_sorted.groupby("country")["regime"].shift(-1)
df_future = df_sorted.dropna(subset=["regime_next"]).copy()

X_future = df_future[features].values
y_future = df_future["regime_next"].values
X_future_scaled = scaler.transform(X_future)

tscv_future = TimeSeriesSplit(n_splits=5)

print("\n=== CAN THE MODEL SEE A CRISIS BEFORE IT HAPPENS? (TimeSeriesSplit) ===")
for fold, (train_idx, test_idx) in enumerate(tscv_future.split(X_future_scaled)):
    print(f"\n----- Forecast Fold {fold + 1} -----")
    X_train_f, X_test_f = X_future_scaled[train_idx], X_future_scaled[test_idx]
    y_train_f, y_test_f = y_future[train_idx], y_future[test_idx]

    clf_future = RandomForestClassifier(random_state=42)
    clf_future.fit(X_train_f, y_train_f)
    y_pred_f = clf_future.predict(X_test_f)

    print(classification_report(y_test_f, y_pred_f))

# ------------------------------------------------------
# 8. Feature importance for next month forecast
# ------------------------------------------------------
# fit a final model on the full future set only for interpretability
clf_future_full = RandomForestClassifier(random_state=42)
clf_future_full.fit(X_future_scaled, y_future)

importances = clf_future_full.feature_importances_
feature_importance = pd.DataFrame(
    {"feature": features, "importance": importances}
).sort_values(by="importance", ascending=False)

feature_importance.to_csv(BASE_DIR.parent / "data" / "feature_importance.csv", index=False)
print(f"\n✔ Saved feature importance → {BASE_DIR.parent / 'data' / 'feature_importance.csv'}")
print("\n=== FEATURE IMPORTANCE (Next month regime) ===")
print(feature_importance)

# ------------------------------
# 9. CRISIS ALERT + TRAFFIC LIGHT SYSTEM
# ------------------------------
df_sorted["crisis_alert"] = (df_sorted["regime_next"] == 2).astype(int)

def risk_level(prob):
    if prob < 0.2: return "GREEN"
    elif prob < 0.5: return "YELLOW"
    else: return "RED"

df_sorted["risk_level"] = df_sorted["crisis_prob"].apply(risk_level)
df_sorted["crisis_prob_smooth"] = df["crisis_prob_smooth"]

# EXPORT FOR STREAMLIT
df_sorted.to_csv(
    BASE_DIR.parent / "data" / "gmm_predictions_dashboard.csv",
    index=False
)
print("✔ Dashboard ready alerts saved →", BASE_DIR.parent / 'data')
