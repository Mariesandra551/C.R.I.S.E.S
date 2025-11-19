"""
Enhanced Visualization for Crisis Prediction
--------------------------------------------
Creates multiple economic plots + summaries:
    1. Top 10 highest-risk observations
    2. Crisis trend over time (line plot)
    3. Country-level average risk
    4. Region-level comparison
    5. Yearly crisis distribution (histogram)
Exports useful CSV tables to support analysis.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ================================
#  PATHS
# ================================
DATA_DIR = "../data"
MODEL_PATH = f"{DATA_DIR}/crisis_model.pkl"
IMPUTER_PATH = f"{DATA_DIR}/imputer.pkl"
SCALER_PATH = f"{DATA_DIR}/scaler.pkl"
DATASET_PATH = f"{DATA_DIR}/merged_cleaned_dataset.csv"

# ================================
# 1. LOAD MODEL & DATA
# ================================
def load_model_and_data():
    print("Loading model and dataset...")
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATASET_PATH)

    # Clean country formatting
    df["country"] = df["country"].str.strip().str.title()
    df = df.loc[:, ~df.columns.str.contains("Unnamed", case=False)]

    print(f"✓ Dataset loaded: {df.shape}")
    return model, imputer, scaler, df

# ================================
# 2. FEATURE PREP
# ================================
def prepare_features(df, imputer, scaler):
    X = df.select_dtypes(include=["float64", "int64"]).drop(columns=["crisis_label"], errors="ignore")
    expected_cols = getattr(imputer, "feature_names_in_", X.columns)
    X = X[[c for c in expected_cols if c in X.columns]]

    X_imputed = pd.DataFrame(imputer.transform(X), columns=X.columns)
    X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=X.columns)
    return X_scaled

# ================================
# 3. CRISIS PROBABILITY
# ================================
def compute_predictions(model, X, df):
    df["crisis_prob"] = model.predict_proba(X)[:, 1]
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    return df

# ================================
# 4. VISUALIZATIONS
# ================================

def plot_top10_risk(df):
    top10 = df.sort_values("crisis_prob", ascending=False).head(10)
    top10.to_csv(f"{DATA_DIR}/top10_crisis_probs.csv", index=False)

    plt.figure(figsize=(10,6))
    plt.barh(top10["country"], top10["crisis_prob"])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Predicted Crisis Probabilities")
    plt.xlabel("Crisis Probability")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/top10_crisis_probs.png")
    plt.close()

def plot_risk_trends_over_time(df):
    df_grouped = df.groupby(["year"])["crisis_prob"].mean()
    plt.figure(figsize=(10,5))
    plt.plot(df_grouped.index, df_grouped.values, marker="o")
    plt.title("Average Crisis Probability Over Time")
    plt.xlabel("Year")
    plt.ylabel("Avg Crisis Probability")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/crisis_trend_over_time.png")
    plt.close()

def plot_country_risk_average(df):
    country_avg = df.groupby("country")["crisis_prob"].mean().sort_values(ascending=False).head(10)
    country_avg.to_csv(f"{DATA_DIR}/avg_risk_by_country.csv")

    plt.figure(figsize=(10,6))
    plt.barh(country_avg.index, country_avg.values)
    plt.gca().invert_yaxis()
    plt.title("Top 10 Countries by Average Crisis Risk")
    plt.xlabel("Average Crisis Probability")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/avg_risk_by_country.png")
    plt.close()

def plot_yearly_risk_histogram(df):
    plt.figure(figsize=(10,6))
    plt.hist(df["year"], bins=20)
    plt.title("Frequency of Crisis-like Years")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/yearly_crisis_histogram.png")
    plt.close()

# ================================
# 5. MAIN EXECUTION
# ================================
def main():
    model, imputer, scaler, df = load_model_and_data()
    X = prepare_features(df, imputer, scaler)
    df = compute_predictions(model, X, df)

    plot_top10_risk(df)
    plot_risk_trends_over_time(df)
    plot_country_risk_average(df)
    plot_yearly_risk_histogram(df)

    print("\n✓ All plots and CSV summaries saved in /data")

if __name__ == "__main__":
    main()
