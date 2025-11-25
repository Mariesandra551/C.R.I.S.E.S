"""
Interactive Streamlit application for exploring global economic stress,
crisis probabilities, and future risk alerts produced by the GMM model.

This dashboard serves as the **front-end interface** for the early warning
system built in `train_model.py` and `visualize_predictions.py`. It allows
users to explore financial risk patterns across countries in an intuitive
and evidence-based format.

Core Features:
1. Load precomputed crisis probabilities and risk alerts
   (`gmm_predictions_dashboard.csv`).
2. Filter by country and risk level (GREEN / YELLOW / RED).
3. Visualize crisis probability trends for any country.
4. Highlight **current high-risk countries** based on latest data.
5. Display **risk-level breakdown by country** for comparison.
6. Show feature importance (economic shock contributions),
   supporting empirical interpretation.
7. Export filtered results for reporting or further analysis.

Usage:
    streamlit run dashboard_app.py

This dashboard supports **academic presentation, policymaking, and financial
intelligence analysis** by transforming model outputs into a clear and
actionable decision-support tool.
"""

import os
import subprocess
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import altair as alt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_DIR / "pipeline.py"
DATA_PATH = BASE_DIR.parent / "data" / "gmm_predictions_dashboard.csv"


if not DATA_PATH.exists(): #runs pipeline on load
    st.warning("Data not found — running pipeline automatically.")
    subprocess.run(["python", str(BASE_DIR / "pipeline.py")], check=True)

# ------------------------------
# RUN PIPELINE ON DEMAND
# ------------------------------
st.sidebar.subheader("⚙ Data Generation")

if st.sidebar.button("Run Full Pipeline (Refresh Data)"):
    with st.spinner("Running pipeline... generating model outputs..."):
        subprocess.run(["python", str(PIPELINE_PATH)], check=True)
    st.success("Pipeline complete — data refreshed! Please reload the dashboard.")


# ------------------------------------
# 1. LOAD DATA
# ------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

st.title("Financial Crisis Early Warning Dashboard")
st.caption("Real-time economic stress detection using GMM + crisis alert system")

# ------------------------------------
# 2. SIDEBAR FILTERS
# ------------------------------------
st.sidebar.header("Filters")
countries = df["country"].unique()
selected_countries = st.sidebar.multiselect("Select countries", countries, default=countries[:4])

risk_filter = st.sidebar.radio(
    "Filter by risk level",
    ["ALL", "GREEN", "YELLOW", "RED"]
)

df_filtered = df[df["country"].isin(selected_countries)]
if risk_filter != "ALL":
    df_filtered = df_filtered[df_filtered["risk_level"] == risk_filter]

st.sidebar.button("Reload dashboard", on_click=lambda: st.rerun())

# ------------------------------------
# 3. HIGHEST CURRENT RISK (Insight)
# ------------------------------------
latest_data = df.sort_values("date").groupby("country").tail(1)
st.subheader("Countries Currently at Highest Risk (latest available month)")
st.dataframe(
    latest_data.sort_values("crisis_prob", ascending=False)[
        ["country", "date", "crisis_prob", "risk_level"]
    ].head(10)
)

# ------------------------------------
# 4. DATA TABLE
# ------------------------------------
st.subheader("Filtered Observations")
st.dataframe(
    df_filtered[
        ["country", "date", "crisis_prob", "regime", "risk_level"]
    ].sort_values(["country", "date"], ascending=[True, False]).head(25)
)

# ------------------------------------
# 5. TREND PLOT (RAW vs SMOOTHED)
# ------------------------------------
st.subheader("Crisis Probability Over Time")

country_sel = st.selectbox("Select country for trend plot", countries)
subset = df[df["country"] == country_sel].copy()

# Select visualization type
plot_type = st.radio(
    "Choose crisis probability visualization",
    ["Raw (original)", "Smoothed (6-month rolling mean)"]
)

fig, ax = plt.subplots(figsize=(8, 4))

if plot_type == "Raw (original)":
    ax.plot(subset["date"], subset["crisis_prob"], marker="o")
    ax.set_title(f"Crisis Probability Trend (Raw): {country_sel}")
else:
    if "crisis_prob_smooth" in subset.columns:
        ax.plot(subset["date"], subset["crisis_prob_smooth"], marker="o")
        ax.set_title(f"Crisis Probability Trend (Smoothed): {country_sel}")
    else:
        st.warning("Smoothed values not found. Run train_model.py again.")
        ax.plot(subset["date"], subset["crisis_prob"], marker="o")

ax.set_xlabel("Date")
ax.set_ylabel("Crisis Probability")
plt.xticks(rotation=45)
st.pyplot(fig)

# -------------------------------
# 6. TRAFFIC LIGHT VIEW
# -------------------------------
st.subheader("Risk Evolution Over Time (Traffic Light View)")

df["year"] = df["date"].dt.year

severity_map = {"GREEN": 0, "YELLOW": 1, "RED": 2}
inv_map      = {0: "GREEN", 1: "YELLOW", 2: "RED"}
df["severity"] = df["risk_level"].map(severity_map)

# Choose worst value (max severity) per year
year_severity = (
    df.groupby(["country", "year"])["severity"]
      .max()
      .unstack(fill_value=0)
      .applymap(lambda x: inv_map[x])
)

# Convert to numeric matrix
heatmap_data = [[severity_map[val] for val in row] for row in year_severity.values]

# CUSTOM COLOR PALETTE (exact match)
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#2ecc71", "#f1c40f", "#e74c3c"])  # green, yellow, red

# Create figure
fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(heatmap_data, cmap=cmap, aspect="equal")  # square cells

# Axis fix
ax.set_yticks(range(len(year_severity.index)))
ax.set_yticklabels(year_severity.index)

ax.set_xticks(range(len(year_severity.columns)))
ax.set_xticklabels(year_severity.columns.astype(int), rotation=45)

# Add thin gridlines to define cells
ax.set_xticks(np.arange(-.5, len(year_severity.columns), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(year_severity.index), 1), minor=True)
ax.grid(which="minor", color='gray', linestyle='-', linewidth=0.3)

# Legend
cbar = plt.colorbar(im, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(["GREEN", "YELLOW", "RED"])

# Titles
plt.title("Risk Evolution by Country and Year (Traffic Light View")
plt.xlabel("Year")
plt.ylabel("Country")

# Display
st.pyplot(fig)

# ------------------------------------
# 7. FEATURE IMPORTANCE (Predictive Model Insight)
# ------------------------------------
feature_path = BASE_DIR.parent / "data" / "feature_importance.csv"
if feature_path.exists():
    st.subheader("Which Economic Shocks Matter Most?")
    feature_imp = pd.read_csv(feature_path)

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(data=feature_imp, x="feature", y="importance", ax=ax)
    ax.set_title("Feature Importance – Drivers of Future Crisis")
    st.pyplot(fig)
else:
    st.warning("Feature importance file not found. Run train_model.py first.")

# ------------------------------------
# 8. DOWNLOAD DATA
# ------------------------------------
st.subheader("Export Filtered Dataset")
csv = df_filtered.to_csv(index=False).encode()
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="filtered_crisis_predictions.csv"
)

st.success("Dashboard ready — explore your early warning system!")

# ------------------------------------
# 9. STATIC MODEL VISUALS (optional)
# ------------------------------------
st.subheader("Model Visual Outputs")

image_dir = BASE_DIR.parent / "data"

images = [
    "greece_gmm_trend_yearly.png",
    "contagion_matrix.png",
    "gmm_regimes.png",
    "regime_timeline.png"
]

for img in images:
    img_path = image_dir / img
    if img_path.exists():
        st.image(str(img_path), caption=img.replace(".png", "").replace("_", " ").title())



if __name__ == "__main__":
    """
    Dashboard for visualizing financial crisis early warning signals.
    Run with:
        streamlit run dashboard_app.py
    """
    pass