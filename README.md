Crisis Risk Identification System for Economic Stability (C.R.I.S.E.S)

A Global Financial Crisis Early-Warning System

🧠 Overview

This system detects and forecasts financial instability across countries using:
✔ Gaussian Mixture Models (GMM)
✔ Regime probability smoothing
✔ Time-series forecasting
✔ Interactive dashboard for real-time analysis

It analyzes macroeconomic shock indicators — bond yields, CDS spreads, and government deficits — to detect economic regimes and crisis risk, helping analysts, policymakers, and researchers anticipate stress before it becomes a crisis.

The emphasis is on clarity and interpretability, showing how basic but well-designed ML tools can complement economic intuition.

🎯 Objectives

Detect macro-financial stress periods across countries

Reveal hidden economic regimes (Normal / Stress / Crisis)

Predict next-month regime transitions

Provide an interactive Streamlit dashboard

Support interpretation with feature importance, contagion heatmaps, and regime evolution

🔍 Methodology
Step	Description
Feature Engineering	Rolling mean and volatility shocks for bond, CDS, and deficit
Regime Detection	GMM identifies hidden economic states
Smoothing	6-month rolling average stabilizes probability curves
Forecasting	Random Forest predicts next-month regime
Alerting	Binary crisis_alert + traffic-light labels
Validation	TimeSeriesSplit prevents data leakage
Visualization	Contagion matrix, regime evolution, trend lines
Dashboard	Streamlit interface for exploration & export
📈 Results

The model successfully detects periods where fiscal imbalance + market stress align with high crisis probability.

Key Predictive Features

Deficit-to-GDP ratio

Debt-to-GDP ratio

Bond yield spreads (Greek vs. German)

The visualizations highlight the exact years markets expressed concern before the official crisis, showing early stress signals.

🧾 Interpretation

Even a simple model can detect early warning signs of financial instability.
This should not be viewed as a crisis predictor, but as a decision-support tool
that flags periods of rising vulnerability.

⚠ Challenges & Solutions
Challenge	Mitigation
Limited and inconsistent data	Interpolation + median imputation
Small dataset	Upsampling of crisis observations
Correlated variables	Use only interpretable indicators
Asymmetric information	Use market-based indicators (CDS)
Delayed fiscal reporting	Combine slow + fast signals
📜 License

This project is for educational and academic purposes only.
You may reuse or adapt with proper citation.

⚙️ How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run Full Pipeline (Generate Data & Model)
python src/pipeline.py

3️⃣ Launch Streamlit Dashboard
streamlit run src/dashboard_app.py
