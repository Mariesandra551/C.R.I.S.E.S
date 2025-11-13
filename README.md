# Early Warning System for the Greek Financial Crisis

## Overview
This project develops a simple, interpretable **early warning system** to detect rising financial risk before the outbreak of the **Greek sovereign debt crisis**.  
By combining **economic theory** with **machine learning**, the model uses fiscal and market indicators—such as deficits, debt-to-GDP, and bond yield spreads—to predict the likelihood of a crisis period.

The emphasis is on **clarity and interpretability**, showing how even basic machine learning tools can complement economic reasoning for crisis forecasting.

---

## Repository Structure

project_root/
│
├── data/
│ ├── *.csv # Raw data files for Greece (debt, deficit, yields, etc.)
│ ├── merged_cleaned_dataset.csv # Output from model.py
│ ├── crisis_model.pkl # Trained logistic regression model
│ ├── imputer.pkl # Median imputer for missing data
│ ├── scaler.pkl # StandardScaler for normalization
│ ├── top10_crisis_probs.csv # Table of top 10 predicted crisis-risk periods
│ └── top10_crisis_probs.png # Visualization of predicted crisis probabilities
│
├── model.py # Cleans and merges raw data into a standardized dataset
├── train_model.py # Trains logistic regression model and saves components
├── Visualize_predictions.py # Generates a visualization and CSV of top crisis probabilities
└── README.md # Project documentation

---

## Objectives
- Build a transparent **early-warning system** for financial instability in Greece.
- Identify which **macroeconomic and market indicators** signal rising risk.
- Demonstrate how **machine learning** can enhance economic analysis without overcomplicating it.

---

## Methodology

### 1. Data Preparation (`model.py`)
- Reads multiple `.csv` files from `/data`
- Automatically detects header rows
- Cleans, renames, and merges data into a single dataset
- Outputs `merged_cleaned_dataset.csv`

### 2. Model Training (`train_model.py`)
- Loads the cleaned dataset
- Creates a binary **crisis label**
- Balances the dataset through upsampling
- Imputes missing values and scales features
- Trains a **Logistic Regression** classifier
- Evaluates results using accuracy and classification metrics
- Saves model artifacts (`.pkl` files) for reuse

### 3. Prediction Visualization (`Visualize_predictions.py`)
- Loads trained model and preprocessing tools
- Predicts **crisis probabilities** for each observation
- Displays top 10 highest-risk years or countries
- Saves both `.csv` and `.png` outputs for presentation

---

## Results

- The model successfully identifies periods where **fiscal imbalance** and **market stress** coincide with high crisis probability.
- Key predictive features include:
  - Deficit-to-GDP ratio
  - Debt-to-GDP ratio
  - Bond yield spreads (Greek vs. German)
- The visualization highlights years that markets perceived as high-risk before the official crisis.

---

## Interpretation
The project demonstrates that even with basic data and a simple logistic regression, it is possible to detect **early warning signs** of financial distress.  
The model should not be viewed as a predictor of exact crises but as a **decision-support tool** that highlights periods of rising vulnerability.

---

## Challenges & Solutions

| Challenge | Mitigation |
|------------|-------------|
| Limited and inconsistent historical data | Used data cleaning and median imputation |
| Small dataset | Applied upsampling of crisis observations |
| Correlated variables | Chose interpretable indicators and simple models |
| Asymmetric information | Relied on market-based indicators like CDS spreads |
| Delayed reporting | Combined fiscal (slow) and market (fast) indicators |

---

## Outputs
- **Data visualization**: `top10_crisis_probs.png`
- **Predicted probabilities**: `top10_crisis_probs.csv`
- **Reusable model files**: `crisis_model.pkl`, `imputer.pkl`, `scaler.pkl`

Example figure:

![Top 10 Crisis Probabilities](data/top10_crisis_probs.png)

---

License

This project is for educational and academic purposes only.
You may reuse or adapt the code with proper citation.

---

## How to Run

### Prerequisites
Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib joblib
Run the pipeline:
bash
Copy code
# Step 1: Clean and merge data
python model.py

# Step 2: Train model
python train_model.py

# Step 3: Visualize predictions
python Visualize_predictions.py
