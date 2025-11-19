import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score



"""
train_model.py
------------------------------
This script trains a binary classification model that predicts whether a sovereign
debt crisis is occurring based on economic indicators.

The pipeline performs the following steps:
1. Load the merged clean dataset.
2. Automatically detect key variable columns: bond yields, CDS spreads, and deficit.
3. Create a binary crisis label based on a threshold (e.g., bond_yield_change < -1).
4. Balance the dataset using upsampling to address class imbalance.
5. Impute and scale numerical features to prepare for training.
6. Train a Logistic Regression model with balanced class weights.
7. Report model performance and feature importance.
8. Save the model, imputer, and scaler for future predictions.

Output files (saved inside /data):
    • crisis_model.pkl   → trained logistic regression model
    • imputer.pkl        → SimpleImputer used during training
    • scaler.pkl         → StandardScaler used during training

This file is meant to be run once after data cleaning is complete
('merged_cleaned_dataset_filled.csv') to create a reusable trained model.
"""

# ------------------------------
# 1. Load dataset
# ------------------------------
print(" Loading dataset...")
df = pd.read_csv("../data/merged_cleaned_dataset_filled.csv")
print(f" Dataset loaded successfully: ({df.shape[0]}, {df.shape[1]})")

# ------------------------------
# 2. Identify key columns
# ------------------------------
change_col = next((c for c in df.columns if "change" in c.lower()), None)
bond_col = next((c for c in df.columns if "bond" in c.lower() or "yield" in c.lower()), None)
cds_col = next((c for c in df.columns if "cds" in c.lower()), None)
deficit_col = next((c for c in df.columns if "deficit" in c.lower()), None)

if not all([change_col, bond_col, cds_col, deficit_col]):
    raise ValueError("Missing one of the key columns: bond, cds, change, or deficit.")

# ------------------------------
# 3. Create crisis_label
# ------------------------------
df["crisis_label"] = (df[change_col] < -1).astype(int)
print("\nCrisis label sample:")
print(df[[change_col, "crisis_label"]].head())

# ------------------------------
# 4. Select features (no manual weights)
# ------------------------------
X = df[[bond_col, cds_col, deficit_col]].copy()
y = df["crisis_label"]

# ------------------------------
# 5. Select features
# ------------------------------
X = df.select_dtypes(include=["float64", "int64"]).drop(columns=["crisis_label"], errors="ignore")
y = df["crisis_label"]

# Drop fully empty columns
X = X.dropna(axis=1, how="all")

# ------------------------------
# 6. Balance dataset
# ------------------------------
print("\nBalancing dataset...")
df_balanced = pd.concat([X, y], axis=1)
majority = df_balanced[df_balanced["crisis_label"] == 0]
minority = df_balanced[df_balanced["crisis_label"] == 1]

if len(minority) == 0:
    raise ValueError("No crisis cases found. The label threshold might be too strict.")

minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

df_balanced = pd.concat([majority, minority_upsampled])
X_bal = df_balanced.drop("crisis_label", axis=1)
y_bal = df_balanced["crisis_label"]

# ------------------------------
# 7. Impute + scale
# ------------------------------
imputer = SimpleImputer(strategy="median")
X_bal_imputed = pd.DataFrame(imputer.fit_transform(X_bal), columns=X_bal.columns)

scaler = StandardScaler()
X_bal_scaled = pd.DataFrame(scaler.fit_transform(X_bal_imputed), columns=X_bal.columns)

# ------------------------------
# 8. Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_bal_scaled, y_bal, test_size=0.2, random_state=42
)

# ------------------------------
# 9. Train model (Logistic Regression)
# ------------------------------
model = LogisticRegression(max_iter=5000, solver="saga", class_weight="balanced")
model.fit(X_train, y_train)

# ------------------------------
# 10. Evaluate model
# ------------------------------
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

print("\nExample crisis probabilities:", y_pred_proba[:10])

# ------------------------------
# 11. Feature Importance (Logistic Regression)
# ------------------------------
coefficients = pd.DataFrame({
    "feature": X_bal.columns,
    "coefficient": model.coef_[0]
}).sort_values(by="coefficient", ascending=False)

print("\nFeature Importance (Logistic Regression):")
print(coefficients)

# Optional export:
# coefficients.to_csv("../data/feature_importance_logistic.csv", index=False)

# ------------------------------
# 12. Save model
# ------------------------------
joblib.dump(model, "../data/crisis_model.pkl")
joblib.dump(imputer, "../data/imputer.pkl")
joblib.dump(scaler, "../data/scaler.pkl")
print("\nModel, imputer, and scaler saved to /data")
