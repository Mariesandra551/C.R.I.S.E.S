import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler



"""
train_model.py
----------------

This script trains a simple machine learning model that serves as an early warning system
for financial crises, using historical economic data from Greece.

The model applies logistic regression to predict whether a given period shows signs of
financial distress (crisis) based on macroeconomic and market indicators such as
deficits, debt levels, and bond yield changes. It prioritizes interpretability over
complexity, aiming to demonstrate how machine learning can complement economic theory
in crisis forecasting.

Main steps:
    1. Load the cleaned dataset produced by model.py.
    2. Automatically detect a column representing economic change and create a binary
       crisis label (1 = crisis, 0 = stable).
    3. Prepare features by selecting numeric columns and dropping empty ones.
    4. Handle data imbalance through upsampling of the minority (crisis) class.
    5. Impute missing values and scale all numeric features.
    6. Split the data into training and testing sets.
    7. Train a logistic regression classifier with class balancing enabled.
    8. Evaluate performance using accuracy and classification metrics.
    9. Save the trained model, imputer, and scaler as reusable .pkl files.

Inputs:
    - ../data/merged_cleaned_dataset.csv : Cleaned dataset created by model.py

Outputs:
    - ../data/crisis_model.pkl  : Trained logistic regression model
    - ../data/imputer.pkl       : Median imputer for missing value handling
    - ../data/scaler.pkl        : StandardScaler for feature normalization

Intended Use:
    This file is part of an academic project (ECON 302) demonstrating how economic
    indicators and simple machine learning can be used to detect early warning signs
    of financial crises in a transparent, explainable way.
"""

# ------------------------------
# 1. Load dataset
# ------------------------------
print(" Loading dataset...")
df = pd.read_csv("../data/merged_cleaned_dataset.csv")
print(f" Dataset loaded successfully: ({df.shape[0]}, {df.shape[1]})")

# ------------------------------
# 2. Create crisis label
# ------------------------------
change_col = None
for col in df.columns:
    if "change" in col.lower():
        change_col = col
        break

if change_col is None:
    raise ValueError("No column containing 'change' found in dataset.")

df["crisis_label"] = (df[change_col] < -40).astype(int)  # slightly relaxed threshold
print(" Crisis label created. Sample:")
print(df[[change_col, "crisis_label"]].head())

# ------------------------------
# 3. Prepare features
# ------------------------------
X = df.select_dtypes(include=["float64", "int64"]).drop(columns=["crisis_label"], errors="ignore")
y = df["crisis_label"]

# Drop fully empty columns
X = X.dropna(axis=1, how="all")

# ------------------------------
# 4. Balance dataset (Upsampling)
# ------------------------------
print(" Balancing dataset...")
df_balanced = pd.concat([X, y], axis=1)
majority = df_balanced[df_balanced["crisis_label"] == 0]
minority = df_balanced[df_balanced["crisis_label"] == 1]

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
# 5. Impute missing values
# ------------------------------
imputer = SimpleImputer(strategy="median")
X_bal_imputed = pd.DataFrame(imputer.fit_transform(X_bal), columns=X_bal.columns)

# ------------------------------
# 6. Feature scaling
# ------------------------------
scaler = StandardScaler()
X_bal_scaled = pd.DataFrame(scaler.fit_transform(X_bal_imputed), columns=X_bal.columns)

# ------------------------------
# 7. Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_bal_scaled, y_bal, test_size=0.2, random_state=42
)
print(f" Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ------------------------------
# 8. Train model
# ------------------------------
model = LogisticRegression(max_iter=5000, solver="saga", class_weight="balanced")
model.fit(X_train, y_train)

print("\nExample crisis probabilities:", model.predict_proba(X_bal_scaled)[:10])


# ------------------------------
# 9. Evaluate
# ------------------------------
y_pred_proba = model.predict_proba(X_test)[:, 1]  
y_pred = (y_pred_proba > 0.5).astype(int)         

print("\n Model Evaluation Results:")
print(classification_report(y_test, y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

print("\nExample crisis probabilities:", y_pred_proba[:10]) 

# ===============================
# Additional Model Performance Checks
# ===============================

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC-AUC Score:", round(roc_auc, 4))

# Precision & Recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))


# ------------------------------
# 10. Save model, imputer, and scaler
# ------------------------------
# Save updated model
joblib.dump(model, "../data/crisis_model.pkl")
joblib.dump(imputer, "../data/imputer.pkl")
joblib.dump(scaler, "../data/scaler.pkl")
print(" Model, imputer, and scaler saved again.")