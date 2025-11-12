import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

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

df["crisis_label"] = (df[change_col] < -3).astype(int)  # slightly relaxed threshold
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

# ------------------------------
# 9. Evaluate
# ------------------------------
y_pred = model.predict(X_test)
print("\n📊 Model Evaluation Results:")
print(classification_report(y_test, y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# ------------------------------
# 10. Save model, imputer, and scaler
# ------------------------------
joblib.dump(model, "../data/crisis_model.pkl")
joblib.dump(imputer, "../data/imputer.pkl")
joblib.dump(scaler, "../data/scaler.pkl")
print("\n Model, imputer, and scaler saved to data/crisis_model.pkl, data/imputer.pkl, data/scaler.pkl")