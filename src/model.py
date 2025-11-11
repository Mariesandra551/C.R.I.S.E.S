import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# Load your CSV file
df = pd.read_csv("../data/Country_data.csv")

#print("COLUMNS:", df.columns.tolist())
#exit()

# Create spread feature
df["SPREAD_10Y"] = df["GR_10Y"] - df["DE_10Y"]

# Select features and labels
features = ["DEFICIT_GDP", "INFLATION", "SPREAD_10Y", "CDS_5Y"]
X = df[features]
y = df["CRISIS LABEL"]

# Train on all except last row, test on last row
X_train = X.iloc[:-1]
y_train = y.iloc[:-1]
X_test = X.iloc[-1:]
y_test = y.iloc[-1:]

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

print(export_text(clf, feature_names=features))
print("Model accuracy:", clf.score(X_test, y_test))