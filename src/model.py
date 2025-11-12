import pandas as pd
import glob
import os

files = glob.glob("data/*.csv")

dfs = []

for file in files:
    print("Processing:", file)

    raw = pd.read_csv(file, header=None)

    # detect header row (first row with the word "Date")
    header_row = raw[raw.apply(lambda row: row.astype(str).str.contains("Date", case=False)).any(axis=1)].index[0]

    temp = pd.read_csv(file, header=header_row)

    # Standardize column names
    temp.columns = (
        temp.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("%", "pct")
        .str.replace("/", "_")
    )

    # Remove duplicated columns
    temp = temp.loc[:, ~temp.columns.duplicated()]

    # Add country column based on filename
    temp["country"] = os.path.basename(file).replace(".csv", "")

    dfs.append(temp)

# Combine all cleaned data
df = pd.concat(dfs, ignore_index=True)

# Convert numeric columns (remove commas, convert to float)
for col in df.columns:
    try:
        df[col] = df[col].astype(str).str.replace(",", "").str.replace("%", "").astype(float)
    except:
        pass

print("\n Final dataset shape:", df.shape)
print(df.head())

df.to_csv("data/merged_cleaned_dataset.csv", index=False)
print("\n Saved cleaned dataset → data/merged_cleaned_dataset.csv")