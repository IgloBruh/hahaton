import pandas as pd
from config import TARGET_COL, DATETIME_COL

df = pd.read_csv("data/3888f9f2-9bda-4b2c-94af-5562668bce86_test_dataset.csv")
print(f"rows: {len(df)}")
print(f"datetime sample: {df[DATETIME_COL].iloc[0]}")
print(f"datetime sample 2: {df[DATETIME_COL].iloc[1]}")
target_nulls = df[TARGET_COL].isna().sum() if TARGET_COL in df.columns else "col absent"
print(f"target nulls: {target_nulls} / {len(df)}")
print(f"cols: {list(df.columns)}")
