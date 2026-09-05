import pandas as pd

file_path = "data/yellow_tripdata_2026-05.parquet"

print("Loading dataset...")

df = pd.read_parquet(file_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())