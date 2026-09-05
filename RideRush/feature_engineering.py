import pandas as pd

# Load cleaned dataset
df = pd.read_parquet("data/cleaned_taxi_data.parquet")

print("Cleaned data loaded successfully!")
print("Original shape:", df.shape)

# Create time features
df["hour"] = df["tpep_pickup_datetime"].dt.hour
df["day"] = df["tpep_pickup_datetime"].dt.day
df["month"] = df["tpep_pickup_datetime"].dt.month
df["day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek

# Weekend feature
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Location features
df["pickup_location"] = df["PULocationID"]
df["dropoff_location"] = df["DOLocationID"]

# Select features
features = [
    "pickup_location",
    "dropoff_location",
    "trip_distance",
    "passenger_count",
    "hour",
    "day",
    "month",
    "day_of_week",
    "is_weekend",
    "trip_duration"
]

df = df[features]

# Save the final feature dataset
df.to_parquet(
    "data/final_features.parquet",
    index=False
)

print("Feature engineering completed successfully!")
print("Final shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nFeatures used:")
print(df.columns.tolist())