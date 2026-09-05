import pandas as pd

# Load the May 2026 taxi dataset
file_path = "data/yellow_tripdata_2026-05.parquet"

df = pd.read_parquet(file_path)

print("Original dataset shape:", df.shape)

# Convert date and time columns
df["tpep_pickup_datetime"] = pd.to_datetime(
    df["tpep_pickup_datetime"],
    errors="coerce"
)

df["tpep_dropoff_datetime"] = pd.to_datetime(
    df["tpep_dropoff_datetime"],
    errors="coerce"
)

# Calculate trip duration in minutes
df["trip_duration"] = (
    df["tpep_dropoff_datetime"]
    - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60

# Remove missing important values
df = df.dropna(
    subset=[
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "trip_distance",
        "passenger_count",
        "trip_duration"
    ]
)

# Remove invalid trip distances
df = df[
    (df["trip_distance"] > 0) &
    (df["trip_distance"] < 100)
]

# Remove invalid trip durations
df = df[
    (df["trip_duration"] > 1) &
    (df["trip_duration"] < 180)
]

# Remove invalid passenger counts
df = df[
    (df["passenger_count"] > 0) &
    (df["passenger_count"] <= 8)
]

# Remove duplicate rows
df = df.drop_duplicates()

print("Cleaned dataset shape:", df.shape)

# Save cleaned dataset
df.to_parquet(
    "data/cleaned_taxi_data.parquet",
    index=False
)

print("Cleaned dataset saved successfully!")