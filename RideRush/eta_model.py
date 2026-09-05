import pandas as pd
import joblib
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# 1. Load feature dataset
df = pd.read_parquet("data/final_features.parquet")

print("Feature dataset loaded successfully!")
print("Dataset shape:", df.shape)


# 2. Select input features
X = df[
    [
        "pickup_location",
        "dropoff_location",
        "trip_distance",
        "passenger_count",
        "hour",
        "day",
        "month",
        "day_of_week",
        "is_weekend"
    ]
]


# 3. Select target
# We want to predict trip duration
y = df["trip_duration"]


# 4. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# 5. Create models
models = {

    "Ridge Regression": Ridge(alpha=1.0),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=12,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
}


# 6. Create models folder
os.makedirs("models", exist_ok=True)


# 7. Train models
best_model = None
best_mae = float("inf")

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    print("MAE :", round(mae, 2), "minutes")
    print("RMSE:", round(rmse, 2), "minutes")
    print("R2  :", round(r2, 2))

    # Select model with lowest MAE
    if mae < best_mae:
        best_mae = mae
        best_model = model


# 8. Save the best model
joblib.dump(
    best_model,
    "models/eta_model.pkl"
)

print("\n==============================")
print("Best ETA model saved!")
print("==============================")
print("File: models/eta_model.pkl")