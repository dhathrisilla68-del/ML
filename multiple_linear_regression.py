import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================

print("Libraries imported successfully.")


# =========================================================
# 2. LOAD DATASET
# =========================================================

file_path = r"C:\Users\S.Dhathri\Desktop\ML\placement_predict_50k Dataset.csv"

df = pd.read_csv(file_path)

print("\nDataset Shape:", df.shape)


# =========================================================
# 3. DISPLAY FIRST 5 ROWS
# =========================================================

print("\nFirst 5 rows:")
print(df.head())


# =========================================================
# 4. COLUMN NAMES
# =========================================================

print("\nColumn Names:")
print(df.columns.tolist())


# =========================================================
# 5. DATA TYPES
# =========================================================

print("\nData Types:")
print(df.dtypes)


# =========================================================
# 6. MISSING VALUE CHECK
# =========================================================

print("\nMISSING VALUE CHECK")

missing_values = df.isnull().sum()

print(missing_values)

total_missing = df.isnull().sum().sum()

print("\nTotal Missing Values:", total_missing)

missing_percentage = (df.isnull().sum() / len(df)) * 100

print("\nMissing Value Percentage:")
print(missing_percentage)


# =========================================================
# 7. DUPLICATE CHECK
# =========================================================

duplicate_count = df.duplicated().sum()

print("\nDUPLICATE CHECK")

print("Number of duplicate rows:", duplicate_count)


# =========================================================
# 8. DEFINE INPUT (X) AND OUTPUT (y)
# =========================================================

# Salary is the value we want to predict
print("\nAVAILABLE COLUMNS:")
print(df.columns.tolist())
X = df.drop("Salary Package", axis=1)
y = df["Salary Package"]

print("\nTarget column:")
print("Salary")


# =========================================================
# 9. REMOVE STUDENT ID
# =========================================================

if "StudentID" in X.columns:
    X = X.drop("StudentID", axis=1)

print("\nInput columns after removing StudentID:")
print(X.columns.tolist())


# =========================================================
# 10. FIND CATEGORICAL AND NUMERICAL COLUMNS
# =========================================================

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()


print("\nCOLUMNS")

print("\nCategorical Columns:")
print(categorical_columns)

print("\nNumerical Columns:")
print(numerical_columns)


# =========================================================
# 11. NUMERICAL PREPROCESSING
# =========================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

print("\nNumerical preprocessing pipeline created.")


# =========================================================
# 12. CATEGORICAL PREPROCESSING
# =========================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

print("Categorical preprocessing pipeline created.")


# =========================================================
# 13. CREATE PREPROCESSOR
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)

print("Preprocessor created successfully.")


# =========================================================
# 14. CREATE LINEAR REGRESSION MODEL
# =========================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)

print("Multiple Linear Regression model created.")


# =========================================================
# 15. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAIN TEST SPLIT")

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# =========================================================
# 16. TRAIN THE MODEL
# =========================================================

model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully.")


# =========================================================
# 17. MAKE PREDICTIONS
# =========================================================

y_pred = model.predict(X_test)

print("\nNumber of predictions:", len(y_pred))

print("\nFirst 10 predictions:")
print(y_pred[:10])


# =========================================================
# 18. MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


# =========================================================
# 19. DISPLAY MODEL PERFORMANCE
# =========================================================

print("\nMODEL PERFORMANCE")

print("Mean Absolute Error (MAE)  :", mae)

print("Mean Squared Error (MSE)   :", mse)

print("Root Mean Squared Error (RMSE):", rmse)

print("R² Score                   :", r2)


# =========================================================
# 20. ACTUAL VS PREDICTED VALUES
# =========================================================

results = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": y_pred
})

print("\nACTUAL VS PREDICTED")

print(results.head(20))


# =========================================================
# 21. ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

minimum = min(y_test.min(), y_pred.min())

maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Salary")

plt.ylabel("Predicted Salary")

plt.title("Actual vs Predicted Salary")

plt.grid(True)

plt.show()