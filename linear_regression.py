import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Libraries imported successfully.")

file_path =r"C:\Users\S.Dhathri\Desktop\ML\placement_predict_50k Dataset.csv"


df = pd.read_csv(file_path)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

X = df[["CGPA"]]

y = df["Salary Package"]

print("\nInput feature:")
print(X.head())

print("\nTarget:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

model = LinearRegression()

model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully.")

y_pred = model.predict(X_test)

print("\nFirst 10 predictions:")
print(y_pred[:10])

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)

print("\nSlope:", model.coef_[0])
print("Intercept:", model.intercept_)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_test,
    y_test,
    alpha=0.5,
    label="Actual"
)

plt.plot(
    X_test,
    y_pred,
    linestyle="--",
    label="Regression Line"
)

plt.xlabel("CGPA")
plt.ylabel("Salary Package")
plt.title("CGPA vs Salary Package")

plt.legend()
plt.grid(True)
plt.show()