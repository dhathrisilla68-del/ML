import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
file_path = r"C:\Users\S.Dhathri\Desktop\ML\placement_predict_50k Dataset.csv"

df = pd.read_csv(file_path)

print('Dataset loaded successfully!')
print('Shape:', df.shape)
df.head()
print('Rows:', df.shape[0])
print('Columns:', df.shape[1])

df.info()
df.describe().T
print(df.isnull().sum())
numeric_columns = df.select_dtypes(
    include=['number']
).columns

categorical_columns = df.select_dtypes(
    include=['object', 'string']
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print('Total missing values after cleaning:', df.isnull().sum().sum())

X = df.drop('Salary Package', axis=1)
y = df['Salary Package']

print('X shape:', X.shape)
print('y shape:', y.shape)
categorical_columns = X.select_dtypes(
    include=['object', 'string']
).columns

encoder = LabelEncoder()

for column in categorical_columns:
    X[column] = encoder.fit_transform(X[column].astype(str))

print('Encoding completed!')
X.head()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print('Training data:', X_train.shape)
print('Testing data :', X_test.shape)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()

model.fit(X_train_scaled, y_train)

print('Model training completed!')
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

coefficients.sort_values('Coefficient', ascending=False)
print('Intercept:', model.intercept_)
y_pred = model.predict(X_test_scaled)

comparison = pd.DataFrame({
    'Actual Salary': y_test.values,
    'Predicted Salary': y_pred
})

comparison.head(10)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print('MODEL PERFORMANCE')
print('-------------------------')
print('MAE :', round(mae, 3))
print('RMSE:', round(rmse, 3))
print('R²  :', round(r2, 3))
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.4)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot([minimum, maximum], [minimum, maximum], linewidth=2)

plt.xlabel('Actual Salary Package')
plt.ylabel('Predicted Salary Package')
plt.title('Actual vs Predicted Salary')
plt.grid(True)
plt.show()
X_simple = df['CGPA'].values
y_simple = df['Salary Package'].values

# Standardize CGPA so Gradient Descent can learn smoothly
X_mean = X_simple.mean()
X_std = X_simple.std()
X_simple_scaled = (X_simple - X_mean) / X_std

print("Number of samples:", len(X_simple))
w_initial = 0.0
b_initial = 0.0

y_initial = w_initial * X_simple_scaled + b_initial

initial_cost = np.mean((y_initial - y_simple) ** 2)

print("Initial Weight:", w_initial)
print("Initial Intercept:", b_initial)
print("Initial Cost:", initial_cost)
x_line = np.linspace(
    X_simple_scaled.min(),
    X_simple_scaled.max(),
    100
)

y_initial_line = w_initial * x_line + b_initial

plt.figure(figsize=(8, 5))

plt.scatter(
    X_simple_scaled,
    y_simple,
    alpha=0.2,
    label="Actual Data"
)

plt.plot(
    x_line,
    y_initial_line,
    linewidth=3,
    label="Initial Line"
)

plt.xlabel("Standardized CGPA")
plt.ylabel("Salary Package")
plt.title("Regression Before Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()
def gradient_descent(X, y, learning_rate=0.01, epochs=100):
    w = 0.0
    b = 0.0

    n = len(y)

    cost_history = []
    weight_history = []
    bias_history = []

    for epoch in range(epochs):

        # 1. Make predictions
        y_pred = w * X + b

        # 2. Calculate error
        error = y_pred - y

        # 3. Calculate gradients
        dw = (2 / n) * np.sum(error * X)
        db = (2 / n) * np.sum(error)

        # 4. Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 5. Calculate new cost
        new_predictions = w * X + b
        cost = np.mean((new_predictions - y) ** 2)

        # Store values
        cost_history.append(cost)
        weight_history.append(w)
        bias_history.append(b)

    return w, b, cost_history, weight_history, bias_history
w, b, cost_history, weight_history, bias_history = gradient_descent(
    X_simple_scaled,
    y_simple,
    learning_rate=0.01,
    epochs=100
)

final_cost = cost_history[-1]

print("Initial Cost :", initial_cost)
print("Final Cost   :", final_cost)
print()
print("Initial Weight :", w_initial)
print("Final Weight   :", w)
print()
print("Initial Intercept :", b_initial)
print("Final Intercept   :", b)
improvement = initial_cost - final_cost
improvement_percent = (improvement / initial_cost) * 100

print("Initial Cost       :", round(initial_cost, 3))
print("Final Cost         :", round(final_cost, 3))
print("Cost Reduction     :", round(improvement, 3))
print("Improvement (%)    :", round(improvement_percent, 2), "%")
y_final_line = w * x_line + b

plt.figure(figsize=(9, 6))

plt.scatter(
    X_simple_scaled,
    y_simple,
    alpha=0.2,
    label="Actual Data"
)

plt.plot(
    x_line,
    y_initial_line,
    linewidth=3,
    label="Before Gradient Descent"
)

plt.plot(
    x_line,
    y_final_line,
    linewidth=3,
    label="After Gradient Descent"
)

plt.xlabel("Standardized CGPA")
plt.ylabel("Salary Package")
plt.title("Linear Regression Before and After Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(8, 5))

plt.plot(cost_history)

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Gradient Descent: Cost Reduction")
plt.grid(True)
plt.show()
plt.figure(figsize=(8, 5))

plt.plot(cost_history)

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Gradient Descent: Cost Reduction")
plt.grid(True)
plt.show()
simple_lr = LinearRegression()

simple_lr.fit(
    X_simple_scaled.reshape(-1, 1),
    y_simple
)

print("Gradient Descent Weight :", round(w, 4))
print("Linear Regression Weight:", round(simple_lr.coef_[0], 4))

print()

print("Gradient Descent Intercept :", round(b, 4))
print("Linear Regression Intercept:", round(simple_lr.intercept_, 4))
learning_rates = [0.001, 0.01, 0.1, 0.5]

plt.figure(figsize=(10, 6))

for lr in learning_rates:

    _, _, costs, _, _ = gradient_descent(
        X_simple_scaled,
        y_simple,
        learning_rate=lr,
        epochs=100
    )

    plt.plot(
        costs,
        label=f"Learning Rate = {lr}"
    )

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Impact of Learning Rate on Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()