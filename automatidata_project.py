# === NYC Taxi Fare Prediction using Multiple Linear Regression ===

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# === Load Dataset ===
df = pd.read_csv("data/taxi_fare_data.csv")  # Update with your actual file name

# === Inspect Data ===
print("Dataset shape:", df.shape)
print(df.info())
print("Missing values:", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

# === Describe Data ===
print(df.describe())

# === Convert Pickup and Dropoff Times to Datetime ===
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')

# === Create Duration Column ===
df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']) / np.timedelta64(1, 'm')

# === Drop Outliers ===
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 200)]
df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 100)]
df = df[(df['duration'] > 0) & (df['duration'] < 180)]

# === Feature Selection ===
features = ['trip_distance', 'duration']
X = df[features]
y = df['fare_amount']

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Feature Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train Linear Regression Model ===
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# === Predictions and Evaluation ===
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# === Visualization ===
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted Fare")
plt.grid(True)
plt.show()
