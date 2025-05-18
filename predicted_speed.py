import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your refined dataset
df = pd.read_csv("fleet_maintenance_dataset_finaled.csv")

# Drop rows where target is NaN
df = df.dropna(subset=["predicted_speed"])

# Fill missing values for numeric columns
df.fillna(df.median(numeric_only=True), inplace=True)

# Ensure 'maintenance_strategy' is treated as a categorical feature
df['maintenance_strategy'] = df['maintenance_strategy'].astype(int)

# Define feature columns and target
features = [
    "engine_health", "oil_quality", 
    "maintenance_strategy", "tire_wear", "days_since_last_maintenance"
]
X = df[features]
y = df["predicted_speed"]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
#joblib.dump(model, "finalised_speed_model1.pkl")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Updated Model Performance on Test Data:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Feature importances
importances = model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
sorted_features = [features[i] for i in sorted_indices]
sorted_importances = importances[sorted_indices]

print("\n🚀 Updated Feature Importances for Speed Prediction:")
for f, i in zip(sorted_features, sorted_importances):
    print(f"{f}: {i:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(sorted_features[::-1], sorted_importances[::-1], color="skyblue")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance - Predicted Speed")
plt.grid(True)
plt.show()

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue", label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Perfect Fit")
plt.xlabel("Actual Speed")
plt.ylabel("Predicted Speed")
plt.title("Model Performance: Actual vs. Predicted Speed")
plt.legend()
plt.grid(True)
plt.show()
