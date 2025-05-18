import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv("fleet_maintenance_dataset_final_refined.csv")  # Change this to your actual dataset path

# Handle missing values
df = df.dropna(subset=["predicted_fuel_efficiency"])  # Remove rows where target is NaN
df.fillna(df.median(numeric_only=True), inplace=True)  # Fill missing feature values

# Select features
features = [
    "engine_health", "oil_quality", "maintenance_strategy", "vehicle_age", "maintenance_interval", "tire_wear"
]

X = df[features]
y = df["predicted_fuel_efficiency"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
regressor = RandomForestRegressor(random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Save the trained model
#joblib.dump(best_model, "finalised_fuel_efficiency_model.pkl")
print("Model saved successfully as fuel_efficiency_model.pkl")

# Make predictions on test set
y_pred = best_model.predict(X_test)

# Additional evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Best Model Parameters: {grid_search.best_params_}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")
print(f"R² Score: {r2:.4f}")

# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue", label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Perfect Fit")
plt.xlabel("Actual Fuel Efficiency")
plt.ylabel("Predicted Fuel Efficiency")
plt.title("Actual vs. Predicted Fuel Efficiency")
plt.legend()
plt.grid(True)
plt.show()

# Load the trained model for real-world prediction
loaded_model = joblib.load("fuel_efficiency_model.pkl")
print("Model loaded successfully!")

# Example unseen real-world data
unseen_data = pd.DataFrame({
    "engine_health": [85],
    "oil_quality": [78],
    "maintenance_strategy": [2],  # Ensure encoding matches your dataset
    "vehicle_age": [5],
    "maintenance_interval": [12000],
    "tire_wear": [30]
})

# Predict fuel efficiency for unseen data
predicted_fuel_efficiency = loaded_model.predict(unseen_data)
print(f"Predicted Fuel Efficiency: {predicted_fuel_efficiency[0]:.2f} km/L")

# Feature Importance Analysis
feature_importances = best_model.feature_importances_
feature_names = X.columns

# Sort features by importance
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_features = [feature_names[i] for i in sorted_indices]
sorted_importances = feature_importances[sorted_indices]

# Print feature importance values
print("\n🚀 Feature Importances for Fuel Efficiency Prediction:")
for feature, importance in zip(sorted_features, sorted_importances):
    print(f"{feature}: {importance:.4f}")

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(sorted_features[::-1], sorted_importances[::-1], color="mediumseagreen")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance - Predicted Fuel Efficiency")
plt.grid(True)
plt.tight_layout()
plt.show()
