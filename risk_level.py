import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("fleet_risk_regression_dataset_final_balanced.csv")

# Features and target
feature_cols = [
    "days_since_last_maintenance",
    "vehicle_age",
    "maintenance_interval",
    "maintenance_strategy",
    "maintenance_quality_score",
    "unexpected_failures"
]
X = df[feature_cols]
y = df["risk_level"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler_risk.pkl")  # Save the scaler

# Train Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test_scaled)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n✅ Model Evaluation")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save the model
joblib.dump(model, "finalised_risk_level_regression_model.pkl")

# ---------- VISUALIZATIONS ---------- #

# 1. Feature Importance
plt.figure(figsize=(8, 5))
sns.barplot(x=model.feature_importances_, y=feature_cols, palette="viridis")
plt.title("Feature Importances (Random Forest Regressor)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# 2. Predicted vs Actual Scatter
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel("Actual Risk Level")
plt.ylabel("Predicted Risk Level")
plt.title("Actual vs Predicted Risk Level")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Residuals Distribution
residuals = y_test - y_pred
plt.figure(figsize=(7, 4))
sns.histplot(residuals, kde=True, color="orange", bins=30)
plt.title("Error Distribution (Residuals)")
plt.xlabel("Actual - Predicted")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Metrics Summary
metrics = {"MAE": mae, "RMSE": rmse, "R² Score": r2}
plt.figure(figsize=(6, 4))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="magma")
plt.title("Risk Level Prediction Metrics")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

# ---------- Strategy-wise Prediction Test ---------- #

print("\n🚗 Testing Risk Predictions for Different Maintenance Strategies:")
base_input = {
    "days_since_last_maintenance": 60,
    "vehicle_age": 5,
    "maintenance_interval": 90,
    "maintenance_quality_score": 85,
    "unexpected_failures": 2,
}

for strategy in [0, 1, 2]:  # 0 = delayed, 1 = immediate, 2 = scheduled
    input_dict = {
        "days_since_last_maintenance": base_input["days_since_last_maintenance"],
        "vehicle_age": base_input["vehicle_age"],
        "maintenance_interval": base_input["maintenance_interval"],
        "maintenance_strategy": strategy,
        "maintenance_quality_score": base_input["maintenance_quality_score"],
        "unexpected_failures": base_input["unexpected_failures"]
    }
    input_df = pd.DataFrame([input_dict])[feature_cols]
    input_scaled = scaler.transform(input_df)
    risk_pred = model.predict(input_scaled)[0]

    strategy_label = {0: "Delayed", 1: "Immediate", 2: "Scheduled"}[strategy]
    print(f"{strategy_label:>10} Strategy → Predicted Risk Level: {risk_pred:.2f}")
