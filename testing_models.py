import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import ttest_rel

# Load models
models = {
    "predicted_maintenance_cost": load_model("fleet_nn_model.keras"),
    "predicted_speed": joblib.load("finalised_speed_model.pkl"),
    "predicted_fuel_efficiency": joblib.load("finalised_fuel_efficiency_model.pkl"),
    "risk_level": joblib.load("finalised_risk_level_regression_model.pkl")
}

# Load the scaler for the predicted_maintenance_cost model
scaler = joblib.load("scaler.pkl")

# Sample input
data = {
    "days_since_last_maintenance": 30,
    "maintenance_interval": 60,
    "last_maintenance_cost": 400,
    "engine_health": 85.0,
    "oil_quality": 70.0,
    "tire_wear": 20.0,
    "vehicle_age": 5.0,
    "maintenance_quality_score": 90.0,
    "unexpected_failures": 1
}

strategies = [0, 1, 2]
results = []

# Predict for each strategy
for strategy in strategies:
    # Prepare input data for each model
    maintenance_cost_input = pd.DataFrame([{
        "days_since_last_maintenance": data["days_since_last_maintenance"],
        "maintenance_strategy": strategy,
        "maintenance_interval": data["maintenance_interval"],
        "last_maintenance_cost": data["last_maintenance_cost"],
        "vehicle_age": data["vehicle_age"]
    }])

    speed_input = pd.DataFrame([{
        "engine_health": data["engine_health"],
        "oil_quality": data["oil_quality"],
        "maintenance_strategy": strategy,
        "tire_wear": data["tire_wear"],
        "days_since_last_maintenance": data["days_since_last_maintenance"]
    }])

    fuel_eff_input = pd.DataFrame([{
        "engine_health": data["engine_health"],
        "oil_quality": data["oil_quality"],
        "maintenance_strategy": strategy,
        "vehicle_age": data["vehicle_age"],
        "maintenance_interval": data["maintenance_interval"],
        "tire_wear": data["tire_wear"]
    }])

    risk_input = pd.DataFrame([{
        "days_since_last_maintenance": data["days_since_last_maintenance"],
        "vehicle_age": data["vehicle_age"],
        "maintenance_interval": data["maintenance_interval"],
        "maintenance_strategy": strategy,
        "maintenance_quality_score": data["maintenance_quality_score"],
        "unexpected_failures": data["unexpected_failures"]
    }])

    # Scale the input data for the maintenance cost model before prediction
    maintenance_cost_scaled_input = scaler.transform(maintenance_cost_input)
    maintenance_cost_pred = float(models["predicted_maintenance_cost"].predict(maintenance_cost_scaled_input)[0][0])

    # Predict other features
    speed_pred = float(models["predicted_speed"].predict(speed_input)[0])
    fuel_eff_pred = float(models["predicted_fuel_efficiency"].predict(fuel_eff_input)[0])
    risk_pred = float(models["risk_level"].predict(risk_input)[0])

    # Store results for each strategy
    results.append([strategy, maintenance_cost_pred, speed_pred, fuel_eff_pred, risk_pred])

# Convert results into a DataFrame for easy display and analysis
headers = ["Strategy", "Maintenance Cost", "Speed", "Fuel Efficiency", "Risk Level"]
df = pd.DataFrame(results, columns=headers)

# Composite Score (lower is better)
df["Score"] = df["Maintenance Cost"] + (df["Risk Level"] * 50) - (df["Speed"] * 10 + df["Fuel Efficiency"] * 20)

# Ranking (ascending for cost/risk/score, descending for speed/efficiency)
df["Maintenance Cost Rank"] = df["Maintenance Cost"].rank(method='min')
df["Speed Rank"] = df["Speed"].rank(method='min', ascending=False)
df["Fuel Efficiency Rank"] = df["Fuel Efficiency"].rank(method='min', ascending=False)
df["Risk Level Rank"] = df["Risk Level"].rank(method='min')
df["Score Rank"] = df["Score"].rank(method='min')

# Best strategy
best_strategy = int(df.loc[df["Score"].idxmin(), "Strategy"])

# Display result table
print("\n📊 Full Results Table:")
print(tabulate(df, headers="keys", tablefmt="fancy_grid", floatfmt=".2f"))

# Statistical tests for percentage changes
print("\n🔍 Percentage Changes Between Strategies:")
numeric = df[["Maintenance Cost", "Speed", "Fuel Efficiency", "Risk Level"]].to_numpy()
percent_change = np.abs(np.diff(numeric, axis=0) / numeric[:-1]) * 100
for i, metric in enumerate(headers[1:]):
    print(f"{metric}: {percent_change[:, i]} %")

# Paired t-test for significance
p_values = []
print("\n📈 Significance Tests (Paired T-Test):")
for i, metric in enumerate(headers[1:]):
    try:
        t_stat, p_val = ttest_rel(numeric[:-1, i], numeric[1:, i])
        p_values.append(p_val)
        print(f"{metric}: T-stat = {t_stat:.3f}, p = {p_val:.6f}")
    except ValueError:
        print(f"{metric}: Skipped (Insufficient variation)")

# JSON-style summary of the best strategy
print("\n📌 Summary:")
summary = {
    "best_strategy": best_strategy,
    "strategies": df[["Strategy", "Maintenance Cost", "Speed", "Fuel Efficiency", "Risk Level", "Score"]].to_dict(orient="records"),
    "note": "Lower score indicates better overall performance (cost-efficient, fast, fuel-saving, and safer)."
}
import json
print(json.dumps(summary, indent=2))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
metrics = ["Maintenance Cost", "Speed", "Fuel Efficiency", "Risk Level"]
colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99"]

for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax = axes[i // 2, i % 2]
    ax.bar(df["Strategy"], df[metric], color=color)
    ax.set_title(metric)
    ax.set_xlabel("Maintenance Strategy")
    ax.set_ylabel(metric)

plt.tight_layout()
plt.show()
