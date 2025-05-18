import pandas as pd
import numpy as np

# Load original data
df = pd.read_csv("fleet_maintenance_dataset_final_refined.csv")

# Normalize features
df["vehicle_age_norm"] = df["vehicle_age"] / df["vehicle_age"].max()
df["days_since_last_maintenance_norm"] = df["days_since_last_maintenance"] / df["days_since_last_maintenance"].max()
df["maintenance_interval_norm"] = df["maintenance_interval"] / df["maintenance_interval"].max()
df["unexpected_failures_norm"] = df["unexpected_failures"] / df["unexpected_failures"].max()
df["quality_score_norm"] = df["maintenance_quality_score"] / 100

# Final balanced components
vehicle_age_component = 0.20 * (df["vehicle_age_norm"] ** 1.6)
days_since_maintenance_component = 0.20 * (df["days_since_last_maintenance_norm"] ** 1.4)
maintenance_interval_component = 0.18 * (df["maintenance_interval_norm"] ** 1.2)
unexpected_failure_component = 0.14 * (df["unexpected_failures_norm"] ** 1.1)
quality_component = 0.14 * ((1 - df["quality_score_norm"]) ** 1.3)

# Reduced strategy weight
strategy_weights = {0: 0.09, 1: -0.01, 2: -0.05}
strategy_component = df["maintenance_strategy"].map(strategy_weights)

# Mild interaction effect
strategy_interaction = 0.03 * df["days_since_last_maintenance_norm"] * strategy_component

# Final risk score computation
risk_score = (
    vehicle_age_component +
    days_since_maintenance_component +
    maintenance_interval_component +
    unexpected_failure_component +
    quality_component +
    strategy_component +
    strategy_interaction +
    np.random.normal(0, 0.01, len(df))  # slight noise
)

# Scale to range 1–100
df["risk_level"] = (risk_score * 100).clip(1, 100).round(2)

# Drop temp columns
df.drop(columns=[col for col in df.columns if col.endswith("_norm")], inplace=True)

# Save refined version
df.to_csv("fleet_risk_regression_dataset_final_balanced.csv", index=False)
print("✅ Final dataset saved as: fleet_risk_regression_dataset_final_balanced.csv")
