import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("finalised_speed_model.pkl")

# ✅ Define the exact feature order used during training
feature_columns = [
    'engine_health',
    'oil_quality',
    'maintenance_strategy',
    'tire_wear',
    'days_since_last_maintenance',
    
]

# ✅ Define real-time input values (replace these as needed)
real_time_input = {
    'engine_health': 85,
    'tire_wear': 23,
    'oil_quality': 70,
    'days_since_last_maintenance': 15
}

# Generate input variants with all 3 maintenance strategies
input_data = []
for strategy in [0, 1, 2]:
    entry = real_time_input.copy()
    entry['maintenance_strategy'] = strategy
    input_data.append(entry)

# Convert to DataFrame and enforce column order
input_df = pd.DataFrame(input_data)[feature_columns]

# Ensure numeric types
input_df = input_df.astype(float)

# ✅ Predict using the model
predictions = model.predict(input_df)

# Display results
strategies = {0: "⛔ Delayed", 1: "✅ Immediate", 2: "🕒 Scheduled"}
print("\n🚀 Real-Time Speed Predictions Under Different Maintenance Strategies:")
for i, strategy in enumerate([0, 1, 2]):
    print(f"{strategies[strategy]:<15} → Predicted Speed: {predictions[i]:.2f} km/h")
