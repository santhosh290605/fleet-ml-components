import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Load Data --------------------
data = pd.read_csv("fleet_maintenance_dataset_final.csv")

# -------------------- Debugging Vehicle Age Column --------------------
# Check if 'vehicle_age' is a valid column
if 'vehicle_age' not in data.columns:
    print("'vehicle_age' column is missing from the dataset.")
else:
    print(f"Shape of 'vehicle_age': {data['vehicle_age'].shape}")
    print(f"Data type of 'vehicle_age': {data['vehicle_age'].dtype}")

# -------------------- Check for NaN and Infinite values --------------------
# Check for any NaN or infinite values in 'vehicle_age' column
if data['vehicle_age'].isnull().any() or np.isinf(data['vehicle_age']).any():
    print("There are NaN or Infinite values in 'vehicle_age'. Cleaning data.")
    data = data.dropna(subset=['vehicle_age'])  # Remove rows with NaN
    data = data[np.isfinite(data['vehicle_age'])]  # Remove rows with infinite values

# -------------------- Feature Engineering --------------------
# Interaction Features (engine health, tire wear, and oil quality)
data['engine_health_tire_wear'] = data['engine_health'] * data['tire_wear']
data['engine_health_oil_quality'] = data['engine_health'] * data['oil_quality']
data['tire_wear_oil_quality'] = data['tire_wear'] * data['oil_quality']

# Polynomial Features (degree 2 interactions for non-linear relationships)
poly = PolynomialFeatures(degree=2, include_bias=False)
interaction_features = poly.fit_transform(data[['days_since_last_maintenance', 'maintenance_interval', 
                                                'last_maintenance_cost', 'vehicle_age']])

# Get feature names from polynomial features
interaction_feature_names = poly.get_feature_names_out(['days_since_last_maintenance', 'maintenance_interval', 
                                                        'last_maintenance_cost', 'vehicle_age'])

# Add interaction features to the dataframe
interaction_df = pd.DataFrame(interaction_features, columns=interaction_feature_names)
data = pd.concat([data, interaction_df], axis=1)

# -------------------- Log Transformation --------------------
# Apply the log transformation to 'vehicle_age' and 'maintenance_interval'
data['log_vehicle_age'] = np.log(data['vehicle_age'] + 1)  # Adding 1 to avoid log(0) issues
data['log_maintenance_interval'] = np.log(data['maintenance_interval'] + 1)

# Create a ratio feature (last maintenance cost / maintenance interval)
data['last_maintenance_cost_ratio'] = data['last_maintenance_cost'] / data['maintenance_interval']

# -------------------- Feature Scaling --------------------
# Scaling numerical features (we'll scale all features except the target variable)
numerical_features = ['days_since_last_maintenance', 'maintenance_interval', 'last_maintenance_cost', 
                      'engine_health', 'tire_wear', 'oil_quality', 'predicted_speed', 
                      'predicted_fuel_efficiency', 'predicted_maintenance_cost', 'vehicle_age', 
                      'unexpected_failures', 'maintenance_quality_score', 'engine_health_tire_wear', 
                      'engine_health_oil_quality', 'tire_wear_oil_quality', 'log_vehicle_age', 
                      'log_maintenance_interval', 'last_maintenance_cost_ratio']

# Including 'maintenance_strategy' as a feature for prediction
numerical_features_with_strategy = numerical_features + ['maintenance_strategy']

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply scaling
data[numerical_features_with_strategy] = scaler.fit_transform(data[numerical_features_with_strategy])

# -------------------- Checking Correlations --------------------
# Checking for strong correlations between features
correlation_matrix = data[numerical_features_with_strategy].corr()

# Plotting the correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# -------------------- Prepare Data for Model --------------------
# Assuming predicted_maintenance_cost is the target variable
X = data.drop(columns=["maintenance_strategy", "predicted_maintenance_cost", "vehicle_id", "day", "risk_level"])
y = data["predicted_maintenance_cost"]

# -------------------- Save the Refined Data --------------------
data.to_csv("refined_fleet_maintenance_refined.csv", index=False)
print("Refined dataset with engineered features saved as 'refined_fleet_maintenance_refined.csv'")

# -------------------- Final Notes --------------------
# This data now contains interaction features, non-linear transformations, scaling, and the necessary maintenance_strategy.
# You can now proceed to train your model on this refined dataset.
