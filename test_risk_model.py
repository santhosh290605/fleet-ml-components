import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessing tools
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")

# Load trained models
log_reg = joblib.load("logistic_regression_model.pkl")
rf_clf = joblib.load("random_forest_model.pkl")
xgb_clf = joblib.load("xgboost_model.pkl")  # Ensure this model is trained and saved

# Feature names
feature_names = ["days_since_last_maintenance", "vehicle_age", "maintenance_interval",
                 "maintenance_strategy", "maintenance_quality_score", "unexpected_failures"]

# Generate synthetic real-time data based on realistic distributions
def generate_real_time_data(num_samples=10):
    np.random.seed(42)  # For reproducibility
    
    synthetic_data = {
        "days_since_last_maintenance": np.random.randint(1, 365, num_samples),
        "vehicle_age": np.random.randint(1, 15, num_samples),
        "maintenance_interval": np.random.randint(30, 180, num_samples),
        "maintenance_strategy": np.random.choice([0, 1, 2], num_samples),  # Assuming 3 strategies
        "maintenance_quality_score": np.random.uniform(0.5, 1.0, num_samples),
        "unexpected_failures": np.random.randint(0, 5, num_samples),
    }
    
    # Convert to DataFrame
    df_synthetic = pd.DataFrame(synthetic_data)
    
    # Apply preprocessing: Handle missing values and scale data
    X_synthetic_imputed = imputer.transform(df_synthetic)
    X_synthetic_scaled = scaler.transform(X_synthetic_imputed)

    return df_synthetic, X_synthetic_scaled

# Generate and display synthetic real-time data
df_real_time, X_real_time_scaled = generate_real_time_data(10)
print("Simulated Real-Time Data:")
print(df_real_time)

# Simulated true labels (for evaluation purposes, assuming a reasonable distribution)
y_true = np.random.choice([0, 1, 2], size=len(df_real_time))  # Fake true values for evaluation

# Predict using each model
y_pred_log_reg = log_reg.predict(X_real_time_scaled)
y_pred_rf = rf_clf.predict(df_real_time)  # Random Forest doesn’t need scaling
y_pred_xgb = xgb_clf.predict(df_real_time)

# Add predictions to DataFrame
df_real_time["Predicted Risk (LogReg)"] = y_pred_log_reg
df_real_time["Predicted Risk (RF)"] = y_pred_rf
df_real_time["Predicted Risk (XGBoost)"] = y_pred_xgb

# Display predictions
print("\nPredictions on Real-Time Data:")
print(df_real_time)

# Evaluate model performance on the synthetic test labels
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Model Performance:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# Evaluate each model
evaluate_model("Logistic Regression", y_true, y_pred_log_reg)
evaluate_model("Random Forest", y_true, y_pred_rf)
evaluate_model("XGBoost", y_true, y_pred_xgb)

# Save the synthetic data with predictions for analysis
df_real_time.to_csv("real_time_predictions.csv", index=False)
print("\nPredictions saved to 'real_time_predictions.csv'.")
