import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, Input
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.manifold import TSNE
import joblib

# Load dataset
data = pd.read_csv("refined_fleet_maintenance_dataset_2.csv")

# Define features and target
X = data[["days_since_last_maintenance", "maintenance_strategy", "maintenance_interval",
          "last_maintenance_cost", "vehicle_age"]]
y = data["predicted_maintenance_cost"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Build optimized neural network
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(256), BatchNormalization(), LeakyReLU(), Dropout(0.2),
    Dense(128), BatchNormalization(), LeakyReLU(), Dropout(0.2),
    Dense(64), BatchNormalization(), LeakyReLU(), Dropout(0.2),
    Dense(32), BatchNormalization(), LeakyReLU(),
    Dense(1)
])

# Compile model
model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4),
              loss="huber", metrics=["mae"])

# Callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)

# Train the model
history = model.fit(X_train_scaled, y_train,
                    validation_data=(X_test_scaled, y_test),
                    epochs=200, batch_size=32,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)

# Evaluate performance
y_pred_train = model.predict(X_train_scaled).flatten()
y_pred_test = model.predict(X_test_scaled).flatten()

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

print(f"\nOptimized Neural Network Performance:")
print(f"  Train R² Score: {train_r2:.2f}")
print(f"  Test R² Score: {test_r2:.2f}")
print(f"  MAE: {mae:.2f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAPE: {mape:.2f}%")

# Save model
model.save("fleet_nn_model.keras")
print("\nOptimized Neural Network Model and Preprocessor saved!")

# Load model
loaded_model = keras.models.load_model("fleet_nn_model.keras")
scaler = joblib.load("scaler.pkl")
test_predictions = loaded_model.predict(X_test_scaled).flatten()
final_test_r2 = r2_score(y_test, test_predictions)
print(f"\nLoaded Model Test R²: {final_test_r2:.2f}")


# Ensure same column order as during training
feature_columns = ["days_since_last_maintenance", "maintenance_strategy", 
                   "maintenance_interval", "last_maintenance_cost", "vehicle_age"]

# Real-time base values
real_time_base = {
    "days_since_last_maintenance": 20,
    "maintenance_interval": 30,
    "last_maintenance_cost": 100,
    "vehicle_age": 2
}


print("\nReal-Time Predictions for All Maintenance Strategies:")
for strategy in [0, 1, 2]:
    # Create DataFrame in correct column order
    real_time_input = pd.DataFrame([{
        "days_since_last_maintenance": real_time_base["days_since_last_maintenance"],
        "maintenance_strategy": strategy,
        "maintenance_interval": real_time_base["maintenance_interval"],
        "last_maintenance_cost": real_time_base["last_maintenance_cost"],
        "vehicle_age": real_time_base["vehicle_age"]
    }])[feature_columns]  # <- This ensures correct order

    # Scale and predict
    real_time_input_scaled = scaler.transform(real_time_input)
    predicted_cost = loaded_model.predict(real_time_input_scaled).flatten()[0]
    print(f"  Strategy {strategy}: Predicted Maintenance Cost = ${predicted_cost:.2f}")


# t-SNE visualization
print("\nGenerating t-SNE plot...")
X_full_scaled = scaler.transform(X)
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
X_tsne = tsne.fit_transform(X_full_scaled)

tsne_df = pd.DataFrame({
    "TSNE-1": X_tsne[:, 0],
    "TSNE-2": X_tsne[:, 1],
    "Predicted Cost": y
})

plt.figure(figsize=(10, 6))
sns.scatterplot(data=tsne_df, x="TSNE-1", y="TSNE-2", hue="Predicted Cost",
                palette="coolwarm", s=60, edgecolor='k')
plt.title("t-SNE Visualization of Maintenance Cost Prediction Features")
plt.legend(title="Predicted Cost", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
