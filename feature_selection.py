import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("fleet_maintenance_dataset_updated.csv")

# Convert risk_level to numerical if it's categorical
if df["risk_level"].dtype == "object":
    df["risk_level"] = df["risk_level"].astype("category").cat.codes  # Convert to numerical categories

# Compute Spearman correlation
correlation_matrix = df.corr(method="spearman")

# Extract correlation values for risk_level
risk_correlation = correlation_matrix["risk_level"].sort_values(ascending=False)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(risk_correlation.to_frame(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Spearman Correlation with Risk Level")
plt.show()

# Print top correlated features
print("Top Features Correlated with Risk Level:")
print(risk_correlation)
