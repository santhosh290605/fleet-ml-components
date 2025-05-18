import pandas as pd

# Load your dataset (adjust the file path accordingly)
# Assuming your dataset is a CSV file
df = pd.read_csv('fleet_maintenance_dataset_final.csv')

# Display basic statistics of the dataset
print("Basic statistics of the dataset:")
print(df.describe())

# Check the column containing the predicted maintenance cost
# Adjust column name if necessary
maintenance_cost_column = 'predicted_maintenance_cost'  # Replace with the correct column name

# Get the minimum and maximum values of the predicted maintenance cost
min_value = df[maintenance_cost_column].min()
max_value = df[maintenance_cost_column].max()

# Print the min and max values
print(f"Minimum predicted maintenance cost: {min_value}")
print(f"Maximum predicted maintenance cost: {max_value}")

# You may also want to check the distribution (optional, useful for normalization strategy)
import matplotlib.pyplot as plt
df[maintenance_cost_column].hist(bins=50)
plt.title('Distribution of Predicted Maintenance Cost')
plt.xlabel('Maintenance Cost')
plt.ylabel('Frequency')
plt.show()

# Additional step to visualize how skewed or balanced the distribution is
# This helps in understanding if a log transformation or other scaling methods are needed
