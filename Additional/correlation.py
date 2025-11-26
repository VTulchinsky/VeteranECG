import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Excel file
file_path = "combined_dataset-selected-0-prev.xlsx" #"combined_dataset-selected.xlsx" #"combined_dataset-selected-db.xlsx"
df = pd.read_excel(file_path)

# Drop the target column
df = df.drop(columns=["STRESS"])
# Drop completely empty columns and rows
df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)

# Compute pairwise correlations while ignoring empty cells
correlation_matrix = df.corr(numeric_only=True)
correlation_matrix*=10

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".0f", linewidths=0.5, square=True)
plt.title("Pairwise Correlation Matrix")
plt.show()