import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Input your own ROA data for 10 stocks over n years
# Remember to put commas between the values
roa_data = np.array([
    [0.5324, 0.5742, 0.3855, 0.0817, 0.1720],  # Stock 1：NVDA
    [0.10, 0.11, 0.12, 0.13, 0.14], # Stock 2:
    [0.20, 0.18, 0.19, 0.21, 0.22], # Stock 3:
    [0.15, 0.16, 0.17, 0.18, 0.19], # Stock 4:
    [0.25, 0.24, 0.23, 0.22, 0.21], # Stock 5:
    [0.30, 0.31, 0.32, 0.33, 0.34], # Stock 6:
    [0.22, 0.23, 0.24, 0.25, 0.26], # Stock 7:
    [0.18, 0.17, 0.16, 0.15, 0.14], # Stock 8:
    [0.28, 0.29, 0.30, 0.31, 0.32], # Stock 9:
    [0.35, 0.36, 0.37, 0.38, 0.39]  # Stock10:
])

# Calculate the average ROA for each stock
avg_roa = np.mean(roa_data, axis=1)  # shape: (10,)
# Print the average ROA for each stock
print("Average ROA for each stock:")
for i, roa in enumerate(avg_roa, 1):
    print(f"Stock {i}: {roa:.4f}")

# Transpose the data to have stocks as rows and years as columns
roa_data_t = roa_data.T  # shape = (5, 10)

# Calculate the portfolio diversity based on correlation
# Using the correlation matrix to measure diversity
def portfolio_diversity(roa_data_t):
    df = pd.DataFrame(roa_data_t)
    corr_matrix = df.corr()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    avg_corr = upper_tri.stack().mean()
    diversity = 1 - avg_corr
    return diversity, corr_matrix

# Calculate the portfolio diversity
# The diversity value will be between 0 (no diversity) and 1 (maximum diversity)
# A higher value indicates a more diverse portfolio 
diversity, corr_matrix = portfolio_diversity(roa_data_t)
print("Portfolio Diversity (correlation-based):", diversity)

# =============== Visualization ===============
# Visualize the average ROA for each stock
plt.figure(figsize=(8, 6))
plt.bar([f"Stock{i+1}" for i in range(len(avg_roa))], avg_roa, color='skyblue')
plt.title("Average ROA per Stock")
plt.ylabel("Average ROA")
plt.xticks(rotation=45)
for i, v in enumerate(avg_roa):
    plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')
plt.tight_layout()
# Save the plot
plt.savefig("average_roa_per_stock.png") 

# Visualize the correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
plt.title("ROA Correlation Heatmap")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
# Save the correlation heatmap
plt.savefig("roa_correlation_heatmap.png")

