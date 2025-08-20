import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Input your own ROA data for 10 stocks over n years
# Remember to put commas between the values
stock_names = [
    "NFLX", "NVDA", "V", "JNJ", "COST",
    "AMZN", "GOOGL", "MSFT", "TSLA", "JPM"
]

roa_data = np.array([
    [0.1505, 0.1272, 0.0893, 0.0756, 0.0923, 0.0782],  # Stock 1：NFLX
    [0.5324, 0.5742, 0.3855, 0.0817, 0.1720, 0.1280],  # Stock 2: NVDA
    [0.1705, 0.1617, 0.1557, 0.1461, 0.1206, 0.1150],  # Stock 3: V
    [0.0761, 0.0813, 0.0840, 0.0751, 0.0755, 0.0762],  # Stock 4: JNJ
    [0.0879, 0.0836, 0.0798, 0.0801, 0.0739, 0.0673],  # Stock 5: COST
    [0.0770, 0.0744, 0.0465, 0.0189, 0.0419, 0.0524],  # Stock 6: AMZN
    [0.1679, 0.1674, 0.1373, 0.1291, 0.1449, 0.0865],  # Stock 7: GOOGL
    [0.1420, 0.1420, 0.1480, 0.1424, 0.1492, 0.1376],  # Stock 8: MSFT
    [0.0291, 0.0419, 0.0588, 0.1185, 0.0713, 0.0282],  # Stock 9: TSLA
    [0.0130, 0.0148, 0.0131, 0.0102, 0.0136, 0.0096]   # Stock 10: JPM
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
plt.bar(stock_names, avg_roa, color='skyblue')
plt.title("Average ROA per Stock")
plt.ylabel("Average ROA")
plt.xticks(rotation=45)
for i, v in enumerate(avg_roa):
    plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')
plt.tight_layout()
# Save the plot
plt.savefig("average_roa_per_stock.png") 

# Visualize the correlation matrix
# Visualize the correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, 
            cbar_kws={"shrink": .8}, 
            xticklabels=stock_names, 
            yticklabels=stock_names)
plt.title("ROA Correlation Heatmap")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
# Save the correlation heatmap
plt.savefig("roa_correlation_heatmap.png")
# plt.show()

