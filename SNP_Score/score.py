import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# ----------------------
# Preprocessing Features
# ----------------------
def preprocess_features(df, drop_threshold=0.5):
    """
    process features: drop columns with too many missing values or low variance, fill missing values
    """
    # Convert "-" to NaN
    df = df.replace("-", np.nan)

    # 1. Drop columns with too many missing values
    df = df.dropna(thresh=len(df) * drop_threshold, axis=1)

    # 2. Fill missing values using forward/backward fill (time series friendly)
    df = df.fillna(method="ffill").fillna(method="bfill")

    # 3. Drop columns with low variance (essentially no information)
    df = df.loc[:, df.var() > 1e-6]

    return df

# ----------------------
# Read historical data
# ----------------------
historical = pd.read_excel("historical_stock_data.xlsx", na_values=["-"])

X = historical.drop(columns=["Stock Name", "Period Ending", "Price Gained"])
y = historical["Price Gained"]

# Preprocess features
X_clean = preprocess_features(X)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# ----------------------
# Lasso Regression
# ----------------------
lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)

# Extract coefficients
coef_df = pd.DataFrame({
    "Feature": X_clean.columns,
    "Weight": lasso.coef_
}).sort_values(by="Weight", ascending=False)

print("Lasso learned weights (non-zero):")
print(coef_df[coef_df["Weight"] != 0])

# ----------------------
# Predict current data
# ----------------------
current = pd.read_excel("current_stock_data.xlsx", na_values=["-"])
X_current = current.drop(columns=["Stock Name", "Period Ending"])

# Apply the same preprocessing to current data (ensure consistent columns)
X_current = X_current[X_clean.columns]   # Only keep columns that were in the training set
X_current = preprocess_features(X_current)

X_current_scaled = scaler.transform(X_current)

pred_price_gain = lasso.predict(X_current_scaled)
current["Predicted Price Gained"] = pred_price_gain

# SNP Score (0–10 normalization)
min_val, max_val = pred_price_gain.min(), pred_price_gain.max()
current["SNP Score"] = 10 * (pred_price_gain - min_val) / (max_val - min_val)

print(current[["Stock Name", "Predicted Price Gained", "SNP Score"]])

# ----------------------
# Visualization
# ----------------------

# Figure 1: Feature Importance (Lasso Weights), only non-zero weights!
plt.figure(figsize=(12, 6))
nz_coef = coef_df[coef_df["Weight"] != 0]
plt.barh(nz_coef["Feature"], nz_coef["Weight"], color="skyblue")
plt.axvline(0, color="black", linewidth=1)
plt.title("Feature Importance (Lasso Weights)", fontsize=14)
plt.xlabel("Weight")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance_lasso.png") 

# Figure 2: SNP Score Ranking
plt.figure(figsize=(10, 6))
current_sorted = current.sort_values("SNP Score", ascending=False)
plt.bar(current_sorted["Stock Name"], current_sorted["SNP Score"], color="lightgreen")
plt.title("SNP Score Ranking (0-10)", fontsize=14)
plt.ylabel("SNP Score")
plt.ylim(0, 10)
plt.tight_layout()
plt.savefig("snp_score_ranking.png")

# ----------------------
# Export Results into Excel
# ----------------------
current[["Stock Name", "Predicted Price Gained", "SNP Score"]].to_excel("snp_score_result.xlsx", index=False)
print("Results exported to snp_score_result.xlsx")

# Sample output:
#   Stock Name  Predicted Price Gained  SNP Score
# 0       NFLX                0.761320   7.671456
# 1       NVDA                0.066773   0.566674
# 2          V                0.128370   1.196778
# 3        JNJ                0.077217   0.673506
# 4       COST                0.314422   3.099971
# 5       AMZN                0.209280   2.024430
# 6      GOOGL                0.011376   0.000000
# 7       MSFT                0.162609   1.547012
# 8       TSLA                0.988954  10.000000
# Results exported to snp_score_result.xlsx