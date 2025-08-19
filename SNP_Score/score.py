import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# ===================== Read data =====================
# Load historical data from Excel file
# You can specify sheet name with sheet_name parameter
historical_df = pd.read_excel('historical_stock_data.xlsx', sheet_name=0)  

# Load current data from a separate Excel file
current_df = pd.read_excel('current_stock_data.xlsx', sheet_name=0)

# Alternative: Read specific sheets by name
# historical_df = pd.read_excel('historical_stock_data.xlsx', sheet_name='Historical')
# current_df = pd.read_excel('current_stock_data.xlsx', sheet_name='Current')

# Automatically identify feature columns (remove non-numeric columns)
non_feature_cols = ['Period Ending']
feature_cols = [col for col in historical_df.columns if col not in non_feature_cols]

# Handle missing values
historical_df[feature_cols] = historical_df[feature_cols].fillna(0)
current_df[feature_cols] = current_df[feature_cols].fillna(0)

# Filter historical data: rows starting with FY (if needed)
if 'Period Ending' in historical_df.columns:
    historical_df = historical_df[historical_df['Period Ending'].str.startswith('FY')].copy()

print(f"Historical data shape: {historical_df.shape}")
print(f"Current data shape: {current_df.shape}")
print(f"Feature columns: {len(feature_cols)} features")

# ===================== Choose target column =====================
# choose the target column for SNP Score calculation
target_col = 'Price gained'

# ===================== Calculate SNP Score =====================
def compute_snp_score(historical_df, current_df, feature_cols, target_col=target_col, alpha=0.1, score_scale=10):
    """
    Compute SNP Score for current data based on historical data
    
    Parameters:
    - historical_df: Historical training data
    - current_df: Current data to predict
    - feature_cols: List of feature column names
    - target_col: Target variable name
    - alpha: Lasso regularization parameter
    - score_scale: Scale for the final score (0 to score_scale)
    
    Returns:
    - current_df_with_score: Current data with SNP scores
    - weights_dict: Feature weights from Lasso regression
    """
    
    # Check if target column exists in historical data
    if target_col not in historical_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in historical data")
    
    # X is the feature matrix, y is the target variable
    X = historical_df[feature_cols]
    y = historical_df[target_col]
    
    print(f"Training on {len(X)} historical samples")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Lasso regression
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)

    # Get feature weights
    weights = lasso.coef_
    weights_dict = {col: w for col, w in zip(feature_cols, weights)}
    
    # Transform current data using the same scaler
    X_current = current_df[feature_cols]
    X_current_scaled = scaler.transform(X_current)

    # Calculate raw scores
    raw_scores = X_current_scaled @ weights
    
    # Normalize scores to 0-score_scale range
    if len(raw_scores) > 1:
        scores_normalized = score_scale * (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        # If only one sample, use raw score
        scores_normalized = raw_scores
    
    # Add scores to current dataframe
    current_df_with_score = current_df.copy()
    current_df_with_score['SNP_Score'] = scores_normalized
    
    return current_df_with_score, weights_dict

# Calculate SNP Score and weights
try:
    snp_df, snp_weights = compute_snp_score(historical_df, current_df, feature_cols, target_col)
    
    # Print results
    print("\n" + "="*50)
    print("FEATURE WEIGHTS (Lasso Regression):")
    print("="*50)
    for k, v in snp_weights.items():
        if abs(v) > 0.001:  # Only show significant weights
            print(f"{k:30}: {v:8.4f}")
    
    print("\n" + "="*50)
    print("SNP SCORES FOR CURRENT DATA:")
    print("="*50)
    
    # Display results with company names if available
    if 'Company' in snp_df.columns:
        result_cols = ['Company', 'SNP_Score']
    elif 'Period Ending' in snp_df.columns:
        result_cols = ['Period Ending', 'SNP_Score']
    else:
        result_cols = ['SNP_Score']
    
    print(snp_df[result_cols].to_string(index=False))
    
    # Save results to Excel
    snp_df.to_excel('snp_scores_results.xlsx', index=False)
    print(f"\nResults saved to 'snp_scores_results.xlsx'")
    
except Exception as e:
    print(f"Error: {e}")
    print("Please check your Excel files and column names.")
    print("Make sure you have installed openpyxl: pip install openpyxl")