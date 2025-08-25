import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 读取 Excel
df = pd.read_excel("daily.xlsx")

# 显式指定日期格式并转换
df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d', errors='coerce')

# 检查日期转换结果
print("Date column after conversion:")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Number of valid dates: {df['Date'].notna().sum()}")

# 删除无效日期行
df = df.dropna(subset=['Date'])

# 设置 Date 为时间索引
df.set_index('Date', inplace=True)

# 确保按时间排序
df.sort_index(inplace=True)

# 选取一支股票，比如 TSLA
tsla = df['TSLA'].dropna()

print(f"TSLA data points: {len(tsla)}")
print(f"TSLA date range: {tsla.index.min()} to {tsla.index.max()}")

# 可视化历史数据
plt.figure(figsize=(12, 6))
tsla.plot(title="TSLA Price History", color='blue')
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------
# 1. 确定 ARIMA(p,d,q) 参数
# ------------------------
# ACF/PACF 图分析
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(tsla.dropna(), lags=20, ax=ax1, title="Autocorrelation Function (ACF)")
plot_pacf(tsla.dropna(), lags=20, ax=ax2, title="Partial Autocorrelation Function (PACF)")
plt.tight_layout()
plt.show()

# ------------------------
# 2. 建立 ARIMA 模型
# ------------------------
try:
    # 首先检查数据的平稳性
    print("Building ARIMA model...")
    model = ARIMA(tsla, order=(2,1,2))  # (p,d,q)
    arima_result = model.fit()
    print(arima_result.summary())
    
    # ------------------------
    # 3. 预测未来 5 天
    # ------------------------
    forecast_steps = 5
    forecast = arima_result.forecast(steps=forecast_steps)
    
    # 创建未来日期索引
    last_date = tsla.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                periods=forecast_steps, freq='D')
    
    # 创建预测结果DataFrame
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': forecast
    })
    
    print("\nFuture Predictions:")
    print(forecast_df)
    
    # ------------------------
    # 4. 可视化结果
    # ------------------------
    plt.figure(figsize=(14, 6))
    
    # 绘制最近50天的历史数据
    recent_data = tsla[-50:]
    plt.plot(recent_data.index, recent_data.values, 
             label="Historical Prices", color='blue', linewidth=2)
    
    # 绘制预测
    plt.plot(future_dates, forecast, 
             label="ARIMA Forecast", color='red', 
             linestyle="--", marker='o', linewidth=2)
    
    plt.title("TSLA Price Forecast using ARIMA Model", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("tsla_arima_forecast.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存预测结果
    forecast_df.to_excel("tsla_arima_predictions.xlsx", index=False)
    print(f"\nPredictions saved to 'tsla_arima_predictions.xlsx'")
    
except Exception as e:
    print(f"Error in ARIMA modeling: {e}")
    print("Try adjusting the ARIMA parameters (p,d,q)")