import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Read Excel file
df = pd.read_excel("daily.xlsx")
series = df.reset_index()[["Date", "TSLA"]]
series.columns = ["ds", "y"]

# Modeling with Prophet
model = Prophet()
model.fit(series)

# Predicting future 5 days
future = model.make_future_dataframe(periods=5) # you can change the number of days (5) here
forecast = model.predict(future)

# Create the plot
fig = model.plot(forecast)
plt.title("TSLA Stock Price Prediction using Prophet")
plt.xlabel("Date")  # X-axis label
plt.ylabel("TSLA Price (USD)")  # Y-axis label
plt.grid(True, alpha=0.3)
plt.savefig("prophet_tsla_prediction.png")  # Save the plot as a file

# View the specific values of the predicted data
print("Last 10 predictions:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

# Example output:
# Last 10 predictions:
#             ds      yhat  yhat_lower  yhat_upper
# 244 2025-08-05 -0.279262   -5.616083    5.775348
# 245 2025-08-06  0.707872   -5.073391    6.438329
# 246 2025-08-07 -0.224844   -5.521869    5.604388
# 247 2025-08-08  0.101606   -5.571381    5.934705
# 248 2025-08-11  0.483027   -4.812194    6.284651
# 249 2025-08-12 -0.270038   -5.793829    5.431448
# 250 2025-08-13  0.717095   -4.581099    6.597150
# 251 2025-08-14 -0.215620   -5.744337    5.949095
# 252 2025-08-15  0.110830   -5.193971    5.884669
# 253 2025-08-16  2.220762   -3.651639    7.799672