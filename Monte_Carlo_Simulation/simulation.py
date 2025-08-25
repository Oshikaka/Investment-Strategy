import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
np.random.seed(42)
n_simulations = 500      # number of Monte Carlo runs
n_days = 252             # 1 year of trading days
# initial portfolio value 
initial_value = 100000   # Change this value as needed


# Assume portfolio daily return ~ Normal(mean, std)
mean_daily_return = 0.0005    # ~ 0.05% per day (~12% annual)
std_daily_return = 0.01       # ~ 1% daily volatility

# Run simulations
simulations = np.zeros((n_days, n_simulations))

for i in range(n_simulations):
    daily_returns = np.random.normal(mean_daily_return, std_daily_return, n_days)
    portfolio_values = initial_value * (1 + daily_returns).cumprod()
    simulations[:, i] = portfolio_values

# Plot a few sample paths
plt.figure(figsize=(12,6))
plt.plot(simulations[:, :50], alpha=0.5)  # plot 50 random scenarios
plt.title("Monte Carlo Portfolio Simulation (50 paths)")
plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.savefig("monte_carlo_sample_paths.png")  # Save the plot as a file

# Compute mean portfolio performance across all simulations
mean_performance = simulations.mean(axis=1)

plt.figure(figsize=(12,6))
plt.plot(simulations, color="blue", alpha=0.1)  # all scenarios faint
plt.plot(mean_performance, color="orange", linewidth=2, label="Mean Path")
plt.legend()
plt.title("Monte Carlo Portfolio Simulation with Mean Path")
plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.savefig("monte_carlo_simulation.png")  # Save the plot as a file

final_values = simulations[-1, :]
print("Expected final value:", final_values.mean())
print("5% worst-case (VaR):", np.percentile(final_values, 5))
print("95% best-case:", np.percentile(final_values, 95))

# # Example output:
# Expected final value: 113807.84004690402
# 5% worst-case (VaR): 87247.35765528133
# 95% best-case: 146158.34248134503

# Note:
# The first chart shows 50 random portfolio paths out of 500 simulations.
# The second chart shows all paths in blue, with the average performance in orange.