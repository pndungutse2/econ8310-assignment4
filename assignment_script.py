#Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta

#Step 2: Load Dataset
data = pd.read_csv('cookie_cats.csv')
data['version'] = data['version'].str.strip()

#Step 3: Explore Dataset
print("First few rows of data:")
print(data.head())

print("\nMissing values per column:")
print(data.isnull().sum())

print("\nSummary statistics:")
print(data.describe())

#Step 4: Visualize Game Rounds Distribution
sns.histplot(data['sum_gamerounds'], bins=50)
plt.title('Distribution of Game Rounds Played')
plt.xlabel('Game Rounds')
plt.ylabel('Frequency')
plt.show()

#Step 5: Retention Rate Comparison (Descriptive)
retention_rates_grouped = data.groupby('version')[['retention_1', 'retention_7']].mean()
print("\nAverage Retention Rates by Group:")
print(retention_rates_grouped)

retention_rates_grouped.plot(kind='bar', figsize=(8, 6))
plt.title('Retention Rates by Version (Control vs Treatment)')
plt.ylabel('Average Retention Rate')
plt.xticks(rotation=0)
plt.show()

#Step 6: Prepare Groups for Bayesian Test
control = data[data['version'] == 'gate_30']
treatment = data[data['version'] == 'gate_40']

#Step 7: Define Posterior Function

def calculate_posterior(success_a, trials_a, success_b, trials_b, alpha=1, beta_param=1):
    posterior_a = beta(alpha + success_a, beta_param + trials_a - success_a)
    posterior_b = beta(alpha + success_b, beta_param + trials_b - success_b)
    return posterior_a, posterior_b


#Step 8: Compute Summary Stats
retention_stats = {
    'success_a_1': control['retention_1'].sum(),
    'trials_a_1': control['retention_1'].count(),
    'success_b_1': treatment['retention_1'].sum(),
    'trials_b_1': treatment['retention_1'].count(),
    'success_a_7': control['retention_7'].sum(),
    'trials_a_7': control['retention_7'].count(),
    'success_b_7': treatment['retention_7'].sum(),
    'trials_b_7': treatment['retention_7'].count(),
}

#Step 9: Bayesian Posterior Estimation
posterior_1_a, posterior_1_b = calculate_posterior(
    retention_stats['success_a_1'], retention_stats['trials_a_1'],
    retention_stats['success_b_1'], retention_stats['trials_b_1']
)

posterior_7_a, posterior_7_b = calculate_posterior(
    retention_stats['success_a_7'], retention_stats['trials_a_7'],
    retention_stats['success_b_7'], retention_stats['trials_b_7']
)

#Step 10: Plot Posterior Distributions
x = np.linspace(0, 1, 1000)
plt.figure(figsize=(12, 6))
plt.plot(x, posterior_1_a.pdf(x), label='Control (Retention 1)', color='blue')
plt.plot(x, posterior_1_b.pdf(x), label='Treatment (Retention 1)', color='orange')
plt.title('Posterior Distributions: Retention 1')
plt.xlabel('Retention Rate')
plt.ylabel('Density')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(x, posterior_7_a.pdf(x), label='Control (Retention 7)', color='blue')
plt.plot(x, posterior_7_b.pdf(x), label='Treatment (Retention 7)', color='orange')
plt.title('Posterior Distributions: Retention 7')
plt.xlabel('Retention Rate')
plt.ylabel('Density')
plt.legend()
plt.show()

#Step 11: Monte Carlo Simulation to Estimate Probabilities
samples_1_a = posterior_1_a.rvs(100000)
samples_1_b = posterior_1_b.rvs(100000)
prob_retention1 = np.mean(samples_1_b > samples_1_a)

samples_7_a = posterior_7_a.rvs(100000)
samples_7_b = posterior_7_b.rvs(100000)
prob_retention7 = np.mean(samples_7_b > samples_7_a)

#Step 12: Report Results
print("\nRetention Results Summary:")
print(f"Control Group Retention 1-Day: {retention_stats['success_a_1'] / retention_stats['trials_a_1']:.4f}")
print(f"Treatment Group Retention 1-Day: {retention_stats['success_b_1'] / retention_stats['trials_b_1']:.4f}")
print(f"→ Probability Treatment > Control (1-Day): {prob_retention1:.4f}")

print(f"\nControl Group Retention 7-Day: {retention_stats['success_a_7'] / retention_stats['trials_a_7']:.4f}")
print(f"Treatment Group Retention 7-Day: {retention_stats['success_b_7'] / retention_stats['trials_b_7']:.4f}")
print(f"→ Probability Treatment > Control (7-Day): {prob_retention7:.4f}")

#Step 13: Reflective Summary
print("\nBiggest Challenge:")
print("Understanding the interpretation of Bayesian posterior probabilities and how to translate that into business decision-making was the biggest learning curve.")
