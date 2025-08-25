"""
Steinschneider Lab Group
Madeline Allen
Code last updated: 7/16/2025
Script purpose: explore reliability statistic sensitivity to number of events for a given site and water year

Steinschneider Lab Group

Madeline Allen

Code last updated: 7/16/2025

Goal: understand the sampling variability of reliability statistic – as the number of flow events 
considered in the metric calculation increases – the metric converges towards a given “underlying” performance. 

The reliability is essentially looking at how observations ranked among the ensemble members, and ideally, 
over many events, the distribution of these ranks should be uniform. I can simulate this by simulating 
the empirical nonexceedance probabilities - pick “observed” empirical NEP from a uniform distribution 
from 0 to 1 (i.e., representing a forecasting system that is perfectly reliable). The theoretical NEP 
is then calculated based on number of ens members, and Reliability statistic is calculated by comparing
"empirical" and theoretical NEPs. Since the data generating mechanism represents a "perfect forecast", 
the reliability statistic should converge to zero; the following analysis is used to explore how many 
events are needed to get a measure of reliability that is close to the "true" reliability.

This project utilizes code generated with the assistance of OpenAI's ChatGPT (July, 2025). 
"""

#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
# local
import config
from config import MyLogger

# Set up logger
current_script_name = Path(__file__).stem
logger = MyLogger(config.log_path / f'log_{current_script_name}.txt')

plot_filepath = config.monte_carlo_reliability_folder

# Read in aggregated metric file
agg_type = 'window'
flow = config.high_flow_threshold
filename = config.metric_folder / f"metrics_{agg_type}_Thresh{flow}.csv"
df_metric = pd.read_csv(filename)
qf_ld1 = df_metric[df_metric['lead']==1]


# Parameters
n_ens = 40
n_events = 365
n_sim = 10000

logger.log(f'Running {n_sim} simulations of perfect forecast model using {n_ens} ensemble members and {n_events} events (days).')
logger.log(f'Comparison to real forecast metrics was done using the reliability metrics saved in: {filename}.')

# Theoretical NEPs for ensemble forecasts (fixed)
nep_theoretical = np.linspace(1, n_ens, n_ens) / (n_ens + 1)

# Store results
all_sim_results = []

for sim in range(n_sim):
    # Simulate empirical NEPs for each event (these are assumed to come from a uniform distribution)
    nep_empirical_sim = np.random.uniform(low=0.0, high=1.0, size=n_events)

    # Initialize DataFrame
    df = pd.DataFrame({
        'n_events': np.arange(1, n_events + 1),
        'nep_empirical_sim': nep_empirical_sim,
        'sim': sim
    })

    # Rolling reliability metric (comparing empirical NEPs to uniformity)
    rolling_rel = []
    for i in range(1, n_events + 1):
        subset = df.iloc[:i]
        empirical = subset['nep_empirical_sim'].values
        # Sort for ECDF comparison
        empirical_sorted = np.sort(empirical)
        theoretical_uniform = np.linspace(1, i, i) / (i + 1)

        # Reliability as average absolute difference between empirical NEP and theoretical uniform
        rel = (2 / i) * np.sum(np.abs(empirical_sorted - theoretical_uniform))
        rolling_rel.append(rel)

    df["rolling_rel"] = rolling_rel
    all_sim_results.append(df)

# Combine all simulation results
all_df = pd.concat(all_sim_results, ignore_index=True)

# Compute summary statistics across all simulations
summary_df = (
    all_df
    .groupby("n_events")["rolling_rel"]
    .agg(rel_mean="mean", rel_p5=lambda x: np.percentile(x, 5), rel_p95=lambda x: np.percentile(x, 95))
    .reset_index()
)
csv_name = f'summary_df_rel_stat_90confit.csv'
summary_df.to_csv(plot_filepath / csv_name)
logger.log(f'Saved summary statistics from simulation to {csv_name}.')

dy = np.diff(summary_df["rel_mean"])
dx = np.diff(summary_df["n_events"])
slope = dy / dx

# Absolute value because we care about magnitude of improvement
abs_slope = np.abs(slope)

# Find first place where slope gets "small enough" (threshold adjustable)
threshold = 0.01  
elbow_index = np.where(abs_slope < threshold)[0][0]
n_events_elbow = summary_df["n_events"].iloc[elbow_index]

logger.log(f"Elbow around {n_events_elbow} events")

# This is the elbow above; and was also chosen by examining plot from code above.
suggested_num_events = 10



# Plot reliability over time for all simulations
plt.figure(figsize=(14, 7))
sns.lineplot(data=all_df, x="n_events", y="rolling_rel", hue="sim", legend=False, alpha=0.1)

plt.title(f"Reliability Convergence Across All Simulations (n_sim={n_sim})")
plt.xlabel("Number of Events")
plt.ylabel("Rolling Reliability")
plt.grid(True)
plt.tight_layout()
# Save figure
filename = plot_filepath /'RelConvergence_PerfectModel_Simulated.png'
plt.savefig(filename, dpi=300)
#plt.show()
plt.close()


'''
Compare to reliability calculations from the wet season high flow threshold data
'''

plt.figure(figsize=(6, 6))
plt.scatter('num_events', 'rel',data = qf_ld1, c='dodgerblue', alpha=0.8, s=1)
plt.xlabel(f'Number of events in given WY')
plt.ylabel(f'Reliability')
plt.title('Reliability vs Number of Events (Forecast Data; lead 1)')
plt.tight_layout()
# Save figure
filename = plot_filepath /'Rel_vs_num_events.png'
plt.savefig(filename, dpi=300)
# plt.show()
plt.close()


plt.figure(figsize=(8, 6))
plt.hist(qf_ld1['rel'] , bins=30, color='indigo', edgecolor='black')
plt.title(f'Reliability Metric Histogram (Forecast Data; lead 1)')
plt.xlabel('Value')
plt.ylabel('Frequency')
# Save figure
filename = plot_filepath /'Reliability_histogram.png'
plt.savefig(filename, dpi=300)
# plt.show()
plt.close()


plt.figure(figsize=(8, 6))
plt.hist(qf_ld1['num_events'] , bins=30, color='indigo', edgecolor='black')
plt.title(f'Number of Events Histogram (Forecast Data; lead 1)')
plt.xlabel('Number of Events')
plt.ylabel('Frequency')
# Save figure
filename = plot_filepath /'Num_events_histogram.png'
plt.savefig(filename, dpi=300)
# plt.show()
plt.close()


# average
avg_num_events = np.mean(qf_ld1['num_events'])
med_num_events = np.median(qf_ld1['num_events'])
logger.log(f'Average number of events for 90th percentile flow in dataset: {avg_num_events}')
logger.log(f'Median number of events for 90th percentile flow in dataset: {med_num_events}')

# rough number of days in wy period:
est_num = 8*30
logger.log(f'rough number of days in wy period: {est_num}')



'''
Main plot: statistics from simulation with median from real data
'''

# Plot
plt.figure(figsize=(9, 5))

# Shaded 90% Confidence Interval
plt.fill_between(summary_df["n_events"], summary_df["rel_p5"], summary_df["rel_p95"], color="skyblue", alpha=0.3, label="5th–95th Percentile")

# Mean line
sns.lineplot(data=summary_df, x="n_events", y="rel_mean", color="blue", label="Mean Reliability")
plt.axvline(x=med_num_events, color = 'red', linestyle='--',label=f'Median number of events ({int(med_num_events)})') #(per site, per water year) 
plt.axvline(x=suggested_num_events, color = 'black', linestyle='--',label=f'Selected Threshold ({suggested_num_events})') # Suggested minimum number of events (per site, per water year) to include in analysis 

plt.title("Reliability Statistic Sampling Variability")
plt.xlabel("Number of Events")
plt.ylabel("Reliability")
plt.grid(True)
plt.legend()
plt.tight_layout()
# Save figure
filename = plot_filepath /'RelConvergence_PerfectModel_int90_labels.png'
plt.savefig(filename, dpi=300)
# plt.show()
plt.close()

logger.end()
