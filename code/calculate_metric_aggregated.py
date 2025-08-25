"""
Steinschneider Lab Group
Madeline Allen
Code last updated: 7/16/2025
Script purpose: calculate deterministic and probabilistic metrics at aggregated scale

This project utilizes code generated with the assistance of OpenAI's ChatGPT (July, 2025). 
"""

#import necessary packages
import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
# local
import config
from config import MyLogger

# Set up logger
current_script_name = Path(__file__).stem
logger = MyLogger(config.log_path / f'log_{current_script_name}.txt')

# Variables from config file:
sd = config.start_date # analysis start date
ed= config.end_date # analysis end date
sd_decade = config.start_date_decade
ed_decade = config.end_date_decade
window_month_start = config.window_month_start # beginning of aggregation window (extended wet season)
window_month_end = config.window_month_end # end of aggregation window (extended wet season)
flow_thresholds = config.flow_thresholds # flow thresholds considered (e.g., 90th percentile flow and above)
thresholds_string = ''.join(map(str, flow_thresholds))
lead_times=config.selected_lead_times # lead times to calculate metrics for
lead_string = ''.join(map(str, lead_times))
aggregation_periods = config.aggregation_periods
AR_recon_window = config.AR_recon_window
calculate_probabilistic = config.calculate_probabilistic


# Set up data paths and read in ID table
ID_table = pd.read_csv(config.NWS_USGS_ID_Table_SufficientData)
selected_sites = ID_table['CNRFC_id'].tolist()
data_filepath = config.metric_folder
qf_df_daily = pd.read_csv(config.metric_daily_filename, parse_dates=['time'])

# Make sure there are no daily rows that don't have a forecast issued:
# Flag and remove rows with missing ensemble forecast (Qf)
missing_qf_rows = qf_df_daily[qf_df_daily['Qf'].isna()]
num_missing = len(missing_qf_rows)

if num_missing > 0:
    logger.log(f'WARNING: {num_missing} rows have missing Qf (ensemble mean forecast) and will be removed.')
    logger.log('Sample rows with missing Qf:')
    logger.log(missing_qf_rows[['site', 'time', 'lead']].head().to_string(index=False))
    qf_df_daily = qf_df_daily[~qf_df_daily['Qf'].isna()]
else:
    logger.log('All rows have valid Qf values.')


# List to store timing results
timing_results = []
timing_results.append(f'Elapsed time for metric calculation (aggregated) for sd = {sd}, ed = {ed}: ')
metric_start_time = time.time()

#Date range represented in sd to ed:
dates = config.dates 
years = config.years
sd_year = years[0]
ed_year = years[-1]

# Flags to isolate water year and aggregation window(s) of interest:
# Water year assignment
def get_water_year(date):
    return date.year + 1 if date.month >= 10 else date.year

# Date categories
def is_winter(date):
    return int(date.month in [12, 1, 2, 3])

def is_decade(date):
    return int(pd.to_datetime(sd_decade) <= date <= pd.to_datetime(ed_decade))

# Get list of months in extended seasonal window
def get_month_range(start_month, end_month):
    return list(range(start_month, end_month + 1)) if start_month <= end_month \
           else list(range(start_month, 13)) + list(range(1, end_month + 1))

month_range = get_month_range(window_month_start, window_month_end)

def is_in_window(date):
    return int(date.month in month_range)

# Apply flags to dataset:
# Assign water year and seasonal window flags
qf_df_daily['water_year'] = qf_df_daily['time'].apply(get_water_year)
qf_df_daily['month_window'] = qf_df_daily['time'].apply(is_in_window)
qf_df_daily['decade'] = qf_df_daily['time'].apply(is_decade)

#Calculate climatological values:
# Climatological flow mean and std per site
qf_df_daily['Qo_clim'] = qf_df_daily.groupby('site')['Qo'].transform('mean')
qf_df_daily['Qo_clim_std'] = qf_df_daily.groupby('site')['Qo'].transform('std')
# Climatological squared error
qf_df_daily['SE_clim'] = (qf_df_daily['Qo_clim'] - qf_df_daily['Qo']) ** 2

logger.log(qf_df_daily.head())



def calculate_rel(df_daily_filtered, df_agg, rel_grouping):
    for keys, group in df_daily_filtered.groupby(rel_grouping):
        if len(rel_grouping) == 2:
            site, lead = keys
            rel_indexer = (df_agg['site'] == site) & (df_agg['lead'] == lead)
        else:
            site, lead, water_year = keys
            rel_indexer = (df_agg['site'] == site) & (df_agg['lead'] == lead) & (df_agg['water_year'] == water_year)

        NEP = group['nep'].dropna().values
        if len(NEP) == 0:
            continue

        NEP_ordered = np.sort(NEP)
        NEP_theoretical = np.linspace(1, len(NEP_ordered), len(NEP_ordered)) / (len(NEP_ordered) + 1)
        rel = (2 / len(NEP_ordered)) * np.sum(np.abs(NEP_theoretical - NEP_ordered))
        df_agg.loc[rel_indexer, 'rel'] = rel

    return df_agg

def postprocess_metrics(df_met, df_filtered):
    df_met.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df_met.columns]

    df_met.rename(columns={
        'Qo_clim_mean': 'Qo_clim',
        'Qo_clim_std_mean': 'Qo_clim_std',
        'AE_mean': 'MAE',
        'SE_mean': 'MSE',
        'above_threshold_sum': 'num_events',
    }, inplace=True)

    df_met['NSE'] = 1 - (df_met['SE_sum'] / df_met['SE_clim_sum'])
    df_met['RMSE'] = df_met['MSE'] ** 0.5
    df_met['RMSE_norm'] = df_met['RMSE'] / df_met['Qo_clim']

    if calculate_probabilistic:
        df_met['eCRPS_mean_std'] = df_met['eCRPS_mean'] / df_met['Qo_clim_std']
        df_met['rel'] = np.nan
        if 'water_year' in df_met.columns:
            rel_grouping = ['site', 'lead', 'water_year']
        else:
            rel_grouping = ['site', 'lead']
        calculate_rel(df_daily_filtered=df_filtered, df_agg=df_met, rel_grouping=rel_grouping)

    for ld in lead_times:
        ar_col = f'NCEP_daily_asim_ld{ld}_{AR_recon_window}window_sum'
        df_met[f'AR_ld{ld}_3window_cum_norm'] = df_met[ar_col] / df_met['num_events']

    df_met['AR_recon'] = df_met.apply(
        lambda row: row.get(f'AR_ld{int(row["lead"])}_3window_cum_norm', np.nan),
        axis=1
    )
    if 'water_year' in df_met.columns:
        df_met['GEFSv12'] = df_met['water_year'].apply(lambda x: 1 if x >= 2022 else 0)

    return df_met

def calculate_metrics_for_threshold(flow_threshold1, df_daily, aggregation_type):
    if flow_threshold1 == 0.0:
        df_daily['above_threshold'] = 1
    else:
        threshold_col = f'Qo_threshold{flow_threshold1}'
        df_daily[threshold_col] = df_daily.groupby('site')['Qo'].transform(lambda x: x.quantile(flow_threshold1))
        df_daily['above_threshold'] = (df_daily['Qo'] > df_daily[threshold_col]).astype(int)

    df_daily['window'] = ((df_daily['above_threshold'] == 1) & (df_daily['month_window'] == 1)).astype(int)

    if aggregation_type == 'water_year':
        df_filtered = df_daily[df_daily['above_threshold'] == 1].copy()
    elif aggregation_type == 'window':
        df_filtered = df_daily[df_daily['window'] == 1].copy()
    else:
        raise ValueError(f"Unknown aggregation type: {aggregation_type}")

    group_cols = ['site', 'lead', 'water_year']
    group_cols_decade = ['site', 'lead']

    agg_dict = {
        'Qo_clim': 'mean',
        'Qo_clim_std': 'mean',
        'AE': ['mean', 'sum'],
        'SE': ['mean', 'sum'],
        'SE_clim': 'sum',
        'Qo': 'mean',
        'window': ['sum'],
        'above_threshold': ['sum'],
    }
    for ld in lead_times:
        col = f'NCEP_daily_asim_ld{ld}_{AR_recon_window}window'
        agg_dict[col] = 'sum'
    if calculate_probabilistic:
        agg_dict.update({
            'eCRPS': 'mean',
            'rank': 'mean',
            'nep': 'mean'
        })

    df_metrics = df_filtered.groupby(group_cols).agg(agg_dict).reset_index()
    df_metrics = postprocess_metrics(df_metrics, df_filtered)

    df_filtered_decade = df_filtered[df_filtered['decade'] == 1].copy()
    df_metrics_decade = df_filtered_decade.groupby(group_cols_decade).agg(agg_dict).reset_index()
    df_metrics_decade = postprocess_metrics(df_metrics_decade, df_filtered_decade)


    filename_main = data_filepath / f"metrics_{aggregation_type}_Thresh{flow_threshold1}.csv"
    filename_decade = data_filepath / f"metrics_{aggregation_type}_Thresh{flow_threshold1}_decade.csv"
    df_metrics.to_csv(filename_main, index=False)
    df_metrics_decade.to_csv(filename_decade, index=False)

    logger.log(df_metrics.head())
    logger.log(f"Saved metrics: {filename_main}")
    logger.log(df_metrics_decade.head())
    logger.log(f"Saved metrics: {filename_decade}")

# Run all threshold/aggregation combinations
for flow_threshold1 in flow_thresholds:
    for agg_type in aggregation_periods:
        calculate_metrics_for_threshold(flow_threshold1, qf_df_daily.copy(), agg_type)

logger.end()