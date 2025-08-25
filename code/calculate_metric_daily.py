"""
Steinschneider Lab Group
Madeline Allen
Code last updated: 7/16/2025
Script purpose: calculate deterministic and probabilistic metrics at daily scale

This project utilizes code generated with the assistance of OpenAI's ChatGPT (July, 2025). 
"""

import time
import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
# local
import config
from config import MyLogger

# Set up logger
current_script_name = Path(__file__).stem
logger = MyLogger(config.log_path / f'log_{current_script_name}.txt')

# Metadata from config
sd, ed = config.start_date, config.end_date
dates = config.dates
years = config.years
lead_times = config.selected_lead_times
lead_string = ''.join(map(str, lead_times))
max_ens = config.max_ens_members
max_lead_time = config.max_lead_time
AR_recon_window = config.AR_recon_window
cores_for_download = config.cores_for_download
calculate_probabilistic = config.calculate_probabilistic

# Set up data paths and read in ID table
ID_table = pd.read_csv(config.NWS_USGS_ID_Table_SufficientData)
selected_sites = ID_table['CNRFC_id'].tolist()
data_filepath = config.metric_folder


# Read in USGS data
usgs_data = pd.read_csv(config.USGS_data_filename)
usgs_data['datetime'] = pd.to_datetime(usgs_data['datetime']).dt.tz_localize(None)
usgs_data = usgs_data[(usgs_data['datetime'] >= sd) & (usgs_data['datetime'] <= ed)]
usgs_data = usgs_data[usgs_data['site_no'].isin(ID_table['USGS_id'])]

# Read in CNRFC forecast data
file_paths = [config.CNRFC_forecasts_folder_name / f'cnrfc14d_{year}.nc' for year in years]
qf = xr.open_mfdataset(file_paths, combine='nested', concat_dim='time')
qf = qf.sel(ens=slice(1, max_ens), lead=lead_times, time=slice(sd, ed), site=selected_sites)
qf_df = qf.to_dataframe().reset_index().replace(-999000, pd.NA)

# Merge forecasts and observed flow
merged = pd.merge(ID_table, usgs_data, left_on='USGS_id', right_on='site_no', how='inner')
qf_df_ens = pd.merge(qf_df, merged, left_on=['site', 'time'], right_on=['CNRFC_id', 'datetime'], how='left')
qf_df_ens = qf_df_ens[['site', 'time', 'ens', 'lead', 'Qf', 'Qo']]

# Collapse to ensemble mean forecast for daily dataset
qf_df_daily = qf_df_ens.groupby(['site', 'time', 'lead']).mean(numeric_only=True).drop(columns='ens').reset_index()

# Read in AR recon data
ar = pd.read_csv(config.AR_recon_records_filename, parse_dates=['Date_PST'])
#group by date, sum daily, reindex so all dates in range represented - from min date to max_lead_time days past the max PST date (so ar dropsondes at end of record get counted for days in the future beyond last dropsonde)
ar = ar[['Date_PST', 'NCEP_daily_asim']].groupby('Date_PST').sum().reindex(pd.date_range(ar['Date_PST'].min(), ar['Date_PST'].max() + pd.Timedelta(days=max_lead_time), freq='D')).replace({pd.NA: np.nan}) 

# Calculate number of AR recon dropsondes assimilated AR_recon_window number of days prior to forecast being issued
for lead in lead_times:
    col = f'NCEP_daily_asim_ld{lead}_3window'
    ar[col] = ar['NCEP_daily_asim'].shift(lead).rolling(window=AR_recon_window, min_periods=1).sum()

ar = ar.reset_index().rename(columns={'index': 'Date_PST'})
qf_df_daily = pd.merge(qf_df_daily, ar, left_on='time', right_on='Date_PST', how='left').drop(columns=['Date_PST'])

# Calculate deterministic metrics
qf_df_daily['AE'] = np.abs(qf_df_daily['Qf'] - qf_df_daily['Qo']) #absolute error
qf_df_daily['SE'] = (qf_df_daily['Qf'] - qf_df_daily['Qo']) ** 2 #squared error

# Calculate probabilistic metrics
results_dtype = [('site', object), ('time', 'datetime64[ns]'), ('lead', object), ('rank', 'float64'), ('nep', 'float64'), ('eCRPS', 'float64')]
for col in ['rank', 'nep', 'eCRPS']:
    qf_df_daily[col] = np.nan #initialize as nan

def calculate_prob_metrics(ensemble, obs):
    # Return NaNs if observation is invalid (None, NaN, or zero)
    # or if the ensemble is completely missing
    if obs is None or pd.isna(obs) or obs == 0 or np.all(pd.isna(ensemble)):
        return np.nan, np.nan, np.nan

    # Filter out NaN values from ensemble and sort the valid values
    # Round to 6 decimals for numerical stability
    ens_valid = np.round(np.sort(ensemble[~pd.isna(ensemble)]), 6)

    # If all ensemble members were NaN, return NaNs
    if len(ens_valid) == 0:
        return np.nan, np.nan, np.nan

    # Round observation for consistency with ensemble
    obs = np.round(obs, 6)

    # Construct empirical CDF: positions of sorted values in CDF space
    cdf = np.linspace(1, len(ens_valid), len(ens_valid)) / (len(ens_valid) + 1)

    # Find where the observation fits relative to the sorted ensemble. note - searchsorted returns index where inserting
    # obs wouldn't change the sorted order of ens_valid.
    low = max(0, np.searchsorted(ens_valid, obs, side='left')-1) #max b/c if search sorted returns zero, want zero not -1; -1 b/c want the index of the ens member "to the left" of where obs would be
    up = np.searchsorted(ens_valid, obs, side='right') # index of ens member "to the right" of where obs would be

    # Compute the rank of the observation as the average index of equal values (or closest match)
    avg_idx = np.ceil((low + up) / 2).astype(int) #this rounds up, b/c if low is 3, and up is 4, we want 4 returned - i.e., the index the obs would actually sit at if it was inserted into sorted list

    rank = avg_idx + 1  # Add 1 to match traditional rank indexing (starts at 1)

    # Clamp indices to stay within valid range of ensemble (avoid indexing errors for extreme obs)
    low = min(low, len(ens_valid) - 1)
    up = min(up, len(ens_valid) - 1)

    # Interpolate CDF value (i.e., NEP) for the observed flow; if low and up are the same, just assign it cdf value associated w low to avoid errors
    if ens_valid[low] == ens_valid[up]:
        nep = cdf[low]  # or cdf[up], they should be equal or close in that case
    else:
        nep = np.interp(obs, (ens_valid[low], ens_valid[up]), (cdf[low], cdf[up]))

    # Calculate eCRPS via: eCRPS = (1 / m) * sum1 - (1 / (m * (m - 1))) * sum2
    # m is number of ensemble members
    m = len(ens_valid)

    #guard agains m<2 which would result in divide by zero
    if m < 2:
        eCRPS = np.nan
    else:
        # First term: average absolute error between ensemble members and the observation
        sum1 = np.sum(np.abs(ens_valid - obs))

        # Second term: average pairwise absolute difference among ensemble members
        # Only sum the upper triangle of the difference matrix (excluding diagonal)
        sum2 = np.sum(np.abs(ens_valid[:, None] - ens_valid[None, :])[np.triu_indices(m, k=1)])

        # Final eCRPS calculation using standard formula
        eCRPS = (1 / m) * sum1 - (1 / (m * (m - 1))) * sum2

    return rank, nep, eCRPS


def process_group(group):
    obs = group.iloc[0]['Qo']
    ensemble = pd.to_numeric(group['Qf'], errors='coerce').to_numpy(dtype=np.float64)
    return calculate_prob_metrics(ensemble, obs)

def process_chunk(chunk):
    out = np.empty(len(chunk), dtype=results_dtype)
    for idx, (key, group) in enumerate(chunk):
        out[idx] = (*key, *process_group(group))
    return out

def apply_prob_metrics(df_ens, df_avg):
    grouped = list(df_ens.groupby(['site', 'time', 'lead']))
    chunks = [grouped[i:i+cores_for_download*4] for i in range(0, len(grouped), cores_for_download*4)]
    results = np.empty(len(grouped), dtype=results_dtype)
    with ProcessPoolExecutor(max_workers=cores_for_download) as executor:
        futures = {executor.submit(process_chunk, c): i for i, c in enumerate(chunks)}
        idx = 0
        for fut in as_completed(futures):
            r = fut.result()
            results[idx:idx+len(r)] = r
            idx += len(r)
    df_r = pd.DataFrame(results)
    df_out = pd.merge(df_avg, df_r, on=['site', 'time', 'lead'], how='left', suffixes=('', '_new'))
    for col in ['rank', 'nep', 'eCRPS']:
        df_out[col] = df_out[col].combine_first(df_out[f'{col}_new'])
    return df_out.drop(columns=['rank_new', 'nep_new', 'eCRPS_new'])

if __name__ == '__main__':
    t0 = time.time()
    if calculate_probabilistic:
        qf_df_daily = apply_prob_metrics(qf_df_ens, qf_df_daily)

    # remove rows with NaN in Qf (ensemble mean forecast)    
    qf_df_daily = qf_df_daily.dropna(subset=['Qf'])

    qf_df_daily.to_csv(config.metric_daily_filename, index=False)
    elapsed = time.time() - t0
    with open(data_filepath / f'metric_calc_daily_time_elapsed.txt', 'w') as f:
        f.write(f'Metric calculation elapsed time (sec): {elapsed:.2f}\n')
        f.write(f'Metric calculation elapsed time (min): {elapsed/60:.2f}')
    logger.log(qf_df_daily.head())

logger.log('Daily metric calculation completed successfully.')

logger.end()