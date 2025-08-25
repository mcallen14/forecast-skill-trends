"""
Steinschneider Lab Group
Madeline Allen
Code last updated: 7/16/2025
Script purpose: data cleaning

The following code examines the USGS observed streamflow data and the CNRFC forecasted streamflow data 
to isolate the sites that have sufficient data availability during the time period of interest.
Threshold for data availability can be adjusted in the config.py file; published analysis 
only included sites with 90% data available during study period.

This project utilizes code generated with the assistance of OpenAI's ChatGPT (July, 2025). 
"""

import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
# local
import config
from config import MyLogger

# Set up logger
current_script_name = Path(__file__).stem
logger = MyLogger(config.log_path / f'log_{current_script_name}.txt')

# Config
sd, ed = config.start_date, config.end_date
percent_thresh = config.percent_threshold_data_availability
cnrfc_data_folder_path = config.CNRFC_forecasts_folder_name
years = config.years
threshold_count = round(percent_thresh * len(pd.date_range(start=sd, end=ed, freq='D'))) #number of days that must have data available to meet threshold


# Load site pairing table + USGS observations
ID_table = pd.read_csv(config.NWS_USGS_ID_Table_All)
usgs_data = pd.read_csv(config.USGS_data_filename)
usgs_data['datetime'] = pd.to_datetime(usgs_data['datetime']).dt.tz_localize(None)
usgs_data = usgs_data[(usgs_data['datetime'] >= sd) & (usgs_data['datetime'] <= ed)]

# Load CNRFC forecast data
ds = xr.open_mfdataset(
    [cnrfc_data_folder_path / f"cnrfc{config.max_lead_time}d_{year}.nc" for year in years],
    combine='nested', concat_dim='time'
).sel(ens=1, lead=1, time=slice(sd, ed))

# --- Visualization: Number of sites with data over time ---
non_na_time = ds['Qf'].count(dim='site').to_dataframe(name='non_na_count').reset_index()
plt.figure(figsize=(8, 5))
plt.plot(non_na_time['time'], non_na_time['non_na_count'], marker='o')
plt.xlabel('Date'); plt.ylabel('Sites with Data')
plt.title('CNRFC Data Availability Over Time'); plt.grid(True); plt.xticks(rotation=45)
plt.tight_layout()
plot_file = config.plots_path / f'DataAvailability_OverTime_{sd}_to_{ed}.png'
if plot_file.exists(): plot_file.unlink()
plt.savefig(plot_file, dpi=300)
#plt.show()

# --- Identify CNRFC sites with sufficient data ---
cnrfc_counts = ds['Qf'].count(dim='time').to_dataframe(name='non_na_count').reset_index()
cnrfc_sufficient = set(cnrfc_counts[cnrfc_counts['non_na_count'] >= threshold_count]['site'])

# --- Identify USGS sites with sufficient data ---
usgs_counts = usgs_data.groupby('site_no')['Qo'].count().reset_index()
usgs_sufficient = set(usgs_counts[usgs_counts['Qo'] >= threshold_count]['site_no'])

# --- Intersect & filter ID table ---
usgs_to_cnrfc = set(ID_table[ID_table['USGS_id'].isin(usgs_sufficient)]['CNRFC_id'])
both_sufficient = cnrfc_sufficient & usgs_to_cnrfc

logger.log(f"# CNRFC sites >={percent_thresh*100}% data: {len(cnrfc_sufficient)}")
logger.log(f"# USGS sites >={percent_thresh*100}% data: {len(usgs_sufficient)}")
logger.log(f"# Sites with sufficient data from BOTH: {len(both_sufficient)}")

# Filter and export updated ID table
ID_table_filtered = ID_table[ID_table['CNRFC_id'].isin(both_sufficient)].reset_index(drop=True)

# Optional: Manually exclude outlier sites
manual_exclude = ['MRYC1', 'SRWC1']
to_remove = set(manual_exclude) & set(ID_table_filtered['CNRFC_id'])
ID_table_final = ID_table_filtered[~ID_table_filtered['CNRFC_id'].isin(to_remove)].reset_index(drop=True)

logger.log(f"Excluded {len(to_remove)} manually flagged sites ({to_remove}). Final site count: {len(ID_table_final)}")
ID_table_final.to_csv(config.NWS_USGS_ID_Table_SufficientData, index=False)

logger.end()