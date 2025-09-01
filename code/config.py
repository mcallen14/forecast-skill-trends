"""
Steinschneider Lab Group
Madeline Allen
Code last updated: 7/16/2025
Script purpose: set configurations for analysis; used by every other script in repo.

This script has variables and path names that are used throughout the rest of the scripts in this repo.
Most adjustments to the analysis can be made here, including: time period, data availability requriements,
states and forecast regions, lead times, number of ensemble members, flow thresholds, 
aggregation windows, and metrics analyzed.

This project utilizes code generated with the assistance of OpenAI's ChatGPT (July, 2025). 
"""

from datetime import date, datetime
import pandas as pd
import os
import multiprocessing
from pathlib import Path

# Get path to config.py and then repo base
CONFIG_DIR = Path(__file__).resolve().parent
BASE_DIR = CONFIG_DIR.parent

# Start date and end date for analysis:
start_date = '2013-09-01' # a month before the start of WY2014
end_date = '2025-05-31' 

years = list(range(datetime.strptime(start_date, '%Y-%m-%d').year, datetime.strptime(end_date, '%Y-%m-%d').year + 1))
date_today_sec = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
date_today_month_year = date.today().strftime("%B%Y")
# Date range represented in sd to ed
dates = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y%m%d')

# Function to get the water year
def get_water_year(date):
    year = date.year
    if date.month >= 10:
        return year + 1
    else:
        return year
# Compute start and end water years
sd_water_year = 2014 #get_water_year(datetime.strptime(start_date, '%Y-%m-%d')) #set manually or get dynamically from sd above; since I manually set it to one month before the first WY of interest, this is set manually
ed_water_year = get_water_year(datetime.strptime(end_date, '%Y-%m-%d'))
# print('ed_water_year', ed_water_year)
start_date_decade = f'{sd_water_year - 1}-10-01'  # WY starts in Oct of prior calendar year
end_date_decade = f'{ed_water_year}-09-30'        # WY ends in Sep of the water year

# Create list of WATER years represented in date range
water_years = list(range(sd_water_year, ed_water_year + 1))

# Key variables for NWS-USGS site pair:
states_in_study_region = ["CA", "OR", "NV"]

# Specify regions of interest - CNRFC:
selected_regions = ['klamath', 'NorthCoast', 'RussianNapa', 'UpperSacramento', 'FeatherYuba', 'cacheputah', 'american', 
                    'SalinasPajaro', 'SouthBay', 'SouthernCalifornia', 'SanDiego_Inland', 'Tulare', 'SanJoaquin', 'n_sanjoaquin',
                    'EastSierra', 'Humboldt', 'CentralCoast']
# selected_regions = ['american']

# Max lead time and ensemble member amount of interest:
max_ens_members = 40
max_lead_time = 14
selected_lead_times = [1, 3, 5, 7, 10, 14]

# Specify number of cores to use when downloading CNRFC forecasts (dynamically, based on system):
slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
if slurm_cpus is not None:
    cores_for_download = int(slurm_cpus) - 1  # subtract 1 if you want to leave one free
else:
    import multiprocessing
    cores_avail = multiprocessing.cpu_count()
    cores_for_download = cores_avail - 1
print(f"Using {cores_for_download} cores.")


# Data cleaning:
percent_threshold_data_availability = 0.9 #percentage of time period that must be present

# AR recon:
AR_recon_window = 3 # window for how many days in advance to consider AR recons for a given forecast date

# Flow thresholds:
no_flow_threshold = 0.0
moderate_flow_threshold = 0.5
high_flow_threshold = 0.9
#flow_thresholds = [no_flow_threshold, moderate_flow_threshold, high_flow_threshold]
flow_thresholds = [high_flow_threshold, moderate_flow_threshold]

# calculate probabilistic metrics?
calculate_probabilistic = True 

# Aggregation periods; options are 'water_year' and 'window'; main analysis done with window
aggregation_periods = ['window']

window_month_start = 10 # start month of window aggregation period (i.e., 10 = October)
window_month_end = 5 # end month of window aggregation period (i.e., 5 = May, included in window)

# Metrics:
selected_metrics = ['RMSE', 'eCRPS_mean', 'rel']
selected_metrics_string = '_'.join(map(str, selected_metrics))

# Minimum number of events for a given site-water year combination to be included in analysis;
# mainly relevant when examining high flow events/ less frequent events;
# See Monte Carlo analysis for more details on selecting an appropriate threshold
event_threshold = 10

# Predictor combinations for bayesian hierarchical models:
model_configs = {
    "wet": {"wet": True, "time": False, "AR": False, "GEFS": False},
    "wet_time": {"wet": True, "time": True, "AR": False, "GEFS": False},
    "wet_AR": {"wet": True, "time": False, "AR": True, "GEFS": False},
    "wet_GEFS": {"wet": True, "time": False, "AR": False, "GEFS": True},
    "AR": {"wet": False, "time": False, "AR": True, "GEFS": False},
    "GEFS": {"wet": False, "time": False, "AR": False, "GEFS": True},
    "time": {"wet": False, "time": True, "AR": False, "GEFS": False},
    "wet_time_AR": {"wet": True, "time": True, "AR": True, "GEFS": False},
    "wet_time_GEFS": {"wet": True, "time": True, "AR": False, "GEFS": True},
    "wet_time_AR_GEFS": {"wet": True, "time": True, "AR": True, "GEFS": True},
}

# Bayesian model tuning and sampling params:
prior_samples = 1000
tuning_samples = 10000 #ideally tune = 10000
posterior_samples =10000 #ideally posterior = 10000

number_chains = 4

check_convergence = True

basin_characteristics = True
basin_char_names = ["precip", "elevation", "relief_ratio", "drainage_area", "num_dams", "permeability", "intercept"]
basin_char_columns = ['PPTAVG_BASIN', 'ELEV_MEAN_M_BASIN', 'RRMEAN', 'DRAIN_SQKM_x_log', 'NDAMS_2009', 'PERMAVE']
# display names for basin characteristics
basin_char_names_display = {
    "precip":          "Mean Precip",
    "elevation":       "Mean Elevation",
    "relief_ratio":    "Relief Ratio",
    "drainage_area":   "Drainage Area (log)",
    "num_dams":        "Dams Upstream",
    "permeability":    "Permeability",
    "intercept":       "Intercept",
}

#Key folder paths and file names:
root_folder = Path('analysis')
data_path = BASE_DIR / root_folder / 'data'
log_path = BASE_DIR / root_folder / 'log' # if varying posterior samples, use the following: f'log_{tuning_samples}tune_{posterior_samples}posterior'
plots_path = BASE_DIR / root_folder / 'plots' # if varying posterior samples, use the following: f'plots_{tuning_samples}tune_{posterior_samples}posterior'
bayesian_traces_path = BASE_DIR / root_folder / 'bayesian_traces' # if varying posterior samples, use the following: f'bayesian_traces_{tuning_samples}tune_{posterior_samples}posterior'

# File paths (as Path objects)
CNRFC_forecasts_folder_name = data_path / 'CNRFC_Forecasts'
AR_recon_records_filename = data_path / 'AR_recon_cw3e_records.csv'
NWS_USGS_ID_Table_All = data_path / 'NWS_USGS_ID_Table_All.csv'
NWS_USGS_ID_Table_SufficientData = data_path / 'NWS_USGS_ID_Table_SufficientData.csv'
basin_char_folder = data_path / 'basin_characteristics'
ID_Table_Basin_Char = data_path / 'NWS_USGS_ID_Table_SufficientData_basinchar.csv'
USGS_data_filename = data_path / f'USGS_daily_{start_date}_{end_date}.csv'
metric_folder = data_path / 'metric_calculations'
metric_daily_filename = metric_folder / 'metrics_daily.csv'
monte_carlo_reliability_folder = plots_path / 'MonteCarloReliability'

# Ensure subdirectories exist
for folder in [data_path, log_path, plots_path, bayesian_traces_path, CNRFC_forecasts_folder_name, basin_char_folder, monte_carlo_reliability_folder, metric_folder]:
    folder.mkdir(parents=True, exist_ok=True)

# Logging mechanism:
class MyLogger:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        self.timestamp_start = datetime.now()
        header_line = f"\nScript output started at {self.timestamp_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
        with self.file_path.open('a') as f:
            f.write(header_line)

    def log(self, *args):
        s = ''.join(str(arg) for arg in args)
        print(s)
        with self.file_path.open('a') as file:
            file.write(s + '\n')

    def end(self):
        timestamp_end = datetime.now()
        elapsed = timestamp_end - self.timestamp_start
        footer = (
            f"\nScript output ended at {timestamp_end.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Elapsed time: {str(elapsed)}\n"
        )
        with self.file_path.open('a') as f:
            f.write(footer)