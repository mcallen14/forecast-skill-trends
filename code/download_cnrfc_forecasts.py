"""
Steinschneider Lab Group
Madeline Allen
Code last updated: 7/16/2025
Script purpose: download CNRFC forecasts

The following code was adapted from Jon Herman et al., and this project utilizes code generated with the 
assistance of OpenAI's ChatGPT (July, 2025). 

Code for downloading *hourly* CNRFC data (which includes flow regulations, in contrast to daily ensemble data)
and aggregating to daily average values for daily time window of interest. The daily aggregation is set up to 
match the USGS daily aggreated values in California (midnight to midnight Pacific Time). Note that the first
20 hours of data are removed from the hourly ensemble data (which is 12:00GMT to 12:00GMT) to 
shift it to 8:00GMT - 8:00GMT (which is midnight-midnight Pacific Time); the first aggregated value is 
considered lead 1.

Begin by specifying the date range, forecast groups, and maximum lead time of interest.
Note that this code is set up to pull the site IDs of interest from a table containing 
NWS-USGS site ID pairs, obtained from this site: https://hads.ncep.noaa.gov/USGS/

Text files output as part of this script detail: 1) elapsed time of different steps, 2) the URLs for which 
the data download failed (from which you can extract the date and regions that data was not downloaded for), 
3) the list of dates that have less data available than the specified max lead time number of days, 4) dates 
missing data for all regions.

Key recent updates to the script include ensuring that sites with local flows or full natural flows are not 
selected (e.g., for the site BCAC1, we only want BCAC1 not BCAC1F - the latter is full natural flow).
Additionally, script was modified to examine all possible ensemble members for a given site on a given date, 
and randomly select 40 of the available ensemble members (rather than selecting the first 40) to ensuer that
if ensemble members are ordered in any way, our anlaysis is not impacted by that.

"""

# Import necessary packages
import time 
import numpy as np
import pandas as pd
from datetime import date, timedelta
import glob
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
# local
import config
from config import MyLogger

# Set up logger
current_script_name = Path(__file__).stem
logger = MyLogger(config.log_path / f'log_{current_script_name}.txt')

# Specify start date and end dates for forecast data retrieval:
sd_forecast =  config.start_date
ed_forecast = config.end_date

date_today = date.today().strftime("%B%Y")

# Specify regions of interest:
selected_regions = config.selected_regions

# Specify max lead time of interest:
max_lead_time = config.max_lead_time
max_ens_members = config.max_ens_members

# Path to where data will be stored
data_path = config.CNRFC_forecasts_folder_name


# Specify number of cores to use when downloading:
cores_for_download = config.cores_for_download

updating_status = False # **Haven't verified whether updating works with this script; keep set to False or verify before using


# Read in csv file containing the USGS Stream gauge ID - NWS station ID pairings:
ID_table = pd.read_csv(config.NWS_USGS_ID_Table_All)

# Make a list of all NWS sites from the table:
cnrfc_ids = [str(x) for x in ID_table['CNRFC_id']]

# List to store timing results
timing_results = []
timing_results.append(f'Elapsed time for data download and compilation for range: {sd_forecast} to {ed_forecast}')

'''
Download forecasts
The two functions below download the HOURLY ensemble data (12:00GMT to 12:00GMT) for given date range and given regions 
of interest, remove the first 20 hours to shift to 8:00GMT - 8:00 GMT, aggregate to daily data, and save csv files to 
local storage. Note that regions are downloaded in parallel; make sure cores used for downloading is correctly specified above.
'''

# Define top-level lists
failed_urls = []
limited_data_dates = []
incorrect_start_dates = []

def download_process_region(region, dates, archive_path, failed_urls, limited_data_dates, incorrect_start_dates):
    region_start_time = time.time()

    for d in dates:
        fn = archive_path / f'{d}_{region}.csv'
        if not fn.exists():
            # Construct URL for data download:
            zipurl = f'https://www.cnrfc.noaa.gov/csv/{d}12_{region}_hefs_csv_hourly.zip'
            logger.log(zipurl)
            try:
                df = pd.read_csv(zipurl, index_col=0, parse_dates=True, low_memory=False, skiprows=[1])
                num_rows, num_columns = df.shape
                
                # Check if the first row of the df is the expected date and time; 
                # if not, remove initial rows that shouldn't be there
                date_correct = pd.to_datetime(d) + pd.Timedelta(hours=12) # Convert d to date time with time 12:00
                if df.index[0] != date_correct:
                    #logger.log(f'First date in {region} df was not correct')
                    incorrect_start_dates.append((region, d))
                    try:
                        date_correct_index = df.index.get_loc(date_correct)
                        # Keep rows from date_correct onwards
                        df = df.iloc[date_correct_index:]
                    except KeyError:
                        logger.log(f"Date {date_correct} not found in the data. Skipping this date.")
                        continue

                rows_needed = max_lead_time * 24 + 20

                # Check if there is sufficient data for max lead time specified; log if not
                if num_rows < rows_needed:
                    limited_data_dates.append((region, d))
                    rows_partial_day = (num_rows - 20) % 24
                    df.iloc[-rows_partial_day:] = np.nan

                # Remove the first 20 rows; isolate the hourly values needed for daily aggregate
                df = df.iloc[20:].head(max_lead_time * 24)
                
                
                # Isolate the columns with sites of interest
                # Filter columns that start with a valid 5-character code in cnrfc_ids, 
                # followed by nothing or a period and a number (e.g., ".1", ".2", etc.)
                keep = [
                    c for c in df.columns 
                    if re.match(rf"^({'|'.join(cnrfc_ids)})(\.\d+)?$", c)
                ]
                df = df[keep]

                # For each 5-character site code, randomly select 40 ensemble members
                selected_columns = []
                for site in cnrfc_ids:
                    # Find all columns for the current site
                    site_columns = [col for col in df.columns if col.startswith(site) and re.match(rf"^{site}(\.\d+)?$", col)]
                    # Randomly select max_ens_members ensemble members if more than that number are available
                    if len(site_columns) > max_ens_members:
                        site_columns = np.random.choice(site_columns, size=max_ens_members, replace=False).tolist()
                    selected_columns.extend(site_columns)

                # Keep only the selected columns
                df = df[selected_columns]


                #Convert the -999 to np.nan
                df.replace(-999, np.nan, inplace=True)

                # Convert columns to numeric
                df = df.apply(pd.to_numeric, errors='coerce')

                # Define custom aggregation function to handle NaN values properly
                def custom_agg(x): 
                    return np.nanmean(x) if len(x) > 0 else np.nan 
                
                # Calculate daily aggregated values; time shift of 8 hours for 8:00GMT-8:00GMT averages
                df = df.resample('D', offset=pd.Timedelta(hours=8)).apply(custom_agg)

                df.to_csv(fn)

            except Exception as e:
                logger.log(f'Error: {e}, skipping ...')
                failed_urls.append(zipurl)
            
    region_end_time = time.time()
    region_elapsed_time = region_end_time - region_start_time
    timing_results.append(f'{region}: {region_elapsed_time} seconds') # Log elapsed time for the region

def download_cnrfc_ensemble_hourly(archive_path, updating=False):
    download_start_time = time.time()

    sd = sd_forecast
    if updating:
        all_files = sorted(archive_path.glob('*.csv'))
        if all_files:
            last = all_files[-1]
            sd = re.findall(r'\d+', last.name)[0]
        else:
            sd = sd_forecast


    dates = pd.date_range(start=sd, end=ed_forecast, freq='D').strftime('%Y%m%d')
    regions = selected_regions

    # Download regions in parallel
    with ThreadPoolExecutor(max_workers=cores_for_download) as executor:
        futures = {executor.submit(download_process_region, r, dates, archive_path, failed_urls, limited_data_dates, incorrect_start_dates): r for r in regions}
        for future in as_completed(futures):
            future.result()

    if failed_urls:
        log_filename = data_path / f'CNRFC_data_download_{sd_forecast}_to_{ed_forecast}_accessed_{date_today}_log_failed.txt'
        with open(log_filename, 'w') as f:
            f.write('\n'.join(failed_urls))
        logger.log(f'Failed URLs logged in: {log_filename}')

    if limited_data_dates:
        limited_data_log_filename = data_path / f'CNRFC_data_download_{sd_forecast}_to_{ed_forecast}_accessed_{date_today}_limited_data_dates.txt'
        with open(limited_data_log_filename, 'w') as f:
            for region, date in limited_data_dates:
                f.write(f'{region}: {date}\n')
        logger.log(f'Limited data dates logged in: {limited_data_log_filename}')

    if incorrect_start_dates:
        incorrect_start_dates_df = pd.DataFrame(incorrect_start_dates, columns=['Region', 'Date'])
        incorrect_start_dates_log_filename = data_path / f'CNRFC_data_download_{sd_forecast}_to_{ed_forecast}_accessed_{date_today}_incorrect_start_dates.csv'
        incorrect_start_dates_df.to_csv(incorrect_start_dates_log_filename, index=False)
        logger.log(f'Incorrect start dates logged in: {incorrect_start_dates_log_filename}')

    download_end_time = time.time()
    download_elapsed_time = download_end_time - download_start_time
    timing_results.append(f'Download elapsed time: {download_elapsed_time} seconds')

# Download CNRFC ensemble forecast data:
download_cnrfc_ensemble_hourly(data_path, updating=False)


'''
Compile forecasts by date for all locations
The code below compiles the forecasts - combining regions into one csv for each date within date range.
(With several changes made to the downloading process, this step can be skipped for most purposes, but is 
retained here because it makes the NetCDF creation script slightly more straightforward.)
'''

def compile_forecasts(archive_path, updating=updating_status):
    sd = sd_forecast
    regions = selected_regions
    dates_missing_data = []
    
    if updating:
        all_files = sorted(archive_path.glob('*_All.csv'))
        if all_files:
            last = all_files[-1]
            sd = re.findall(r'\d+', last.name)[0]
        else:
            sd = sd_forecast


    dates = pd.date_range(start=sd, end=ed_forecast, freq='D')

    compile_start_time = time.time()
    
    # Loop through each date to compile data from all regions into one csv
    for d in dates:
        logger.log(f'Compile forecasts from date {d}:')
        dfs = []
        for r in regions:
            fn = archive_path / f'{d.strftime("%Y%m%d")}_{r}.csv'

            if fn.exists():
                dfs.append(pd.read_csv(fn, index_col=0, parse_dates=True))

        if dfs:
            df = pd.concat(dfs, axis=1)
            df.to_csv(archive_path / f'{d.strftime("%Y%m%d")}_All.csv')

        else:
            logger.log(f'Missing all forecasts date {d.strftime("%Y%m%d")}')
            dates_missing_data.append(d.strftime('%Y%m%d'))

    compile_end_time = time.time()
    compile_elapsed_time = compile_end_time - compile_start_time
    timing_results.append(f'data compile: {compile_elapsed_time} seconds')

    if dates_missing_data:
        log_filename_comp = data_path / f'CNRFC_data_compilation_{sd_forecast}_to_{ed_forecast}_compiled_{date_today}_dates_missing_all_data.txt'
        with open(log_filename_comp, 'w') as f:
            f.write('\n'.join(dates_missing_data))
        logger.log(f'Failed URLs logged in: {log_filename_comp}')

# Compile forecasts:
compile_forecasts(data_path, updating=updating_status)

# Write the elapsed times to a text file
with open(data_path / f'CNRFC_data_{sd_forecast}_to_{ed_forecast}_compiled_{date_today}_time_elapsed.txt', 'w') as file:
    file.write('\n'.join(timing_results))

logger.end()