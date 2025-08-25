"""
Steinschneider Lab Group
Madeline Allen
Code last updated: 7/16/2025
Script purpose: compile CNRFC forecasts into netcdf files

The following code was adapted from Jon Herman et al., and this project utilizes code 
generated with the assistance of OpenAI's ChatGPT (July, 2025). 

Code for compiling daily average ensemble streamflow forecast data into a multidimensional array with the following
dimensions: site, time/date, ensemble member, and lead time ('site', 'time', 'ens', 'lead'). The values stored in the
multidimensional are streamflow ('Qf') in cubic feet per second. For a given site and date ('time'), the data associated with
the lead times are the forecasts FOR that specific date (rather than the forecasts made ON that given date), in order to 
facilitate comparisons between observed streamflow and forecasts of various lead times for a given date. The data is compiled 
annually, with one NetCDF file per year.
"""

# Import necessary packages
import time 
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
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


# Modify the code block below to specify the date range and max lead time of interest, and path to data folder
sd_forecast =  config.start_date
ed_forecast = config.end_date

max_lead_time = config.max_lead_time

archive_path = config.CNRFC_forecasts_folder_name

updating_status = False 

# Read in csv file containing the USGS Stream gauge ID - NWS station ID pairs
ID_table = pd.read_csv(config.NWS_USGS_ID_Table_All)



# Make a list of all NWS sites from the table
cnrfc_ids = [str(x) for x in ID_table['CNRFC_id']]
#logger.log(cnrfc_ids)

# List to store timing results
timing_results = []
timing_results.append(f'Elapsed time for netcdf creation for range: {sd_forecast} to {ed_forecast}')

date_today = date.today().strftime("%B%Y")

# Updated version of 14 day extract - flexible max lead time; annual NetCDF files
def extract_byyear(archive_path, updating=updating_status):
    netcdf_start_time = time.time()
    neg_values_log = []
    dates_not_rep = []
    sites = cnrfc_ids
    lead_time = max_lead_time
    max_ens = 40 #70

    years = list(range(datetime.strptime(sd_forecast, '%Y-%m-%d').year, datetime.strptime(ed_forecast, '%Y-%m-%d').year + 1))

    # Create a NetCDF file for each year in the date range
    for year in years:
        # Create temporary sd so that forecasts made in max_lead_time days before that year are included, and
        # create temporary ed so that forecasts made at end of year (out to max_lead_time into the next year) are included
        # Note that these are necessary for data compilations but are later trimmed so that the resulting NetCDF file 
        # contains only values for that given year
        sd = datetime(year, 1, 1).strftime('%Y-%m-%d')
        tmp_sd = (datetime(year, 1, 1) - timedelta(days=lead_time)).strftime('%Y-%m-%d')
        ed = datetime(year, 12, 31).strftime('%Y-%m-%d')
        tmp_ed = (datetime(year, 12, 31) + timedelta(days=lead_time)).strftime('%Y-%m-%d')

        # Construct empty multidimensional array
        dates_dim = pd.date_range(start=tmp_sd, end=tmp_ed, freq='D')
        dates = pd.date_range(start=tmp_sd, end=ed, freq='D')
        empty = np.zeros((len(sites), len(dates_dim), max_ens, lead_time))
        empty[:] = np.nan
        ds = xr.Dataset(data_vars=dict(Qf=(['site', 'time', 'ens', 'lead'], empty)),
                        coords=dict(site=(sites),
                                    time=dates_dim,
                                    ens=np.arange(1, max_ens + 1),
                                    lead=np.arange(1, lead_time + 1)))
        ds['Qf'].attrs = {'units': 'cfs'}

        dates = dates.strftime('%Y%m%d')

        for d in dates:
            file_name = f'{d}_All.csv'
            f = archive_path / file_name
            if f.exists():
                df_Qf = pd.read_csv(f, index_col=0, parse_dates=True)
                df_Qf = df_Qf.head(lead_time)

                count_negative_values = (df_Qf < 0).sum().sum()
                if count_negative_values > 0:
                    neg_values_log.append((d, count_negative_values))

                gmt_times = pd.to_datetime(df_Qf.index, format='%Y-%m-%d %H:%M:%S').values

                for s in sites:
                    df_site = df_Qf.filter(regex=r'^{0}((\.\d+)?)$'.format(s)) * 1000
                    n_lead, n_ens = df_site.shape

                    if n_ens == 0:
                        df_site = df_Qf.filter(regex='%s(.*?)\.' % s) * 1000
                        n_lead, n_ens = df_site.shape

                    # Truncate columns if there are more than max_ens columns - though there should not be, based on current download script
                    if n_ens > max_ens:
                        df_site = df_site.iloc[:, :max_ens]
                        n_ens = max_ens

                    # Extract all ensembles for each lead time (each date listed in csv) within a forecast, 
                    # and insert it into the appropriate portion of the multidimensional array
                    for t_idx, t in enumerate(gmt_times):
                        t_str = np.datetime_as_string(t, unit='D')
                        ix = dict(site=s, time=t_str, ens=np.arange(1, n_ens + 1), lead=t_idx + 1)

                        data = df_site.T.values[:n_ens, t_idx]
                        ds['Qf'].loc[ix] = data

                logger.log(f'Extract Forecasts from date {f}')
            else:
                logger.log(f'Missing all forecasts date {f}')
                dates_not_rep.append(d)
                continue

        # Trim multidimensional array to only contain given year
        ds_trimmed = ds.sel(time=slice(sd, ed))

        netcdf_file_name = 'cnrfc%sd_%s.nc' % (lead_time, year)
        netcdf_file_path = archive_path / netcdf_file_name
        ds_trimmed.to_netcdf(netcdf_file_path, 'w')


    if neg_values_log:
        neg_values_log_filename = archive_path / f'CNRFC_netcdfcreation_{sd_forecast}_to_{ed_forecast}_accessed_{date_today}_neg_values_log.txt'
        with open(neg_values_log_filename, 'w') as f:
            f.write('\n'.join([f"{d}, {count}" for d, count in neg_values_log]))
        logger.log(f'Negative values logged in: {neg_values_log_filename}')


    if dates_not_rep:
        dates_not_rep_filename = archive_path / f'CNRFC_netcdfcreation_{sd_forecast}_to_{ed_forecast}_accessed_{date_today}_dates_not_rep_log.txt'
        with open(dates_not_rep_filename, 'w') as f:
            f.write('\n'.join(dates_not_rep))
        logger.log(f'Dates not represented logged in: {dates_not_rep_filename}')

    netcdf_end_time = time.time()
    netcdf_elapsed_time = netcdf_end_time - netcdf_start_time
    timing_results.append(f'{netcdf_elapsed_time} seconds')

    with open(archive_path/ f'CNRFC_data_{sd_forecast}_to_{ed_forecast}_compiled_{date_today}_time_elapsed.txt', 'w') as file:
        file.write('\n'.join(timing_results))

# Run function; create netcdf file(s)
extract_byyear(archive_path=archive_path, updating=updating_status)

logger.end()