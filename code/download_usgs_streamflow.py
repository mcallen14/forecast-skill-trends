"""
Steinschneider Lab Group
Madeline Allen
Code last updated: 7/16/2025
Script purpose: download USGS observed streamflow

Code utilizes USGS dataretrieval package; for details and documentation see: https://github.com/DOI-USGS/dataretrieval-python

The portion of the following code using the dataretrieval package was adapted from an example on HydroShare
(obtained from https://www.hydroshare.org/resource/c97c32ecf59b4dff90ef013030c54264/)

This script downloads USGS data (daily observed discharge) for all stream gauges that have corresponding NWS station ID numbers where forecasts 
are available by the CNRFC. Several variables are used in this script can be configured in the config.py file, including the 
start date and end date of interest. Note that the single-site USGS stream gauge data retreival is used first
for a limited date range to verify whether data is available, and then the multi-site data retreival is used to actually
download and save the data for the full date range. (Not super efficient and may discard sites that have some available
information in full time window of interest, however multi-side download fails if any single site within that call fails; 
currently in the process of working on a better way to identify and remove problematic sites that will stop the multi-site download from working.)

This project utilizes code generated with the assistance of OpenAI's ChatGPT (July, 2025). 
"""

# Import necessary packages
import pandas as pd
from dataretrieval import nwis
from IPython.display import display
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
# local
import config
from config import MyLogger

# Set up logger
current_script_name = Path(__file__).stem
logger = MyLogger(config.log_path / f'log_{current_script_name}.txt')


#Specify date range of interest:
startDate = config.start_date
endDate = config.end_date


#Specify what data you want (e.g., discharge):
#Note: If you change this line, make sure the columns being exported to csv still make sense, and are named appropriately
parameterCode = "00060" # Discharge

#Read in csv file containing the USGS Stream gauge ID - NWS station ID pairs:
ID_table = pd.read_csv(config.NWS_USGS_ID_Table_All)
#Create list of unique station numbers:
site=[str(x) for x in ID_table['USGS_id'].unique()]

'''
Use the single-site USGS stream gauge data retreival using nwis.get_dv() from the dataretrieval package 
to examine which sites are problematic (and cause the multi-site data download to fail)
'''

#Create empty list to populate with problematic sites:
sites_to_remove = []

# Code in progress: note that this approach for finding problematic sites is limited by the date range specified; if a stream gauge is active for
# part of the time range of interest but is not active during small range specified below, it will not be included.
for site_1 in site:
    try:
        # Attempt to retrieve a few days of data to identify which USGS stream gauge sites are problematic
        dailyStreamflow = nwis.get_dv(sites=site_1, parameterCd=parameterCode, start="2017-01-10", end="2017-01-15") 
    except:
        #logger.log(site_1)
        #Add problematic sites to list
        sites_to_remove.append(site_1)

#Problematic sites to remove:
logger.log(f'Sites to remove: {sites_to_remove}.')
sites_to_remove_df = pd.DataFrame({'FailedSites':sites_to_remove})

#Create new siteNumber list without the problematic USGS stream gauges identified in the code above:
siteNumber = [s for s in site if s not in sites_to_remove]
logger.log(f'Initial number of unique sites: {len(site)}')
logger.log(f'Number of sites identified as problematic: {len(sites_to_remove)}')
logger.log(f'{len(sites_to_remove)} sites removed')
logger.log(f'Final number of unique sites: {len(siteNumber)}')


'''
Download all the unique USGS stream gauge sites that have information available. Note that code below will not work if 
problematic sites are not removed first.
'''
siteNumber = site

#Download data from all unique USGS sites that have information available:
sd = startDate 
ed = endDate

USGS_daily = nwis.get_dv(sites=siteNumber, parameterCd=parameterCode,
                              start=sd, end=ed)
# Note - can add other specifications within nwis.get_dv(), including an arguement for statCd; see documentation for more info

USGS_daily_df = USGS_daily[0]

#Convert the -999999 to pd.NaN 
USGS_daily_df['00060_Mean'] = USGS_daily_df['00060_Mean'].replace(-999999, pd.NA)


# Rename the column before saving
USGS_daily_df_renamed = USGS_daily_df.rename(columns={'00060_Mean': 'Qo'})

# Save to CSV with 'Qo' and its associated quality code
USGS_daily_df_renamed[['Qo', '00060_Mean_cd']].to_csv(config.USGS_data_filename)

logger.end()

