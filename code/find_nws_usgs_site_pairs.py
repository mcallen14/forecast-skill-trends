"""
Steinschneider Lab Group
Madeline Allen
Code last updated: 7/16/2025
Script purpose: Find NWS USGS Site Pairs


The following script downloads tables for states in the study region specified in 
define_key_variables.txt. The tables are used to compile a list of possible sites that 
could have CNRFC forecasts and USGS observed streamflow.

A list of locations that CNRFC issues ensemble streamflow forecasts for is obtained from a CNRFC website,
and the list of possible NWS-USGS sites is refined to only include locations that are also ensembel forecast locations.

The ID table for the final list of sites is exported as a csv.

This project utilizes code generated with the assistance of OpenAI's ChatGPT (July, 2025). 
"""

# load packages:
import requests
import pandas as pd
from io import StringIO
import xml.etree.ElementTree as ET
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
# local
import config
from config import MyLogger

# Set up logger
current_script_name = Path(__file__).stem
logger = MyLogger(config.log_path / f'log_{current_script_name}.txt')


# Access the text files for each state that contain NWS-USGS site pairs; combine and save resulting csv.
def download_and_process_file(url):
    response = requests.get(url)

    if response.status_code == 200:
        logger.log(f"Downloaded from {url} successfully.")
        text_data = response.text
    else:
        logger.log(f"Failed to retrieve file from {url}. Status code: {response.status_code}.")
        return None

    # Load into DataFrame directly from text using StringIO
    df = pd.read_csv(StringIO(text_data), sep='|', header=None)

    # Build column names from first 3 rows
    column_names = [
        ' '.join(
            str(col).strip()
            if not pd.isna(col) and not (isinstance(col, float) or str(col).strip().isdigit())
            else ''
            for col in row
        )
        for row in zip(*df.iloc[:3].values.tolist())
    ]

    df = df.iloc[4:]  # Remove header rows
    df.columns = column_names
    df.reset_index(drop=True, inplace=True)

    return df

# Process each state in memory
dfs = []
for state in config.states_in_study_region:
    url = f"https://hads.ncep.noaa.gov/USGS/{state}_USGS-HADS_SITES.txt"
    df = download_and_process_file(url)
    if df is not None:
        dfs.append(df)

# Combine
combined_df = pd.concat(dfs, ignore_index=True)

# Rename columns
combined_df = combined_df.rename(columns={
    'NWS 5 CHR IDENT': 'CNRFC_id',
    'USGS STATION NUMBER': 'USGS_id',
    ' GOES IDENTIFR': 'GOES_id',
    ' NWS HSA': 'NWS_HSA',
    ' LATITUDE dd mm ss': 'Lat',
    ' LONGITUDE ddd mm ss': 'Lon',
    ' LOCATION NAME': 'Name'
})

# optional - remove columns not used in analysis:
combined_df = combined_df[['CNRFC_id', 'USGS_id']]


'''
Create list of all potential ensemble forecast locations (regradless of archive length) from CNRFC website:
'''


# URL of the XML data
url = 'https://www.cnrfc.noaa.gov/data/kml/ensPoints.xml?random=0.19970156070153156'

# Fetch the XML data
response = requests.get(url)
xml_data = response.content

# Parse the XML
root = ET.fromstring(xml_data)

# Extract data into a list of dictionaries
data = []
for point in root.findall('ensPoints'):
    data.append({
        'amount': point.get('amount'),
        'riverName': point.get('riverName'),
        'stationName': point.get('stationName'),
        'latitude': float(point.get('latitude')),
        'longitude': float(point.get('longitude')),
        'printLat': float(point.get('printLat')),
        'printLon': float(point.get('printLon')),
        'printElev': float(point.get('printElev')),
        'id': point.get('id'),
        'idLow': point.get('idLow')
    })

# Create DataFrame
enspoints_df = pd.DataFrame(data)


'''
Filter the df of NWS-USGS sites (combined_df) to only include those where CNRFC issues ensemble forecasts
'''
# Filter combined_df where CNRFC_id exists in enspoints_df['id']
combined_df_filtered = combined_df[combined_df['CNRFC_id'].isin(enspoints_df['id'])].copy()

# Optional: check the result
logger.log(f"Filtered from {len(combined_df)} to {len(combined_df_filtered)} rows.")

# save:
#combined_df_filtered.to_csv(f'{config.data_path}{config.NWS_USGS_ID_Table_All}', index=False)
combined_df_filtered.to_csv(config.NWS_USGS_ID_Table_All, index=False)
logger.log("Filtered combined CSV saved successfully for NWS-USGS site pairs that also are ensemble forecast locations.")

logger.end()