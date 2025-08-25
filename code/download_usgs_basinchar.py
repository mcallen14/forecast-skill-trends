"""
Steinschneider Lab Group
Madeline Allen
Code last updated: 7/16/2025
Script purpose: download basin characteristic data for selected sites from the Gages-II dataset.

Dataset reference:
Falcone, J., 2011, GAGES-II: Geospatial Attributes of Gages for Evaluating Streamflow: 
U.S. Geological Survey data release, https://doi.org/10.5066/P96CPHOT.
Access link for manual download if needed: https://www.sciencebase.gov/catalog/item/631405bbd34e36012efa304a 

This project utilizes code generated with the assistance of OpenAI's ChatGPT (July, 2025). 
"""

# Import packages
import requests
import numpy as np
import pandas as pd
import zipfile
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
# local
import config
from config import MyLogger

# Set up logger
current_script_name = Path(__file__).stem
logger = MyLogger(config.log_path / f'log_{current_script_name}.txt')


# Constants
basin_char_folder = config.basin_char_folder
ZIP_NAME = 'basinchar_and_report_sept_2011.zip'
SB_ITEM_ID = '631405bbd34e36012efa304a'
SB_URL = f'https://www.sciencebase.gov/catalog/item/{SB_ITEM_ID}?format=json'
ID_table = pd.read_csv(config.NWS_USGS_ID_Table_SufficientData)

# Download metadata
response = requests.get(SB_URL)
item_data = response.json()

# Locate the zip file only
zip_file_info = next(
    (f for f in item_data['files'] if f['name'] == ZIP_NAME), None
)
if not zip_file_info:
    raise Exception("ZIP file not found in ScienceBase item.")

# Download the zip file if it doesn't already exist
zip_path = basin_char_folder / ZIP_NAME
if not zip_path.exists():
    logger.log(f'Downloading {ZIP_NAME}...')
    r = requests.get(zip_file_info['url'])
    with open(zip_path, 'wb') as f:
        f.write(r.content)
else:
    logger.log(f'{ZIP_NAME} already exists.')

# Extract only needed Excel files
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for filename in ['gagesII_sept30_2011_conterm.xlsx', 'gagesII_sept30_2011_var_desc.xlsx']:
        if not (basin_char_folder / filename).exists():
            logger.log(f'Extracting {filename}...')
            zip_ref.extract(filename, basin_char_folder)

# Proceed with your original script
selected_sites = ID_table['USGS_id'].tolist()
logger.log('Number of sites:', len(selected_sites))
gagesii_desc = pd.read_excel(basin_char_folder / 'gagesII_sept30_2011_var_desc.xlsx', engine='openpyxl')
gagesii_og = pd.read_excel(basin_char_folder / 'gagesII_sept30_2011_conterm.xlsx', sheet_name=None, engine='openpyxl')

sheet_names = list(gagesii_og.keys())[:-1]
filtered_data = [
    gagesii_og[sheet][gagesii_og[sheet]['STAID'].isin(selected_sites)]
    for sheet in sheet_names
]

gagesii_selected_sites = filtered_data[0]
for filtered_sheet in filtered_data[1:]:
    gagesii_selected_sites = pd.merge(gagesii_selected_sites, filtered_sheet, on='STAID', how='outer')

gagesii_selected_sites.to_csv(basin_char_folder / f'gagesii_selected_sites.csv')
logger.log('Filtered GAGES-II data saved.')

# Filter to only include key basin characteristics, save updated ID table:

#Filter gagesii_selected_sites by variables of interest:
predictors1 = [ 'LAT_GAGE', 'LNG_GAGE', 'PPTAVG_BASIN', 'ELEV_MEAN_M_BASIN', 'RRMEAN', 'NDAMS_2009', 'PERMAVE', 'DRAIN_SQKM_x'
]

gagesii_selected_variables = gagesii_selected_sites[['STAID']+ predictors1].copy()

# Add log-transformed DRAIN_SQKM_x and drop original
gagesii_selected_variables['DRAIN_SQKM_x_log'] = np.log(gagesii_selected_variables['DRAIN_SQKM_x'])


# Count problematic values
n_inf = np.isinf(gagesii_selected_variables['DRAIN_SQKM_x_log']).sum()
n_nan = gagesii_selected_variables['DRAIN_SQKM_x_log'].isna().sum()

# Replace problematic values with 0
gagesii_selected_variables['DRAIN_SQKM_x_log'] = gagesii_selected_variables['DRAIN_SQKM_x_log'].replace([np.inf, -np.inf], 0)
gagesii_selected_variables['DRAIN_SQKM_x_log'] = gagesii_selected_variables['DRAIN_SQKM_x_log'].fillna(0)

# Print a warning if any replacements were made
if n_inf > 0 or n_nan > 0:
    logger.log(f"Warning: Replaced {n_inf} infinite and {n_nan} NaN values in log(DRAIN_SQKM_x) with 0.")

# Drop the original (non-log) version
gagesii_selected_variables = gagesii_selected_variables.drop(columns=['DRAIN_SQKM_x'])


# Merge with ID table
ID_table_basin_char = pd.merge(
    ID_table,
    gagesii_selected_variables,
    left_on='USGS_id',
    right_on='STAID',
    how='left'
)

# help - get rid of STAID now that i've merged
ID_table_basin_char = ID_table_basin_char.drop(columns=['STAID'])

# Save final merged table
ID_table_basin_char.to_csv(config.ID_Table_Basin_Char, index=False)
logger.log(f'Merged ID table with basin characteristics saved to: {config.ID_Table_Basin_Char}')
logger.end()
