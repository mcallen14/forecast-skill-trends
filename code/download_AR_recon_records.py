"""
Steinschneider Lab Group
Madeline Allen
Code last updated: 7/16/2025
Script purpose: AR Recon Data Extraction

Extract the data from the various tabs of the tables on the CW3E AR Recon overview site: https://cw3e.ucsd.edu/arrecon_overview/

The following code extracts the data, collapses the two rows of headers into one to make it easier to access data later, 
and then adds a column 'ncep_daily_asim' that is the sum of all dropsondes integrated into NCEP that day. 
Note about timezone conversions: the far left Date column of the AR recon record tables on the website are associated with a
date at 00Z, and the three assimilation windows associated with that row are from 18Z the prior day, 00Z and 06Z the current day 
(all of which align with one PST date). We add another date column that represents the date in PST of the 12Z ensemble streamflow
forecast issued by CNRFC that would have incorporated the data from these three assimilation windows.

This project utilizes code generated with the assistance of OpenAI's ChatGPT (July, 2025). 

"""

import requests
from bs4 import BeautifulSoup
import csv
from datetime import date, datetime, timedelta
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
# local
import config
from config import MyLogger

# Set up logger
current_script_name = Path(__file__).stem
logger = MyLogger(config.log_path / f'log_{current_script_name}.txt')



url = "https://cw3e.ucsd.edu/arrecon_overview/"

response = requests.get(url)
response.raise_for_status()
soup = BeautifulSoup(response.text, 'html.parser')

# Prepare to collect all clean rows
refined_rows = []
header_combined = None

# Select all divs with id starting with 'tabs_desc_24498_' and a number > 1
data_divs = soup.find_all("div", id=re.compile(r"tabs_desc_24498_\d+"))
for div in data_divs:
    div_id = div.get("id")
    match = re.match(r"tabs_desc_24498_(\d+)", div_id)
    if not match or int(match.group(1)) <= 1:
        continue  # skip summary or invalid tab

    table = div.find("table")
    if table is None:
        continue

    rows = table.find_all('tr')
    first_two_processed = False
    first_two_rows = 0
    ncep_column_index = None
    date_column_index = None

    for row in rows:
        cells = row.find_all(['td', 'th'])
        data = []

        for cell in cells:
            text = cell.get_text(strip=True)
            if text.startswith('Dropsondes'):
                data.extend([text] * 3)
            elif text.startswith('Assimilated Drops'):
                data.extend([text] * 3)
            else:
                data.append(text)

        if not first_two_processed:
            if first_two_rows == 0:
                header_combined = data
            elif first_two_rows == 1:
                for i in range(1, len(data)):
                    header_combined[i] = f"{header_combined[i]}_{data[i]}"
                header_combined.append("NCEP_daily_asim")
                header_combined.append("Date_PST")
                first_two_processed = True
            first_two_rows += 1
        else:
            # Handle short rows
            if len(data) < len(header_combined) - 2:
                data.extend([''] * (len(header_combined) - 2 - len(data)))

            # NCEP value cleaning
            try:
                ncep_value = data[header_combined.index("Assimilated Drops (18/00/06 UTC)*_NCEP")]
                clean_values = re.sub(r"\([^)]*\)", "", ncep_value).split('/')
                ncep_daily_asim = sum(int(val.strip()) for val in clean_values if val.strip().isdigit())
            except Exception:
                ncep_daily_asim = None
            data.append(ncep_daily_asim)

            # Date cleaning
            try:
                date_str = data[header_combined.index("Date")]
                date_str = re.sub(r"\s+", "", date_str)  # remove spaces like "00 UTC 29 Feb 2024"
                date_utc = datetime.strptime(date_str, "%HUTC%d%b%Y")
                date_pst = (date_utc - timedelta(hours=8)).date()
            except Exception:
                date_pst = None
            data.append(date_pst)

            if data[0].startswith("00 UTC") or data[0].startswith("12 UTC"):
                refined_rows.append(data)

# Finally write to one cleaned CSV file
with open(config.data_path / config.AR_recon_records_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_combined)
    writer.writerows(refined_rows)

logger.log("CSV file written: all AR recon records for available WY.")

logger.end()