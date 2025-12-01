"""
Steinschneider Lab Group
Madeline Allen
Code last updated: 7/30/2025
Script purpose: Additional visualizations

This project utilizes code generated with the assistance of OpenAI's ChatGPT (July, 2025). 
"""


import re
import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
import geopandas as gpd
import contextily as ctx
import arviz as az
import arviz.labels as azl
import xarray as xr
import seaborn as sns 
import copy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
# local
import config
from config import MyLogger

# Set up logger
current_script_name = Path(__file__).stem
logger = MyLogger(config.log_path / f'log_{current_script_name}.txt')


# import from config
from config import (
    bayesian_traces_path,
    plots_path,
    selected_metrics,
    selected_lead_times,
    aggregation_periods,
    flow_thresholds,
    model_configs,
    metric_folder,
    data_path
)

selected_metrics_decadal = ['RMSE_norm', 'eCRPS_mean_std', 'rel']


ID_table = pd.read_csv(config.ID_Table_Basin_Char)
selected_sites = ID_table['CNRFC_id'].unique()

'''
Examine ensemble forecasts for specific site & date range
'''
def read_in_ens_forecasts():
    """
    Read in ensemble forecasts and USGS observed data, merge them, and prepare for visualization.
    """
    #Read in ens forecasts:
    # Read in USGS data
    usgs_data = pd.read_csv(config.USGS_data_filename)
    usgs_data['datetime'] = pd.to_datetime(usgs_data['datetime']).dt.tz_localize(None)
    usgs_data = usgs_data[(usgs_data['datetime'] >= config.start_date) & (usgs_data['datetime'] <= config.end_date)]
    usgs_data = usgs_data[usgs_data['site_no'].isin(ID_table['USGS_id'])]

    # Read in CNRFC forecast data
    file_paths = [config.CNRFC_forecasts_folder_name / f'cnrfc14d_{year}.nc' for year in config.years]
    qf = xr.open_mfdataset(file_paths, combine='nested', concat_dim='time')
    qf = qf.sel(ens=slice(1, config.max_ens_members), lead=config.selected_lead_times, time=slice(config.start_date, config.end_date), site=selected_sites)
    qf_df = qf.to_dataframe().reset_index().replace(-999000, pd.NA)

    # Merge forecasts and observed flow
    merged = pd.merge(ID_table, usgs_data, left_on='USGS_id', right_on='site_no', how='inner')
    qf_df_ens = pd.merge(qf_df, merged, left_on=['site', 'time'], right_on=['CNRFC_id', 'datetime'], how='left')
    qf_df_ens = qf_df_ens[['site', 'time', 'ens', 'lead', 'Qf', 'Qo']]

    #add average ensemble forecast:
    qf_df_ens['Qf_mean'] = qf_df_ens.groupby(['site', 'time', 'lead'])['Qf'].transform('mean')

    return qf_df_ens

def visualize_ens_forecast(qf_df_ens, 
                           site='AHOC1', 
                           lead=1, 
                           start_date_vis='2022-10-01', 
                           end_date_vis='2023-05-31'):
    """
    Visualize ensemble forecasts for a specific site and lead time.

    Parameters:
    - qf_df_ens: DataFrame with ensemble forecast, mean forecast, and observed flow.
    - site: CNRFC site code.
    - lead: Lead time in days to filter.
    - start_date: Start date (string or datetime).
    - end_date: End date (string or datetime).
    """
    # Convert to datetime if not already
    start_date_vis = pd.to_datetime(start_date_vis)
    end_date_vis = pd.to_datetime(end_date_vis)

    # Filter data
    df_plot = qf_df_ens[
        (qf_df_ens['site'] == site) &
        (qf_df_ens['lead'] == lead) &
        (qf_df_ens['time'] >= start_date_vis) &
        (qf_df_ens['time'] <= end_date_vis)
    ]

    if df_plot.empty:
        print("No data available for selected filters.")
        return

    # Create figure
    plt.figure(figsize=(9, 6))

    # Plot each ensemble member
    for ens_num in df_plot['ens'].unique():
        ens_data = df_plot[df_plot['ens'] == ens_num]
        plt.plot(ens_data['time'], ens_data['Qf'], color='lightblue', alpha=0.5, linewidth=1)

    # Plot ensemble mean
    df_mean = df_plot.drop_duplicates(subset=['site', 'time', 'lead'])[['time', 'Qf_mean']]
    plt.plot(df_mean['time'], df_mean['Qf_mean'], label='Ensemble Mean', color='blue', linewidth=2)

    # Plot observed flow
    df_obs = df_plot.drop_duplicates(subset=['site', 'time', 'lead'])[['time', 'Qo']]
    plt.plot(df_obs['time'], df_obs['Qo'], label='Observed Flow', color='black', linestyle='--', linewidth=2)

    # Labels and legend
    plt.title(f'Ensemble Forecasts for Site {site} (Lead {lead})', fontsize=14)
    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Flow (cfs)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = folder_path / f"ens_forecast_vis_{site}_ld{lead}_{start_date_vis.strftime('%Y-%m-%d')}_{end_date_vis.strftime('%Y-%m-%d')}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()


'''
Plot geographic variation in basin characteristics
'''

basin_char_columns = config.basin_char_columns # ['PPTAVG_BASIN', 'ELEV_MEAN_M_BASIN', 'RRMEAN', 'DRAIN_SQKM_x_log', 'NDAMS_2009', 'PERMAVE']
# display names for basin characteristics:
basin_char_titles = ["Mean Precipitation","Mean Elevation","Relief Ratio","Drainage Area (log)","Dams Upstream","Permeability"]

basin_char_cbar_labels = ["Mean Annual Precip [cm]","Mean Watershed Elevation [m]","Relief Ratio [-]","Drainage Area [log-sqare-miles]","Number of Dams Upstream","Average Permeability [inches/hour]"]


import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
import numpy as np

def plot_basin_char_maps(ID_table, basin_char_columns, basin_char_titles, basin_char_cbar_labels, folder_path):
    """
    Plot maps of 6 basin characteristics for 97 sites using the viridis colormap.

    Parameters:
    - ID_table (DataFrame): Contains site info and basin characteristics
    - basin_char_columns (list): Column names in the DataFrame
    - basin_char_titles (list): Short names for subplot titles
    - basin_char_cbar_labels (list): Full names with units for colorbars
    - folder_path (Path): Where to save the output figure
    """
    assert len(basin_char_columns) == 6
    assert len(basin_char_titles) == 6
    assert len(basin_char_cbar_labels) == 6

    gdf = gpd.GeoDataFrame(
        ID_table.copy(),
        geometry=gpd.points_from_xy(ID_table['LNG_GAGE'], ID_table['LAT_GAGE']),
        crs="EPSG:4269"
    ).to_crs(epsg=3857)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    cmap = plt.get_cmap('viridis')

    for i, (col, short_title, cbar_label) in enumerate(zip(basin_char_columns, basin_char_titles, basin_char_cbar_labels)):
        gdf_col = gdf.copy()

        vmin_10, vmax_90 = np.percentile(gdf_col[col], [10, 90])
        norm = Normalize(vmin=vmin_10, vmax=vmax_90)

        gdf_col.plot(
            ax=axes[i],
            color=[cmap(norm(val)) for val in gdf_col[col]],
            edgecolor='k',
            markersize=50,
            alpha=0.9
        )
        ctx.add_basemap(axes[i], source=ctx.providers.Esri.WorldTopoMap, alpha=0.6, attribution="")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(short_title, fontsize=13)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=axes[i], orientation='vertical', fraction=0.035)
        cbar.set_label(cbar_label, fontsize=11)

    fig.suptitle(f'Basin Characteristics at {len(selected_sites)} CNRFC Sites', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(folder_path / 'basin_characteristics_map.png', dpi=300, bbox_inches='tight')
    # plt.show()


qf_df_ens = read_in_ens_forecasts()

print(qf_df_ens.head())

for agg_type in aggregation_periods:
    for flow in [0.9]: #flow_thresholds:
         # make folders for plots: 
        folder_path = plots_path / f'plots_{agg_type}_Thresh{flow}/'
        folder_path.mkdir(parents=True, exist_ok=True)
        # plot ens forecast visualization:
        visualize_ens_forecast(qf_df_ens, 
                            site='AHOC1', 
                            lead=1, 
                            # start_date_vis='2022-12-01', 
                            # end_date_vis='2023-01-30'
                            start_date_vis='2022-10-01', 
                            end_date_vis='2023-05-31'

                           )
    
        plot_basin_char_maps(ID_table, basin_char_columns, basin_char_titles, basin_char_cbar_labels, folder_path)

logger.end()