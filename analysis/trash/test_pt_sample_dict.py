# %% [markdown]
# ## Import libraries

# %%
# basic libs
import os
import numpy as np
import pandas as pd
import datetime

# data processing libs
import rasterio as rio
import xarray as xr
import rioxarray
import json
import requests
from pyproj import CRS
from osgeo import gdal
from math import ceil, floor

# plotting libs
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from odc.stac import stac_load
import planetary_computer
import pystac_client
import rich.table
import warnings
warnings.filterwarnings("ignore")



# %% [markdown]
# ## Some parameters

# %%

input_path = r"G:\Shared drives\Ryoko and Hilary\SMSigxSMAP\analysis\1_data"
output_path = r"G:\Shared drives\Ryoko and Hilary\SMSigxSMAP\analysis\3_data_out"
appears_path = r"APPEEARS_subsetting"
SMAPL3_path = r".\SPL3SMP_E"
SMAPL4_path = r".\SPL4SMGP"
SMAPL4_grid_path = r".\SMAPL4SMGP_EASEreference"
MODIS_path = r".\MOD15A2H"
# os.chdir("G:\Shared drives\Ryoko and Hilary\SMSigxSMAP\analysis")

network_name = "OZNET"

# %% [markdown]
# ## 1. Load EASE grid

# %%
fn = "SMAP_L4_SM_lmc_00000000T000000_Vv7032_001.h5"
file_path = os.path.join(input_path, SMAPL4_grid_path, fn)
# if os.path.exists(file_path):
#     print('The file exists')
# else:
#     print('The file does NOT exist')

g = gdal.Open(file_path)
subdatasets = g.GetSubDatasets()

varname_lat = "cell_lat"
full_varname_lat = f'HDF5:"{file_path}"://{varname_lat}'

varname_lon = "cell_lon"
full_varname_lon = f'HDF5:"{file_path}"://{varname_lon}'

varname_ease_column = "cell_column"
full_varname_ease_column = f'HDF5:"{file_path}"://{varname_ease_column}'

varname_ease_row = "cell_row"
full_varname_ease_row = f'HDF5:"{file_path}"://{varname_ease_row}'

ease_lat = rioxarray.open_rasterio(full_varname_lat)
ease_lon = rioxarray.open_rasterio(full_varname_lon)
ease_column = rioxarray.open_rasterio(full_varname_ease_column)
ease_row = rioxarray.open_rasterio(full_varname_ease_row)


# %% [markdown]
# ## Load Appears sample request

# %%
file_path = os.path.join(input_path, appears_path, network_name, f'{network_name}-request.json')
with open(file_path, 'r') as infile:
    request_content = json.load(infile)
coordinates = request_content['params']['coordinates']
dates = request_content['params']['dates']
coordinates
# dates

# %% [markdown]
# ## Loop for the target coordinates (currently set i=0)

# %%
for i in range(len(coordinates)): 
    target_lat = coordinates[i]['latitude']
    target_lon = coordinates[i]['longitude']
    target_station = coordinates[i]['category'].split()[0]
    
    print(f'processing{i}/{len(coordinates)} station: {target_station}')

    # %% [markdown]
    # ## Load APPEEARS output (SMAPL3)

    # %%
    file_path = os.path.join(input_path, appears_path, network_name, f'{network_name}-SPL3SMP-E-005-results.csv')
    SMAPL3_pt_sample = pd.read_csv(file_path)
    SMAPL3_pt_sample = SMAPL3_pt_sample[(SMAPL3_pt_sample['Latitude'] == target_lat) & (SMAPL3_pt_sample['Longitude'] == target_lon)].copy()
    SMAPL3_pt_sample.columns

    # %%
    df_ts_smap_am = SMAPL3_pt_sample[['Date', 'SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_AM_soil_moisture','SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag']].copy()
    df_ts_smap_am['Date'] = pd.to_datetime(df_ts_smap_am['Date'])
    df_ts_smap_am.set_index('Date', inplace=True)
    bad_data_idx_smap = df_ts_smap_am[(df_ts_smap_am['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag'] != 0.0) & (df_ts_smap_am['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag'] != 8.0)].index
    df_ts_smap_am.drop(bad_data_idx_smap, inplace=True)
    df_ts_smap_am_daily = df_ts_smap_am['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_AM_soil_moisture'].resample('D', axis=0).mean()

    df_ts_smap_pm = SMAPL3_pt_sample[['Date', 'SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm','SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm']].copy()
    df_ts_smap_pm['Date'] = pd.to_datetime(df_ts_smap_pm['Date'])
    df_ts_smap_pm.set_index('Date', inplace=True)
    bad_data_idx_smap = df_ts_smap_pm[(df_ts_smap_pm['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm'] != 0.0) & (df_ts_smap_pm['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm'] != 8.0)].index
    df_ts_smap_pm.drop(bad_data_idx_smap, inplace=True)
    df_ts_smap_pm_daily = df_ts_smap_pm['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm'].resample('D', axis=0).mean()

    df_ts_sync = pd.merge(df_ts_smap_am_daily, df_ts_smap_pm_daily, how='inner', left_index=True, right_index=True)
    if df_ts_sync.empty:
        print(f'probably {target_station} did not have any good data')
        continue
    
    print(df_ts_sync.head())
    df_ts_sync['soil_moisture_smapL3'] = df_ts_sync[['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_AM_soil_moisture','SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm']].mean(axis=1, skipna=True).copy()
    df_ts_sync['soil_moisture_smapL3'] = df_ts_sync['soil_moisture_smapL3'].resample('D', axis=0).mean().copy()


    # %% [markdown]
    # ## Load APPEEARS output (SMAPL4)

    # %%
    file_path = os.path.join(input_path, appears_path, network_name, f'{network_name}-SPL4SMGP-006-results.csv')
    SMAPL4_pt_sample = pd.read_csv(file_path)
    SMAPL4_pt_sample = SMAPL4_pt_sample[(SMAPL4_pt_sample['Latitude'] == target_lat) & (SMAPL4_pt_sample['Longitude'] == target_lon)].copy()
    print(SMAPL4_pt_sample.columns)

    SMAPL4_pt_sample[['SPL4SMGP_006_Geophysical_Data_precipitation_total_surface_flux_0', 
                    'SPL4SMGP_006_Geophysical_Data_precipitation_total_surface_flux_1',
                    'SPL4SMGP_006_Geophysical_Data_precipitation_total_surface_flux_2',
                    'SPL4SMGP_006_Geophysical_Data_precipitation_total_surface_flux_3',
                    'SPL4SMGP_006_Geophysical_Data_precipitation_total_surface_flux_4',
                    'SPL4SMGP_006_Geophysical_Data_precipitation_total_surface_flux_5',
                    'SPL4SMGP_006_Geophysical_Data_precipitation_total_surface_flux_6',
                    'SPL4SMGP_006_Geophysical_Data_precipitation_total_surface_flux_7']].plot()

    # %%
    df_ts_smap_precip = SMAPL4_pt_sample[['Date', 'SPL4SMGP_006_Geophysical_Data_precipitation_total_surface_flux_0']].copy()
    df_ts_smap_precip = df_ts_smap_precip.rename({'SPL4SMGP_006_Geophysical_Data_precipitation_total_surface_flux_0': 'precip'}, axis='columns')
    df_ts_smap_precip['Date'] = pd.to_datetime(df_ts_smap_precip['Date'])
    df_ts_smap_precip.set_index('Date', inplace=True)
    df_ts_smap_precip.plot()
    df_ts_sync = pd.merge(df_ts_sync, df_ts_smap_precip, how='inner', left_index=True, right_index=True)

    noprecip = df_ts_smap_precip['precip'] < 0.00002
    df_ts_sync['noprecip'] = noprecip

    df_ts_sync

    # %% [markdown]
    # ## Get corresponding EASE grid to the sample request

    # %%
    distance = np.sqrt((target_lat-ease_lat[0].values)**2+(target_lon-ease_lon[0].values)**2)

    minElement  = np.where(abs(distance) == np.nanmin(abs(distance)))
    print(np.nanmin(distance))

    if len(minElement[0])!=1:
        print('There are more than two closest cells')
        
    lat_center = ease_lat[0].values[minElement]
    lon_center = ease_lon[0].values[minElement]
    ease_center_column = ease_column[0].values[minElement]
    ease_center_row = ease_row[0].values[minElement]

    print(f'The closest cell to the point ({target_lat}, {target_lon}) is\
        ({lat_center[0]}, {lon_center[0]}:\
        EASE GRID ({ease_center_row[0]}, {ease_center_column[0]})),\
        d={distance[minElement][0]} degrees')

    bbox_lat_max = (ease_lat[0].values[minElement]+ease_lat[0].values[minElement[0][0]-1][minElement[1][0]])/2
    bbox_lat_min = (ease_lat[0].values[minElement]+ease_lat[0].values[minElement[0][0]+1][minElement[1][0]])/2
    bbox_lon_max = (ease_lon[0].values[minElement]+ease_lon[0].values[minElement[0][0]][minElement[1][0]+1])/2
    bbox_lon_min = (ease_lon[0].values[minElement]+ease_lon[0].values[minElement[0][0]][minElement[1][0]-1])/2

    bounding_box = f'{bbox_lon_min[0]},{bbox_lat_min[0]},{bbox_lon_max[0]},{bbox_lat_max[0]}'
    print(bounding_box)
    bounding_dict = {target_station: bounding_box}
    if 'bounding_dicts' not in globals():
        bounding_dicts = {i :bounding_dict}
    else:
        bounding_dicts.update({i :bounding_dict})
    
with open(os.path.join(input_path, appears_path, network_name, "bounding_box.json"), "w") as outfile:
    json.dump(bounding_dicts, outfile)