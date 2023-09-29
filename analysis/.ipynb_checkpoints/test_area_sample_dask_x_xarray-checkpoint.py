# %% [markdown]
# # Area sampling using Dask and Xarray
# Computational improvement based on test_area_sample_PETandLtheta.ipynb

# %%
from dask.distributed import Client
import xarray as xr
import numpy as np
from datetime import datetime
import os
import rasterio as rio
import glob
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from osgeo import gdal
import cartopy.crs as ccrs

# %%
###### Define constants ###########

# Changable
plot_results = False

# Area
network_name = "California"
minx = -124.5
miny = 32.5
maxx = -114
maxy = 42.5
bbox = {'minx':minx, 'maxx':maxx, 'miny':miny, 'maxy':maxy}

# Chunks
# https://blog.dask.org/2021/11/02/choosing-dask-chunk-sizes
# Chunk size between 100MB and 1GB are generally good
chunks = {'x': 100, 'y': 100}
pet_chunks = {'longitude': 100, 'latitude': 100}

# Thresholds
lower_quantile_thresh = 0.25
upper_quantile_thresh = 0.75
precip_thresh = 0.00002

# Dates
startDate = datetime(2016, 1, 1)
endDate = datetime(2017, 1, 1)

# Non-changable
SMAPL4_times = ['0130', '0430', '0730', '1030', '1330', '1630', '1930', '2230'] # 3-hourly data

###### PATH ###########
input_path = r"..\1_data"
output_path = r"..\3_data_out"
SMAPL3_path = "SPL3SMP_E"
SMAPL4_path = "SPL4SMGP"
SMAPL4_grid_path = "SMAPL4SMGP_EASEreference"
PET_path = "PET"

# %%
os.environ['NUMEXPR_MAX_THREADS'] = '48'
client = Client(n_workers=12, threads_per_worker=4, memory_limit='auto')
client
# See https://distributed.dask.org/en/stable/client.html
# https://distributed.dask.org/en/stable/api.html#distributed.LocalCluster
# https://superfastpython.com/threadpool-number-of-workers/


# n_workers should be #workes 
# 64 CPU cores on this machine
# The number of worker threads in the ThreadPool is not related to the number of CPUs or CPU cores in your system.

# A good rule of thumb is to create arrays with a minimum chunksize of at least one million elements 

# %% [markdown]
# ## Read data 

# %% [markdown]
# ### Read SMAP L4 data

# %%
def _preprocess_SMAPL4(ds):
    # Assign missing time dimension
    startTime = datetime.strptime(ds.rangeBeginningDateTime.split(".")[0], '%Y-%m-%dT%H:%M:%S')
    endTime = datetime.strptime(ds.rangeEndingDateTime.split(".")[0], '%Y-%m-%dT%H:%M:%S')
    midTime = startTime + (startTime - endTime)/2
    ds = ds.assign_coords(time=midTime)
    return ds

# %%
# Get a list of files 
# 1 file is 3GB
chunks = {'x': 1200, 'y': 1200, 'time':1, 'band':1}
SMAPL4_fn_pattern = f'SMAP_L4_SM_gph_*.nc'
# SMAPL4_fn_pattern = f'SMAP_L4_SM_gph_{startDate.year}01*.nc' ####### CHNAGE LATER: testing with 2016 Jan 1-9 data ####### 
SMAPL4_file_paths = glob.glob(rf'{input_path}/{SMAPL4_path}/{SMAPL4_fn_pattern}')
# TODO/IMPROVEMENT #3: open_mfdataset(parallel=True) is not really making things super fast. Need to otimize Clients

# # https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html#xarray.open_mfdataset

# %%
# Load data
ds_SMAPL4_3hrly = xr.open_mfdataset(SMAPL4_file_paths, group='Geophysical_Data', engine="rasterio", preprocess=_preprocess_SMAPL4, chunks=chunks, combine='nested', concat_dim='time', parallel=True)

# %%
ds_SMAPL4_3hrly.precipitation_total_surface_flux

# %%
# Re-assign x and y coordinates
SMAPL4_template_fn = r"G:\Araki\SMSigxSMAP\1_data\SPL4SMGP\SMAP_L4_SM_gph_20180911T103000_Vv7032_001_HEGOUT.nc"
SMAPL4_template = xr.open_dataset(SMAPL4_template_fn)
ds_SMAPL4_3hrly = ds_SMAPL4_3hrly.assign_coords(x=SMAPL4_template['x'][:], y=SMAPL4_template['y'][:]*(-1))

# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines()
# ds_SMAPL4.precipitation_total_surface_flux.sel(time='2016-01-01 01:30:00').plot(ax=ax)
# ds_SMAPL4.precipitation_total_surface_flux

# %%
ds_SMAPL4_3hrly = ds_SMAPL4_3hrly.sel(x=slice(minx, maxx), y=slice(miny, maxy)).copy()

if plot_results:
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(color='white')
    ds_SMAPL4_3hrly.sel(time='2016-01-01 01:30:00').precipitation_total_surface_flux.plot(ax=ax)
    ds_SMAPL4_3hrly

# %% [markdown]
# ### Read SMAP L3 data

# %%
def _preprocess_SMAPL3(ds):
    # Assign missing time dimension
    # Doesn't care about hour amd minutes, as it is daily data
    startTime = datetime.strptime(ds.rangeBeginningDateTime.split("T")[0], '%Y-%m-%d')
    ds = ds.assign_coords(time=startTime)
    return ds

# %%
# Get a list of files 
# Test with 2016 Jan 1-9 data first
chunks = {'x': 1200, 'y': 1200, 'time':1, 'band':1}
SMAPL3_fn_pattern = f'SMAP_L3_SM_P_E_2015*.nc'
# SMAPL3_fn_pattern = f'SMAP_L3_SM_P_E_{startDate.year}01*.nc' ####### CHNAGE LATER: testing with 2016 Jan 1-9 data #######
SMAPL3_file_paths = glob.glob(rf'{input_path}/{SMAPL3_path}/{SMAPL3_fn_pattern}')
# Load data
ds_SMAPL3 = xr.open_mfdataset(SMAPL3_file_paths, preprocess=_preprocess_SMAPL3, engine="rasterio", chunks=chunks, combine="nested", concat_dim="time", parallel=True)

# %%
ds_SMAPL3 = ds_SMAPL3.sel(x=slice(minx, maxx), y=slice(maxy, miny))
ds_SMAPL3.rio.write_crs('epsg:4326')
ds_SMAPL3
# 3.3 sec for 1 mo of data

# %%
if plot_results:
    ds_SMAPL3.sel(time='2016-01-03').soil_moisture.plot()
    ds_SMAPL3.soil_moisture
# TODO/IMPROVEMENT: Add dropna(how=all) somewhere to skip calculation of the ocean etc.

# %% [markdown]
# ### Read Singer PET data

# %%
# Get a list of files 
PET_fn_pattern = f'*_daily_pet.nc'
PET_file_paths = glob.glob(rf'{input_path}/{PET_path}/{PET_fn_pattern}')

# Load data
ds_PET = xr.open_mfdataset(PET_file_paths, combine="nested", chunks=pet_chunks, concat_dim="time", parallel=True)
ds_PET['pet']

# %%
# Clip to California
ds_PET = ds_PET.rename({'latitude': 'y', 'longitude':'x'})
ds_PET = ds_PET.sel(x=slice(minx, maxx), y=slice(maxy, miny)).copy()

# Interpolate to SMAP grid
ds_PET.rio.write_crs('epsg:4326', inplace=True)
PET_resampled = ds_PET['pet'].interp(coords={'x': ds_SMAPL3['x'], 'y': ds_SMAPL3['y']}, method='linear', kwargs={'fill_value': np.nan})
ds_SMAPL3['PET'] = PET_resampled

# Plot
if plot_results:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ds_PET.pet.sel(time='2016-01-01').plot(vmax=6, ax=ax1)
    PET_resampled.sel(time='2016-01-01').plot(vmax=6, ax=ax2)

# See: https://docs.xarray.dev/en/stable/generated/xarray.DataArray.interp.html
# If I have to reproject, see: https://github.com/corteva/rioxarray/issues/119

# %% [markdown]
# ## Processing data

# %% [markdown]
# ### Get daily mean values

# %%
# SMAP L4
ds_SMAPL4 = ds_SMAPL4_3hrly.chunk(chunks={'x': 50, 'y': 50})
ds_SMAPL4 = ds_SMAPL4_3hrly.precipitation_total_surface_flux.resample(time='D', skipna=True, keep_attrs=True).mean('time')
ds_SMAPL4.rio.write_crs('epsg:4326', inplace=True)
ds_SMAPL4 = ds_SMAPL4.sel(band=1).rio.reproject_match(ds_SMAPL3)
del ds_SMAPL4_3hrly

# %%
ds_SMAPL3 = ds_SMAPL3.chunk(chunks={'x': 50, 'y': 50})
ds_SMAPL3.soil_moisture

# %%
# SMAP L3
# Mask low-quality data
ds_SMAPL3['soil_moisture_am_masked'] = ds_SMAPL3.soil_moisture.where((ds_SMAPL3.retrieval_qual_flag == 0) | (ds_SMAPL3.retrieval_qual_flag == 8))
ds_SMAPL3['soil_moisture_pm_masked'] = ds_SMAPL3.soil_moisture_pm.where((ds_SMAPL3.retrieval_qual_flag_pm == 0) | (ds_SMAPL3.retrieval_qual_flag_pm == 8))
stacked_data = ds_SMAPL3[['soil_moisture_am_masked', 'soil_moisture_pm_masked']].to_array(dim='new_dim')
ds_SMAPL3['soil_moisture_daily'] = stacked_data.mean(skipna=True, dim="new_dim")

# %%
if plot_results:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ds_SMAPL4.sel(time='2016-01-01').plot(ax=ax1)
    ds_SMAPL3.soil_moisture_daily.sel(time='2016-01-01').plot(ax=ax2)

# %% [markdown]
# ### Calculate dS/dt

# %%
# Mask by precipitation
# https://geohackweek.github.io/nDarrays/09-masking/
precip_mask = ds_SMAPL4.where(ds_SMAPL4 < precip_thresh)

# Insert dummy soil moisture record where (precipitation is present) && (soil moisture record does not exist)
# In this case, drydown pattern is disrupted and shouldn't be calculated. 
# So I put extremely large values for those records, calculate dS, and drop the dS afterwards
no_sm_record_but_precip_present = ds_SMAPL4.where((precip_mask.isnull()) & (ds_SMAPL3['soil_moisture_daily'].isnull()))
ds_SMAPL3['sm_for_dS_calc'] = ds_SMAPL3['soil_moisture_daily'].where(no_sm_record_but_precip_present.isnull(), 9999)

# print(precip_mask.sel(x=sample_x, y=sample_y, method='nearest').values.T)
# print(ds_SMAPL3['soil_moisture_daily'].sel(x=sample_x, y=sample_y, method='nearest').values.T)
# print(no_sm_record_but_precip_present.sel(x=sample_x, y=sample_y, method='nearest').values.T)
# print(sm_for_dS_calc.sel(x=sample_x, y=sample_y, method='nearest').values.T)

# %%
# Calculate dS
ds_SMAPL3['dS'] = ds_SMAPL3['sm_for_dS_calc'].bfill(dim="time", limit=5).diff(dim="time").where(ds_SMAPL3['sm_for_dS_calc'].notnull().shift(time=+1))

# Drop the dS where  (precipitation is present) && (soil moisture record does not exist)
ds_SMAPL3['dS'] = ds_SMAPL3['dS'].where((ds_SMAPL3['dS'] > -1) & (ds_SMAPL3['dS'] < 1))

# Calculate dt
non_nulls = ds_SMAPL3['sm_for_dS_calc'].isnull().cumsum(dim='time')
nan_length = non_nulls.where(ds_SMAPL3['sm_for_dS_calc'].notnull()).bfill(dim="time")+1 - non_nulls +1
ds_SMAPL3['dt'] = nan_length.where(ds_SMAPL3['sm_for_dS_calc'].isnull()).fillna(1)

# Calculate dS/dt
ds_SMAPL3['dSdt'] = ds_SMAPL3['dS']/ds_SMAPL3['dt']
ds_SMAPL3['dSdt'] = ds_SMAPL3['dSdt'].shift(time=-1)

if plot_results:
    ds_SMAPL3['dSdt'].sel(time='2016-01-03').plot()

# %%
# Mask where precipitation is on the day 1 of soil moisture measruement
ds_SMAPL3['dSdt'] = ds_SMAPL3['dSdt'].where(precip_mask.notnull())

# print(ds_SMAPL3['dSdt'].sel(x=sample_x, y=sample_y, method='nearest').values.T)
# print(ds_SMAPL3_masked.sel(x=sample_x, y=sample_y, method='nearest').values.T)
# print(test.sel(x=sample_x, y=sample_y, method='nearest').values)

if plot_results:
    sample_x = -114
    sample_y = 34
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))
    ds_SMAPL3['soil_moisture_daily'].sel(x=sample_x, y=sample_y, method='nearest').interp(method='linear').plot.scatter(ax=ax1)
    ds_SMAPL3['dSdt'].sel(x=sample_x, y=sample_y, method='nearest').interp(method='linear').plot.scatter(ax=ax2)
    ds_SMAPL4.sel(x=sample_x, y=sample_y, method='nearest').plot.scatter(ax=ax3)

# %% [markdown]
# ## Fit regression b/w dS/dt & S for upper/lower PET quantile

# %% [markdown]
# ### Get upper/lower PET quantile

# %%
# Get PET quantile values 
ds_SMAPL3['PET'] = ds_SMAPL3['PET'].chunk({'time': len(ds_SMAPL3['PET'].time), 'x': 'auto', 'y': 'auto'})
ds_quantile = ds_SMAPL3['PET'].where(precip_mask.notnull()).quantile(dim="time", q=[lower_quantile_thresh, upper_quantile_thresh])

if plot_results:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ds_quantile.sel(quantile=lower_quantile_thresh).plot(ax=ax1, vmax=6)
    ds_quantile.sel(quantile=upper_quantile_thresh).plot(ax=ax2, vmax=6)

# %%
ds_SMAPL3['PET_upper_mask'] = ds_SMAPL3['PET'].where(ds_SMAPL3['PET'] >= ds_quantile.sel(quantile=upper_quantile_thresh))
ds_SMAPL3['PET_lower_mask'] = ds_SMAPL3['PET'].where(ds_SMAPL3['PET'] <= ds_quantile.sel(quantile=lower_quantile_thresh))

# %%
if plot_results:
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    ds_SMAPL3['PET'].sel(x=sample_x, y=sample_y, method='nearest').plot.scatter(ax=ax1, color='blue')
    ds_SMAPL3['PET_upper_mask'].sel(x=sample_x, y=sample_y, method='nearest').plot.scatter(ax=ax1, color='green')
    ds_SMAPL3['PET_lower_mask'].sel(x=sample_x, y=sample_y, method='nearest').plot.scatter(ax=ax1, color='red')

# %% [markdown]
# ### Fit regression line

# %%
# Get the minimum soil mositure values over the observation period for a given pixel
sm_min = ds_SMAPL3.soil_moisture_daily.min(dim="time")

# %%
# Shift x values 
ds_SMAPL3['shifted_sm'] = ds_SMAPL3.soil_moisture_daily - sm_min
ds_SMAPL3['neg_dSdt'] = ds_SMAPL3['dSdt'] * (-1)
input_sm_upper = ds_SMAPL3.where((ds_SMAPL3['PET_upper_mask'].notnull()) & (ds_SMAPL3['neg_dSdt'] > 0))
input_sm_lower = ds_SMAPL3.where((ds_SMAPL3['PET_lower_mask'].notnull()) & (ds_SMAPL3['neg_dSdt'] > 0))

if plot_results:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ds_SMAPL3.soil_moisture_daily.sel(x=sample_x, y=sample_y, method='nearest').interp(method='linear').plot(ax=ax1)
    ds_SMAPL3.shifted_sm.sel(x=sample_x, y=sample_y, method='nearest').interp(method='linear').plot(ax=ax2)
    print(sm_min.sel(x=sample_x, y=sample_y, method='nearest').values)

# %%
if plot_results:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    input_sm_upper.PET.sel(time='2016-01-01').plot(ax=ax1)
    input_sm_lower.PET.sel(time='2016-01-01').plot(ax=ax2)

# %%
def fit_regression_through_origin(input_sm):

    # Fit regression of linear line through the origin
    # the slope a is calculated is: a = sum(xi * yi) / sum((xi)^2)
    # If the weight is assumed to be w=1/x**2 (in case of this data)
    # a_out = sum(y/x) / len(x)
    # Proofs in: 
    # https://onlinelibrary.wiley.com/doi/10.1111/1467-9639.00136
    # http://sites.msudenver.edu/ngrevsta/wp-content/uploads/sites/416/2020/02/Notes_07.pdf
    # https://www.jstor.org/stable/2527698?seq=2

    # numerator = input_sm.shifted_sm * input_sm.neg_dSdt
    # denominator = input_sm.shifted_sm * input_sm.shifted_sm
    # denominator_masked = denominator.where((~numerator.isnull()))
    # a = numerator.sum(dim="time", skipna=True) / denominator_masked.sum(dim="time", skipna=True)

    numerator = input_sm.neg_dSdt/input_sm.shifted_sm
    denominator = numerator.notnull().sum(dim='time')
    a = numerator.sum(dim="time", skipna=True) / denominator

    # Calculate error metrics 

    # https://web.ist.utl.pt/~ist11038/compute/errtheory/,regression/regrthroughorigin.pdf
    # R2 = sum(Yi_modeled^2)/sum(Yi_observed^2)

    # https://rpubs.com/aaronsc32/regression-through-the-origin
    # http://sites.msudenver.edu/ngrevsta/wp-content/uploads/sites/416/2020/02/Notes_07.pdf
    # SSE = sum(Yi_obs ^2) - a_i^2 * sum(xi_obs^2)
    # MSE = SSE/ (n-1)

    y2 =  ds_SMAPL3.neg_dSdt *  ds_SMAPL3.neg_dSdt
    n = denominator.where(~numerator.isnull()).time.shape[0]
    SSE = y2.where(numerator.notnull()).sum(dim="time", skipna=True) - a * denominator.where(numerator.notnull()).sum(dim="time", skipna=True)
    MSE = SSE / (n-1)
    
    # https://pubs.cif-ifc.org/doi/pdf/10.5558/tfc71326-3
    # https://dynamicecology.wordpress.com/2017/04/13/dont-force-your-regression-through-zero-just-because-you-know-the-true-intercept-has-to-be-zero/

    return a, MSE


# %%
a_upper, MSE_upper = fit_regression_through_origin(input_sm_upper)
a_lower, MSE_lower = fit_regression_through_origin(input_sm_lower)

# %%
if plot_results:
    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    a_upper.sel(band=1).plot(ax=ax1, vmin=-1, vmax=1)
    a_lower.sel(band=1).plot(ax=ax2, vmin=-1, vmax=1)
    MSE_upper.sel(band=1).plot(ax=ax3, vmin=0)
    MSE_lower.sel(band=1).plot(ax=ax4, vmin=0)

# %%
a_diff = a_upper - a_lower
if plot_results:
    a_diff.plot()

# %%
# Plot 
if plot_results:
    sample_x = -120
    sample_y = 35.3
    S = ds_SMAPL3.shifted_sm.sel(x=sample_x, y=sample_y, method='nearest').values
    dSdt = ds_SMAPL3.neg_dSdt.sel(x=sample_x, y=sample_y, method='nearest').values
    PET = ds_SMAPL3.PET.sel(x=sample_x, y=sample_y, method='nearest').values
    S_min = sm_min.sel(x=sample_x, y=sample_y, method='nearest').values
    a_upper_sel = a_upper.sel(x=sample_x, y=sample_y, method='nearest').values
    a_lower_sel = a_lower.sel(x=sample_x, y=sample_y, method='nearest').values
    a_diff_sel = a_diff.sel(x=sample_x, y=sample_y, method='nearest').values

    print(S.T)
    print(dSdt.T)
    print(dSdt.T)
    print(S_min)
    print(a_upper_sel)
    print(a_lower_sel)
    print(a_diff_sel)


# %%
if plot_results:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    sc1 = ax1.scatter(S, dSdt, c=PET)
    plt.colorbar(sc1)
    x = np.linspace(0, np.nanmax(S),100)
    y_lower = a_lower_sel*x # (x+S_min)
    y_upper = a_upper_sel*x
    ax1.plot(x, y_lower, '-r')
    ax1.plot(x, y_upper, '-b')

    sc = ax2.scatter(S+S_min, dSdt, c=PET)
    plt.colorbar(sc)
    x = np.linspace(0, np.nanmax(S+S_min),100)
    y_lower = a_lower_sel*x - a_lower_sel*S_min 
    y_upper = a_upper_sel*x - a_upper_sel*S_min 
    ax2.plot(x, y_lower, '-r')
    ax2.plot(x, y_upper, '-b')

# %%
meanSM = ds_SMAPL3.soil_moisture_daily.mean(dim="time")
meanP = ds_SMAPL4.mean(dim="time")

# %% [markdown]
# # Save results

# %%
results = xr.Dataset({'a_diff': a_diff.sel(band=1), 
                      'a_upper': a_upper.sel(band=1), 'a_lower': a_lower.sel(band=1), 
                      'MSE_upper': MSE_upper.sel(band=1), 'MSE_lower': MSE_lower.sel(band=1),
                      'PET_upper': ds_quantile.sel(quantile=upper_quantile_thresh).drop('quantile'), 
                      'PET_lower': ds_quantile.sel(quantile=lower_quantile_thresh).drop('quantile'), 
                      'meanSM': meanSM.sel(band=1),
                      'meanP': meanP})
results.rio.write_crs('epsg:4326')
results = results.drop_vars(["band", "/crs", "projection_information", "quantile", "spatial_ref"])

# %%
out_path = r'G:\Araki\SMSigxSMAP\3_data_out\a_diff_202303'
results.to_netcdf(os.path.join(out_path, 'results.nc'))

# %%
print(f'Finished running at {datetime.now()}')


