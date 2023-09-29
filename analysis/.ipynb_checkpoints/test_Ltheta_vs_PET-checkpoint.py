# %% [markdown]
# ## Import libraries

# %%
import json
import requests
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # matplotlib is not installed automatically
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression

# %% [markdown]
# ## Load target station info 
network_name = 'OZNET'

input_path = r"..\1_data"
appeears_path = r"APPEEARS_subsetting"
SMAPL3_path = r"SPL3SMP_E"
SMAPL4_path = r"SPL4SMGP"
SMAPL4_grid_path = r"SMAPL4SMGP_EASEreference"
MODIS_path = r"MOD15A2H"
PET_path = r"dPET_data_4RA"

print(os. getcwd())

# %%

file_path = os.path.join(input_path, appeears_path, network_name, f'{network_name}-request.json')
with open(file_path, 'r') as infile:
    request_content = json.load(infile)

coordinates = request_content['params']['coordinates']

# %%
# dates
# for coordinate in range(len(coordinates)):
coordinate = 0
target_lat = coordinates[coordinate]['latitude']
target_lon = coordinates[coordinate]['longitude']
target_station = coordinates[coordinate]['category']
print(f'Currently processing station: {target_station}')

output_path = os.path.join(r".\3_data_out", target_station)
if not os.path.exists(output_path):
    os.makedirs(output_path)

# %% [markdown]
# ## Read synched data

# %%
file_path = os.path.join(input_path, appeears_path, network_name.upper(), f'timeseries_synced_{target_station}.csv')
# if os.path.exists(file_path):
#     print('Files for the station exists')
#     pass
# else:
#     print('Files for the station DOES NOT exist')
#     continue
ds_synced = pd.read_csv(file_path)
ds_synced['Date'] = pd.to_datetime(ds_synced['Date'])
ds_synced.set_index('Date', inplace=True)
ds_synced

# %% [markdown]
# ## Read PET data

# %%
file_path = os.path.join(input_path, PET_path, target_station)
file_list = os.listdir(file_path)


print(file_list)
PET = pd.read_csv(os.path.join(file_path, file_list[0]), header=None)
day_num = PET.index.to_numpy()
year = file_list[0][-8:-4]
PET['Date'] = pd.to_datetime(PET.index, unit='D', origin=pd.Timestamp(year))
PET.set_index('Date', inplace=True)


# %%
for i, file in enumerate(file_list):
    PET = pd.read_csv(os.path.join(file_path, file), header=None)
    year = file[-8:-4]
    PET['Date'] = pd.to_datetime(PET.index, unit='D', origin=pd.Timestamp(year))
    PET.set_index('Date', inplace=True)
    if i==0:
        ds_PET = PET
    else:
        ds_PET = pd.concat([ds_PET, PET])
ds_PET = ds_PET.sort_index()
ds_PET = ds_PET.rename(columns = {0:'PET'})
ds_PET

# %%
ds_synced2 = pd.merge(ds_synced, ds_PET, how='inner', left_index=True, right_index=True)
ds_synced2

# %%
# prep
lat = target_lat
lon = target_lon

smap_color = '#ff7f0e'
precip_color = '#779eb2'

label_SM = r"$\theta [m^3/m^3]$"
label_dSdt = r"$-d\theta/dt$"
label_lai = r"$LAI [m^2/m^2]$"
label_PET = r"$PET [mm/d]$"

title = f"{network_name}: {target_station}\n({lat:.2f}, {lon:.2f})"
save_title = f"{network_name}_{target_station}"
plt.rcParams['font.size'] = 12

# %%
# Get drydown timeseries for plotting
ds_synced2['dSdt'] = ds_synced2['soil_moisture_smapL3'].diff()
ds_synced2['dSdt'][ds_synced2['dSdt']>0] = np.nan
ds_synced2['dSdt(t+1)'] = ds_synced2['dSdt'].shift(periods=-1).copy()
ds_synced2[['soil_moisture_smapL3','dSdt(t+1)', 'noprecip', 'MODISmeanLAI_SMAPgrid']].head(30)

ds_synced2['values_while_drydown'] = ds_synced2['soil_moisture_smapL3']
drydown_period = ds_synced2['dSdt(t+1)'].notna()
drydown_period = drydown_period.shift(periods=+1) | drydown_period
ds_synced2['values_while_drydown'][drydown_period==False] = np.nan
noprecip_with_buffer = (ds_synced2['noprecip']==True) | (ds_synced2['noprecip'].shift(periods=-1)==True)
ds_synced2['values_while_drydown'][noprecip_with_buffer==False] = np.nan

# %%
# ds_synced2['PET_quantile']  = pd.qcut(ds_synced2['PET'][ds_synced2['noprecip']], 10, labels=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ds_synced2['PET_quantile']  = pd.qcut(ds_synced2['PET'][ds_synced2['noprecip']], 5, labels=[20, 40, 60, 80, 100])
# print(ds_synced2['PET'].quantile())

# print(ds_synced2[['PET','PET_quantile']])
ds_synced2['PET_quantile'].head(10)
# plot(ds_synced2['PET_quantile'])
x_upper = ds_synced2['soil_moisture_smapL3'][ds_synced2['PET_quantile']==100].values
y_upper = ds_synced2['dSdt(t+1)'][ds_synced2['PET_quantile']==100].values *-1
nans_upper = np.isnan(x_upper) | np.isnan(y_upper)

# x_lower = ds_synced2['soil_moisture_smapL3'][ds_synced2['PET_quantile']==10].values
# y_lower = ds_synced2['dSdt(t+1)'][ds_synced2['PET_quantile']==10].values *-1
x_lower = ds_synced2['soil_moisture_smapL3'][ds_synced2['PET_quantile']==20].values
y_lower = ds_synced2['dSdt(t+1)'][ds_synced2['PET_quantile']==20].values *-1
nans_lower = np.isnan(x_lower) | np.isnan(y_lower)


x_upper = np.delete(x_upper, nans_upper)
y_upper = np.delete(y_upper, nans_upper)
x_lower = np.delete(x_lower, nans_lower)
y_lower = np.delete(y_lower, nans_lower)

#%%

# %% Fig regression

def fit_linear_line(x, y, forced_intercept_x, weight_regression=True, weight_type=None, plot_results=False):
    ## Using Scikit learn

    # Shift x values by forced_intercept_x
    new_x = x - forced_intercept_x
    new_x = new_x.reshape((-1,1))

    # Linear regression wighout weight
    model = LinearRegression().fit(x.reshape((-1,1)), y)

    # r = model.score(new_x_upper,y_upper)
    a = model.coef_
    b = -1 * a * forced_intercept_x
    
    if not weight_regression:
        a_out = a
        b_out = b

    if weight_regression:

        if weight_type == 'residuals':
            # Residuals
            pre_residuals = y - (a*x+b)

            # Linear regression with weight
            model_weighted = LinearRegression(fit_intercept=False).fit(new_x, y, sample_weight=1/pre_residuals**2)
        elif weight_type == 'x':
            model_weighted = LinearRegression(fit_intercept=False).fit(new_x, y, sample_weight=1/(x - forced_intercept_x))
        elif weight_type == 'x2':
            model_weighted = LinearRegression(fit_intercept=False).fit(new_x, y, sample_weight=1/(x - forced_intercept_x)**2)
            
        a_weighted = model_weighted.coef_
        b_weighted = -1 * a_weighted * forced_intercept_x
        a_out = a_weighted
        b_out = b_weighted

    residuals = y - (a_out*x+b_out)
        
    if plot_results:
        plt.plot(a_out*x+b_out,residuals,'o')
        plt.title(f'Weight {weight_regression} - {weight_type}')
        plt.show()
        
    return a_out, b_out, residuals
    
## Using Scikit learn
# forced_intercept = 0.02 # The minimum possible SMAP value [m3/m3] according to Akbar et al., (2018)
forced_intercept = ds_synced2['soil_moisture_smapL3'].min()

m_upper, b_upper, res_upper = fit_linear_line(x=x_upper, y=y_upper, forced_intercept_x=forced_intercept, weight_regression=True, weight_type='residuals', plot_results=True)
m_lower, b_lower, res_lower = fit_linear_line(x=x_lower, y=y_lower, forced_intercept_x=forced_intercept, weight_regression=True, weight_type='residuals', plot_results=True)

m_upper_residual_method = m_upper.copy()
m_lower_residual_method = m_lower.copy()

m_upper, b_upper, res_upper = fit_linear_line(x=x_upper, y=y_upper, forced_intercept_x=forced_intercept, weight_regression=True, weight_type='x', plot_results=True)
m_lower, b_lower, res_lower = fit_linear_line(x=x_lower, y=y_lower, forced_intercept_x=forced_intercept, weight_regression=True, weight_type='x',plot_results=True)

m_upper_LR_prop_x = m_upper.copy()
m_lower_LR_prop_x = m_lower.copy()

m_upper, b_upper, res_upper = fit_linear_line(x=x_upper, y=y_upper, forced_intercept_x=forced_intercept, weight_regression=True, weight_type='x2', plot_results=True)
m_lower, b_lower, res_lower = fit_linear_line(x=x_lower, y=y_lower, forced_intercept_x=forced_intercept, weight_regression=True, weight_type='x2',plot_results=True)

m_upper_LR_prop_x2 = m_upper.copy()
m_lower_LR_prop_x2 = m_lower.copy()

# %% 

def WRO_weight_x_simple(x, y, forced_intercept_x, weight_regression=True, weight_type=None, plot_results=True):
    # Shift x values by forced_intercept_x
    new_x = x - forced_intercept_x

    # Linear regression with weight
    if not weight_regression:
        a_out = sum(new_x*y) / sum(new_x**2)
    else:
        if weight_type == 'x':
            a_out = sum(y) / sum(new_x)
        elif weight_type == 'x2':
            a_out = sum(y/new_x) / len(new_x)

    residuals = y - a_out*new_x

    if plot_results:
        plt.plot(a_out*x,residuals,'o')
        plt.title(f'Weight {weight_regression} - {weight_type}')
        plt.show()

    return a_out, residuals

a_upper, res_upper = WRO_weight_x_simple(x=x_upper, y=y_upper, weight_regression=False, forced_intercept_x=forced_intercept, plot_results=True)
a_lower, res_lower = WRO_weight_x_simple(x=x_lower, y=y_lower, weight_regression=False, forced_intercept_x=forced_intercept, plot_results=True)

m_upper_original = a_upper.copy()
m_lower_original = a_lower.copy()

a_upper, res_upper = WRO_weight_x_simple(x=x_upper, y=y_upper, forced_intercept_x=forced_intercept, weight_type='x', plot_results=True)
a_lower, res_lower = WRO_weight_x_simple(x=x_lower, y=y_lower, forced_intercept_x=forced_intercept, weight_type='x',plot_results=True)

m_upper_simple_prop_x = a_upper.copy()
m_lower_simple_prop_x = a_lower.copy()

a_upper, res_upper = WRO_weight_x_simple(x=x_upper, y=y_upper, forced_intercept_x=forced_intercept, weight_type='x2',plot_results=True)
a_lower, res_lower = WRO_weight_x_simple(x=x_lower, y=y_lower, forced_intercept_x=forced_intercept, weight_type='x2',plot_results=True)

m_upper_simple_prop_x2 = a_upper.copy()
m_lower_simple_prop_x2 = a_lower.copy()


# %%

new_x_upper = x_upper - forced_intercept
fig, ax = plt.subplots(1)
ax.plot(new_x_upper,y_upper,'o')
ax.plot(new_x_upper,m_upper_residual_method*new_x_upper, label=f'Residual (a={m_upper_residual_method[0]:.2f})')
ax.plot(new_x_upper,m_upper_LR_prop_x*new_x_upper, label=f'W=1/x (a={m_upper_LR_prop_x[0]:.2f})')
ax.plot(new_x_upper,m_upper_LR_prop_x2*new_x_upper, label=f'W=1/x2 (a={m_upper_LR_prop_x2[0]:.2f})')
ax.plot(new_x_upper,m_upper_original*new_x_upper, label=f'Simple WOS (a={m_upper_original:.2f})')
ax.plot(new_x_upper,m_upper_simple_prop_x*new_x_upper, label=f'Simple WOS, w=1/x (a={m_upper_simple_prop_x:.2f})')
ax.plot(new_x_upper,m_upper_simple_prop_x2*new_x_upper, label=f'Simple WOS, w=1/x2 (a={m_upper_simple_prop_x2:.2f})')
fig.legend()
# fig.show()

new_x_lower = x_lower - forced_intercept
fig2, ax2 = plt.subplots(1)
ax2.plot(x_lower - forced_intercept,y_lower,'o')
ax2.plot(new_x_lower,m_lower_residual_method*new_x_lower, label=f'Residual (a={m_lower_residual_method[0]:.2f})')
ax2.plot(new_x_lower,m_lower_LR_prop_x*new_x_lower, label=f'W=1/x (a={m_lower_LR_prop_x[0]:.2f})')
ax.plot(new_x_lower,m_lower_LR_prop_x2*new_x_lower, label=f'W=1/x2 (a={m_lower_LR_prop_x2[0]:.2f})')
ax2.plot(new_x_lower,m_lower_original*new_x_lower, label=f'Simple WOS (a={m_lower_original:.2f})')
ax2.plot(new_x_lower,m_lower_simple_prop_x*new_x_lower, label=f'Simple WOS, w=1/x (a={m_lower_simple_prop_x:.2f})')
ax2.plot(new_x_lower,m_lower_simple_prop_x2*new_x_lower, label=f'Simple WOS, w=1/x2 (a={m_lower_simple_prop_x2:.2f})')
fig2.legend()
# plt.show()


# %%




# %%
m_upper[0]
# %% [markdown]
## 2D plot  (PET as Z-axis)
# %%
from pylab import cm


title = f"{network_name}: {target_station} ({lat:.2f}, {lon:.2f})\nSlope upper-lower = {m_upper[0]-m_lower[0]:.3f}"

fig = plt.figure(figsize=(15, 5))
fig.tight_layout(pad=5)
sm = ds_synced2['soil_moisture_smapL3'][ds_synced2['noprecip']].values
neg_dSdt = ds_synced2['dSdt(t+1)'][ds_synced2['noprecip']].values*-1
pet = ds_synced2['PET'][ds_synced2['noprecip']].values
pet_quantile = ds_synced2['PET_quantile'][ds_synced2['noprecip']].values
cmap_descrete = cm.get_cmap('Oranges', 5)

ax1 =  fig.add_subplot(1,2,1)
scatter = ax1.scatter(x=sm, y=neg_dSdt, c=pet_quantile, cmap= cmap_descrete, marker='o', alpha=0.5, label='SMAP L4')
xax = ax1.xaxis
# ax1.set_title(title)
ax1.set_xlabel(label_SM)
ax1.set_ylabel(label_dSdt)
ax1.set_xlim([0, 0.60])
ax1.set_ylim([0, 0.200])
cbar = plt.colorbar(scatter, ax=ax1)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel("PET quantile", rotation=270)
fig.autofmt_xdate()

ax2 =  fig.add_subplot(1,2,2)

m_upper = m_upper_simple_prop_x2
m_lower = m_lower_simple_prop_x2

scatter1 = ax2.scatter(x=x_upper, y=y_upper, c='sienna', marker='o', alpha=0.5, label='Upper')
scatter2 = ax2.scatter(x=x_lower, y=y_lower, c='bisque', marker='o', alpha=0.5, label='Lower')
x_array = np.arange(0, max(x_lower), 0.01)
plt.plot(x_array, m_upper*x_array+b_upper, '-k')
plt.plot(x_array, m_lower*x_array+b_lower, '-k')
# plt.scatter(x_lower[~nans_lower], y_lower[~nans_lower], marker='o', s=50,facecolors='none',edgecolors='grey')
# plt.scatter(x_upper[~nans_upper], y_upper[~nans_upper], marker='o', s=50,facecolors='none',edgecolors='grey')
xax = ax2.xaxis
# ax2.set_title(title)
ax2.set_xlabel(label_SM)
ax2.set_ylabel(label_dSdt)
ax2.set_xlim([0, 0.60])
ax2.set_ylim([0, 0.200])
ax2.legend()
# cbar = plt.colorbar(scatter, ax=ax2)
# cbar.ax.get_yaxis().labelpad = 15
# cbar.ax.set_ylabel("PET quantile", rotation=270)
fig.autofmt_xdate()
fig.suptitle(title)

fig.savefig(os.path.join(output_path, f'Ltheta_PET_fit.png'))




# %% Snipped


# #%%
# # https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal
# def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
#     """
#     Input :
#     s: 1d-array, data signal from which to extract high and low envelopes
#     dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
#     split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
#     Output :
#     lmin,lmax : high/low envelope idx of input signal s
#     """

#     # locals min      
#     lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
#     # locals max
#     lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    

#     if split:
#         # s_mid is zero if s centered around x-axis or more generally mean of signal
#         s_mid = np.mean(s) 
#         # pre-sorting of locals min based on relative position with respect to s_mid 
#         lmin = lmin[s[lmin]<s_mid]
#         # pre-sorting of local max based on relative position with respect to s_mid 
#         lmax = lmax[s[lmax]>s_mid]


#     # global max of dmax-chunks of locals max 
#     lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
#     # global min of dmin-chunks of locals min 
#     lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
#     return lmin,lmax

# #%%

# ## Upper quantile
# sort_data_upper = pd.DataFrame(data={'x': x_upper, 'y': y_upper})
# s = sort_data_upper.sort_values(by='x')['y'].values
# t = sort_data_upper.sort_values(by='x')['x'].values
# high_idx, low_idx = hl_envelopes_idx(s, dmax=2)

# # Without weight
# m_upper, b_upper = np.polyfit(t[low_idx], s[low_idx], deg=1)
# # Residuals
# m_res, b_res = np.polyfit(t[low_idx], abs(m_upper*t[low_idx]+b_upper-s[low_idx]), deg=1)
# # With weight
# weight_res = m_res*t[low_idx]+b_res
# m_upper, b_upper = np.polyfit(t[low_idx], s[low_idx], deg=1, w=1/weight_res**2)

# ## Lower quantile
# sort_data_lower = pd.DataFrame(data={'x': x_lower, 'y': y_lower})
# s = sort_data_lower.sort_values(by='x')['y'].values
# t = sort_data_lower.sort_values(by='x')['x'].values
# high_idx, low_idx = hl_envelopes_idx(s, dmax=2)

# # Wightout weight
# m_lower, b_lower = np.polyfit(t[low_idx], s[low_idx], deg=1)
# # Residuals
# m_res, b_res = np.polyfit(t[low_idx], abs(m_lower*t[low_idx]+b_lower-s[low_idx]), deg=1)
# # With weight
# weight_res = m_res*t[low_idx]+b_res
# m_lower, b_lower = np.polyfit(t[low_idx], s[low_idx], deg=1, w=1/weight_res**2)
# %%
