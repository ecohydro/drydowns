# Import libraries
import json
import requests
import os
from ismn.interface import ISMN_Interface
from ismn.meta import Depth
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # matplotlib is not installed automatically
from datetime import datetime
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.signal import find_peaks
from math import sqrt

# Specify current directory and create output directory if it does not exist
os.chdir("G:/Shared drives/Ryoko and Hilary/SMSigxSMAP/analysis/0_code")

def calc_kerneldensity(ts, bandwidth_fact=1, plot=False):

    # Prep
    kd_method = 'gaussian'

    bandwidths = np.arange(0.005, 0.1, 0.001)
    maxsm = np.max(ts.dropna().values)
    minsm = np.min(ts.dropna().values)
    x_plot = np.linspace(minsm, maxsm, 100)[:, np.newaxis]
    x_test = ts.dropna().values[:, np.newaxis]

    # Calculation
    kde0 = KernelDensity(kernel=kd_method)
    grid = GridSearchCV(kde0, {'bandwidth': bandwidths})
    grid.fit(x_test)
    kde_optimal = grid.best_estimator_
    log_dens_optimal = kde_optimal.score_samples(x_plot)
    log_dens_smoothed = KernelDensity(kernel=kd_method, bandwidth=kde_optimal.bandwidth * bandwidth_fact).fit(
        x_test).score_samples(x_plot)

    peaks_smoothed, properties = find_peaks(np.exp(log_dens_smoothed), height=0, prominence=0)
    peaks_optimal, properties = find_peaks(np.exp(log_dens_optimal), height=0, prominence=0)
    print(properties)
    wilting_point = np.min(x_plot[:, 0][peaks_optimal])
    field_capacity = np.max(x_plot[:, 0][peaks_smoothed])
    n_modes = len(peaks_smoothed)

    if plot:
        # Plot the histogram
        if ts.name == 'smap':
            color = '#ff7f0e'
        elif ts.name == 'ismn':
            color = '#1f77b4'

        fig, ax = plt.subplots()
        ax.hist(x_test[:, 0], label='Observed', color=color, bins=50, alpha=0.5, density=True, stacked=True)
        ax.plot(x_plot[:, 0], np.exp(log_dens_optimal), color="k", linestyle='--', label=f'{kd_method} (bandwidth={kde_optimal.bandwidth})')
        ax.plot(x_plot[:, 0], np.exp(log_dens_smoothed), color="k", linestyle='-', label=f'{kd_method} (bandwidth=optimal * {bandwidth_fact})')
        ax.plot(x_plot[:, 0][peaks_smoothed], np.exp(log_dens_smoothed)[peaks_smoothed], marker='x', markersize=12, markeredgecolor="k", linestyle = 'None', label='Peaks')
        ax.plot(x_plot[:, 0][peaks_optimal], np.exp(log_dens_optimal)[peaks_optimal], marker='x', markersize=12,
                markeredgecolor="k", linestyle='None', label=None)

        fig.legend()
        ax.set_xlabel("Volmetric soil water content [m^3/m^3]")
        ax.set_ylabel("Normalized frequency [-]")
        ax.set_title(ts.name)

    return wilting_point, field_capacity, n_modes, fig, ax


class SMAPxISMN():
    def __init__(self, input_path_ismn, input_path_smap, insitu_network_name, out_path):
        self.input_path_ismn = input_path_ismn
        self.input_path_smap = input_path_smap
        self.insitu_network_name = insitu_network_name
        self.out_path = out_path
        self.smap_color = '#ff7f0e'
        self.ismn_color = '#1f77b4'

    def load_data(self, data_treatment_option=None, plot=False):

        self.data_treatment_option = data_treatment_option

        #################
        #   READ DATA   #
        #################

        # Read SMAP data
        df_SMAP = pd.read_csv(self.input_path_smap)
        df_ts_smap = df_SMAP[df_SMAP['ID']==0].copy() #TODO: make this more generic
        df_ts_smap['Date'] = pd.to_datetime(df_ts_smap['Date'])
        df_ts_smap.set_index('Date', inplace=True)

        # Read ISMN data
        ismn_data = ISMN_Interface(self.input_path_ismn)

        # Find the ISMN station corresponding to the SMAP extraction
        ismn_grid = ismn_data.collection.grid
        gpis, lons, lats = ismn_grid.get_grid_points()
        station, dist = ismn_data.collection.get_nearest_station(df_SMAP['Longitude'][0], df_SMAP['Latitude'][0])
        print(f'Station {station.name} is {int(dist)} metres away from the passed coordinates:')
        if int(dist) >= 100:
            warnings.warn('The sensor is too far away')

        # Get the ISMN sensor installed shallower than 0.05 cm
        for sensor in station.iter_sensors(variable='soil_moisture',depth=Depth(0., 0.05)):
            print(sensor)
            df_ts_ismn = sensor.read_data()

        ###################################
        #   QUALITY CONTROL OF THE DATA   #
        ###################################

        # ISMN
        bad_data_idx_ismn = df_ts_ismn[df_ts_ismn['soil_moisture_flag']!='G'].index
        df_ts_ismn.drop(bad_data_idx_ismn, inplace=True)
        df_ts_ismn_daily = df_ts_ismn.resample('D', axis=0).mean()

        # SMAP
        """
        # 1. Use the retrieval_qual_flag field to identify retrievals in the soil_moisture field estimated to be of 
        recommended quality. A retrieval_qual_flag value of either 0 or 8 indicates high-quality retrievals 
        (8 because a failed F/T retrieval does not affect soil moisture retrieval). 
        """

        df_ts_smap_am = df_ts_smap[['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_AM_soil_moisture', 'SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag']].copy()
        bad_data_idx_smap = df_ts_smap_am[(df_ts_smap_am['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag'] != 0.0) & (df_ts_smap_am['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag'] != 8.0)].index
        df_ts_smap_am.drop(bad_data_idx_smap, inplace=True)
        del bad_data_idx_smap

        df_ts_smap_pm = df_ts_smap[['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm', 'SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm']].copy()
        bad_data_idx_smap = df_ts_smap_pm[(df_ts_smap_pm['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm'] != 0.0) & (df_ts_smap_pm['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm'] != 8.0)].index
        df_ts_smap_pm.drop(bad_data_idx_smap, inplace=True)
        del bad_data_idx_smap

        #########################################
        #   MERGE DATA AND AGGREGATE TO DAILY   #
        #########################################

        # SMAP, get the average of AM and PM data
        df_ts_smap_daily = pd.merge(df_ts_smap_am, df_ts_smap_pm, how='inner', left_index=True, right_index=True)
        df_ts_smap_daily['soil_moisture_smap'] = df_ts_smap_daily[['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_AM_soil_moisture','SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm']].mean(axis=1, skipna=True)
        df_ts_smap_daily = df_ts_smap_daily['soil_moisture_smap'].resample('D', axis=0).mean()

        # Merge and sync ISMN and SMAP data
        df_ts_sync = pd.merge(df_ts_smap_daily, df_ts_ismn_daily, how='inner', left_index=True, right_index=True)
        df_ts_sync.rename(columns={"soil_moisture": "ismn", "soil_moisture_smap": "smap"}, inplace=True)

        # Filter out the portion of timeseries where either ISMN or SMAP data is unavailable
        df_ts_sync['rolling_ismn'] = df_ts_sync['ismn'].rolling(window='7d', center=True, closed='both').mean()
        df_ts_sync['rolling_smap'] = df_ts_sync['smap'].rolling(window='7d', center=True, closed='both').mean()
        no_data_idx = df_ts_sync[df_ts_sync['rolling_ismn'].isnull() | df_ts_sync['rolling_smap'].isnull()].index
        df_ts_sync['ismn'][no_data_idx]=np.NaN
        df_ts_sync['smap'][no_data_idx]=np.NaN

        # Extra data treatment
        if self.data_treatment_option == 'limit_data_to_smap_available_dates':
            no_data_idx = df_ts_sync[df_ts_sync['smap'].isnull()].index
            df_ts_sync['ismn'][no_data_idx] = np.NaN

        if self.data_treatment_option == 'linear_interpolation_on_smap':
            # Interpolate up to 3 days window on before and after (in total, 1 week window)
            df_ts_before_interp = df_ts_sync.copy()
            df_ts_sync['smap'] = df_ts_sync['smap'].interpolate(method='linear', limit=3, limit_direction='both')

        # Variables to return
        self.soil_moisture_ts = df_ts_sync
        self.soil_moisture_Ndata_smap = df_ts_sync['smap'].count() # Count the number of data which is not NaN. len(df_ts_sync['ismn'])
        self.both_data_notnull_idx = df_ts_sync[df_ts_sync['smap'].notnull() & df_ts_sync['ismn'].notnull()].index

        self.plottitle = f"{station.name} station, OZNET, Australia\n({df_SMAP['Longitude'][0]}, {df_SMAP['Latitude'][0]})"
        self.station = station
        self.longitude = df_SMAP['Longitude'][0]
        self.latitude = df_SMAP['Latitude'][0]

        if plot:
            # Plot the timesereis of data

            fig, ax = plt.subplots()
            if self.data_treatment_option == 'limit_data_to_smap_available_dates':
                line1, = ax.plot(df_ts_sync['ismn'], 'o', markersize=4, alpha=0.5, label='In-situ')
            else:
                line1, = ax.plot(df_ts_sync['ismn'], label='In-situ')
            if self.data_treatment_option == 'linear_interpolation_on_smap':
                line2, = ax.plot(df_ts_sync['smap'], alpha=0.5, label='SMAP')
            else:
                line2, = ax.plot(df_ts_sync['smap'], 'o', markersize=4, alpha=0.5, label='SMAP')
            fig.legend()
            xax = ax.xaxis
            ax.set_title(f"{station.name} station, OZNET, Australia\n({df_SMAP['Longitude'][0]}, {df_SMAP['Latitude'][0]})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Volumetric soil water content [m^3/m^3]")
            # formatter = mdates.DateFormatter("%Y") ### formatter of the date
            # locator = mdates.YearLocator() ### where to put the labels
            # ax.xaxis.set_major_formatter(formatter) ## calling the formatter for the x-axis
            # ax.xaxis.set_major_locator(locator) ## calling the locator for the x-axis
            fig.autofmt_xdate()
            fig.savefig(os.path.join(self.out_path, 'test_ts.png'))
            del fig, ax

            # Plot the histogram
            fig, ax = plt.subplots()
            ax.hist(df_ts_sync['ismn'], label='In-situ',  bins=30, alpha = 0.5, density=True, stacked=True, color=self.ismn_color)
            ax.hist(df_ts_sync['smap'], label='SMAP',     bins=30, alpha = 0.5, density=True, stacked=True, color=self.smap_color)
            fig.legend()
            ax.set_title(f"{station.name} station, OZNET, Australia\n({df_SMAP['Longitude'][0]}, {df_SMAP['Latitude'][0]})")
            ax.set_xlabel("Volmetric soil water content [m^3/m^3]")
            ax.set_ylabel("Normalized frequency [-]")
            fig.savefig(os.path.join(self.out_path, 'test_hist.png'))
            del fig, ax

            # Plot the close-up
            if self.data_treatment_option == 'linear_interpolation_on_smap':
                date_limit_start = '2015-06-01'
                date_limit_end = '2016-05-31'
                df_limit_range = df_ts_sync.loc[date_limit_start:date_limit_end].copy()
                fig, ax = plt.subplots()
                line1, = ax.plot(df_limit_range['ismn'], label='In-situ', color=self.ismn_color)
                line2, = ax.plot(df_limit_range['smap'], alpha=0.5, label='SMAP', color=self.smap_color)
                line3, = ax.plot(df_ts_before_interp['smap'].loc[date_limit_start:date_limit_end], 'o', markersize=4, alpha=0.5, label='SMAP', color=self.smap_color)
            fig.legend()
            xax = ax.xaxis
            ax.set_title(
                f"{station.name} station, OZNET, Australia\n({df_SMAP['Longitude'][0]}, {df_SMAP['Latitude'][0]})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Volumetric soil water content [m^3/m^3]")
            # formatter = mdates.DateFormatter("%Y") ### formatter of the date
            # locator = mdates.YearLocator() ### where to put the labels
            # ax.xaxis.set_major_formatter(formatter) ## calling the formatter for the x-axis
            # ax.xaxis.set_major_locator(locator) ## calling the locator for the x-axis
            fig.autofmt_xdate()
            fig.savefig(os.path.join(self.out_path, f'test_ts_close_up_{date_limit_start}_{date_limit_end}.png'))
            del fig, ax

    def calc_bias(self):
        bias = sum(self.soil_moisture_ts['smap'][self.both_data_notnull_idx]-self.soil_moisture_ts['ismn'][self.both_data_notnull_idx])/len(self.both_data_notnull_idx)
        return bias

    def count_N(self):
        return len(self.both_data_notnull_idx)

    def calc_RMSE(self):
        RMSE = mean_squared_error(self.soil_moisture_ts['ismn'][self.both_data_notnull_idx], self.soil_moisture_ts['smap'][self.both_data_notnull_idx], squared=False)
        return RMSE

    def calc_ubRMSE(self):
        RMSE = mean_squared_error(self.soil_moisture_ts['ismn'][self.both_data_notnull_idx],
                                  self.soil_moisture_ts['smap'][self.both_data_notnull_idx], squared=False)
        bias = sum(self.soil_moisture_ts['smap'][self.both_data_notnull_idx] - self.soil_moisture_ts['ismn'][
            self.both_data_notnull_idx]) / len(self.both_data_notnull_idx)
        ubRMSE = sqrt(RMSE**2 - bias**2)
        return ubRMSE

    def calc_R(self, plot=False):
        R = np.corrcoef(self.soil_moisture_ts['ismn'][self.both_data_notnull_idx], self.soil_moisture_ts['smap'][self.both_data_notnull_idx])
        if plot:
            fig, ax = plt.subplots()
            scatter = ax.scatter(self.soil_moisture_ts['ismn'][self.both_data_notnull_idx].values, self.soil_moisture_ts['smap'][self.both_data_notnull_idx].values)
            ax.set_title("Volumetric soil water content [m^3/m^3]")
            ax.set_xlabel("In-situ")
            ax.set_ylabel("SMAP")
            ax.axis('square')
            fig.savefig(os.path.join(self.out_path, 'test_scatter.png'))
            del fig, ax
        return R[0][1]

    def calc_dist(self, plot=True):
        # Get the field_capacity and wilting points

        # When the data is coarse (e.g., SMAP data without data interpolation)
        #     bandwidth_fact = 1.5
        # When the data is daily (e.g., ISMN, or SMAP data after interpolation)
        #     bandwidth_fact = 3

        # SMAP
        wilting_point_smap, field_capacity_smap, n_modes_smap, fig, ax = calc_kerneldensity(self.soil_moisture_ts['smap'], bandwidth_fact = 2, plot=True)
        fig.savefig(os.path.join(self.out_path, 'test_kernel_smap.png'))
        del fig, ax

        # ISMN
        wilting_point_ismn, field_capacity_ismn, n_modes_ismn, fig, ax = calc_kerneldensity(self.soil_moisture_ts['ismn'],  bandwidth_fact = 3, plot=True)
        fig.savefig(os.path.join(self.out_path, 'test_kernel_ismn.png'))
        del fig, ax

        print(f"SMAP: wilting point: {wilting_point_ismn:.2f}, field capacity: {field_capacity_ismn:.2f}, Number of modes: {n_modes_ismn}")
        print(f"ISMN: wilting point: {wilting_point_smap:.2f}, field capacity: {field_capacity_smap:.2f}, Number of modes: {n_modes_smap}")

        if plot:

            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True,  figsize=(10, 5))
            if self.data_treatment_option=='limit_data_to_smap_available_dates':
                ax1.plot(self.soil_moisture_ts['ismn'], 'o', alpha=0.5, label='In-situ', color='#1f77b4')
            else:
                ax1.plot(self.soil_moisture_ts['ismn'], '-', label='In-situ', color='#1f77b4')
            ax1.axhline(y=wilting_point_ismn, color='#1f77b4')
            ax1.axhline(y=field_capacity_ismn, color='#1f77b4')
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Volumetric soil water content [m^3/m^3]")
            # ax2.set_ylabel("Volumetric soil water content [m^3/m^3]")

            if self.data_treatment_option == 'linear_interpolation_on_smap':
                ax2.plot(self.soil_moisture_ts['smap'], label='SMAP', color='#ff7f0e')
            else:
                ax2.plot(self.soil_moisture_ts['smap'], 'o', label='SMAP', color='#ff7f0e',  alpha=0.5)
            ax2.axhline(y=wilting_point_smap, color='#ff7f0e')
            ax2.axhline(y=field_capacity_smap, color='#ff7f0e')
            ax2.set_xlabel("Time")

            fig.legend()
            fig.suptitle(self.plottitle)
            fig.autofmt_xdate()

            fig.savefig(os.path.join(self.out_path, 'test_kernel_ts.png'))
            del fig, ax1, ax2

def main():

    input_path_smap = "../1_data/SMAP/OZNET/Point-Example-SPL3SMP-E-005-results.csv"
    input_path_ismn = "G:/Shared drives/Ryoko and Hilary/SMSigxISMN/analysis/0_data_raw/ISMN"
    out_path = "../3_data_out/linear_interpolation_on_smap"

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    mySMAP = SMAPxISMN(input_path_smap=input_path_smap, input_path_ismn=input_path_ismn, insitu_network_name='OZNET', out_path=out_path)
    mySMAP.load_data(data_treatment_option='linear_interpolation_on_smap', plot=True)
    # data_treatment_option = 'limit_data_to_smap_available_dates', 'linear_interpolation_on_smap'
    # print(f"  Bias [m^3/m^3]: {mySMAP.calc_bias()}")
    # print(f"  RMSE [m^3/m^3]: {mySMAP.calc_RMSE()}")
    # print(f"ubRMSE [m^3/m^3]: {mySMAP.calc_ubRMSE()}")
    # print(f"     R       [-]: {mySMAP.calc_R()}")
    # print(f"     N   [count]: {mySMAP.count_N()}")
    mySMAP.calc_dist(plot=True)

if __name__ == '__main__':
    main()




# TODO: check the timezone UTC? local time?
# All measurements are converted to Coordinated Universal Time (UTC).
# TODO: generalize more