import os
import warnings
import threading
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime


from .data import Data
from .event import Event
from .mylogger import getLogger

# Create a logger
log = getLogger(__name__)


anc_dict = {
    'precip' : 'SPL4SMGP',
    'PET' : 'PET',
}


col_map = {
    'precipitation_total_surface_flux': 'precip',
    'pet': 'PET',
    'Soil_Moisture_Retrieval_Data_AM_soil_moisture': 'SWC_AM',
    'Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm': 'SWC_PM',
    'Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag': 'SWC_AM_QC',
    'Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm': 'SWC_PM_QC',
    'Soil_Moisture_Retrieval_Data_AM_surface_flag': 'SWC_AM_surf_QC',
    'Soil_Moisture_Retrieval_Data_PM_surface_flag_pm': 'SWC_PM_surf_QC',
}

def get_filename(varname, EASE_row_index, EASE_column_index):
    """Get the filename of the datarod"""
    filename = f"{varname}_{EASE_row_index:03d}_{EASE_column_index:03d}.csv"
    return filename


# def set_time_index(df, index_name="time"):
#     """Set the datetime index to the pandas dataframe"""
#     df[index_name] = pd.to_datetime(df[index_name])
#     return df.set_index("time")


class SMAPData(Data):
    """Class that handles datarods (Precipitation, SM, PET data) for an EASE pixel"""

    def __init__(self, cfg, EASEindex) -> None:
        self.id = EASEindex
        super().__init__(cfg)
        # _______________________________________________________________________________
        # Attributes

        # Read inputs
        # self.cfg = cfg
        # self.id = EASEindex
        # self.EASE_row_index = EASEindex[0]
        # self.EASE_column_index = EASEindex[1]

        # Define soil depth [m]
        # self.z = 0.05

        # Get the directory name
        # self.data_dir = cfg.get('data_dir')  # cfg["PATHS"]["data_dir"]
        # self.datarods_dir = cfg.get('datarods_dir')  #cfg["PATHS"]["datarods_dir"]

        # Get the start and end time of the analysis
        date_format = "%Y-%m-%d"
        # self.start_date = datetime.strptime(cfg["EXTENT"]["start_date"], date_format)
        # self.end_date = datetime.strptime(cfg["EXTENT"]["end_date"], date_format)
        self._start_date = datetime.strptime(self.cfg.get('start_date'), date_format)
        self._end_date = datetime.strptime(self.cfg.get('end_date'), date_format)

        # TODO: Update to get soil texture + field capacity
        # self.soil_texture = None
        # self.theta_fc = self.max_sm
        # self.n = np.nan
        # _______________________________________________________________________________
        # Datasets
        # _df = self.get_concat_datasets()
        # self.df = self.calc_dSdt(_df)
        # self.df = self.get_data()
    
    # def _get_data(self):
    #     _df = self.get_concat_datasets()
    #     return self.calc_dSdt(_df)

    # def get_concat_datasets(self):
    #     """Get datarods for each data variable, and concatinate them together to create a pandas dataframe"""

    #     # ___________________
    #     # Read each datasets
    #     sm = self.get_soil_moisture()
    #     pet = self.get_pet()
    #     p = self.get_precip()

    #     # ___________________
    #     # Concat all the datasets
    #     _df = pd.merge(sm, pet, how="outer", left_index=True, right_index=True)
    #     df = pd.merge(_df, p, how="outer", left_index=True, right_index=True)

    #     return df
    
    # def add_data_cols(self, cols):
        # self.df = pd.merge(self.df, cols, how="outer", left_index=True, right_index=True)

    def get_data(self, varname="SPL3SMP"):
        """Get a datarod of soil moisture data for a pixel"""

        # Get variable dataframe
        # _df = self.get_dataframe(varname=varname)
        df = self.read_datarod(varname=varname)

        # Use retrieval flag to quality control the data
        condition_bad_data_am = (
            df["Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag"] != 0.0
        ) & (df["Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag"] != 8.0)
        condition_bad_data_pm = (
            df["Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm"] != 0.0
        ) & (df["Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm"] != 8.0)
        df.loc[
            condition_bad_data_am, "Soil_Moisture_Retrieval_Data_AM_soil_moisture"
        ] = np.nan
        df.loc[
            condition_bad_data_pm, "Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm"
        ] = np.nan
        # TODO: Rename columns

        # If there are two different versions of 2015-03-31 data --- remove this
        df = df.loc[~df.index.duplicated(keep="first")]

        # Resample to regular time interval
        df = df.resample("D").asfreq()

        # Merge the AM and PM soil moisture data into one daily timeseries of data
        df['SWC'] = df[
            [
                "Soil_Moisture_Retrieval_Data_AM_soil_moisture",
                "Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm",
            ]
        ].mean(axis=1, skipna=True)

        # Rename columns
        df.rename(columns=col_map, inplace=True)

        # # Get max and min values
        # self.min_sm = df.SWC.min(skipna=True)
        # # Instead of actual max values, take the 95% percentile as max_sm # df.SWC.max(skipna=True)
        # self.max_sm = df.SWC.quantile(0.95)
        # df.SWC.max(skipna=True)
        # self.s_thresh = df.SWC.quantile(
        #     0.95
        # )
        # df["SWC_before_masking"] = df["SWC"].copy()
        # # Mask out the timeseries when sm is larger than 90% percentile value
        # df.loc[df["SWC"] > self.max_sm, "SWC"] = np.nan
        # self.range_sm = self.max_sm - self.min_sm
        # if not (np.isnan(self.min_sm) or np.isnan(self.max_sm)):
        #     df["normalized_sm"] = (df.SWC - self.min_sm) / self.range_sm

        return df

    def read_datarod(self, varname, dt_col='time'):
        """Read the datarod for a variable of interest"""

        fn = get_filename(
            varname,
            EASE_row_index=self.id[0], #self.EASE_row_index,
            EASE_column_index=self.id[1], #self.EASE_column_index,
        )
        df = pd.read_csv(
            os.path.join(self.cfg.get('data_dir'), varname, fn)
        )
        # Set datetime index + crop
        df[dt_col] = pd.to_datetime(df[dt_col])
        df.set_index(dt_col, inplace=True, drop=True)

        start_date = datetime.strptime(self.cfg.get('start_date'), "%Y-%m-%d")
        end_date = datetime.strptime(self.cfg.get('end_date'), "%Y-%m-%d")

        return df[start_date : end_date]

    def get_pet(self, col='PET'):
        # df = self.read_datarod(varname=varname)
        # # Drop unnecessary columns
        # df = df.drop(columns=['x', 'y'])
        # # Resample to regular time intervals
        # return df.resample('D').asfreq()
        if col not in self.df.columns:
            self._add_data_col(
                'PET', col_map={'pet': 'PET'}
            )
        return self.df[col]
    
    # def get_precip(self, varname='SPL4SMGP'):
    def get_precip(self, col='precip'):
        # df = self.read_datarod(varname=varname)
        # # Drop unnecessary columns
        # df = df.drop(columns=['x', 'y'])
        if col not in self.df.columns:
            self._add_data_col(
                col=col, col_map={'precipitation_total_surface_flux': 'precip'}
            )
            # Convert precipitation from kg/m2/s to mm/day -> 1 kg/m2/s = 86400 mm/day
            self.df.precip = self.df.precip * 86400
        return self.df[col]


    def add_data_cols(self, cols, col_map=col_map):
        for col in cols:
            self._add_data_col(col)

    def _add_data_col(self, col, col_map=col_map):
        if col in anc_dict.keys():
            var = anc_dict[col]
        # Read data
        df = self.read_datarod(varname=var)
        # Drop unnecessary columns
        df = df.drop(columns=['x', 'y'])
        # Resample to regular time intervals
        df = df.resample('D').asfreq()
        # Rename columns
        if col_map is not None:
            df.rename(columns=col_map, inplace=True)
        # Merge the data
        self.df = self.df.join(df, how='outer')
        # self.df = self.df.merge(df, how='outer', left_index=True, right_index=True)


#-------------------------------------------------------------------------------



    # def calc_dsdt(self, df):
    #     """Calculate d(Soil Moisture)/dt"""

    #     # Allow detecting soil moisture increment even if there is no SM data in between before/after rainfall event
    #     # df["sm_for_dS_calc"] = df["SWC_unmasked"].ffill()
    #     sm = df['SWC'].ffill()

    #     # Calculate dS
    #     # df['ds'] 
    #     ds = (
    #         sm
    #         .bfill(limit=5)
    #         .diff()
    #         .where(sm.notnull().shift(periods=+1))
    #     )

    #     # Drop the dS where  (precipitation is present) && (soil moisture record does not exist)
    #     # df['ds'] = df['ds'].where((df['ds'] > -1) & (df['ds'] < 1))
    #     ds = ds.where((ds > -1) & (ds < 1))

    #     # Calculate dt
    #     non_nulls = sm.isnull().cumsum()
    #     nan_length = (
    #         non_nulls.where(sm.notnull()).bfill() + 1 - non_nulls + 1
    #     )
    #     # df["dt"] = nan_length.where(sm.isnull()).fillna(1)
    #     dt = nan_length.where(sm.isnull()).fillna(1)

    #     # Calculate dS/dt
    #     # df['ds_dt'] = df['ds'] / df["dt"]
    #     df['ds_dt'] = ds / dt
    #     df['ds_dt'] = df['ds_dt'].shift(periods=-1)

    #     df.loc[
    #         df["SWC"].shift(-1).isna(), 'ds_dt'
    #     ] = np.nan
    #     df['ds_dt'] = df['ds_dt'].ffill(limit=5)

    #     return df


        # if not (np.isnan(self.min_sm) or np.isnan(self.max_sm)):
        #     df["normalized_sm"] = (df.SWC - self.min_sm) / self.range_sm

        # return df
    

#-------------------------------------------------------------------------------
# From EventSeparator
#-------------------------------------------------------------------------------
    def separate_events(self, ):
        # Add precip column
        self.get_precip()
        # Mask values
        self.mask_values(self.df, 'SWC', self.max_sm)
        # Calculate dS/dt
        self.calc_dsdt(self.df, 'SWC_unmasked')
        # Find events
        events = self.find_events()
        if events.empty:
            self.events_df = None
            self.events = None
            return None
        self.events_df = self.extract_events(events)
        # Filter events
        self.filter_events(self._params['duration'])
        # Create events
        self.events = self.create_events(self.events_df)
        return self.events


    def mask_values(self, df, col, max_sm):
        """
        Mask out the values in a column when they are larger than the max_sm value
        """
        df["SWC_unmasked"] = df['SWC'].copy()
        df.loc[df[col] > max_sm, col] = np.nan
        # return df
        

    def calc_dsdt(self, df, col='SWC_unmasked'):
        """Calculate d(Soil Moisture)/dt"""

        # Allow detecting soil moisture increment even if there is no SM data in 
        # between before/after rainfall event
        sm = df[col].ffill()

        # Calculate ds
        df["ds"] = (sm.bfill(limit=5).diff().where(
            sm.notnull().shift(periods=+1)
        ))

        # Drop the ds where (precipitation is present) & (soil moisture record does not exist)
        df["ds"] = df["ds"].where((df["ds"] > -1) & (df["ds"] < 1))

        # Calculate dt
        non_nulls = sm.isnull().cumsum()
        nan_length = non_nulls.where(sm.notnull()).bfill() + 1 - non_nulls + 1
        df["dt"] = nan_length.where(sm.isnull()).fillna(1)

        # Calculate ds/dt
        df["ds_dt"] = df["ds"] / df["dt"]
        df["ds_dt"] = df["ds_dt"].shift(periods=-1)

        df.loc[df[col].shift(-1).isna(), "ds_dt"] = np.nan

        df["ds_dt"] = df["ds_dt"].ffill(limit=5)

        return df



    def find_events(self, diff_col='ds_dt', col='SWC_unmasked'):
        # Find start dates of events
        start_dates = self.find_starts(diff_col=diff_col, col=col)
        # Find end dates of events
        end_dates = self.find_ends(start_dates, diff_col=diff_col)
        # Get event dates
        event_dates = pd.DataFrame({'start_date': start_dates, 'end_date': end_dates})
        return event_dates


    def find_starts(self, diff_col='ds_dt', col='SWC_unmasked', threshold=None):
        start_dates = self._find_starts(diff_col=diff_col, threshold=threshold)
        adjusted = self._look_back(start_dates, col=col)
        adjusted = self._look_ahead(adjusted, col=col)

        return adjusted

    def _find_starts(self, diff_col='ds_dt', threshold=None):
        if not threshold:
            threshold = self._params['target_rmsd'] * 2

        neg = self.df[diff_col] < 0
        pos = self.df[diff_col] > threshold

        start_dates = self.df.index[neg & np.concatenate(([False], pos[:-1]))]
        # self.df['start_date'] = neg & np.concatenate(([False], pos[:-1]))
        return start_dates
    

    def _look_back(self, start_dates, col='SWC_unmasked'):
        adjusted = []
        # Loop for event start dates
        for i, start_date in enumerate(start_dates):
            # Look back up to 6 timesteps to seek for sm value which is not nan
            for j in range(0, 6):  
                current_date = start_date - pd.Timedelta(days=j)

                # If rainfall exceeds threshold, stop there
                if self.df.loc[current_date].precip > self._params['precip_thresh']:
                    # If SM value IS NOT nap.nan, update the event start date to this timestep
                    if not np.isnan(self.df.loc[current_date, col]):
                        update_date = current_date
                    # If SM value IS nap.nan, don't update the event start date value
                    else:
                        update_date = start_date
                    break

                # If ds > 0, stop there
                if self.df.loc[current_date]['ds'] > 0:
                    update_date = start_date
                    break

                # If reached to the NON-nan SM value, update start date value to this timestep
                if ((i - j) >= 0) or (not np.isnan(self.df.loc[current_date, col])):
                    update_date = current_date
                    break
            # start_dates[i] = update_date
            adjusted.append(update_date)
        return adjusted
    
    def _look_ahead(self, start_dates, col='SWC_unmasked'):

        nan_dates = self.df.loc[start_dates].index[self.df.loc[start_dates, col].isna()]

        adjusted = start_dates.copy()
        for i, start_date in enumerate(nan_dates):
            update_date = start_date
            # Look ahead up to 6 timesteps to seek for sm value which is not nan, 
            # or start of the precip event
            for j in range(0, 6):  
                current_date = start_date + pd.Timedelta(days=j)
                # If Non-nan SM value is detected, update start date value to this timstep
                if current_date > self._end_date:
                    update_date = current_date
                    break

                if not pd.isna(self.df.loc[current_date, col]):
                    update_date = current_date
                    break
            ind = start_dates.get_loc[start_date]
            adjusted[ind] = update_date
        return adjusted



    def find_ends(self, start_dates, diff_col='ds_dt', col='SWC_unmasked'):
        end_dates = [
            self.find_event_end(
                i, start_dates, diff_col=diff_col) for i in range(len(start_dates)
            )
        ]
        return pd.DatetimeIndex(end_dates)


    def find_event_end(self, i, start_dates, diff_col='ds_dt', col='SWC_unmasked'):
        start_date = start_dates[i]
        end_date = start_dates
        for j in range(1, len(self.df)):
            current_date = start_date + pd.Timedelta(days=j)

            if current_date > self._end_date:
                break

            if np.isnan(self.df.loc[current_date, col]):
                continue
            
            check_diff = self.df.loc[current_date, diff_col] >= self._params['noise_thresh']
            check_precip = self.df.loc[current_date, 'precip'] > self._params['precip_thresh']
            check_event_start = current_date in start_dates

            if check_diff or check_precip or check_event_start:
                end_date = current_date
                break
            else:
                continue
        return end_date
    


    def filter_events(self, min_dur=5):
        self.events_df = self.events_df[
            self.events_df['soil_moisture'].apply(lambda x: pd.notna(x).sum())
            >= min_dur
        ].copy()
        self.events_df.reset_index(drop=True, inplace=True)


    # def create_events(self, events_df):
    #     events = [
    #         Event(
    #             **row.to_dict(),
    #             theta_w = self.min_sm,
    #             theta_star = self.max_sm, # TODO: Get this from config, mask if > fc
    #             z = self.z,
    #             event_data = self.get_event_data(row.start_date, row.end_date)
    #         ) 
    #         for i, row in events_df[['start_date','end_date','soil_moisture']].iterrows()
    #     ]
    #     return events

    # def get_event_data(self, start, end, cols=['precip','PET']):
    #     new_cols = [col for col in cols if col not in self.df.columns]
    #     if new_cols:
    #         self.add_data_cols(new_cols)
    #     return self.df.loc[start:end]


    # def _get_theta_star(self):
    #     if self.cfg.get('theta_star').lower() in ['theta_fc', 'fc', 'field capacity']:
    #         theta_star = self.theta_fc
    #     elif self.cfg.get('theta_star').lower() in ['max_sm', 'maximum', 'max']:
    #         theta_star = self.max_sm
    #     else:
    #         log.info(
    #             f"Unknown theta_star value: {self.cfg.get('theta_star')}. 
    #             Setting to 95th percentile value"
    #         )
    #         theta_star = self.max_sm
    #     return theta_star



    def plot_events(self):
        fig, (ax11, ax12) = plt.subplots(2, 1, figsize=(20, 5))

        self.df.SWC_unmasked.plot(ax=ax11, alpha=0.5)
        ax11.scatter(
            self.df.SWC_unmasked[
                self.df['start_date']
            ].index,
            self.df.SWC_unmasked[
                self.df['start_date']
            ].values,
            color="orange",
            alpha=0.5,
        )
        ax11.scatter(
            self.df.SWC_unmasked[
                self.df['end_date']
            ].index,
            self.df.SWC_unmasked[
                self.df['end_date']
            ].values,
            color="orange",
            marker="x",
            alpha=0.5,
        )
        self.df.precip.plot(ax=ax12, alpha=0.5)

        # Save results
        filename = f"{self.id[0]:03d}_{self.id[1]:03d}_eventseparation.png"
        output_dir2 = os.path.join(self.output_dir, "plots")
        if not os.path.exists(output_dir2):
            # Use a lock to ensure only one thread creates the directory
            with threading.Lock():
                # Check again if the directory was created while waiting
                if not os.path.exists(output_dir2):
                    os.makedirs(output_dir2)

        fig.savefig(os.path.join(output_dir2, filename))
        plt.close()
