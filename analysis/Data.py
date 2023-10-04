import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings

def get_filename(varname, EASE_row_index, EASE_column_index):
    filename = f"{varname}_{EASE_row_index:03d}_{EASE_column_index:03d}.csv"
    return filename

def set_time_index(df, index_name = 'time'):
    df[index_name] = pd.to_datetime(df[index_name])
    return df.set_index('time')

class Data():
    def __init__(self, cfg, EASEindex) -> None:

        # Read inputs
        self.cfg = cfg
        self.EASE_row_index = EASEindex[0]
        self.EASE_column_index = EASEindex[1]

        # Get the directory name
        self.data_dir = cfg["PATHS"]["data_dir"]

        # Setup variable names 
        self.p_varname = "SPL4SMGP"
        self.pet_varname = "PET"

        # Get the start and end time of the analysis
        self.start_time = datetime(cfg["EXTENT"]["start_date"])
        self.end_date = datetime(cfg["EXTENT"]["end_date"])

        # Read in datasets
        _df = self.get_concat_datasets()
        self.df = self.calc_dSdt(_df)



    def get_concat_datasets(self):

        # Read each datasets
        sm = self.get_soil_moisture(self.cfg)
        pet = self.get_pet(self.cfg)
        p = self.get_precipitation(self.cfg)

        # Concat all the datasets
        _df = pd.merge(sm , pet, how='outer', left_index=True, right_index=True)
        df = pd.merge(_df , p, how='outer', left_index=True, right_index=True)

        return df

    def calc_dSdt(self, df):
        # TODO: put this precip mask back once I've got precip data
        # precip_mask = ds_synced['precip'].where(ds_synced['precip'] < precip_thresh)
        # no_sm_record_but_precip_present = ds_synced['precip'].where((precip_mask.isnull()) & (ds_synced['soil_moisture_daily'].isnull()))

        # Allow detecting soil moisture increment even if there is no SM data in between before/after rainfall event
        df['sm_for_dS_calc'] = df['soil_moisture_daily'].ffill() 

        # Calculate dS
        df['dS'] = df['sm_for_dS_calc'].bfill(limit=5).diff().where(df['sm_for_dS_calc'].notnull().shift(periods=+1))

        # Drop the dS where  (precipitation is present) && (soil moisture record does not exist)
        df['dS'] = df['dS'].where((df['dS'] > -1) & (df['dS'] < 1))

        # Calculate dt
        non_nulls = df['sm_for_dS_calc'].isnull().cumsum()
        nan_length = non_nulls.where(df['sm_for_dS_calc'].notnull()).bfill()+1 - non_nulls +1
        df['dt'] = nan_length.where(df['sm_for_dS_calc'].isnull()).fillna(1)

        # Calculate dS/dt
        df['dSdt'] = df['dS']/df['dt']
        df['dSdt'] = df['dSdt'].shift(periods=-1)

        df.loc[df['soil_moisture_daily'].shift(-1).isna(), 'dSdt'] = np.nan

        return df


    def get_soil_moisture(self, varname = "SPL3SMP"):

        # Read data
        fn = get_filename(varname, self.EASE_row_index, self.EASE_column_index)
        _df = pd.read_csv(fn)
        _df = set_time_index(_df, index_name='time')

        # Use retrieval flag to quality control the data
        condition_bad_data_am = (_df['Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag'] != 0.0) & (df['Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag'] != 8.0)
        condition_bad_data_pm = (_df['Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm'] != 0.0) & (df['Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm'] != 8.0)
        _df.loc[condition_bad_data_am, 'Soil_Moisture_Retrieval_Data_AM_soil_moisture'] = np.nan
        _df.loc[condition_bad_data_pm, 'Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm'] = np.nan

        # If there is two different versions of 2015-03-31 data --- remove this 
        duplicate_labels = df.index.duplicated(keep=False)
        df = df.loc[~df.index.duplicated(keep='first')]

        # Resample to regular time interval
        df = df.resample('D').asfreq()

        # Check if the data is all nan or not

            
        # Merge the AM and PM soil moisture data into one daily timeseries of data
        df['soil_moisture_daily'] = df[['Soil_Moisture_Retrieval_Data_AM_soil_moisture','Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm']].mean(axis=1, skipna=True)
        df['normalized_S'] = (df.soil_moisture_daily - min_sm)/(max_sm - min_sm)

        return df


    def get_pet(self, varname = "SPL4SMGP"):
        

        return df

    def get_precipitation(self, varname = "PET"):
        
        return df
