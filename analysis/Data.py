import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import os
import warnings


def get_filename(varname, EASE_row_index, EASE_column_index):
    """Get the filename of the datarod"""
    filename = f"{varname}_{EASE_row_index:03d}_{EASE_column_index:03d}.csv"
    return filename


def set_time_index(df, index_name="time"):
    """Set the datetime index to the pandas dataframe"""
    df[index_name] = pd.to_datetime(df[index_name])
    return df.set_index("time")


class Data:
    """Class that handles datarods (Precipitation, SM, PET data) for a EASE pixel"""

    def __init__(self, cfg, EASEindex) -> None:
        # _______________________________________________________________________________
        # Attributes

        # Read inputs
        self.cfg = cfg
        self.EASE_row_index = EASEindex[0]
        self.EASE_column_index = EASEindex[1]

        # Get the directory name
        self.data_dir = cfg["PATHS"]["data_dir"]
        self.datarods_dir = cfg["PATHS"]["datarods_dir"]

        # Get the start and end time of the analysis
        date_format = "%Y-%m-%d"
        self.start_date = datetime.strptime(cfg["EXTENT"]["start_date"], date_format)
        self.end_date = datetime.strptime(cfg["EXTENT"]["end_date"], date_format)

        # _______________________________________________________________________________
        # Datasets
        _df = self.get_concat_datasets()
        self.df = self.calc_dSdt(_df)

    def get_concat_datasets(self):
        """Get datarods for each data variable, and concatinate them together to create a pandas dataframe"""

        # ___________________
        # Read each datasets
        sm = self.get_soil_moisture()
        pet = self.get_pet()
        p = self.get_precipitation()

        # ___________________
        # Concat all the datasets
        _df = pd.merge(sm, pet, how="outer", left_index=True, right_index=True)
        df = pd.merge(_df, p, how="outer", left_index=True, right_index=True)

        return df

    def get_dataframe(self, varname):
        """Get the pandas dataframe for a datarod of interest

        Args:
            varname (string): name of the variable: "SPL3SMP", "PET", "SPL4SMGP"

        Returns:
            dataframe: Return dataframe with datetime index, cropped for the timeperiod for a variable
        """

        fn = get_filename(
            varname,
            EASE_row_index=self.EASE_row_index,
            EASE_column_index=self.EASE_column_index,
        )
        _df = pd.read_csv(os.path.join(self.data_dir, self.datarods_dir, varname, fn))

        # Set time index and crop
        _df = set_time_index(_df, index_name="time")
        return _df[self.start_date : self.end_date]

    def get_soil_moisture(self, varname="SPL3SMP"):
        """Get a datarod of soil moisture data for a pixel"""

        # Get variable dataframe
        _df = self.get_dataframe(varname=varname)

        # Use retrieval flag to quality control the data
        condition_bad_data_am = (
            _df["Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag"] != 0.0
        ) & (_df["Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag"] != 8.0)
        condition_bad_data_pm = (
            _df["Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm"] != 0.0
        ) & (_df["Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm"] != 8.0)
        _df.loc[
            condition_bad_data_am, "Soil_Moisture_Retrieval_Data_AM_soil_moisture"
        ] = np.nan
        _df.loc[
            condition_bad_data_pm, "Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm"
        ] = np.nan

        # If there is two different versions of 2015-03-31 data --- remove this
        df = _df.loc[~_df.index.duplicated(keep="first")]

        # Resample to regular time interval
        df = df.resample("D").asfreq()

        # Merge the AM and PM soil moisture data into one daily timeseries of data
        df["soil_moisture_daily"] = df[
            [
                "Soil_Moisture_Retrieval_Data_AM_soil_moisture",
                "Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm",
            ]
        ].mean(axis=1, skipna=True)

        # Get max and min values
        self.min_sm = df.soil_moisture_daily.min(skipna=True)
        self.max_sm = df.soil_moisture_daily.max(skipna=True)
        if not (np.isnan(self.min_sm) or np.isnan(self.max_sm)):
            df["normalized_sm"] = (df.soil_moisture_daily - self.min_sm) / (
                self.max_sm - self.min_sm
            )

        return df

    def get_pet(self, varname="PET"):
        """Get a datarod of PET data for a pixel"""

        # Get variable dataframe
        _df = self.get_dataframe(varname=varname)

        # Drop unnccesary dimension
        _df = _df.drop(columns=["x", "y"])

        # Resample to regular time intervals
        return _df.resample("D").asfreq()

    def get_precipitation(self, varname="SPL4SMGP"):
        """Get a datarod of precipitation data for a pixel"""

        # Get variable dataframe
        _df = self.get_dataframe(varname=varname)

        # Drop unnccesary dimension and change variable name
        _df = _df.drop(columns=["x", "y"]).rename(
            {"precipitation_total_surface_flux": "precip"}, axis="columns"
        )

        # Convert precipitation from kg/m2/s to mm/day -> 1 kg/m2/s = 86400 mm/day
        _df.precip = _df.precip * 86400

        # Resample to regular time interval
        return _df.resample("D").asfreq()

    def calc_dSdt(self, df):
        """Calculate d(Soil Moisture)/dt"""

        # TODO: put this precip mask back once I've got precip data
        # precip_mask = ds_synced['precip'].where(ds_synced['precip'] < precip_thresh)
        # no_sm_record_but_precip_present = ds_synced['precip'].where((precip_mask.isnull()) & (ds_synced['soil_moisture_daily'].isnull()))

        # Allow detecting soil moisture increment even if there is no SM data in between before/after rainfall event
        df["sm_for_dS_calc"] = df["soil_moisture_daily"].ffill()

        # Calculate dS
        df["dS"] = (
            df["sm_for_dS_calc"]
            .bfill(limit=5)
            .diff()
            .where(df["sm_for_dS_calc"].notnull().shift(periods=+1))
        )

        # Drop the dS where  (precipitation is present) && (soil moisture record does not exist)
        df["dS"] = df["dS"].where((df["dS"] > -1) & (df["dS"] < 1))

        # Calculate dt
        non_nulls = df["sm_for_dS_calc"].isnull().cumsum()
        nan_length = (
            non_nulls.where(df["sm_for_dS_calc"].notnull()).bfill() + 1 - non_nulls + 1
        )
        df["dt"] = nan_length.where(df["sm_for_dS_calc"].isnull()).fillna(1)

        # Calculate dS/dt
        df["dSdt"] = df["dS"] / df["dt"]
        df["dSdt"] = df["dSdt"].shift(periods=-1)

        df.loc[df["soil_moisture_daily"].shift(-1).isna(), "dSdt"] = np.nan

        return df
