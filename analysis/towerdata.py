import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
import warnings
import threading
from MyLogger import getLogger
import logging

import fluxtower
from Data import Data
from towerevent import TowerEvent

# Create a logger
log = getLogger(__name__)


# class TowerData(Data):
#     """Class that handles datarods (Precipitation, SM, PET data) for a flux tower"""

#     def __init__(self, cfg, tower=None, tower_id=None, timestep='DD') -> None:
#         # _______________________________________________________________________________
#         # Attributes

#         # Read inputs
#         self.cfg = cfg

#         # data_dir
#         self.data_dir = cfg["PATHS"]["data_dir"]
        
#         # tower
#         if tower:
#             self._tower = tower
#         else:
#             self._tower = fluxtower.FluxNetTower(self.get_filename(tower_id, timestep))
        
#         # start_date and end_date
#         self.start_date = self._tower.data.index.min().to_pydatetime()
#         self.end_date = self._tower.data.index.max().to_pydatetime()

#         # col_dict
#         self._cols, self._col_dict = self.get_cols()        

#         # data
#         self.soil_data = self.get_data()


    
#     def get_filename(self, tower_id, timestep='DD'):
#         """Get the filename of the datarod"""
#         pattern = re.compile(rf'FLX_{tower_id}_FLUXNET2015_FULLSET_{timestep}_(.*)\.csv$')
#         for fn in os.listdir(self.data_dir):
#             if re.match(pattern, fn):
#                 return fn
#         return None
    
#     def get_cols(self, var_cols=['SWC', 'P', 'TA', 'VPD', 'LE', 'ET', 'e_a']):
#         col_list = []
#         col_dict = {}
#         for var in var_cols:
#             col_list += self._tower.get_var_cols(variable=var, exclude='ERA')
#             cols = self._tower.get_var_cols(variable=var, exclude='ERA')
#             col_dict[var] = {
#                 'var_cols' : [col for col in cols if 'QC' not in col], 
#                 'qc_cols' : [col for col in cols if 'QC' in col]
#             }

#         return col_list, col_dict

#     def get_data(self) -> list:
#         sm_cols = self._col_dict['SWC']['var_cols']
#         soil_data = [SoilSensorData(self.cfg, self._tower, col) for col in sm_cols]
#         return soil_data


class ThreadNameHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            # Add thread name to the log message
            record.threadName = threading.current_thread().name
            super(ThreadNameHandler, self).emit(record)
        except Exception:
            self.handleError(record)


class SoilSensorData(Data):

    def __init__(self, cfg, tower, sensor_col,):
        # cfg
        self.cfg = cfg

        # tower
        self._tower = tower
        # id
        self.id = (self._tower.id, sensor_col)

        # info about sensor
        self.info = self._tower.var_info.get(sensor_col).copy()

        # z (depth of sensor) [m]
        self.z = float(self.info['HEIGHT']) * -1.

        # data
        self.df = self.get_sensor_data(sensor_col)

        # min, max, range
        self.min_sm = self.df.SWC.min()
        self.max_sm = self.df.SWC.quantile(
            self.cfg['DATA'].getfloat('max_sm_frac', 1.0)
        )
        self.range_sm = self.max_sm - self.min_sm

        # Calculate diff + normalize values.
        self.calc_diff()
        self.normalize()


        # Moved from TowerEventSeparator
        #----------------------------------------------------------------------
        self._params = self.get_params()

        # self.events = self.separate_events()

        current_thread = threading.current_thread()
        current_thread.name = ( f"{self.id[0]}, {self.id[1]}" )
        self.thread_name = current_thread.name



    def get_sensor_data(self, sensor_col):
        # Copy soil moisture data
        df = self._tower.data[['TIMESTAMP']+[sensor_col]].copy()
        df.set_index('TIMESTAMP', inplace=True, drop=False)
        df.index.name = 'DATE'
        # Rename to standard column name
        df.rename(columns={sensor_col: 'SWC'}, inplace=True)

        # Convert units to m3/m3
        df['SWC'] = df['SWC']/100

        # Update info dict
        self.info.update({'VARNAME' : sensor_col})
        self.info.update({'unit': 'm3 m-3'})

        return df
    
    def add_data_cols(self, cols):
        # self.df = pd.concat([self.df, cols], axis=1)
        self.df = self.df.join(self._tower.data.set_index('TIMESTAMP')[cols])

    def calc_diff(self):
        self.df['SWC_diff'] = self.df['SWC'].diff() #/ df['TIMESTAMP'].diff().dt.days


    def normalize(self):
        self.df['norm_sm'] = (self.df['SWC'] - self.min_sm) / self.range_sm


#-------------------------------------------------------------------------------
# From TowerEventSeparator
#-------------------------------------------------------------------------------
    def get_params(self):
        """
        Get the parameters for event separation
        """
        _noise_thresh = (self.max_sm - self.min_sm) * self.cfg.getfloat(
            "EVENT_SEPARATION", "frac_range_thresh"
        )
        
        params = {
            'precip_thresh': self.cfg.getfloat("EVENT_SEPARATION", "precip_thresh"),
            # 'target_rmsd': self.cfg.getfloat("EVENT_SEPARATION", "target_rmsd"),
            # 'start_diff' : self.cfg.getfloat("EVENT_SEPARATION", "start_thresh"),
            # 'end_diff' : np.minimum(
            #     noise_thresh, self.cfg.getfloat("EVENT_SEPARATION", "target_rmsd") * 2
            # ),
            'noise_thresh' : np.minimum(
                _noise_thresh, self.cfg.getfloat("EVENT_SEPARATION", "target_rmsd") * 2
            ),
            'duration' : self.cfg.getint("EVENT_SEPARATION", "min_duration"),
        }
        return params


    def separate_events(self, cols=['P_F', 'ET_F_MDS']):

        self.mask_values(self.df, 'SWC', self.max_sm)
        self.calc_dsdt(self.df, 'SWC_masked')

        events = self.find_events()
        if events.empty:
            self.events_df = None
            self.events = None
            return None
            # return None
        self.events_df = self.extract_events(events)
        
        # events[['min_sm', 'max_sm']] = self.df.groupby('Event')['SWC'].agg(['min', 'max'])
        # events['range_sm'] = events.max_sm - events.min_sm

        # self.filter_events(self.min_duration)
        self.events = self.create_events(self.events_df)

        # if self.plot:
        #     self.plot_events()

        # return self.events

    def mask_values(self, df, col, max_sm):
        df[col+'_masked'] = df[col].mask(df[col] > max_sm).bfill()
    
    def calc_dsdt(self, df, col):
        df['ds_dt'] = df[col].diff()

    def find_events(self, diff_col='ds_dt',): #min_dur, start_diff, end_diff):
        # Find start dates of events
        start_dates = self.find_starts(
            diff_col, min_dur=self._params['duration'], threshold=self._params['noise_thresh']
        )
        # Find end dates of drydown events
        end_dates = self.find_ends(
            start_dates, diff_col, self._params['noise_thresh']
        )
        # Get event dates
        event_dates = pd.DataFrame({'start_date': start_dates, 'end_date': end_dates})
        return event_dates


    def find_starts(self, diff_col='ds_dt', min_dur=4, threshold=0.):
        # Mask for differences below threshold
        mask_diff = self.df[diff_col].shift(-1) < -threshold
        # Indexer for rolling window (forward-looking)
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=min_dur)
        # Calculate number of upcoming days where differences are below threshold + 
        # mask for duration >= min_dur
        mask_dur = mask_diff.rolling(window=indexer, min_periods=min_dur).sum() >= min_dur

        events = (~mask_dur).cumsum()
        valid = events[mask_dur]
        # labels = {event: new_event for new_event, event in enumerate(valid.unique(), start=1)}
        # df['EVENT'] = np.nan
        # df.loc[mask_dur, 'EVENT'] = events[mask_dur].map(labels)
        start_dates = self.df.groupby(valid).TIMESTAMP.min().reset_index(drop=True)
        return start_dates

    def find_ends(self, start_dates, diff_col, threshold=0.):
        end_dates = start_dates.apply(self.find_event_end, args=(diff_col, threshold,))
        return end_dates

    def find_event_end(self, start_date, diff_col, threshold=0.0):
        end = (self.df[start_date:][diff_col].shift(-1) > threshold).idxmax()
        return end


    def label_events(self, event_dates):
        self.df['Event'] = np.nan
        for i, row in event_dates.iterrows():
            self.df.loc[row['start_date']:row['end_date'], 'Event'] = int(i)

    def extract_events(self, event_dates, col='SWC'):
        if 'Event' not in self.df.columns:
            self.label_events(event_dates)
        events = event_dates.join(
            self.df[self.df.Event.notnull()].groupby('Event')[col].apply(list)
        ).reset_index(drop=True)
        events.rename(columns={col: 'soil_moisture'}, inplace=True)

        return events

    def get_event_data(self, start, end, cols=['P_F', 'ET_F_MDS']):
        new_cols = [col for col in cols if col not in self.df.columns]
        if new_cols:
            self.add_data_cols(new_cols)
        
        return self.df.loc[start:end]

    def get_precip(self, p_col='P_F',):
        if p_col not in self.df.columns:
            self.add_data_cols([p_col])
        return self.df[p_col]
    
    def get_et(self, et_col='ET_F_MDS',):
        if et_col not in self.df.columns:
            self.add_data_cols([et_col])
        return self.df[et_col]

    def filter_events(self):
        # raise NotImplementedError
        pass

    def create_events(self, events_df):
        events = [
            TowerEvent(
                **row.to_dict(),
                theta_w = self.min_sm,
                theta_star = self.max_sm,
                event_data = self.get_event_data(row.start_date, row.end_date)
            )
            for i, row in events_df[['start_date','end_date','soil_moisture']].iterrows()
        ]
        return events

    def plot_event(self):
        raise NotImplementedError

