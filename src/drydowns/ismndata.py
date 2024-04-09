import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
import warnings
import threading
import logging

import fluxtower

from .data import Data
from .towerdata import SoilSensorData
from .towerevent import TowerEvent
from .soil import Soil, soils

from .mylogger import getLogger

# Create a logger
log = getLogger(__name__)


class ISMNSoilData(SoilSensorData):

    def __init__(self, cfg, sensor, station):
        # super().__init__(cfg, sensor)
        # cfg
        self.cfg = cfg

        # tower
        self._sensor = sensor
        self._station = station
        # id
        self.id = (sensor.station, sensor.name)

        # info about sensor
        self.info = sensor.meta

        # soil info
        self.soil_texture = self.info.get('soil_texture')
        # self.n = self.soil_info.get('porosity')

        # z (depth of sensor) [m]
        # self.z = float(self.info['HEIGHT']) * -1.
        self.z = float(self.info.get('depth_to'))

        # data
        self.df = self.get_sensor_data()

        # theta_fc
        if self.soil_texture.lower() in soils.keys():
            self.theta_fc = Soil(texture=self.soil_texture).theta_fc
        else:
            log.info(
                f"Texture '{self.soil_texture}' not found in soil database. Setting theta_fc to 0.3."
            )

        # min, max, range
        self.min_sm = self.df.SWC.min()
        self.max_sm = self.df.SWC.quantile(
            # self.cfg['DATA'].getfloat('max_sm_frac', 1.0)
            self.cfg.getfloat('max_sm_frac', 1.0)
        )
        self.range_sm = self.max_sm - self.min_sm

        # self.max_sm = self.theta_fc

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

    def describe(self):
        return {k : v for k,v in self.__dict__.items() if isinstance(v,(str,float,int))}

    # def get_sensor_data(self, sensor_col):
    def get_sensor_data(self):
        # Copy soil moisture data

        sensor_col = self._station.sensor_cols[self._sensor.name]

        start = self.info.get('timerange_from')
        end = self.info.get('timerange_to')

        # df = self._station.data.loc[
        #     (self._station.data.index >= start) & (self._station.data.index < end),
        #     [sensor_col]
        # ].copy()
        df = self._station.daily.loc[
            (self._station.daily.index >= start) & (self._station.daily.index < end),
            [sensor_col]
        ].copy()
        # df = self._tower.data[['TIMESTAMP']+[sensor_col]].copy()
        # df.set_index('TIMESTAMP', inplace=True, drop=False)
        df.index.name = 'DATE'
        # Rename to standard column name
        df.rename(columns={sensor_col: 'SWC'}, inplace=True)

        # Resample to daily
        df = df.resample('D').asfreq()
        # Convert units to m3/m3
        # df['SWC'] = df['SWC']/100
        df.insert(0,'TIMESTAMP',df.index)

        # Update info dict
        self.info.update({'VARNAME' : sensor_col})
        self.info.update({'unit': 'm3 m-3'})

        return df


    def add_data_cols(self, cols):
        # self.df = pd.concat([self.df, cols], axis=1)
        self.df = self.df.join(self._station.daily[cols])

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
        # _noise_thresh = (self.range_sm) * self.cfg.getfloat(
        #     "EVENT_SEPARATION", "frac_range_thresh"
        # )
        
        # params = {
        #     'precip_thresh': self.cfg.getfloat("EVENT_SEPARATION", "precip_thresh"),
        #     # 'target_rmsd': self.cfg.getfloat("EVENT_SEPARATION", "target_rmsd"),
        #     # 'start_diff' : self.cfg.getfloat("EVENT_SEPARATION", "start_thresh"),
        #     # 'end_diff' : np.minimum(
        #     #     noise_thresh, self.cfg.getfloat("EVENT_SEPARATION", "target_rmsd") * 2
        #     # ),
        #     'noise_thresh' : np.minimum(
        #         _noise_thresh, self.cfg.getfloat("EVENT_SEPARATION", "target_rmsd") * 2
        #     ),
        #     'duration' : self.cfg.getint("EVENT_SEPARATION", "min_duration"),
        # }
        _noise_thresh = (self.range_sm) * self.cfg.getfloat("frac_range_thresh")
        params = {
            'precip_thresh': self.cfg.getfloat("precip_thresh"),
            'noise_thresh' : np.minimum(
                _noise_thresh, self.cfg.getfloat("target_rmsd") * 2
            ),
            'duration' : self.cfg.getint("min_duration"),
        }
        return params


    def separate_events(self, cols=['P_F', 'ET_F_MDS']):

        # self.mask_values(self.df, 'SWC', self.max_sm)
        self.mask_values(self.df, 'SWC', self.theta_fc)
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
        new_cols = [col for col in cols if col not in self.df.columns and col in self._station.daily.columns]
        if new_cols:
            self.add_data_cols(new_cols)
        
        return self.df.loc[start:end]

    # def get_precip(self, p_col='P_F',):
    #     if p_col not in self.df.columns:
    #         self.add_data_cols([p_col])
    #     return self.df[p_col]
    
    # def get_et(self, et_col='ET_F_MDS',):
    #     if et_col not in self.df.columns:
    #         self.add_data_cols([et_col])
    #     return self.df[et_col]

    def filter_events(self):
        # raise NotImplementedError
        pass

    def create_events(self, events_df):
        events = [
            TowerEvent(
                **row.to_dict(),
                theta_w = self.min_sm,
                # theta_star = self.max_sm,
                theta_star = self.theta_fc,
                z = self.z, 
                event_data = self.get_event_data(row.start_date, row.end_date)
            )
            for i, row in events_df[['start_date','end_date','soil_moisture']].iterrows()
        ]
        return events

    def plot_event(self):
        raise NotImplementedError



