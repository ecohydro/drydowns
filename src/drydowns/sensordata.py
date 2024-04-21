import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from pathlib import Path
# from datetime import datetime
# import warnings
# import threading
# import logging

import fluxtower

from .data import Data
from .event import Event
from .soil import Soil, soils

from .mylogger import getLogger

# Create a logger
log = getLogger(__name__)


# col_map = {
#     'P_F': 'precip',
#     'ET_F_MDS': 'ET',
# }
col_dict = {
    'precip' : 'P_F',
    'PET' : 'ET_F_MDS',
}

# col_map = dict((v,k) for k,v in col_dict.items())
col_map = {
    'P_F': 'precip',
    'ET_F_MDS': 'PET',
    'P_1_1' : 'precip', # TODO: get other precip cols + take avg (for ISMN data)
    'pet' : 'PET'
}

# class ThreadNameHandler(logging.StreamHandler):
#     def emit(self, record):
#         try:
#             # Add thread name to the log message
#             record.threadName = threading.current_thread().name
#             super(ThreadNameHandler, self).emit(record)
#         except Exception:
#             self.handleError(record)


class SensorData(Data):
    def __init__(self, cfg, station, sensor_id):
        # id
        self.id = (station.id, sensor_id)
        # info
        self.info = self._get_meta(sensor_id)

        super().__init__(cfg, sensor_id=sensor_id)
        # cfg
        # self.cfg = cfg

        # # id
        # self.id = (station.id, sensor_id)

        # # info
        # self.info = self._get_meta(sensor_id)

        # z
        # self.z = 0.05 #np.nan

        # texture
        self.soil_texture = self.info.get('soil_texture')

        # soil, theta_fc, porosity (n)
        # if self.soil_texture.lower() in soils.keys():
        #     self._soil = Soil(texture=self.soil_texture)
        #     # self.theta_fc = self._soil.theta_fc
        #     # self.n = self._soil.n
        # else:
        #     # Eventually, need to go get soil texture...
        #     log.info(
        #         f"Texture '{self.soil_texture}' not found in soil database. "
        #     )
        #     self._soil = None

        # data
        # self.df = self.get_data()

        # # min, max, range
        # self.min_sm = self.df.SWC.min()
        # self.max_sm = self.df.SWC.quantile(
        #     # self.cfg['DATA'].getfloat('max_sm_frac', 1.0)
        #     self.cfg.getfloat('max_sm_frac', 1.0)
        # )
        # self.range_sm = self.max_sm - self.min_sm

        # self.max_sm = self.theta_fc

        # Calculate diff + normalize values.
        self.calc_diff()
        self.normalize()

        # Moved from TowerEventSeparator
        #----------------------------------------------------------------------
        # self._params = self.get_params()

        # self.events = self.separate_events()

        # current_thread = threading.current_thread()
        # current_thread.name = ( f"{self.id[0]}, {self.id[1]}" )
        # self.thread_name = current_thread.name
    
    def _get_meta(self, sensor_id):
        # raise NotImplementedError
        return None

    # def get_data(self):
    #     # raise NotImplementedError
    #     return None

    def add_data_cols(self, cols):
        # self.df = pd.concat([self.df, cols], axis=1)
        raise NotImplementedError

    # def calc_diff(self):
    #     self.df['SWC_diff'] = self.df['SWC'].diff() #/ df['TIMESTAMP'].diff().dt.days


#-------------------------------------------------------------------------------
# From TowerEventSeparator
#-------------------------------------------------------------------------------
    # def get_params(self):
    #     """
    #     Get the parameters for event separation
    #     """
    #     # _noise_thresh = (self.range_sm) * self.cfg.getfloat(
    #     #     "EVENT_SEPARATION", "frac_range_thresh"
    #     # )
        
    #     # params = {
    #     #     'precip_thresh': self.cfg.getfloat("EVENT_SEPARATION", "precip_thresh"),
    #     #     # 'target_rmsd': self.cfg.getfloat("EVENT_SEPARATION", "target_rmsd"),
    #     #     # 'start_diff' : self.cfg.getfloat("EVENT_SEPARATION", "start_thresh"),
    #     #     # 'end_diff' : np.minimum(
    #     #     #     noise_thresh, self.cfg.getfloat("EVENT_SEPARATION", "target_rmsd") * 2
    #     #     # ),
    #     #     'noise_thresh' : np.minimum(
    #     #         _noise_thresh, self.cfg.getfloat("EVENT_SEPARATION", "target_rmsd") * 2
    #     #     ),
    #     #     'duration' : self.cfg.getint("EVENT_SEPARATION", "min_duration"),
    #     # }
    #     _noise_thresh = (self.range_sm) * self.cfg.getfloat("frac_range_thresh")
    #     params = {
    #         'precip_thresh': self.cfg.getfloat("precip_thresh"),
    #         'noise_thresh' : np.minimum(
    #             _noise_thresh, self.cfg.getfloat("target_rmsd") * 2
    #         ),
    #         'duration' : self.cfg.getint("min_duration"),
    #     }
    #     return params


    def separate_events(self): #, cols=['P_F', 'ET_F_MDS']):

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

        self.filter_events()
        # self.filter_events(self.min_duration)
        self.events = self.create_events(self.events_df)

        # if self.plot:
        #     self.plot_events()

        # return self.events

    def mask_values(self, df, col, max_sm,):
        df[col+'_masked'] = df[col].mask(df[col] > max_sm).bfill()
    
    def calc_dsdt(self, df, col):
        df['ds_dt'] = df[col].diff()

    def find_events(self, diff_col='ds_dt',): #min_dur, start_diff, end_diff):
        # Find start dates of events
        start_dates = self.find_starts(
            diff_col, min_dur=self._params['duration'], min_diff=0.5#self._params['min_diff'] #threshold=self._params['noise_thresh']
        )
        # Find end dates of drydown events
        end_dates = self.find_ends(
            start_dates, diff_col, min_diff=self._params['min_diff']
        )
        # Get event dates
        event_dates = pd.DataFrame({'start_date': start_dates, 'end_date': end_dates})
        return event_dates


    def find_starts(self, diff_col='ds_dt', min_dur=4, min_diff=0.5):#threshold=0.):
        # Mask for differences below threshold
        threshold = self._calc_ds_dt_threshold(min_diff)
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

    def find_ends(self, start_dates, diff_col, min_diff=0.5):#threshold=0.):
        end_dates = start_dates.apply(self.find_event_end, args=(diff_col, min_diff,))
        return end_dates

    def find_event_end(self, start_date, diff_col, min_diff=0.5):#threshold=None):
        # Get dtheta/dt threshold (min diff allowed during drydown)
        # (Added to avoid long tails)
        threshold = self._calc_ds_dt_threshold(min_diff)
        end = (self.df[start_date:][diff_col].shift(-1) > -threshold).idxmax()
        return end
    
    def _calc_ds_dt_threshold(self, min_diff=0.5):
        # Calculate the threshold for ds/dt
        return 1 * min_diff / (self.z * 1000)


    # def label_events(self, event_dates):
    #     self.df['Event'] = np.nan
    #     for i, row in event_dates.iterrows():
    #         self.df.loc[row['start_date']:row['end_date'], 'Event'] = int(i)

    # def extract_events(self, event_dates, col='SWC'):
    #     if 'Event' not in self.df.columns:
    #         self.label_events(event_dates)
    #     events = event_dates.join(
    #         self.df[self.df.Event.notnull()].groupby('Event')[col].apply(list)
    #     ).reset_index(drop=True)
    #     events.rename(columns={col: 'soil_moisture'}, inplace=True)

    #     return events

    # def get_event_data(self, start, end, cols=['P_F', 'ET_F_MDS']):
    #     new_cols = [col for col in cols if col not in self.df.columns]
    #     if new_cols:
    #         self.add_data_cols(new_cols)
        
    #     return self.df.loc[start:end]

    # def get_precip(self, p_col='P_F',):
    #     if p_col not in self.df.columns:
    #         self.add_data_cols([p_col])
    #     return self.df[p_col]
    
    # def get_et(self, et_col='ET_F_MDS',):
    #     if et_col not in self.df.columns:
    #         self.add_data_cols([et_col])
    #     return self.df[et_col]

    # def filter_events(self):
    #     # raise NotImplementedError
    #     pass

    # def create_events(self, events_df):
    #     events = [
    #         Event(
    #             **row.to_dict(),
    #             theta_w = self.min_sm,
    #             # theta_star = self.max_sm,
    #             theta_star = self.theta_fc,
    #             z = self.z, 
    #             event_data = self.get_event_data(row.start_date, row.end_date)
    #         )
    #         for i, row in events_df[['start_date','end_date','soil_moisture']].iterrows()
    #     ]
    #     return events

    # def plot_event(self):
    #     raise NotImplementedError
    


class TowerSensorData(SensorData):
    _col_dict = {
        'precip' : 'P_F',
        'PET' : 'ET_F_MDS',
    }

    def __init__(self, cfg, tower, sensor_grp):
        # tower
        self._tower = tower
        super().__init__(cfg, station=tower, sensor_id=sensor_grp)
        # id
        # self.id = (self._tower.id, sensor_grp)
        # info
        # self.info = self._get_meta(sensor_grp)

        # z (depth of sensor) [m]
        self.z = float(self.info['HEIGHT']) * -1.

        self.df = self.get_data(sensor_grp)

        # soil_info
        # self.soil_info = self._tower.soil_info
        # self.soil_texture = self.soil_info.get('soil_texture')
        self.n = self.info.get('porosity', self._soil.n)
        # self.soil_texture = self._tower.soil_info.get('soil_texture')

        # data
        # self.df = self.get_data()

    def _get_meta(self, sensor_id):
        meta = self._tower.grp_info.get(sensor_id).copy()
        meta.update(self._tower.soil_info)
        return meta
        # return self._tower.grp_info.get(sensor_id).copy()

    def get_data(self, sensor_id):
        # Copy soil moisture data
        sensor_col = self._tower.grp_info.get(sensor_id).get('VARNAME')

        start = self._tower.grp_info.get(sensor_id).get('DATE') # self.info.get('DATE')
        end = self._get_end_date(sensor_id, sensor_col)         # 

        df = self._tower.data.loc[
            (self._tower.data.TIMESTAMP >= start) & (self._tower.data.TIMESTAMP < end),
            ['TIMESTAMP'] + [sensor_col]
        ].copy()
        # df = self._tower.data[['TIMESTAMP']+[sensor_col]].copy()
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

    def get_precip(self, p_col='precip',):
        if p_col not in self.df.columns:
            self.add_data_cols([p_col])
        return self.df[p_col]
    
    def get_et(self, et_col='PET',):
        if et_col not in self.df.columns:
            self.add_data_cols([et_col])
        return self.df[et_col]

    def add_data_cols(self, cols):
        col_names = [self._col_dict.get(col) for col in cols if col in self._tower.data.columns]
        # self.df = pd.concat([self.df, cols], axis=1)
        self.df = self.df.join(self._tower.data.set_index('TIMESTAMP')[col_names])
        self.df.rename(columns=col_map, inplace=True)

    # def get_event_data(self, start, end, cols=['precip', 'PET']):
    #     new_cols = [col for col in cols if col not in self.df.columns]
    #     if new_cols:
    #         self.add_data_cols(new_cols)
        
    #     return self.df.loc[start:end]

    def _get_end_date(self, grp, col):
        try:
            ind = self._tower.var_info[col]['GROUP_ID'].index(grp)
            end_date = self._tower.var_info[col]['DATE'][ind+1]
        except:
            end_date = self._tower.data.iloc[-1].TIMESTAMP + pd.Timedelta(days=1)
        return end_date




class ISMNSensorData(SensorData):
    _col_dict = {
        'precip' : 'P',
        'PET' : 'PET',
        'LAI' : 'LAI'
    }

    
    def __init__(self, cfg, station, sensor_name):
        # station
        self._station = station
        super().__init__(cfg, station=station, sensor_id=sensor_name)

        # id
        # self.id = (self._station.name, sensor_name)

        # info
        # self.info = self._get_meta(sensor_name)

        # z (depth of sensor) [m]
        self.z = float(self.info.get('depth_to'))

        # soil info
        # self.soil_texture = self.info.get('soil_texture')
        
        # data
        self.df = self.get_data(sensor_name)

        # theta_fc
        # if self.soil_texture.lower() in soils.keys():
        #     self.theta_fc = Soil(texture=self.soil_texture).theta_fc
        # else:
        #     log.info(
        #         f"Texture '{self.soil_texture}' not found in soil database. Setting theta_fc to 0.3."
        #     )

        # min, max, range
        # self.min_sm = self.df.SWC.min()
        # self.max_sm = self.df.SWC.quantile(
        #     # self.cfg['DATA'].getfloat('max_sm_frac', 1.0)
        #     self.cfg.getfloat('max_sm_frac', 1.0)
        # )
        # self.range_sm = self.max_sm - self.min_sm

        # self.max_sm = self.theta_fc

    def _get_meta(self, sensor_id):
        
        sensor_meta = self._station.var_info[
            (self._station.var_info.sensor == sensor_id)
        ].squeeze().to_dict()
        
        return sensor_meta

    # def get_data(self, sensor_col):
    def get_data(self, sensor_id):
        # Copy soil moisture data

        sensor_col = self._station.sensor_cols[sensor_id]

        start = self.info.get('timerange_from')
        end = self.info.get('timerange_to')

        # df = self._station.data.loc[
        #     (self._station.data.index >= start) & (self._station.data.index < end),
        #     [sensor_col]
        # ].copy()
        df = self._station.daily.loc[
            (self._station.daily.index >= start) & (self._station.daily.index < end),
            [sensor_col, sensor_col+'_masked', sensor_col+'_QC']
        ].copy()
        # df = self._tower.data[['TIMESTAMP']+[sensor_col]].copy()
        # df.set_index('TIMESTAMP', inplace=True, drop=False)
        df.index.name = 'DATE'
        # Rename to standard column name
        df.rename(
            # columns={sensor_col+suff : 'SWC' + suff for suff in ['', '_masked', '_QC']},
            columns={
                sensor_col: 'SWC_raw', 
                sensor_col + '_masked' : 'SWC', # temporarily just using the masked values. # TODO: Update this somehow.
                sensor_col + '_QC' : 'SWC_QC'
            },
            # {sensor_col: 'SWC', sensor_col + '_QC' : 'SWC_QC'}, 
            inplace=True
        )

        # Resample to daily
        df = df.resample('D').asfreq()
        # Convert units to m3/m3
        # df['SWC'] = df['SWC']/100
        df.insert(0,'TIMESTAMP',df.index)

        # Update info dict
        self.info.update({'VARNAME' : sensor_col})
        self.info.update({'unit': 'm3 m-3'})

        return df
    
    # def get_precip(self, p_col='precip',):
    #     if p_col not in self.df.columns:
    #         self.add_data_cols([p_col])
    #     return self.df[p_col]
    
    # def get_et(self, et_col='PET',):
    #     if et_col not in self.df.columns:
    #         self.add_data_cols([et_col])
    #     return self.df[et_col]

    # def get_anc(self, )


    def add_data_cols(self, cols):
        # Check for ancillary data
        col_names = [self._col_dict.get(col) for col in cols]
        anc = self._station.get_anc_data(col_names)
        self.df = self.df.join(anc)
        # Check for cols in original dataframe
        cols = list(set(cols) - set(anc.columns))
        cols = [col for col in cols if col in self._station.daily.columns]
        # self.df = pd.concat([self.df, cols], axis=1)
        self.df = self.df.join(self._station.daily[cols])


    # def get_event_data(self, start, end, cols=['precip', 'PET']):
    #     new_cols = [
    #         col for col in cols if col not in self.df.columns #and col in self._station.daily.columns
    #     ]
    #     if new_cols:
    #         self.add_data_cols(new_cols)
        
    #     return self.df.loc[start:end]