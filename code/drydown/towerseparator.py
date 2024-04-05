import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from Event import Event
from towerevent import TowerEvent
from EventSeparator import EventSeparator

import warnings
import threading
from MyLogger import getLogger, modifyLogger
import logging

# Create a logger
log = getLogger(__name__)


class ThreadNameHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            # Add thread name to the log message
            record.threadName = threading.current_thread().name
            super(ThreadNameHandler, self).emit(record)
        except Exception:
            self.handleError(record)



# TODO: Move these functions into Data class (I don't think there's really a need
# for this class??)
class TowerEventSeparator(EventSeparator):

    def __init__(self, cfg, data):
    
        self.cfg = cfg
    
        self.verbose = cfg["MODEL"]["verbose"].lower() in ["true", "yes", "1"]
    
        self.data = data
    
        self._params = self.get_params()

        # self.events = self.separate_events()


        current_thread = threading.current_thread()
        current_thread.name = (f"{self.data.id[0]}, {self.data.id[1]}")
        self.thread_name = current_thread.name

    def get_params(self):
        _noise_thresh = (self.data.max_sm - self.data.min_sm) * self.cfg.getfloat(
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

    def separate_events(self):

        self.mask_values(self.data.df, 'SWC', self.data.max_sm)
        self.calc_dsdt(self.data.df, 'SWC_masked')

        events = self.find_events()
        if events.empty:
            self.events_df = None
            self.events = None
            return None
            # return None
        self.events_df = self.extract_events(events)
        
        # events[['min_sm', 'max_sm']] = self.data.df.groupby('Event')['SWC'].agg(['min', 'max'])
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
        mask_diff = self.data.df[diff_col].shift(-1) < -threshold
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
        start_dates = self.data.df.groupby(valid).TIMESTAMP.min().reset_index(drop=True)
        return start_dates

    def find_ends(self, start_dates, diff_col, threshold=0.):
        end_dates = start_dates.apply(self.find_event_end, args=(diff_col, threshold,))
        return end_dates

    def find_event_end(self, start_date, diff_col, threshold=0.0):
        end = (self.data.df[start_date:][diff_col].shift(-1) > threshold).idxmax()
        return end


    def label_events(self, event_dates):
        self.data.df['Event'] = np.nan
        for i, row in event_dates.iterrows():
            self.data.df.loc[row['start_date']:row['end_date'], 'Event'] = int(i)

    def extract_events(self, event_dates, col='SWC'):
        if 'Event' not in self.data.df.columns:
            self.label_events(event_dates)
        events = event_dates.join(
            self.data.df[self.data.df.Event.notnull()].groupby('Event')[col].apply(list)
        ).reset_index(drop=True)
        events.rename(columns={col: 'soil_moisture'}, inplace=True)

        return events


    def filter_events(self):
        # raise NotImplementedError
        pass

    def create_events(self, events_df):
        events = [
            TowerEvent(
                **row.to_dict(),
                theta_w = self.data.min_sm,
                theta_star = self.data.max_sm,
            )
            for i, row in events_df[['start_date','end_date','soil_moisture']].iterrows()
        ]
        return events

    def plot_event(self):
        raise NotImplementedError
