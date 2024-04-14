import os
import warnings
import threading
import logging

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


from .mylogger import getLogger

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


class Data:

    def __init__(self, cfg, **kwargs):
        # _______________________________________________________________________________
        # Attributes

        # cfg
        self.cfg = cfg

        # id
        # self.id = None

        # z (depth of soil column) [m]
        self.z = 0.05

        # df
        self.df = self.get_data(**kwargs)
        
        # min, max, range
        self.min_sm = self.df.SWC.min(skipna=True)
        self.max_sm = self.df.SWC.quantile(
            self.cfg.getfloat('max_sm_frac', 1.0)
        )
        self.range_sm = self.max_sm - self.min_sm

        # EventSeparator
        self._params = self.get_params()

        current_thread = threading.current_thread()
        current_thread.name = ( f"{self.id[0]}, {self.id[1]}" )
        self.thread_name = current_thread.name
        # # Not working at the moment ...
        # custom_handler = ThreadNameHandler()
        # log = modifyLogger(name=__name__, custom_handler=custom_handler)

    def get_data(self, **kwargs):
        return None
        # raise NotImplementedError
    
    def normalize(self):
        self.df['norm_sm'] = (self.df['SWC'] - self.min_sm) / self.range_sm


#-------------------------------------------------------------------------------
# EventSeparator
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
            'target_rmsd' : self.cfg.getfloat("target_rmsd"),
            'duration' : self.cfg.getint("min_duration"),
        }
        return params
    
    def label_events(self, event_dates):
        self.df['Event'] = np.nan
        for i, row in event_dates.iterrows():
            self.df.loc[row['start_date']:row['end_date'], 'Event'] = int(i)

    def extract_events(self, event_dates, col='SWC'):
        # Label events
        if 'Event' not in self.df.columns:
            self.label_events(event_dates)
        # Get data
        events = event_dates.join(
            self.df[self.df.Event.notnull()].groupby('Event')[col].apply(list)
        ).reset_index(drop=True)
        events.rename(columns={col: 'soil_moisture'}, inplace=True)

        return events