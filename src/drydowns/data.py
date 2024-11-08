import os
import warnings
import threading
import logging

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from .event import Event
from .mylogger import getLogger
from .soil import Soil, soils
from .utils_figs import plot_drydown

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
        self._depth = np.nan

        # info
        self.info = self._get_meta(**kwargs)

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

    def _get_meta(self):
        # raise NotImplementedError
        return None

    @property
    def _soil(self):
        if not hasattr(self, '__soil'):
            self.__soil = self._init_soil()
        return self.__soil
    
    def _init_soil(self):
        if self.soil_texture.lower() in soils.keys():
            return Soil(texture=self.soil_texture)
        else:
            log.info(
                f"Texture '{self.soil_texture}' not found in soil database. "
            )
            return None

    # def _set_soil_properties(self):
    #     if self.soil_texture.lower() in soils.keys():
    #         self._soil = Soil(texture=self.soil_texture)
    #         self.theta_fc = self._soil.theta_fc
    #         self.n = self._soil.n
    #     else:
    #         # Eventually, need to go get soil texture...
    #         log.info(
    #             f"Texture '{self.soil_texture}' not found in soil database. "
    #         )
    #         self._soil = None
    #         self.theta_fc = self.max_sm
    #         self.n = np.nan
    
    @property
    def theta_fc(self):
        if not hasattr(self, '_theta_fc'):
            try:
                self._theta_fc = self._soil.theta_fc
            except:
                log.info(
                    f"Cannot set field capacity from soil object. Setting field"
                    " capacity to maximum soil moisture."
                )
                self._theta_fc = self.max_sm
        return self._theta_fc
    
    @theta_fc.setter
    def theta_fc(self, value):
        # if not hasattr(self, '_theta_fc'):
        #     try:
        #         self._theta_fc = self._soil.theta_fc
        #     except:
        #         self._theta_fc = self.max_sm
        # else:
        self._theta_fc = value

    @property
    def n(self):
        if not hasattr(self, '_n'):
            try:
                self._n = self._soil.n
            except:
                log.info(
                    f"Cannot set porosity from soil object. Setting porosity to nan."
                )
                self._n = np.nan
        return self._n

    @n.setter
    def n(self, value):
        # if not hasattr(self, '_n'):
        #     try:
        #         self._n = self._soil.n
        #     except:
        #         self._n = np.nan
        # else:
        self._n = value


    def get_data(self, **kwargs):
        return None
        # raise NotImplementedError
    
    def calc_diff(self):
        self.df['SWC_diff'] = self.df['SWC'].diff() #/ df['TIMESTAMP'].diff().dt.days


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
            'min_diff' : self.cfg.getfloat("min_diff", 0.5),
        }
        return params
    
    def separate_events(self):
        raise NotImplementedError

    def find_events(self):
        raise NotImplementedError

    def find_starts(self):
        raise NotImplementedError

    def find_ends(self):
        raise NotImplementedError


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

        # self.get_end_dsdt(events, diff_col='ds_dt')

        return events

    # def get_end_dsdt(self, events_df, diff_col='ds_dt'):
    #      end_dsdt = self.df.loc[events_df.end_date, diff_col].reset_index()
    #     #  events_df['end_dsdt'] = end_dsdt[diff_col]
    #      events_df['end_dsdt_mm'] = end_dsdt[diff_col] * 1000 * self.z
    
    
    def filter_events(self):
        # raise NotImplementedError
        pass

    def create_events(self, events_df):
        events = [
            Event(
                **row.to_dict(),
                theta_w = self.min_sm,
                theta_star = self._get_theta_star(),
                z = self.z,
                depth = self._depth,
                event_data = self.get_event_data(row.start_date, row.end_date)
            ) 
            for i, row in events_df[['start_date','end_date','soil_moisture']].iterrows()
        ]
        return events

    def get_event_data(self, start, end, cols=['precip', 'PET', 'LAI', 'GPP'], buffer=0):
        new_cols = [col for col in cols if col not in self.df.columns]
        if new_cols:
            self.add_data_cols(new_cols)
        
        buffer = pd.to_timedelta(buffer, 'D')

        # return self.df.loc[start:end]
        return self.df.loc[start-buffer:end+buffer]


    def _get_theta_star(self):
        if self.cfg.get('theta_star').lower() in ['theta_fc', 'fc', 'field capacity']:
            theta_star = self.theta_fc
        elif self.cfg.get('theta_star').lower() in ['max_sm', 'maximum', 'max']:
            theta_star = self.max_sm
        else:
            log.info(
                f"Unknown theta_star value: {self.cfg.get('theta_star')}."
                "Setting to 95th percentile value."
            )
            theta_star = self.max_sm
        return theta_star
   

    def plot_drydowns(self, kwargs={}):
        df = pd.concat([
            self.get_plot_data(event, buffer=0) for event in self.events
        ]).drop_duplicates()
        hue = 'Event'
        axs = plot_drydown(df=df, hue=hue, **kwargs)
        return axs


    def plot_drydown(self, event=None, buffer=5, show_bounds=True, kwargs={}):
        df = self.get_plot_data(event, buffer=buffer)
        hue = None

        axs = plot_drydown(df=df, hue=hue, **kwargs)

        axs = plot_drydown(axs=axs, df=self.get_plot_data(event, buffer=0), **kwargs)
        # ylim0 = axs[0].get_ylim()
        # ylim1 = axs[1].get_ylim()
        # axs[0].vlines(
        #     [0,len(df)], *ylim0, 
        #     color='k', linestyle='--', linewidth=0.8
        # )
        # axs[1].vlines(
        #     [df.theta.loc[event.start_date], df.theta.loc[event.end_date]], 
        #     *ylim1, color='k', linestyle='--', linewidth=0.8
        # )
        # axs[0].set_ylim(ylim0)
        # axs[1].set_ylim(ylim1)

        return axs


    def get_plot_data(self, event, buffer=5):
        data = self.df.loc[
            event.start_date - pd.Timedelta(days=buffer):event.end_date + pd.Timedelta(days=buffer)
        ]
        data['theta'] = data['SWC'].values
        data['dtheta'] = -data.theta.diff()
        data['dtheta_mm'] = data.dtheta * event.z * 1000
        data['t'] = (data.index - event.start_date).days
        return data