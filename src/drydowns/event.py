import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from .mylogger import getLogger
# from .event import Event

# Create a logger
log = getLogger(__name__)


class Event:
    def __init__(
        self, #event_dict
        start_date, end_date, soil_moisture, #norm_sm, min_sm, max_sm
        theta_w, theta_star, z,
        event_data=None,
        depth=np.nan,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.soil_moisture = np.asarray(soil_moisture)

        self.event_data = event_data

        self.pet = self.calc_pet() #np.nan
        self.z = z
        self._depth = depth
        # Model params        
        self.theta_star = theta_star
        self.theta_w = theta_w

        norm_sm = self.normalize()

        self.event_min = np.nanmin(self.soil_moisture)
        self.event_max = np.nanmax(self.soil_moisture)
        self.event_range = self.event_max - self.event_min
        # # Read the data
        # self.start_date = event_dict["start"]
        # self.end_date = event_dict["end"]
        # self.soil_moisture = np.asarray(event_dict["SWC"])
        # # norm_sm = np.asarray(event_dict["norm_sm"])
        # # self.pet = np.average(event_dict["PET"])
        # self.min_sm = event_dict["min_sm"]
        # self.max_sm = event_dict["max_sm"]

        # Prepare the inputs
        t = np.arange(0, len(self.soil_moisture), 1)
        self.t = t[~np.isnan(self.soil_moisture)]
        self.theta = self.soil_moisture[~np.isnan(self.soil_moisture)]
        self.theta_norm = norm_sm[~np.isnan(self.soil_moisture)]
        
        # self._ancillary = None

    def describe(self):
        return {
            'z_m' : self._depth,
            'dz' : self.z,
            'event_start': self.start_date,
            'event_end': self.end_date,
            'duration': (self.end_date - self.start_date).days,
            'time': self.t,
            'theta': self.theta,
            # 'event_min': self.event_min,
            # 'event_max': self.event_max,
            # 'event_range': self.event_range,
            # 'pet': self.pet,
        }

    def normalize(self):
        norm_sm = (self.soil_moisture - self.theta_w) / (self.theta_star - self.theta_w)
        return norm_sm

    def calc_precip(self, p_col='precip'):
        precip = self.event_data[p_col].sum()
        return precip
    
    def calc_gpp(self, gpp_col='GPP'):
        if gpp_col in self.event_data.columns:
            gpp = self.event_data[gpp_col].sum()
        else:
            gpp = np.nan
        return gpp
    
    def calc_pet(self, et_col='PET'):
        # TODO: Check for ET col + set default if DNE
        if self.event_data.empty or et_col not in self.event_data.columns or self.event_data[et_col].isnull().all():
            # pet = np.nan
            pet = 5.0
        else:
            pet = self.event_data[et_col].max() 
        # NOTE: Should be initial value (well, really should be calculated), 
        # but using this for now bc PET > AET, so if max value isn't initial, this
        # ensures highest AET value.
        return pet
    
    def get_et(self, et_col='PET'):
        et = self.event_data[et_col].to_numpy()
        return et

    @property
    def exponential(self):
        return self._exponential
    
    @exponential.setter
    def exponential(self, value):
        self._exponential = value

    @property
    def q(self):
        return self._q
    
    @q.setter
    def q(self, value):
        self._q = value
    
    @property
    def sigmoid(self):
        return self._sigmoid
    
    @sigmoid.setter
    def sigmoid(self, value):
        self._sigmoid = value

    # def add_results(self, model):
    def add_results(self, mod_type, results):
        setattr(self, mod_type, results)
    

    # def add_attributes(
    #     self, model_type="", popt=[], r_squared=np.nan, y_opt=[], 
    #     force_PET=False, fit_theta_star=False
    # ):
    #     if model_type == "exponential":
    #         self.exponential = {
    #             "delta_theta": popt[0],
    #             # "theta_0": popt[0] + popt[1],
    #             "theta_w": popt[1],
    #             "tau": popt[2],
    #             "r_squared": r_squared,
    #             "theta_opt": y_opt.tolist(),
    #             # "k" : (self.theta_star - popt[1]) / popt[2],
    #             # "ET_max" : (self.z * 1000) * ((self.theta_star - popt[1]) / popt[2])
    #         }
    #         self.exponential.update({
    #                 "theta_0": self.exponential['delta_theta'] + self.exponential['theta_w'],
    #                 "k": (self.theta_star - self.exponential['theta_w']) / self.exponential['tau'],
    #                 "ET_max": (self.z * 1000) * (self.theta_star - self.exponential['theta_w']) / self.exponential['tau'],
    #         })

    #     if model_type == "q":
    #         if fit_theta_star:
    #             self.q = {
    #                 "delta_theta" : popt[0],
    #                 # "theta_0" : popt[0] + self.theta_w,
    #                 "k": popt[1],
    #                 "q": popt[2],
    #                 "theta_star": popt[3],
    #                 # "delta_theta": popt[2],
    #                 # "theta_0" : popt[2] + self.theta_w,
    #                 "r_squared": r_squared,
    #                 "theta_opt": y_opt.tolist(),
    #                 # "ET_max" : (self.z * 1000) * popt[1]
    #             }
    #         else:
    #             if not force_PET:
    #                 self.q = {
    #                     "delta_theta" : popt[0],
    #                     # "theta_0" : popt[0] + self.theta_w,
    #                     # Multiplying by (theta_star - theta_w) denormalizes k
    #                     "k": popt[1] * (self.theta_star - self.theta_w), #popt[0],
    #                     "q": popt[2], #popt[1],
    #                     # "delta_theta": popt[2],
    #                     # "theta_0" : popt[2] + self.theta_w,
    #                     "r_squared": r_squared,
    #                     "theta_opt": y_opt.tolist(),
    #                     # "ET_max" : (self.z * 1000) * popt[1]
    #                 }
    #             else:
    #                 self.q = {
    #                     "delta_theta": popt[0],
    #                     # "theta_0" : popt[0] + self.theta_w,
    #                     "q": popt[1],
    #                     "r_squared": r_squared,
    #                     "theta_opt": y_opt.tolist(),
    #                 }
    #         self.q.update({
    #             "theta_0": self.q['delta_theta'] + self.theta_w
    #         })
    #         if 'k' in self.q:
    #             self.q.update({
    #                 "ET_max": (self.z * 1000) * self.q['k']
    #             })

    #     if model_type == "sigmoid":
    #         self.sigmoid = {
    #             "theta_50": popt[0],
    #             "k": popt[1],
    #             "a": popt[2],
    #             "r_squared": r_squared,
    #             "theta_opt": y_opt.tolist(),
    #         }
    
    @property
    def ancillary(self):
        if not hasattr(self, '_ancillary') or self._ancillary is None:
            # self._ancillary = self.get_anc_data()
            self.get_anc_data()
        return self._ancillary
    
    # @ancillary.setter
    # def ancillary(self, value):
    #     if self._ancillary is None:
    #         self._ancillary = self.get_anc_data()
            

    def get_anc_data(self):

        cols = [
            'precip_total60d', 'precip_total_month',
            'pet_total60d', 'pet_total_month',
            'LAI', 
        ]

        anc = {
            col : self.event_data.iloc[0][col] for col in cols if col in self.event_data.columns
        }
        try: 
            anc.update({
                'AI_month' : anc['precip_total_month'] / anc['pet_total_month'],
                'AI_60d' : anc['precip_total60d'] / anc['pet_total60d'],
            })
        except KeyError:
            pass

        setattr(self, '_ancillary', anc)