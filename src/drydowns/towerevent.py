import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from .mylogger import getLogger
from .event import Event

# Create a logger
log = getLogger(__name__)


class TowerEvent(Event):
    def __init__(
        self, #event_dict
        start_date, end_date, soil_moisture, #norm_sm, min_sm, max_sm
        theta_w, theta_star, z,
        event_data=None
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.soil_moisture = np.asarray(soil_moisture)

        self.event_data = event_data

        self.pet = self.calc_pet() #np.nan
        self.z = z
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


    def normalize(self):
        norm_sm = (self.soil_moisture - self.theta_w) / (self.theta_star - self.theta_w)
        return norm_sm

    def calc_precip(self, p_col='P_F'):
        precip = self.event_data[p_col].sum()
        return precip
    
    def calc_pet(self, et_col='ET_F_MDS'):
        # TODO: Check for ET col + set default if DNE
        if self.event_data.empty or et_col not in self.event_data.columns:
            # pet = np.nan
            pet = 5.0
        else:
            pet = self.event_data[et_col].max() 
        # NOTE: Should be initial value (well, really should be calculated), 
        # but using this for now bc PET > AET, so if max value isn't initial, this
        # ensures highest AET value.
        return pet
    
    def get_et(self, et_col='ET_F_MDS'):
        et = self.event_data[et_col].to_numpy()
        return et


    def add_attributes(
        self, model_type="", popt=[], r_squared=np.nan, y_opt=[], force_PET=False, fit_theta_star=False
    ):
        if model_type == "exponential":
            self.exponential = {
                "delta_theta": popt[0],
                # "theta_0": popt[0] + popt[1],
                "theta_w": popt[1],
                "tau": popt[2],
                "r_squared": r_squared,
                "theta_opt": y_opt.tolist(),
                # "k" : (self.theta_star - popt[1]) / popt[2],
                # "ET_max" : (self.z * 1000) * ((self.theta_star - popt[1]) / popt[2])
            }
            self.exponential.update({
                    "theta_0": self.exponential['delta_theta'] + self.exponential['theta_w'],
                    "k": (self.theta_star - self.exponential['theta_w']) / self.exponential['tau'],
                    "ET_max": (self.z * 1000) * (self.theta_star - self.exponential['theta_w']) / self.exponential['tau'],
            })

        if model_type == "q":
            if fit_theta_star:
                self.q = {
                    "delta_theta" : popt[0],
                    # "theta_0" : popt[0] + self.theta_w,
                    "k": popt[1],
                    "q": popt[2],
                    "theta_star": popt[3],
                    # "delta_theta": popt[2],
                    # "theta_0" : popt[2] + self.theta_w,
                    "r_squared": r_squared,
                    "theta_opt": y_opt.tolist(),
                    # "ET_max" : (self.z * 1000) * popt[1]
                }
            else:
                if not force_PET:
                    self.q = {
                        "delta_theta" : popt[0],
                        # "theta_0" : popt[0] + self.theta_w,
                        "k": popt[1], #popt[0],
                        "q": popt[2], #popt[1],
                        # "delta_theta": popt[2],
                        # "theta_0" : popt[2] + self.theta_w,
                        "r_squared": r_squared,
                        "theta_opt": y_opt.tolist(),
                        # "ET_max" : (self.z * 1000) * popt[1]
                    }
                else:
                    self.q = {
                        "delta_theta": popt[0],
                        # "theta_0" : popt[0] + self.theta_w,
                        "q": popt[1],
                        "r_squared": r_squared,
                        "theta_opt": y_opt.tolist(),
                    }
            self.q.update({
                "theta_0": self.q['delta_theta'] + self.theta_w
            })
            if 'k' in self.q:
                self.q.update({
                    "ET_max": (self.z * 1000) * self.q['k']
                })

        if model_type == "sigmoid":
            self.sigmoid = {
                "theta_50": popt[0],
                "k": popt[1],
                "a": popt[2],
                "r_squared": r_squared,
                "theta_opt": y_opt.tolist(),
            }
