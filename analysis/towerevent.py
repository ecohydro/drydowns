import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from MyLogger import getLogger
from Event import Event

# Create a logger
log = getLogger(__name__)


class TowerEvent(Event):
    def __init__(
        self, #event_dict
        start_date, end_date, soil_moisture, #norm_sm, min_sm, max_sm
        theta_w, theta_star,
        event_data=None
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.soil_moisture = np.asarray(soil_moisture)

        self.event_data = event_data

        self.pet = self.calc_pet() #np.nan

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
        pet = self.event_data[et_col].max() 
        # NOTE: Should be initial value (well, really should be calculated), 
        # but using this for now bc PET > AET, so if max value isn't initial, this
        # ensures highest AET value.
        return pet


    def add_attributes(
        self, model_type="", popt=[], r_squared=np.nan, y_opt=[], force_PET=False
    ):
        if model_type == "exponential":
            self.exponential = {
                "theta_0" : popt[0] + popt[1],
                # "delta_theta": popt[0],
                "theta_w": popt[1],
                "tau": popt[2],
                "r_squared": r_squared,
                "theta_opt": y_opt.tolist(),
            }

        if model_type == "q":
            if not force_PET:
                self.q = {
                    "k": popt[0],
                    "q": popt[1],
                    # "delta_theta": popt[2],
                    "theta_0" : popt[2] + self.theta_w,
                    "r_squared": r_squared,
                    "theta_opt": y_opt.tolist(),
                }
            else:
                self.q = {
                    "q": popt[0],
                    # "delta_theta": popt[1],
                    "theta_0" : popt[1] + self.theta_w,
                    "r_squared": r_squared,
                    "theta_opt": y_opt.tolist(),
                }

        if model_type == "sigmoid":
            self.sigmoid = {
                "theta_50": popt[0],
                "k": popt[1],
                "a": popt[2],
                "r_squared": r_squared,
                "theta_opt": y_opt.tolist(),
            }
