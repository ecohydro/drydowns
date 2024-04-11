import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from .mylogger import getLogger

# Create a logger
log = getLogger(__name__)


class Event:
    def __init__(self, index, event_dict):
        # Read the data
        self.index = index
        self.start_date = event_dict["event_start"]
        self.end_date = event_dict["event_end"]
        self.soil_moisture = np.asarray(event_dict["soil_moisture_daily"])
        
        norm_sm = np.asarray(event_dict["normalized_sm"])
        self.pet = np.average(event_dict["PET"])
        self.z = 0.05

        # self.min_sm = event_dict["min_sm"]
        # self.max_sm = event_dict["max_sm"]
        self.theta_w = event_dict["min_sm"]
        self.theta_star = event_dict["max_sm"]

        # Prepare the attributes
        self.event_min = np.nanmin(self.soil_moisture)
        self.event_max = np.nanmax(self.soil_moisture)
        self.event_range = self.event_max - self.event_min

        # Prepare the inputs
        t = np.arange(0, len(self.soil_moisture), 1)
        self.t = t[~np.isnan(self.soil_moisture)]
        # self.y = self.soil_moisture[~np.isnan(self.soil_moisture)]
        # self.norm_y = norm_sm[~np.isnan(self.soil_moisture)]
        self.theta = self.soil_moisture[~np.isnan(self.soil_moisture)]
        self.theta_norm = norm_sm[~np.isnan(self.soil_moisture)]

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
                # "y_opt": y_opt.tolist(),
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
                    # "y_opt": y_opt.tolist(),
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
                        # "y_opt": y_opt.tolist(),
                        "theta_opt": y_opt.tolist(),
                        # "ET_max" : (self.z * 1000) * popt[1]
                    }
                else:
                    self.q = {
                        "delta_theta": popt[0],
                        # "theta_0" : popt[0] + self.theta_w,
                        "q": popt[1],
                        "r_squared": r_squared,
                        # "y_opt": y_opt.tolist(),
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
                "theta50": popt[0],
                "k": popt[1],
                "a": popt[2],
                "r_squared": r_squared,
                # "y_opt": y_opt.tolist(),
                "theta_opt": y_opt.tolist(),
            }
