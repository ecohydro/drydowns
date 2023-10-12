import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class Event:
    def __init__(self, event_dict):
        # Read the data
        self.index = event_dict.index
        self.start_date = event_dict["event_start"]
        self.end_date = event_dict["event_end"]
        soil_moisture_subset = np.asarray(event_dict["soil_moisture_daily"])
        norm_soil_moisture_subset = np.asarray(event_dict["normalized_S"])
        self.pet = np.average(event_dict["PET"])

        # Prepare the attributes
        self.subset_sm_range = np.nanmax(soil_moisture_subset) - np.nanmin(
            soil_moisture_subset
        )
        self.subset_min_sm = np.nanmin(soil_moisture_subset)

        # Prepare the inputs
        t = np.arange(0, len(soil_moisture_subset), 1)
        self.x = t[~np.isnan(soil_moisture_subset)]
        self.y = soil_moisture_subset[~np.isnan(soil_moisture_subset)]
        self.norm_y = norm_soil_moisture_subset[~np.isnan(soil_moisture_subset)]

    def add_attributes(self, model_type, popt, r_squared, y_opt):
        if model_type == "exponential":
            self.exponential = {
                "delta_theta": popt[0],
                "theta_w": popt[1],
                "tau": popt[2],
                "r_squared": r_squared,
                "y_opt": y_opt.tolist(),
            }

        if model_type == "q":
            self.q = {
                "k": popt[0],
                "q": popt[1],
                "delta_theta": popt[2],
                "r_squared": r_squared,
                "y_opt": y_opt.tolist(),
            }
