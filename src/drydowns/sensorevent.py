import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from .mylogger import getLogger

"""

Name:           sensorevent.py
Compatibility:  Python 3.7.0
Description:    Description of what program does

URL:            https://

Requires:       list of libraries required

Dev ToDo:       None

AUTHOR:         Ryoko Araki (initial dev); Bryn Morgan (refactor)
ORGANIZATION:   University of California, Santa Barbara
Contact:        raraki@ucsb.edu
Copyright:      (c) Ryoko Araki & Bryn Morgan 2024


"""

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
