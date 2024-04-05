from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mylogger import getLogger
import threading
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from utils import is_true

import model as model

# Create a logger
log = getLogger(__name__)


class TowerModel(model.DrydownModel):
    def __init__(self, cfg, data, event):
        
        super().__init__(cfg, data, event)

    
    

        
        