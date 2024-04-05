# Import libraries
import json
import requests
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # matplotlib is not installed automatically
from datetime import datetime
import warnings

os.chdir(r'G:\Shared drives\Ryoko and Hilary\SMSigxSMAP')
pt_id = 0
in_path = r'.\analysis\1_data\SMAP\KENYA\GEE'

pcp = pd.read_csv(os.path.join(in_path, f'glas_pcp_pt{pt_id}.csv'))
print(pcp.head())
