# Import libraries
import json
import requests
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # matplotlib is not installed automatically
from datetime import datetime
import warnings


fn = r"G:\Shared drives\Ryoko and Hilary\SMSigxSMAP\analysis\1_data\CalVal\NSIDC-0712_67013601_SMAPL2SMP_T16500_20150401_20180323.txt"

# Read the header
with open(fn , 'r') as f:
    first_line = f.readline()
    second_line = f.readline()
    third_line = f.readline()

second_line = second_line.replace('\n', '')
second_line = second_line.replace(' ', '')
header = second_line.split(',')

# Create the dataframe
df_calval = pd.read_csv(fn, skiprows=2, sep=',')
df_calval = df_calval.set_axis(header, axis=1, copy=False)
df_calval.rename(columns={"Yr": "year", "Mo": "month", 'Day':'day', 'Hr':'hour', 'Min':'minute'}, inplace=True)
df_calval['Date'] = pd.to_datetime(df_calval[['year', 'month', 'day', 'hour', 'minute']])
df_calval.set_index('Date', inplace=True)

print(df_calval.head())