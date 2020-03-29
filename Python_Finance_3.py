#------------------3/28/2020----------------------------------
import numpy as np
import pandas as pd
import pickle 

def process_data_for_labels(ticker):
    hm_days = 7 
    df = pd.read_csv('sp500_joined_closes.csv',index_col=0)

