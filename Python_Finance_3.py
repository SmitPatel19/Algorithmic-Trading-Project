#------------------3/28/2020----------------------------------
from collections import Counter
import numpy as np
import pandas as pd
import pickle 

def process_data_for_labels(ticker):
    hm_days = 7 #   <-----------JUST CHANGE THIS SINGLE VALUE TO LOOK THAT MANY DAYS INTO THE FUTURE
    df = pd.read_csv('sp500_joined_closes.csv',index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i)- df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers,df

#process_data_for_labels('XOM')

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1 #BUY
        if col < -requirement:
            return -1 #SELL
    return 0 #HOLD

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)],
                                               ))
    
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))
    
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    Y = df['{}_target'.format(ticker)].values

    return X, Y, df

extract_featuresets('XOM')

#----------------------3/29/2020-------------------------------------------------

