# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 20:39:45 2020

@author: Haochen Li
"""

import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas_datareader as dr
from pandas_datareader import data
from datetime import datetime
import cvxopt as opt
from cvxopt import blas, solvers


# set working path
os.getcwd()
os.chdir(r'C:\Temp\SDA')
os.getcwd()

def save_sp500_tickers():
   resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
   soup = bs.BeautifulSoup(resp.text, 'lxml')
   table = soup.find('table', {'class': 'wikitable sortable'})
   tickers = []
   sectors = []
   for row in table.findAll('tr')[1:]:
       ticker = row.findAll('td')[0].text
       ticker = ticker[:-1]
       if "." in ticker:
            ticker = ticker.replace('.','-')
            print('ticker replaced to', ticker) 
       
       tickers.append(ticker)  
       sector = row.findAll('td')[3].text
       if "\n" in sector:
           sector = sector.replace('\n','')
           print('sector replaced to', sector)
       sectors.append(sector)  
   with open("sp500tickers.pickle","wb") as f:
       pickle.dump(tickers,f)
       
   return tickers, sectors

tickers,sectors = save_sp500_tickers()

# save_sp500_tickers()
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = datetime(2020, 1, 20)
    end = dt.datetime.now()
    
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            #df = df.drop("Symbol", axis=1)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


get_data_from_yahoo()

# Compile data into one sheet
def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')


compile_data()


# Compute returns 
df = pd.read_csv('sp500_joined_closes.csv')
df.set_index('Date', inplace = True)

#look if there are values missing
print(df.isnull().sum())

#drop the stock if more than 5 prices are missing, otherwise replace the missing values with the values in the previous row
df=df.dropna(axis=1,thresh=5)
df=df.fillna(axis=1, method='ffill')
print(df.isnull().sum())

df
prices = df
prices


# Calculate the log returns
log_r = np.log(prices / prices.shift(1))
log_r = log_r.drop(axis = 0, index = ['2020-01-21'])
log_r 

# Compute the annualised returns
annual_r = log_r.mean() * 252
annual_r.name = 'annual return log'
annual_r
 
# Calculate the covariance matrix
cov_matrix = log_r.cov() * 252
cov_matrix

# Calculate the volatility
var = log_r.var() * 252
Std = np.sqrt(var)
Std.name = 'Std'

# Compile the dataset we need

pd_sectors = pd.Series(sectors, index = tickers)
pd_sectors.name = 'sector'

dataset = pd.concat([pd_sectors, annual_r, Std], axis = 1)
dataset

# save the dataset
dataset.to_csv('sp500_dataset.csv')


