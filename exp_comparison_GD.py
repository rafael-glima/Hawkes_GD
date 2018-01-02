#!/usr/bin/env python 

import pandas as pd
import numpy as np
import urllib2
import datetime as dt
import matplotlib.pyplot as plt
import scipy.io
import math
import random as rd
import time
import csv
import re
import requests


llh_GD_EXP = []
llh_GD_PWL = []
llh_GD_QEXP = []

eps = np.finfo(float).eps;

# Remove 'DELL' and 'EMC'

stocks = ['CSCO','GOOGL','HPQ','INTC','IBM','MSFT','ORCL','TXN','XRX','AAPL'];

# def get_google_data(symbol, period, window):
#     url_root = 'http://www.google.com/finance/getprices?i='
#     url_root += str(period) + '&p=' + str(window)
#     url_root += 'd&f=d,o,h,l,c,v&df=cpct&q=' + symbol
#     response =  urllib2.urlopen(url_root)
#     data = response.read().split('\n')
#     #actual data starts at index = 7
#     #first line contains full timestamp,
#     #every other line is offset of period from timestamp
#     parsed_data = []
#     anchor_stamp = ''
#     end = len(data)
#     for i in range(7, end):
#         cdata = data[i].split(',')
#         if 'a' in cdata[0]:
#             #first one record anchor timestamp
#             anchor_stamp = cdata[0].replace('a', '')
#             cts = int(anchor_stamp)
#         else:
#             try:
#                 coffset = int(cdata[0])
#                 cts = int(anchor_stamp) + (coffset * period)
#                 parsed_data.append((dt.datetime.fromtimestamp(float(cts)), float(cdata[1]), float(cdata[2]), float(cdata[3]), float(cdata[4]), float(cdata[5])))
#             except:
#                 pass # for time zone offsets thrown into data
#     df = pd.DataFrame(parsed_data)
#     print(type(df))
#     df.columns = ['ts', 'o', 'h', 'l', 'c', 'v']
#     df.index = df.ts
#     del df['ts']
#     #print df
#     return df

def get_google_data(ticker, period=60, days=1):
    """
    Retrieve intraday stock data from Google Finance.
    Parameters
    ----------
    ticker : str
        Company ticker symbol.
    period : int
        Interval between stock values in seconds.
    days : int
        Number of days of data to retrieve.
    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the opening price, high price, low price,
        closing price, and volume. The index contains the times associated with
        the retrieved price values.
    """

    uri = 'http://www.google.com/finance/getprices' \
          '?i={period}&p={days}d&f=d,o,h,l,c,v&df=cpct&q={ticker}'.format(ticker=ticker,
                                                                          period=period,
                                                                          days=days)
    page = requests.get(uri)
    reader = csv.reader(page.content.splitlines())
    #reader = csv.reader(codecs.iterdecode(page.content.splitlines(), "utf-8"))
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    rows = []
    times = []
    for row in reader:
        if re.match('^[a\d]', row[0]):
            if row[0].startswith('a'):
                start = datetime.datetime.fromtimestamp(int(row[0][1:]))
                times.append(start)
            else:
                times.append(start+datetime.timedelta(seconds=period*int(row[0])))
            rows.append(map(float, row[1:]))
    if len(rows):

        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'),
                            columns=columns)
    else:

        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'))

hour_begin = 5

hour_end = 23

hour_diff = hour_end - hour_begin

for i in range(0,len(stocks)):

    print(stocks[i])

    df = get_google_data(stocks[i],120,30)

    print ('testing df: ' + repr(df))

    df['Price'] = df.o

    df['Time'] = [dt.datetime.strftime(x, '%Y-%m-%d %H:%M:%S') for x in df.index]

    df['Date'] = [x[:10] for x in df.Time]

    df['Time'] = [x[11:19] for x in df.Time]

    df['Hour'] = df.Time.str[:2]

    print('df.Hour: ' + repr(df.Hour))

    df.Hour = df.Hour.astype(np.int64)

    print('Type conversion of df.Hour: ' + repr(df.Hour))

    # df = df.loc[(df["Hour"] > hour_begin - 1 )&(df["Hour"] < hour_end + 1)].values

    # #df = df[(df.Hour > hour_begin - 1 )&(df.Hour < hour_end + 1)]

    # print('after time checking: ' + repr(df['Hour'] > hour_begin - 1))

    #df.Time = df.Time.str[:8]

    FMT = '%H:%M:%S'

    #df['Time'].map(DT.datetime.strptime(x,FMT))

    #df['Time'].apply(lambda x: DT.datetime.strptime(x,FMT))

    def to_seconds(s):
        hr, min, sec = [float(x) for x in s.split(':')]
        return hr*3600. + min*60. + sec - hour_begin*3600.

    df['Time'] = [to_seconds(x) for x in df.Time]

    df['Jump'] = df.Price.diff(1)

    df['Jump'][0] = 0;

    print(df.Jump)

    minval = df['Price'].min()

    df = df[abs(df['Jump']) >= 0.002]#*minval]#*minval]

    #df.Time = np.cumsum(df.Time)

    minTime = df.ix[0,'Time'] #df['Time'].iloc[0]

    df.Time = df.Time - minTime #df.Time.iloc[0]

    #df['Time'] = [x - minTime for x in df.Time]

    #df.Jump[-1]

    # df['Day'] = df['Date'].str[5:7]

    # df['Month'] = df['Date'].str[8:10]

    # df['Year'] = df['Date'].str[0:4]

    # df.Day = df.Day.astype(np.float64)

    # df.Month = df.Month.astype(np.float64)

    # df.Year = df.Year.astype(np.float64)

    # df.Month = df.Month - df.Month.iloc[0]

    # df.Year = df.Year - df.Year.iloc[0]

    # df.Day = df.Day - df.Day.iloc[0] + 30*(df.Month - df.Month.iloc[0]) + 365*

    FMTD = '%Y-%m-%d'

    df['Days'] = [dt.datetime.strptime(x,FMTD) - dt.datetime.strptime(df.Date.iloc[0],FMTD) for x in df.Date]

    df.Days = [x.days for x in df.Days]

    df.Days = df.Days.astype(np.float64)

    df.Time = df.Time + df.Days*(hour_diff*3600)

    #plt.plot(df.Time,df.Jump,'-')

    df.Jump = 1

    # df.Jump = np.cumsum(df.Jump)

    df['Intervals'] = df.Time.diff(1)

    #df.set_value(0,'Intervals',0)

    df.Intervals.iloc[0] = 0

    # Getting the scale-independent parameter from histogram

    hist, bin_edges = np.histogram(df.Intervals[1:],bins=100)

    cumhist = np.cumsum(hist)

    indtau = np.where(cumhist > 0.95*np.sum(hist))

    indtau = indtau[0]

    if len(indtau) < 2:
        taumax = bin_edges[indtau[0]]
    else:
        taumax = bin_edges[indtau[0]+1]

    delta = taumax/20

    N = int(taumax/delta);

    seq = df.Time

    seq = seq.as_matrix()

    seq = seq[:-1]

    T = seq[-1] - seq[0]

    Delta = len(seq)/T

    Param_GD_EXP = trainGD_EXP(seq)

    Delta_GD_EXP = Param_GD_EXP['EXP_coeffs']

    Param_GD_PWL = trainGD_PWL(seq)

    Delta_GD_PWL = Param_GD_PWL['PWL_coeffs']

    Param_GD_QEXP = trainGD_QEXP(seq)

    Delta_QEXP = Param_GD_QEXP['QEXP_coeffs']

    print('EXP llh: ' + repr(Param_GD_EXP['final_llh']))

    print('PWL llh: ' + repr(Param_GD_PWL['final_llh']))

    print('QEXP llh: ' + repr(Param_GD_QEXP['final_llh']))

    llh_GD_EXP = np.append(llh_GD_EXP, Param_GD_EXP['final_llh'])

    llh_GD_PWL = np.append(llh_GD_PWL, Param_GD_PWL['final_llh'])

    llh_GD_QEXP = np.append(llh_GD_QEXP, Param_GD_QEXP['final_llh'])

plt.plot(range(len(llh_GD_EXP)), llh_GD_EXP, 'r:', linewidth=7.0, label='Loglikelihood_EXP')

plt.plot(range(len(llh_GD_PWL)), llh_GD_PWL, 'g:', linewidth=7.0, label='Loglikelihood_PWL')

plt.plot(range(len(llh_GD_QEXP)), llh_GD_QEXP, 'b:', linewidth=7.0, label='Loglikelihood_QEXP')

plt.legend(loc='lower right')

plt.savefig('Comparison_Stock_GD.png')

plt.close()
