import pandas as pd
import numpy as np
import urllib2
import datetime as dt
import matplotlib.pyplot as plt

import scipy.io
import math
import random as rd

import time

from K1_Est import K1_Est
from K1_Class import K1_Class
from EXP_Class import EXP_Class
from PWL_Class import PWL_Class
from SQR_Class import SQR_Class
from SNS_Class import SNS_Class
import config.cfg as cfg 


frequency_K1 = np.zeros(4,)
frequency_K2 = np.zeros((4,8))

llh_K1 = []
llh_K2 = []

eps = np.finfo(float).eps;

# Remove 'DELL' and 'EMC' 

stocks = ['CSCO','GOOGL','HPQ','INTC','IBM','MSFT','ORCL','TXN','XRX','AAPL'];

def get_google_data(symbol, period, window):
    url_root = 'http://www.google.com/finance/getprices?i='
    url_root += str(period) + '&p=' + str(window)
    url_root += 'd&f=d,o,h,l,c,v&df=cpct&q=' + symbol
    response = urllib2.urlopen(url_root)
    data = response.read().split('\n')
    #actual data starts at index = 7
    #first line contains full timestamp,
    #every other line is offset of period from timestamp
    parsed_data = []
    anchor_stamp = ''
    end = len(data)
    for i in range(7, end):
        cdata = data[i].split(',')
        if 'a' in cdata[0]:
            #first one record anchor timestamp
            anchor_stamp = cdata[0].replace('a', '')
            cts = int(anchor_stamp)
        else:
            try:
                coffset = int(cdata[0])
                cts = int(anchor_stamp) + (coffset * period)
                parsed_data.append((dt.datetime.fromtimestamp(float(cts)), float(cdata[1]), float(cdata[2]), float(cdata[3]), float(cdata[4]), float(cdata[5])))
            except:
                pass # for time zone offsets thrown into data
    df = pd.DataFrame(parsed_data)
    print(type(df))
    df.columns = ['ts', 'o', 'h', 'l', 'c', 'v']
    df.index = df.ts
    del df['ts']
    #print df
    return df

hour_begin = 5

hour_end = 23

hour_diff = hour_end - hour_begin

for i in range(0,len(stocks)):

    print stocks[i]

    df = get_google_data(stocks[i],120,15)

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

    print df.Jump

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

    x = np.linspace(0,taumax-delta,N);

    kest = K1_Est(seq,taumax,delta);

    #omg = 3;
    #kest = [0.7*np.sin(omg*y)+ rd.gauss(0,0.03) for y in x];
    #kest = np.array(kest);
    #kest[math.ceil(np.pi/(omg*delta)):] = 0;

    # Test ###########
    #+ echo=False

    # markerline, stemlines, baseline = plt.stem(x, kest, '-')
    # plt.setp(markerline, 'markerfacecolor', 'b')
    # plt.setp(baseline, 'color', 'k', 'linewidth', 2)
    # plt.setp(stemlines, 'color', 'k')
    # plt.xlabel('Time')
    # plt.ylabel('Magnitude')
    # plt.title('Discretized Kernel Estimation')

    #' ## K1-Estimation
    #' Results from the first level of kernel decomposition (EXP,PWL,SQR,SNS)

    #+ echo=False
    K1_Param = K1_Class(kest,taumax,delta,seq,T,Delta);

    K1_Index = K1_Param['K1_Index']

    if K1_Param['K1_Type'] == 'EXP':

        fitkernel = K1_Param['fitEXP'];

        llh_K1 = np.append(llh_K1,K1_Param['Likelihood'][K1_Index])

        # fitted = plt.plot(x, fitkernel, '-')
        # plt.setp(fitted,linewidth=2.0)
        # markerline, stemlines, baseline = plt.stem(x, kest, '-')
        # plt.setp(markerline, 'markerfacecolor', 'b')
        # plt.setp(baseline, 'color', 'k', 'linewidth', 2)
        # plt.setp(stemlines, 'color', 'k')
        # plt.xlabel('Time')
        # plt.ylabel('Magnitude')
        # plt.title('Fitted EXP Kernel')

    elif K1_Param['K1_Type'] == 'PWL':

        fitkernel = K1_Param['fitPWL'];

        llh_K1 = np.append(llh_K1,K1_Param['Likelihood'][K1_Index])

        # fitted = plt.plot(x, fitkernel, '-')
        # plt.setp(fitted,linewidth=2.0)
        # markerline, stemlines, baseline = plt.stem(x, kest, '-')
        # plt.setp(markerline, 'markerfacecolor', 'b')
        # plt.setp(baseline, 'color', 'k', 'linewidth', 2)
        # plt.setp(stemlines, 'color', 'k')
        # plt.xlabel('Time')
        # plt.ylabel('Magnitude')
        # plt.title('Fitted PWL Kernel')

    elif K1_Param['K1_Type'] == 'SQR':

        fitkernel = K1_Param['fitSQR'];

        llh_K1 = np.append(llh_K1,K1_Param['Likelihood'][K1_Index])

        # plt.plot(x, fitkernel, '-')
        # markerline, stemlines, baseline = plt.stem(x, kest, '-')
        # plt.setp(markerline, 'markerfacecolor', 'b')
        # plt.setp(baseline, 'color', 'k', 'linewidth', 2)
        # plt.setp(stemlines, 'color', 'k')
        # plt.xlabel('Time')
        # plt.ylabel('Magnitude')
        # plt.title('Fitted SQR Kernel')

    elif K1_Param['K1_Type'] == 'SNS':

        fitkernel = K1_Param['fitSNS'];

        llh_K1 = np.append(llh_K1,K1_Param['Likelihood'][K1_Index])

        # plt.plot(x, fitkernel, '-')
        # markerline, stemlines, baseline = plt.stem(x, kest, '-')
        # plt.setp(markerline, 'markerfacecolor', 'b')
        # plt.setp(baseline, 'color', 'k', 'linewidth', 2)
        # plt.setp(stemlines, 'color', 'k')
        # plt.xlabel('Time')
        # plt.ylabel('Magnitude')
        # plt.title('Fitted SNS Kernel')

    else:
        print('No category suits K1')

    #' ## K2-Estimation
    #' Results from the multiplicative and additive decomposition level:

    #+ echo=False
    if K1_Param['K1_Type'] == 'EXP':

        K2_Param = EXP_Class(K1_Param,kest,taumax,delta,seq,T,Delta)

        K2_Index = K2_Param['K2_Index']

        llh_K2 = np.append(llh_K2,K2_Param['Likelihood'][K2_Index])

    elif K1_Param['K1_Type'] == 'PWL':

        K2_Param = PWL_Class(K1_Param,kest,taumax,delta,seq,T,Delta)

        K2_Index = K2_Param['K2_Index']

        llh_K2 = np.append(llh_K2,K2_Param['Likelihood'][K2_Index])

    elif K1_Param['K1_Type'] == 'SQR':

        K2_Param = SQR_Class(K1_Param,kest,taumax,delta,seq,T,Delta)

        K2_Index = K2_Param['K2_Index']

        llh_K2 = np.append(llh_K2,K2_Param['Likelihood'][K2_Index])

    elif K1_Param['K1_Type'] == 'SNS':

        K2_Param = SNS_Class(K1_Param,kest,taumax,delta,seq,T,Delta)

        K2_Index = K2_Param['K2_Index']

        llh_K2 = np.append(llh_K2,K2_Param['Likelihood'][K2_Index])

    else:
        print('No category suits K2')

    fitkernel = K2_Param['fitkernel'];

    # fitted = plt.plot(x, fitkernel, '-')
    # plt.setp(fitted,linewidth=2.0)
    # markerline, stemlines, baseline = plt.stem(x, kest, '-')
    # plt.setp(markerline, 'markerfacecolor', 'b')
    # plt.setp(baseline, 'color', 'k', 'linewidth', 2)
    # plt.setp(stemlines, 'color', 'k')
    # plt.xlabel('Time')
    # plt.ylabel('Magnitude')
    # plt.title('Fitted K2')

    #' ## Comparison

    #+ echo=False

    eta = 1.3 # Regularization parameter

    if K1_Param['K1_Type'] == 'EXP':

        Res_K1 = K1_Param['Res_EXP']
        statcriter_K1 = K1_Param['EXP_statcriter']

    elif K1_Param['K1_Type'] == 'PWL':

        Res_K1 = K1_Param['Res_PWL']
        statcriter_K1 = K1_Param['PWL_statcriter']

    elif K1_Param['K1_Type'] == 'SQR':

        Res_K1 = K1_Param['Res_SQR']
        statcriter_K1 = K1_Param['SQR_statcriter']

    elif K1_Param['K1_Type'] == 'SNS':

        Res_K1 = K1_Param['Res_SNS']
        statcriter_K1 = K1_Param['SNS_statcriter']

    else:
        print('No category suits K1')

    K1 = K1_Param['K1_Index']

    K2 = K2_Param['K2_Index']

    frequency_K1[K1] += 1

    frequency_K2[K1,K2] += 1

    Res_K2 = K2_Param['Res_K2']
    statcriter_K2 = K2_Param['K2_statcriter']

    if not (statcriter_K1 or statcriter_K2):

        print('No kernel category satisfies the stationarity criteria')

    elif (not statcriter_K1) and statcriter_K2:

        print('Considering the two levels of decomposition, the best kernel fitting is ' + repr(K2_Param['K2_Type']))

    elif statcriter_K1 and (not statcriter_K2):

        print('Considering the two levels of decomposition, the best kernel fitting is ' + repr(K1_Param['K1_Type']))

    else:

        if Res_K1/eta <= Res_K2:

            print('Considering the two levels of decomposition, the best kernel fitting is ' + repr(K1_Param['K1_Type']))

        else:

            print('Considering the two levels of decomposition, the best kernel fitting is ' + repr(K2_Param['K2_Type']))

print('frequency_K1: ' + repr(frequency_K1))
print('frequency_K2: ' + repr(frequency_K2))
print('llh_K1: ' + repr(llh_K1))
print('llh_K2: ' + repr(llh_K2))

f = open('frequency_per_kernel_stock.txt','w')
f.write(repr(frequency_K1))
f.write(repr(frequency_K2))
f.write(repr(llh_K1))
f.write(repr(llh_K2))
f.close()