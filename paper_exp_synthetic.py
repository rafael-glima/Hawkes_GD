
import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
import random as rd

import pandas as pd
import datetime as DT
import time

from joblib import Parallel, delayed
import multiprocessing
import parmap
from multiprocessing import Pool, freeze_support
import itertools

from K1_Est import K1_Est
from K1_Class import K1_Class
from EXP_Class import EXP_Class
from PWL_Class import PWL_Class
from SQR_Class import SQR_Class
from SNS_Class import SNS_Class

from SimHawkesProcesses import simHP

input_data = scipy.io.loadmat('4Kern_newSNS_10seqT100000_highMuandfreq0.15.mat')

eps = np.finfo(float).eps

n_of_seq = 10

resolution = 40

frequency_K1 = np.zeros((4,4), dtype= np.int)
frequency_K2 = np.zeros((4,4,8), dtype= np.int)

llh_K1 = []
llh_K2 = []

res_1st_lev = []
res_2nd_lev = []

taumax_avg = np.array([0.,0.,0.,0.])
delta_avg = np.array([0.,0.,0.,0.])
T_avg = np.array([0.,0.,0.,0.])
Delta_avg = np.array([0.,0.,0.,0.])

kest_matrix = np.zeros((4,n_of_seq,resolution), dtype=np.float64)

kest_avg = np.zeros((4,resolution), dtype=np.float64)

for i in range(1,5):

	ind_seq = 'Seq' + repr(i)

	print ind_seq

	for j in range(0,n_of_seq):

		print j

		seq = input_data[ind_seq][j][0][0]

		#seq = seq[0:2000]

		T = seq[-1] - seq[0]

		Delta = len(seq)/T

		Intervals = np.diff(seq)

		hist, bin_edges = np.histogram(Intervals,bins=100)

		cumhist = np.cumsum(hist)

		indtau = np.where(cumhist > 0.95*np.sum(hist))

		indtau = indtau[0]

		if len(indtau) < 2:
			taumax = bin_edges[indtau[0]]
		else:
			taumax = bin_edges[indtau[0]+1]

		#taumax = 1.2/Delta #3.2

		#taumax = 6.6

		#if i==3:
			#taumax = 3.7

		print ('taumax: ' + repr(taumax) + '\n')

		delta = taumax/resolution

		#N = int(taumax/delta)

		N = resolution

		x = np.linspace(0,taumax-delta,N);

		#kest = K1_Est(seq,taumax,delta);

		######################################################################

		#                            K1 Est

		#####################################################################

		# Loading sequence of time events

		# First Order Statistics

		siz = len(seq);
		T = seq[-1]-seq[0];

		Delta = siz/T;

		# Discretized Covariance Estimation

		eps = np.finfo(float).eps;

		h = delta;

		tau = 0;

		j = 0;

		#N = int(taumax/delta);

		num_cores = multiprocessing.cpu_count()

		t = range(0,N)

		def par_cov(seq,delta,h,Delta,T,taumax,t):

			print t

			EMtMt = 0;

			tau = t*delta;

			for ind in range(0,int(np.nan_to_num(int((T-taumax)/h)-1))):

				c1 = ((seq >= ind*h) & (seq <= (ind+1)*h)).sum();
				c2 = ((seq >= (ind*h+tau)) & (seq <= ((ind+1)*h+tau))).sum();

				EMtMt = EMtMt + (c1-Delta*h)*(c2-Delta*h);
			
			return EMtMt/(T-taumax);

		cov = Parallel(n_jobs=num_cores)(delayed(par_cov)(seq,delta,h,Delta,T,taumax,i) for i in t)

		#N = int(taumax/h);

		covr = cov[1:];
		covr = covr[::-1];

		cov1 = np.append(covr,cov);

		# Un-KDEstimation of the Discretized Covariance

		fftcov = np.fft.fftshift(cov1)

		vbyg = np.zeros(len(cov1));

		vbyg[N-1] = cov1[N-1]/h;

		for ind in range(1,N):

			# OMEGA CONSTANT!!!!!!!

			omega = 2*np.pi*(ind)/(N*taumax);
			gh = (4/(h*omega**2))*(np.sin(omega*h/2))**2;
			vbyg[ind+N-1] = fftcov[ind+N-1]/gh;

		vbygr = vbyg[N:];
		vbygr = vbygr[::-1];

		vbyg1 = np.append(vbygr,vbyg[N-1:]);

		onepphi2 = [y / Delta for y in vbyg1];

		logophi = np.log(np.sqrt(np.abs(onepphi2+eps)));

		logophi2 = [2*y for y in logophi];

		phiest = np.append(np.exp(logophi2[:N-1]),np.exp(logophi[N-1]));
		phiest = np.append(phiest,np.zeros(N-1));

		kest = np.fft.ifftshift(phiest);

		kest = kest[N-1:]

		kest[0] = kest[2];

		kest[1] = kest[2];


		#####################################################################


		# kest = np.asarray([  0.        ,  0.033,   0.03290132,   0.02020813,
  #        0.02254365,   0.02554656,   0.04257098,   0.06426612,
  #        0.07293955,   0.08361437,   0.07292792,   0.0792631 ,
  #        0.08392905,   0.09827107,   0.07290693,   0.10693228,
  #        0.11459853,   0.12793511,   0.11792066,   0.12892129,
  #        0.12624624,   0.13491131,   0.10989242,   0.09688157,
  #        0.09287582,   0.10654387,   0.06652651,   0.0745266 ,
  #        0.06652207,   0.05251677,   0.03984601,   0.0348441 ,
  #        0.03484334,   0.04184369,   0.01950678,   0.01650616,
  #        0.01650597,   0.01683919,   0.02317297,   0.01717237], dtype=np.float64)

		print type(kest)

		# print('T: ' + repr(T))

		# print('Delta: ' + repr(Delta))

		# print('hist: ' + repr(hist))

		# print('\n' + 'bin_edges: ' + repr(bin_edges) + '\n')

		# print('\n' + 'indtau: ' + repr(indtau) + '\n')

		# print('\n' + 'taumax: ' + repr(taumax) + '\n')

		print('kest: ' + repr(kest))


		kest_avg[i-1,:] += kest

		kest_matrix[i-1,j,:] = kest

		taumax_avg[i-1] += taumax

		delta_avg[i-1] += delta

		T_avg[i-1] += T

		Delta_avg[i-1] += Delta

		K1_Param = K1_Class(kest,taumax,delta,seq,T,Delta);

		K1_Index = K1_Param['K1_Index']

		if K1_Param['K1_Type'] == 'EXP':

			fitkernel = K1_Param['fitEXP'];

			llh_K1 = np.append(llh_K1,K1_Param['Likelihood'])

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
			llh_K1 = np.append(llh_K1,K1_Param['Likelihood'])

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
			llh_K1 = np.append(llh_K1,K1_Param['Likelihood'])

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
			llh_K1 = np.append(llh_K1,K1_Param['Likelihood'])
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

			llh_K2 = np.append(llh_K2,K2_Param['Likelihood'])

		elif K1_Param['K1_Type'] == 'PWL':

			K2_Param = PWL_Class(K1_Param,kest,taumax,delta,seq,T,Delta)
			
			K2_Index = K2_Param['K2_Index']

			llh_K2 = np.append(llh_K2,K2_Param['Likelihood'])

		elif K1_Param['K1_Type'] == 'SQR':

			K2_Param = SQR_Class(K1_Param,kest,taumax,delta,seq,T,Delta)
			
			K2_Index = K2_Param['K2_Index']

			llh_K2 = np.append(llh_K2,K2_Param['Likelihood'])

		elif K1_Param['K1_Type'] == 'SNS':

			K2_Param = SNS_Class(K1_Param,kest,taumax,delta,seq,T,Delta)
			
			K2_Index = K2_Param['K2_Index']

			llh_K2 = np.append(llh_K2,K2_Param['Likelihood'])

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

		eta = 1.2 # Regularization parameter

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

		frequency_K1[i-1,K1] += 1

		frequency_K2[i-1,K1,K2] += 1

		Res_K2 = K2_Param['Res_K2']
		statcriter_K2 = K2_Param['K2_statcriter']

		res_1st_lev = np.append(res_1st_lev,Res_K1)
		res_2nd_lev = np.append(res_2nd_lev,Res_K2)

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


print ('frequency_K1:' + repr(frequency_K1) + '\n')
print('frequency_K2:' + repr(frequency_K2) + '\n')
print('llh_K1: ' + repr(llh_K1) + '\n')
print('llh_K2: ' + repr(llh_K2) + '\n')
print('res_1st_lev: ' + repr(res_1st_lev) + '\n')
print('res_2nd_lev: ' + repr(res_2nd_lev) + '\n')
print('kest_matrix: ' + repr(kest_matrix) + '\n')

kest_avg = kest_avg/n_of_seq

taumax_avg = taumax_avg/n_of_seq
delta_avg = taumax_avg/resolution #delta_avg/n_of_seq
T_avg = T_avg/n_of_seq
Delta_avg = Delta_avg/n_of_seq

print('\n' + 'Now, getting the classification for aggregated estimations: ' + '\n')

for i in range(0,4):
	ind_seq = 'Seq' + repr(i+1)
	seq = input_data[ind_seq][0][0][0]
	#seq = seq[:2000]
	Delta = len(seq)/(seq[-1]+eps)
	K1_Param = K1_Class(kest_avg[i,:],taumax_avg[i],delta_avg[i],seq,T_avg[i],Delta_avg[i]);
	sim_seq = simHP(1, K1_Param, len(seq), taumax_avg[i], Delta)
	print('seq.shape: ' + repr(seq.shape))
	print('sim_seq.shape: ' + repr(sim_seq.shape))
	plt.plot(seq, sim_seq)
	plt.plot(seq,seq)
	plt.savefig('Goodness_of_Fit_Kernel_' + repr(i) + '.png')
	#plt.show()
	plt.close()

print('kest_avg: ' + repr(kest_avg) + '\n')

f = open('frequency_per_kernel_4Kern_newSNS_10seqT100000_highMuandfreq_5thtry.mat.txt','w')
f.write('frequency_K1: ' + repr(frequency_K1) + '\n')
f.write('frequency_K1: ' + repr(frequency_K2) + '\n')
f.write('llh_K1: ' + repr(llh_K1) + '\n')
f.write('llh_K2: ' + repr(llh_K2) + '\n')
f.write('res_1st_lev: ' + repr(res_1st_lev) + '\n')
f.write('res_2nd_lev: ' + repr(res_2nd_lev) + '\n')
f.write('kest_matrix: ' + repr(kest_matrix) + '\n')
f.write('kest_avg: ' + repr(kest_avg) + '\n')
f.write('taumax: ' + repr(taumax) + '\n')
f.write('resolution: ' + repr(resolution) + '\n')
f.close()

