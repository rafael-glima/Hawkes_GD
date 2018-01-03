#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import random as rd
import time
import csv
import re
import requests
import pandas as pd
import time

from trainGD_EXP import trainGD_EXP
from trainGD_PWL import trainGD_PWL
from trainGD_QEXP import trainGD_QEXP

input_data = scipy.io.loadmat('4Kern_newSNS_10seqT5000_highMuandfreq.mat')

eps = np.finfo(float).eps

n_of_seq = 10

llh_GD_EXP = []
llh_GD_PWL = []
llh_GD_QEXP = []

for i in range(1,3):

	ind_seq = 'Seq' + repr(i)

	print(ind_seq)

	for j in range(0,n_of_seq):

		print(j)

		seq = input_data[ind_seq][j][0][0]

		#seq = seq[0:3000]

		seq = seq[:-1]

		T = seq[-1] - seq[0]

		Delta = len(seq)/T

		print("Training EXP model")

		Param_GD_EXP = trainGD_EXP(seq)

		Delta_GD_EXP = Param_GD_EXP['EXP_coeffs']

		print("Training PWL model")

		Param_GD_PWL = trainGD_PWL(seq)

		Delta_GD_PWL = Param_GD_PWL['PWL_coeffs']

		print("Training QEXP model")

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
