import scipy.io
from scipy.optimize import minimize
import numpy as np
from scipy.integrate import quad
#import numpy.random as np.random

def trainGD_EXP(seq):

	alpha_0 = np.random.rand()
	beta_0 = 2*alpha_0
	mu_0 = np.random.rand()

	# input_data = scipy.io.loadmat('4Kern_newSNS_10seqT100000_highMuandfreq0.15.mat')
	# seq = input_data['Seq2'][1][0][0]

	# seq = seq[:300]

	T = seq[-1]-seq[0]
	Delta = len(seq)/T

	#seq = np.array([1.,2.,3.,4.,5.])

	bnds = ((0,None),(0,None),(0,None))

	def logGD_EXP(EXP_coeffs):

		def funcexp(x,alpha,beta):
			return alpha*np.exp(-beta*x)

		alpha = EXP_coeffs[1];

		beta = EXP_coeffs[2];

		mu = EXP_coeffs[0]

		if (alpha/beta < 1.) and (alpha/beta >= 0.):
			mu = (1.-alpha/beta)*Delta;
		else:
			mu = 0. 
			return np.inf

		intens = np.zeros(len(seq));

		compens = mu*T;

		for i in range(0,len(seq)):

			intens[i] += mu;

			compens += (alpha/beta)*(1-np.exp(-beta*(T-seq[i])))#quad(funcexp,0,T-seq[i], args=(alpha,beta))[0]

			for j in range(0,i):

				intens[i] += alpha*np.exp(-beta*(seq[i] - seq[j]))			

		print ('Loglikelihood Train GD: ' + repr(np.sum(np.nan_to_num(np.log(intens))) - compens) + '\n')

		return - np.sum(np.nan_to_num(np.log(intens))) + compens

	par = minimize(logGD_EXP, [mu_0, alpha_0, beta_0], method='Nelder-Mead', tol=1e-2, options={'maxiter':10})

	print('Final Parameters: '+ repr(par.x)+'\n')

	fin_llh = logGD_EXP(par.x)

	fin_llh = (-1)*fin_llh

	K1_Param = {'EXP_coeffs': par.x, 'K1_Type': 'EXP', 'EXP_statcriter': par.x[1]/par.x[2], 'final_llh': fin_llh}

	return K1_Param

