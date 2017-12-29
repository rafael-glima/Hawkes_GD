import scipy.io
from scipy.optimize import minimize
import numpy as np
from scipy.integrate import quad
#import numpy.random as np.random

def trainGD_EXP(seq):

	alpha_0 = np.random.rand()
	beta_0 = 2*alpha_0
	q_0 = 1. + np.random.rand()
	mu_0 = np.random.rand()

	# input_data = scipy.io.loadmat('4Kern_newSNS_10seqT100000_highMuandfreq0.15.mat')
	# seq = input_data['Seq2'][1][0][0]

	# seq = seq[:300]

	T = seq[-1]-seq[0]
	Delta = len(seq)/T

	#seq = np.array([1.,2.,3.,4.,5.])

	bnds = ((0,None),(0,None),(0,None))

	def logGD_QEXP(EXP_coeffs):

		def funcqexp(x,alpha,beta,q):

			print('q: '+ repr(q))

			if math.isclose(q, 1., rel_tol=1e-3, abs_tol=0.0):

				return alpha*np.exp(-beta*x)

			elif (q != 1.) and (1 + (1-q)*beta*x > 0):

				return np.exp(alpha*(1+(q-1)*beta*x),1/1-q)

			else:

				return 0

		alpha = EXP_coeffs[1];

		beta = EXP_coeffs[2];

		q = EXP_Coeffs[3]

		mu = EXP_coeffs[0]


		if math.isclose(q, 1., rel_tol=1e-3, abs_tol=0.0):

			if (alpha/beta < 1.) and (alpha/beta >= 0.):

				mu = (1.-alpha/beta)*Delta;

			else:

				mu = 0. 
				return np.inf

		elif (q != 1.) and (1 + (q-1)*beta*x > 0.):

			if (alpha*(q-1)/(2-q) < 1.) and (alpha*(q-1)/(2-q) > 0.):

				mu = (1.- alpha*(q-1)/(2-q))*Delta;

			else:

				mu = 0.
				return np.inf

		else:

			mu = Delta

		intens = np.zeros(len(seq));

		compens = mu*T;

		for i in range(0,len(seq)):

			intens[i] += mu;

			if math.isclose(q, 1., rel_tol=1e-3, abs_tol=0.0):

				compens += (alpha/beta)*(1-np.exp(-beta*(T-seq[i])))

			elif (q != 1.) and (1 + (q-1)*beta*x > 0.):

				compens += alpha*((1-q)/(2-q))*(np.exp(1+(q-1)*beta*(T-seq[i]), (2-q)/(1-q))-1)

			else:

				compens += 0.

			#compens += (alpha/beta)*(1-np.exp(-beta*(T-seq[i])))#quad(funcexp,0,T-seq[i], args=(alpha,beta))[0]

			for j in range(0,i):

				if math.isclose(q, 1., rel_tol=1e-3, abs_tol=0.0):

					intens[i] += alpha*np.exp(-beta*(seq[i] - seq[j]))

				elif (q != 1.) and (1 + (q-1)*beta*x > 0.):

					intens[i] += alpha*((1-q)/(2-q))*np.exp((1+(q-1)*beta*(seq[i]-seq[j])),(2-q)/(1-q))

				else:

					intens[i] += 0.

		print ('Loglikelihood Train GD: ' + repr(np.sum(np.nan_to_num(np.log(intens))) - compens) + '\n')

		return - np.sum(np.nan_to_num(np.log(intens))) + compens

	par = minimize(logGD_EXP, [mu_0, alpha_0, beta_0, q_0], method='Nelder-Mead', tol=1e-2, options={'maxiter':10})

	print('Final Parameters: '+ repr(par.x)+'\n')

	K1_Param = {'EXP_coeffs': par.x, 'K1_Type': 'EXP', 'EXP_statcriter': par.x[1]/par.x[2]}

	return K1_Param