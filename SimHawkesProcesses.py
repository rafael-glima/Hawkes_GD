
import numpy as np
import numpy.random as rand
import random
from KernelFunc import KernelFunc
from SupIntensity import SupIntensity

def simHP(level, para, maxjumps, taumax, Delta):

	K1_Param = para

	if K1_Param['K1_Type'] == 'EXP':

		statcriter = K1_Param['EXP_statcriter']

	if K1_Param['K1_Type'] == 'PWL':

		statcriter = K1_Param['PWL_statcriter']

	if K1_Param['K1_Type'] == 'SQR':

		statcriter = K1_Param['SQR_statcriter']

	if K1_Param['K1_Type'] == 'SNS':

		statcriter = K1_Param['SNS_statcriter']

	if statcriter >= 1.:
#		print('statcriter: ' + repr(statcriter))

		print('Error: The sequence could not be modeled, because the estimated kernel is not stable.')

		mu = Delta #return np.zeros((maxjumps,))
	else: 

		mu = Delta*(1-statcriter)

	#mu = Delta

	#mu = Delta*(1-statcriter)

	#mu = max(Delta*(1-statcriter),0.0001)
	

	#print('mu: '+repr(mu))
	#sim_seq = np.array([rand.exponential(1/mu)])

	sim_seq = np.array([random.expovariate(mu)])

	n_of_jumps = 1

	time = sim_seq[0]

	while (n_of_jumps < maxjumps):

		l = taumax

		# For Python 3

		u = rand.random()

		# For Python 2

		# u = rand.random

		mt = SupIntensity(para, sim_seq, mu, taumax)

		#print('mt: ' + repr(mt))

		#dt = rand.exponential(1/mt)

		dt =random.expovariate(mt)

		intens_dt = SupIntensity(para, np.append(sim_seq, time+dt), mu, taumax)
		
		mt2 = SupIntensity(para,sim_seq - time, mu, taumax) # np.append(sim_seq, time), mu, taumax)

		#print('intens_dt: ' + repr(intens_dt))

		#print('intens_dt/mt2: '+repr(intens_dt/mt2))

		#print('l: ' + repr(l))

		#print('dt: ' + repr(dt))

		#print('u: ' + repr(u))
		
		if (dt < l) and ((intens_dt/mt2) > u) :

			time += dt

			sim_seq = np.append(sim_seq, time)

			n_of_jumps += 1
			
			#print('n_of_jumps: ' + repr(n_of_jumps))
		else:

			time += l


	return sim_seq
