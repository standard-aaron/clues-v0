import warnings
import argparse
#import hmm
import tree_utils
import numpy as np
import scipy.stats as stats
import time
import glob
from Bio import Phylo
from io import StringIO

#import numpy as np
from numba import njit,jit,int32,float32,int64,float64,typeof

@njit('float64(float64[:])',cache=True)
def _logsumexp(a):
    a_max = np.max(a)

    tmp = np.exp(a - a_max)

    s = np.sum(tmp)
    out = np.log(s)

    out += a_max
    return out

@njit('float64(float64[:],float64[:])',cache=True)
def _logsumexpb(a,b):

    a_max = np.max(a)

    tmp = b * np.exp(a - a_max)

    s = np.sum(tmp)
    out = np.log(s)

    out += a_max
    return out

@njit('float64(float64)',cache=True)
def _log_phi(z):
	logphi = -0.5 * np.log(2.0* np.pi) - 0.5 * z * z
	return logphi

# cdef Phi(double z):
# 	cdef double a1 =  0.254829592
# 	cdef double a2 = -0.284496736
# 	cdef double a3 =  1.421413741
# 	cdef double a4 = -1.453152027
# 	cdef double a5 =  1.061405429
# 	cdef double p  =  0.3275911

# 	cdef int sign = 1
# 	if (z<0):
# 		sign = -1
# 	z = fabs(z)/sqrt(2.0)

# 	cdef double t = 1.0/(1.0 + p*z)
# 	cdef double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-z*z)
# 	return 0.5 * (1.0 + sign * y)

@njit('float64(float64,float64)',cache=True)
def _eta(alpha,beta):
	return alpha * beta / (alpha * (np.exp(beta) - 1) + beta * np.exp(beta))

@njit('float64(float64,float64,float64,float64)',cache=True)
def _griffiths_log_prob_coal_counts(a,b,t,N):
	t *= 1.0/(2.0*N)
	n = a
	alpha = 0.5*n*t
	beta = -0.5*t
	h = _eta(alpha,beta)
	mu = 2.0 * h * t**(-1.0)
	var = 2.0*h*t**(-1.0) * (h + beta)**2
	var *= (1.0 + h/(h+beta) - h/alpha - h/(alpha + beta) - 2.0*h)
	var *= beta**-2

	std = np.sqrt(var)
	lp = np.zeros(int(a))
	for bprime in range(1,int(a)):
		lp[bprime-1] = _log_phi((float(bprime)-mu)/std)

	return _log_phi((b-mu)/std) - _logsumexp(lp)

@njit('float64(int64,int64,float64,float64)',cache=True)
def _tavare_log_prob_coal_counts(a, b, t, n):

	lnC1 = 0.0
	lnC2=0.0
	C3=1.0

	for y in range(0,int(b)):
		lnC1 += np.log((b+y)*(a-y)/(a+y))

	s = -b*(b-1.0)*t/4.0/n

	for k in range(int(b)+1,int(a)+1):
		k1 = k - 1.0
		lnC2 += np.log((b+k1)*(a-k1)/(a+k1)/(k-b))
		C3 *= -1.0
		val = -k*k1*t/4.0/n + lnC2 + np.log((2.0*k-1.0)/(k1+b))
		if True:
			loga = s
			logc = val
			if (logc > loga):
				tmp = logc
				logc = loga
				loga = tmp
			s = loga + np.log(1.0 + C3*np.exp(logc - loga))
	for i in range(2,int(b)+1):
		s -= np.log(i)

	return s + lnC1

@njit('float64(int64,int64,int64,int64,int64,int64,float64,float64,float64,int64)',cache=True)
def _tavare_structured_coal(Cder0,Cder1,Canc0,Canc1,Cmix0,Cmix1,Nnow,x,t,isTavare):

		logCondProb = 0.0
		if x == 0.0:
				if Cder1 > 1:
						return -np.inf
				if Cmix0 == 1:
						return 0.0
				b = Cmix1
				a = Cmix0

				if isTavare:
					logCondProb += _tavare_log_prob_coal_counts(a,b,t,Nnow)
				else:
					logCondProb += _griffiths_log_prob_coal_counts(a,b,t,Nnow)
		else:
				if Cmix1 == Canc1 and Cder0 > 0.0:
						return -np.inf

				for (N,a,b) in zip([Nnow*x,Nnow*(1-x)],[Cder0,Canc0],[Cder1,Canc1]):
						if a == 1.0 or a == 0.0:
								continue
						elif N == 0:
								return -np.inf
						if isTavare:
							logCondProb += _tavare_log_prob_coal_counts(a,b,t,N)
						else:
							logCondProb += _griffiths_log_prob_coal_counts(a,b,t,N)
		return logCondProb

@njit('float64[:,:](float64[:,:],float64[:,:])',cache=True)
def _log_prob_mat_mul(A,B):
    # multiplication of probability matrices in log space
    C = np.zeros((A.shape[0],B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i,j] = _logsumexp( A[i,:] + B[:,j])
            if np.isnan(C[i,j]):
                C[i,j] = np.NINF
        ## special sauce...
        C[i,:] -= _logsumexp(C[i,:])
    return C

@njit('float64[:,:](float64[:,:],int64)',cache=True)
def _log_matrix_power(X,n):
    ## find log of exp(X)^n (pointwise exp()!) 

    # use 18 because you are fucked if you want trans
    # for dt > 2^18...
    #print('Calculating matrix powers...')

    maxlog2dt = 18
    assert(np.log(n)/np.log(2) < maxlog2dt)
    assert(X.shape[0] == X.shape[1])
    b = 1
    k = 0
    matrices = np.zeros((X.shape[0],X.shape[1],maxlog2dt))
    matrices[:,:,0] = X
    
    while b < n:
        #print(b,k)
        k += 1
        b += 2**k
        # square the last matrix
        matrices[:,:,k] = _log_prob_mat_mul(matrices[:,:,k-1],
                                           matrices[:,:,k-1])
    leftover = n
    Y = np.NINF * np.ones((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):
        Y[i,i] = 0
        
    while leftover > 0:
        #print(n-leftover,k)
        if 2**k <= leftover:
            Y = _log_prob_mat_mul(Y,matrices[:,:,k])
            leftover -= 2**k
        k -= 1
        
    return Y

@njit('float64[:](int64,float64,float64,float64[:],float64[:],float64[:],float64[:],int64)',cache=True)
def _log_trans_prob(i,N,s,FREQS,z_bins,z_logcdf,z_logsf,dt):
	# 1-generation transition prob based on Normal distn
	
	p = FREQS[i]
	lf = len(FREQS)
	logP = np.NINF * np.ones(lf)

	if p <= 0.0:
		logP[0] = 0
	elif p >= 1.0:
		logP[lf-1] = 0
		return logP
	else:
		#plo = (FREQS[i]+FREQS[i-1])/2
		#phi = (FREQS[i]+FREQS[i+1])/2
		if s != 0:
			mu = p - s*p*(1.0-p)/np.tanh(2*N*s*(1-p))*dt
			# mulo = plo - s*plo*(1.0-plo)/np.tanh(2*N*s*(1-plo))*dt
			# muhi = phi - s*phi*(1.0-phi)/np.tanh(2*N*s*(1-phi))*dt
		else:
			mu = p - p * 1/(2.0*N)*dt
			# mulo = plo - plo * 1/(2.0*N)*dt
			# muhi = phi - phi * 1/(2.0*N)*dt
		sigma = np.sqrt(p*(1.0-p)/(2.0*N)*dt)
		# sigmalo = np.sqrt(plo*(1.0-plo)/(2.0*N)*dt)
		# sigmahi = np.sqrt(phi*(1.0-phi)/(2.0*N)*dt)
                      
		pi0 = np.interp(np.array([(FREQS[0]-mu)/sigma]),z_bins,z_logcdf)[0]
		pi1 = np.interp(np.array([(FREQS[lf-1]-mu)/sigma]),z_bins,z_logsf)[0]

		# pi0lo = np.interp(np.array([(np.mean(FREQS[:2])-mulo)/sigmalo]),z_bins,z_logcdf)[0]
		# pi1lo = np.interp(np.array([(FREQS[lf-1]-mulo)/sigmalo]),z_bins,z_logsf)[0]
		# pi0hi = np.interp(np.array([(np.mean(FREQS[:2])-muhi)/sigmahi]),z_bins,z_logcdf)[0]
		# pi1hi = np.interp(np.array([(FREQS[lf-1]-muhi)/sigmahi]),z_bins,z_logsf)[0]
		#print(i,p,plo,phi,pi0lo,pi0hi,pi1lo,pi1hi)
		x = np.array([0.0,pi0,pi1])
		b = np.array([1.0,-1.0,-1.0])
		middleNorm = _logsumexpb(x,b)
        
		# x = np.array([0.0,pi0lo,pi1lo,0.0,pi0hi,pi1hi])
		# b = np.array([0.5,-0.5,-0.5,0.5,-0.5,-0.5])
		# middleNorm = _logsumexpb(x,b)

		middleP = np.zeros(lf-2)
		for j in range(1,lf-1):
			if j == 0:
				mhi = FREQS[0]
			else:
				mlo = np.mean(np.array([FREQS[j],FREQS[j-1]]))
			if j == lf-2:
				mhi = FREQS[j+1]
			else:
				mhi = np.mean(np.array([FREQS[j],FREQS[j+1]]))

			l1 = np.interp(np.array([(mlo-mu)/sigma]),z_bins,z_logcdf)[0]
			l2 = np.interp(np.array([(mhi-mu)/sigma]),z_bins,z_logcdf)[0]
			middleP[j-1] = _logsumexpb(np.array([l1,l2]),np.array([-1.0,1.0]))                    
			# l1lo = np.interp(np.array([(mlo-mulo)/sigmalo]),z_bins,z_logcdf)[0]
			# l2lo = np.interp(np.array([(mhi-mulo)/sigmalo]),z_bins,z_logcdf)[0]

			# l1hi = np.interp(np.array([(mlo-muhi)/sigmahi]),z_bins,z_logcdf)[0]
			# l2hi = np.interp(np.array([(mhi-muhi)/sigmahi]),z_bins,z_logcdf)[0]
			# middleP[j-1] = _logsumexpb(np.array([l1lo,l2lo,l1hi,l2hi]),np.array([-0.5,0.5,-0.5,0.5]))

		#print(p,pi0,pi1,middleNorm)
		#logP[0] = _logsumexpb(np.array([pi0lo,pi0hi]),np.array([0.5,0.5]))
		#logP[1:lf-1] = middleP
		#logP[lf-1] = _logsumexpb(np.array([pi1lo,pi1hi]),np.array([0.5,0.5]))
		#logP[0] = _logsumexpb(np.array([pi0lo,pi0hi]),np.array([0.5,0.5]))
		#logP[1:lf-1] = middleP
		#logP[lf-1] = _logsumexpb(np.array([pi1lo,pi1hi]),np.array([0.5,0.5]))

		logP[0] = pi0
		logP[1:lf-1] = middleP
		logP[lf-1] = pi1

	return logP

@njit('float64[:,:](float64,float64,float64[:],float64[:],float64[:],float64[:],int64)',cache=True)
def _nstep_log_trans_prob(N,s,FREQS,z_bins,z_logcdf,z_logsf,dt):
	lf = len(FREQS)
	p1 = np.zeros((lf,lf))

	# load rows into p1
	for i in range(lf):
		row = _log_trans_prob(i,N,s,FREQS,z_bins,z_logcdf,z_logsf,1)
		p1[i,:] = row

	# exponentiate matrix
	# exponentiate matrix
	pn = _log_matrix_power(p1,int(dt))
	return pn

@njit('float64[:,:](float64[:],float64[:],float64[:,:],float64[:],int64,float64[:],int64,float64[:],float64[:],float64[:])',cache=True)
def _forward_algorithm(selOverTime,dts,C,N,dpfi,freqs,coalModelChangepoint,z_bins,z_logcdf,z_logsf):

	'''
	Moves backward in time from present to past
	'''

	prevst = -999999
	prevNt = -1
	prevdt = -1
	lf = freqs.shape[0]
	alpha = np.zeros(lf)
	prevAlpha = np.zeros(lf)
	T = len(selOverTime)
	#if fb:
	alphaMat = np.zeros((lf,T))
	
	prevCder0 = -1
	prevCder1 = -1
	prevCanc0 = -1
	prevCanc1 = -1
	prevCmix0 = -1
	prevCmix1 = -1
	currTrans = np.zeros((lf,lf))
	for t,st in enumerate(selOverTime):
		Nt = N[t]
		dt = dts[t]
		isTavare = int(np.sum(dts[:t]) > coalModelChangepoint)
		if prevNt != Nt or prevst != st or prevdt != dt:
			# recalculate freq transition matrix
			#for i in range(lf):
			#	row = _log_trans_prob(i,Nt,st,freqs,z_bins,z_logcdf,z_logsf,dt)
			#	currTrans[i,:] = row

			currTrans = _nstep_log_trans_prob(Nt,st,freqs,z_bins,z_logcdf,z_logsf,dt)

		Cder0 = C[0,t]
		Cder1 = C[0,t+1]
		Canc0 = C[1,t]
		Canc1 = C[1,t+1]
		Cmix0 = C[2,t]
		Cmix1 = C[2,t+1]

		if t == 0:
			alpha = -np.inf * np.ones(lf)
			alpha[dpfi] = _tavare_structured_coal(Cder0,Cder1,Canc0,Canc1,Cmix0,Cmix1,Nt,freqs[dpfi],dt,isTavare)
			#print(alpha)
			alphaMat[:,0] = alpha
		else:
			#if prevNt != Nt or prevst != st or prevdt != dt:
			#	print('Marginalizing hidden states...')
			if prevNt != Nt or prevst != st or prevdt != dt\
							or Cder0 != prevCder0 \
							or Cder1 != prevCder1 \
							or Canc0 != prevCanc0 \
							or Canc1 != prevCanc1 \
							or Cmix0 != prevCmix0 \
							or Cmix1 != prevCmix1:
				coalVec = np.zeros(lf)
				for j in range(lf):
					coalVec[j] = _tavare_structured_coal(Cder0,Cder1,Canc0,Canc1,Cmix0,Cmix1,Nt,freqs[j],dt,isTavare)

			
			for j in range(lf):

				alpha[j] = _logsumexp(prevAlpha+currTrans[:,j]) + coalVec[j]
				if np.isnan(alpha[j]):
					alpha[j] = -np.inf
			alphaMat[:,t] = alpha

		prevAlpha = alpha
		prevst = st
		prevNt = Nt
		prevdt = dt

	return alphaMat	

@njit('float64[:,:](float64[:],float64[:],float64[:,:],float64[:],int64,float64[:],int64,float64[:],float64[:],float64[:])',cache=True)
def _backward_algorithm(selOverTime,dts,C,N,dpfi,freqs,coalModelChangepoint,z_bins,z_logcdf,z_logsf):

	'''
	Moves forward in time from present to past
	'''

	prevst = -999999
	prevNt = -1
	prevdt = -1
	lf = freqs.shape[0]
	beta = np.zeros(lf)
	T = len(selOverTime)
	#if fb:
	betaMat = np.zeros((lf,T))
	prevBeta = np.NINF * np.ones(lf)
	prevBeta[0] = 0
	betaMat[:,T-1] = prevBeta

	currTrans = np.zeros((lf,lf))

	prevCder0 = -1
	prevCder1 = -1
	prevCanc0 = -1
	prevCanc1 = -1
	prevCmix0 = -1
	prevCmix1 = -1
	for tprime,st in enumerate(selOverTime[::-1]):

		tcoal = T-tprime
		t = tcoal-1
		dt = dts[t]
		#if fb:
		
		#import pdb; pdb.set_trace()
		Nt = N[tcoal-1]
		isTavare = int(np.sum(dts[:t]) > coalModelChangepoint)
		if prevNt != Nt or prevst != st or prevdt != dt:
			# recalculate freq transition matrix
			#for i in range(lf):
			#	row = _log_trans_prob(i,Nt,st,freqs,z_bins,z_logcdf,z_logsf,dt)
			#	currTrans[i,:] = row
			currTrans = _nstep_log_trans_prob(Nt,st,freqs,z_bins,z_logcdf,z_logsf,dt)


		Cder0 = C[0,tcoal-1]
		Cder1 = C[0,tcoal]
		Canc0 = C[1,tcoal-1]
		Canc1 = C[1,tcoal]
		Cmix0 = C[2,tcoal-1]
		Cmix1 = C[2,tcoal]

		if prevNt != Nt or prevst != st or prevdt != dt\
						or Cder0 != prevCder0 \
						or Cder1 != prevCder1 \
						or Canc0 != prevCanc0 \
						or Canc1 != prevCanc1 \
						or Cmix0 != prevCmix0 \
						or Cmix1 != prevCmix1:
			coalVec = np.zeros(lf)
			for j in range(lf):
				coalVec[j] = _tavare_structured_coal(Cder0,Cder1,Canc0,Canc1,Cmix0,Cmix1,Nt,freqs[j],dt,isTavare)

		for i in range(lf):
			beta[i] = _logsumexp(prevBeta + currTrans[i,:] + coalVec)
			if np.isnan(beta[i]):
				beta[i] = -np.inf

		betaMat[:,t] = beta
			
		prevCder0 = Cder0
		prevCder1 = Cder1
		prevCanc0 = Canc0
		prevCanc1 = Canc1
		prevCmix0 = Cmix0
		prevCmix1 = Cmix1
		prevBeta = beta
		prevst = st
		prevNt = Nt
		prevdt = dt

	return betaMat

@njit('float64[:](float64[::1],float64[::1],float64[:,::1],float64[::1],int64,float64[::1],int64,float64[::1],float64[::1],float64[::1])',cache=True)
def forward_backward_algorithm(selOverTime,diffTimes,C,N,dpfi,freqs,coalModelChangepoint,
									z_bins,z_logcdf,z_logsf):
	print('Running forward-backward...')
	alphaMat = _forward_algorithm(selOverTime,diffTimes,C,N,dpfi,freqs,coalModelChangepoint,z_bins,z_logcdf,z_logsf)
	betaMat = _backward_algorithm(selOverTime,diffTimes,C,N,dpfi,freqs,coalModelChangepoint,z_bins,z_logcdf,z_logsf)
	lf = betaMat.shape[0]
	T = betaMat.shape[1]
	xPostMean = np.zeros(T)
	logPostMarg = np.zeros((lf,T))
	#import pdb; pdb.set_trace()
	for t in range(T):
		logPostMarg[:,t] = alphaMat[:,t] + betaMat[:,t] - _logsumexp(alphaMat[:,t] + betaMat[:,t])
		
		xPostMean[t] = np.sum(np.exp(logPostMarg[:,t]) * freqs)
		#xPostMean[t] = freqs[logPostMarg[:,t].argmax()]
	return xPostMean

@njit('float64(float64[:],float64[:],float64[:,:],float64[:],int64,float64[:],int64,float64[:],float64[:],float64[:])',cache=True)
def _likelihood_wrapper(selOverTime,diffTimes,C,N,dpfi,freqs,coalModelChangepoint,
							z_bins,z_logcdf,z_logsf):
	alphaMat = _forward_algorithm(selOverTime,diffTimes,C,N,dpfi,freqs,coalModelChangepoint,z_bins,z_logcdf,z_logsf)
	T = alphaMat.shape[1]
	return alphaMat[0,T-1]

@njit('Tuple((float64[:],float64[:]))(	float64[:], \
										float64[:], \
										float64[:], \
										float64[:,:],\
										float64[:],\
										int64,\
										float64[:],\
										float64[:],\
										float64[:],\
										float64[:],\
										int64)',
				cache=True)
def likelihood_grid_const(S,times,diffTimes,C,N,dpfi,freqs,z_bins,z_logcdf,z_logsf,coalModelChangepoint=0):

	print('Calculating likelihood surface and MLE...')
	T = len(times)-1

	logL = np.zeros(len(S))
	for (i,s) in enumerate(S):
		selOverTime = np.ones(T)
		selOverTime *= s
		logL[i] = _likelihood_wrapper(selOverTime,diffTimes,C,N,dpfi,freqs,coalModelChangepoint,z_bins,z_logcdf,z_logsf)
		print('[',int(100*i/len(S)),'%] logL at s = ',s,':',logL[i])
	## run maximum-likelihood
	i_ml = np.argmax(logL)
	mleSelOverTime = np.ones(T)
	mleSelOverTime *= S[i_ml]

	return (logL, mleSelOverTime)

@njit('Tuple((float64[:,:],float64[:]))(float64[:],float64[:],float64[:],float64[:],float64[:,:],float64[:],int64,float64[:],float64[:],float64[:],float64[:],int64,int64)',cache=True)
def likelihood_grid_pulse(S,pulseTimes,times,diffTimes,C,N,dpfi,freqs,z_bins,z_logcdf,z_logsf,coalModelChangepoint=0,pulseLen=100):

	print('Calculating likelihood surface and MLE...')
	T = len(times)-1

	# test for a pulse at each time specified in pulseTimes
	ctimes = np.ascontiguousarray(times)
	logLp = np.zeros((len(S), len(pulseTimes)))
	for j,tb in enumerate(pulseTimes):
		ipulse0 = int(np.digitize(np.array([tb]),ctimes)[0])
		ipulse1 = int(np.digitize(np.array([tb+pulseLen]),ctimes)[0])
		for i,s in enumerate(S):
				selOverTime = np.zeros(T)
				selOverTime[ipulse0:ipulse1] = s
				logLp[i,j] = _likelihood_wrapper(selOverTime,diffTimes,C,N,dpfi,freqs,coalModelChangepoint,z_bins,z_logcdf,z_logsf)
				#print('[%d%%] logL at s = %.2e, t = %d-%d :\t%.2f'%(int(100*(i*len(pulseTimes) + j)/(len(S)*len(pulseTimes))),s,tb,tb+pulseLen,logL[i,j]))
				print('[',int(100*(i*len(pulseTimes) + j)/(len(S)*len(pulseTimes))),'%] logL at s = ',s,' , t = ',tb,':',logLp[i,j])
	## run maximum-likelihood
	iam = logLp.argmax()
	i_ml0 = iam // T
	i_ml1 = iam % T
	pt_ml = pulseTimes[i_ml1]
	ipulse0_ml = int(np.digitize(np.array([pt_ml]),ctimes)[0])
	ipulse1_ml = int(np.digitize(np.array([pt_ml+pulseLen]),ctimes)[0])
	mleSelOverTime = np.zeros(T)
	mleSelOverTime[ipulse0_ml:ipulse1_ml] = S[i_ml0]

	return (logLp, mleSelOverTime)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=
                'CLUES calculates an estimate of the log-likelihood ratio of '+
                'selection along a grid of sites (fixed derived and/or segregating)'+
                'across a set of homologous sequences.')
# mandatory inputs:
parser.add_argument('treesFile',type=str,help='A file of local trees at the site of \
						interest. Extract these using arg-summarize (a tool in ARGweaver)')
parser.add_argument('sitesFile',type=str,help='The data file used as input to sample ARGs \
						in ARGweaver (.sites format)')
parser.add_argument('popFreq',type=float,help='Population frequency of the SNP of interest.')

# options:
parser.add_argument('-q','--quiet',action='store_true')
parser.add_argument('-o','--output',dest='outFile',type=str,default=None)
parser.add_argument('-posn','--posn',type=int,help='posn of the site of interest; only \
						necessary for segsites.',default=50000)
parser.add_argument('-derivedAllele','--derivedAllele',type=str,help='specifies which \
						allele you assume to be the derived type.',default='G')
parser.add_argument('-debug','--debug',action='store_true')

#advanced options
parser.add_argument('-timeScale','--timeScale',type=float,help='Multiply the coal times \
					 	in bedFile by this factor to get in terms of generations; e.g. use \
					 	this on trees in units of 4N gens (--timeScale <4*N>)',default=1)
parser.add_argument('-ancientHap','--ancientHap',type=str,help='specifies the label of \
						the ancient haplotype that you may include in the .sites file',default='0')
parser.add_argument('-noAncientHap','--noAncientHap',action='store_true')
parser.add_argument('-thin','--thin',type=int,default=5)
parser.add_argument('--burnin',type=int,default=0)
parser.add_argument('-ssv','--ssv',action='store_true',help='Expands to test for sweeps \
						from a standing variants (SSVs) and outputs estimates of t_s, f_s -- \
						 the time and frequency at which selection begins to act on the allele')
parser.add_argument('--selCoeff',type=float,default=None)
parser.add_argument('--tSel',type=float,default=None)
parser.add_argument('-prune','--prune',type=str,default=None,help='file that specifies the \
						leaf nodes you wish to prune (ignore) from the genealogies')
parser.add_argument('--approx',type=float,help='the generation after which (bkwds in time) \
						we will use Tavares exact formula, rather than the Griffiths approx.',default=0)
###############################

args = parser.parse_args()

bedFile = args.treesFile
outFile = args.outFile

# time discretization
DISC = [[0,1],
		[500,1],
		[1000,1],
		[2000,1],
		[5000,1000],
		[10000,1000],
		[10**5,-1]]

#DISC = [[0,100],
#		[100,1000],
#		[1000,10000],
#		[10**5,-1]]

times = []
for (i,row) in enumerate(DISC[:-1]):
	t0 = row[0]
	t1 = DISC[i+1][0]
	jump = row[1]
	times.append(np.arange(t0,t1,jump))
times.append(np.array([DISC[-1][0]]))
times = np.concatenate(tuple(times))
diffTimes = np.diff(times)
T = len(diffTimes)

#griffiths -> tavare changepoint
changept = np.digitize(args.approx,times)

# frequency space discretization
d = 150
a=0.5
b=a

freqs = stats.beta.ppf(np.linspace(0,1,d),a,b)

dpfi = np.digitize(args.popFreq,freqs)

# read in global Phi(z) lookups
z_bins = np.genfromtxt('z_bins.txt')
z_logcdf = np.genfromtxt('z_logcdf.txt')
z_logsf = np.genfromtxt('z_logsf.txt')

# grid of S values
sMin=10**-5
sMax=0.1
deleterious=0
base = 1.3
steps = 20
w = (1+base**np.arange(steps))/np.sum(1+base**np.arange(steps))*(sMax - sMin)
if deleterious:
	S = np.concatenate((-1*(np.cumsum(w) + sMin)[::-1],np.array([sMin]),np.cumsum(w) + sMin))
else:
	S = np.concatenate((np.array([0]),np.cumsum(w) + sMin))

#pulse times to test for
pulseTimes=np.array([0,400,800,1200])
#pulseTimes=np.array([])


# population size trajectory
N = 10.0**4*np.ones(T)

## parse the .sites file to get the allelic states at the site of interest
if args.noAncientHap:
	ancientHap = None
else:
	ancientHap = args.ancientHap

indLists = tree_utils._derived_carriers_from_sites(args.sitesFile,
						args.posn,
						derivedAllele=args.derivedAllele,
						ancientHap=ancientHap)

derInds = indLists[0]
ancInds = indLists[1]
ancHap = indLists[2]

f = open(bedFile,'r')

lines = f.readlines()
lines = [line for line in lines if line[0] != '#' and line[0] != 'R' and line[0] != 'N'][args.burnin::args.thin]
numImportanceSamples = len(lines)

for (k,line) in enumerate(lines):
	nwk = line.rstrip().split()[-1]
	derTree =  Phylo.read(StringIO(nwk),'newick')
	ancTree = Phylo.read(StringIO(nwk),'newick')
	mixTree = Phylo.read(StringIO(nwk),'newick')

	n = len(derInds)
	m = len(ancInds)

	Cder,Canc,Cmix = tree_utils._get_branches_all_classes(derTree,ancTree,mixTree,
												derInds,ancInds,ancHap,n,m,
												args.sitesFile,times,
												timeScale=args.timeScale,prune=args.prune)
	C = [Cder,Canc,Cmix]
	for l in range(len(C)):
		C[l] += [C[l][-1] for _ in range(T - len(C[l]) + 1)]
	#C[0] = [n] + C[0]
	#C[1] = [m] + C[1]
	#C[2] = [m+1] + C[2]
	C = np.array(C)
	print(times)
	print(C[0])

	#@njit('Tuple((float64[:],float64[:]))(float64[:],float64[:],float64[:],float64[:,:],float64[:],int64,float64[:],float64[:],float64[:],float64[:],int64,int64)',cache=True)
	#def likelihood_grid_const(S,times,diffTimes,C,N,dpfi,freqs,z_bins,z_logcdf,z_logsf,coalModelChangepoint=0,pulseLen=100):

	logL, mleSelOverTime = likelihood_grid_const(
								S,
								times.astype(float),
								diffTimes.astype(float),
								C.astype(float),
								N,
								int(dpfi),
								freqs,
								z_bins,
								z_logcdf,
								z_logsf,
								0)

	xPostMean = forward_backward_algorithm(
								mleSelOverTime,
								diffTimes.astype(float),
								C.astype(float),
								N.astype(float),
								int(dpfi),
								freqs,
								0,
								z_bins,
								z_logcdf,
								z_logsf)


	prevt = -1
	tf = 2500
	for t,x in zip(times[:-1],xPostMean):
		if t // int(tf/25) > prevt // int(tf/25):
			print('t = %d :\t x = %.3f'%(t,x))
		prevt = t
		if t > tf:
			break





