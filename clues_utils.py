from __future__ import division
import numpy as np
from scipy.special import comb, logsumexp
from scipy import special
from scipy import stats

def coal_times(clades):
        # get number of leaf nodes

        [left,right] = clades
        lbl =  float(left.branch_length)
        rbl =  float(right.branch_length)

        #print lbl, rbl
        if len(left.clades) == 0 and len(right.clades) == 0:
            return [rbl]

        elif len(left.clades) == 0:
            right_times = coal_times(right.clades)
            return [lbl] + right_times

        elif len(right.clades) == 0:
            left_times = coal_times(left.clades)
            return [rbl] + left_times

        else:
            left_times = coal_times(left)
            right_times = coal_times(right)

            if lbl < rbl:
                return [lbl + left_times[0]] + left_times + right_times
            else:
                return [rbl + right_times[0]] + left_times + right_times

def branch_counts(coalTimes, timePts, eps=1):
	## return number of lineages at each time point
	n = len(coalTimes) + 1
	C = [n]
	
	for tp in timePts:
		i = 0
		for (j,ct) in enumerate(coalTimes[i:]):
			if ct >= tp + eps:
				i += j
				C.append( n-j )
				break
	return C


def log_falling_factorial(a,k):
	if k >= 1:
		return np.sum(np.log(np.arange(a-k+1,a+1)))
	else:
		return 0

def log_rising_factorial(a,k):
	if k >= 1:
		return np.sum(np.log(np.arange(a,a+k)))
	else:
		return 0


def eta(alpha,beta):
    return alpha * beta / (alpha * (np.exp(beta) - 1) + beta * np.exp(beta))

def griffiths_log_prob_coal_counts(a,b,t,N):
    t = t/(2*N)
    n = a
    alpha = 1/2*n*t
    beta = -1/2*t
    h = eta(alpha,beta)
    mu = 2 * h * t**(-1)
    var = 2*h*t**(-1) * (h + beta)**2
    var *= (1 + h/(h+beta) - h/alpha - h/(alpha + beta) - 2*h)
    var *= beta**-2
    std = np.sqrt(var)
    return stats.norm.logpdf(b,mu,std) - logsumexp(stats.norm.logpdf(np.arange(1,a+1),mu,std))


def tavare_log_prob_coal_counts(a, b, t, n):

    lnC1 = 0.0
    lnC2=0.0
    C3=1.0

    for y in range(0,b):
        lnC1 += np.log((b+y)*(a-y)/(a+y))

    s = -b*(b-1)*t/4.0/n

    for k in range(b+1,a+1):
        
        k1 = k - 1
        lnC2 += np.log((b+k1)*(a-k1)/(a+k1)/(k-b))
        C3 *= -1.0
        val = -k*k1*t/4.0/n + lnC2 + np.log((2.0*k-1)/(k1+b))
        if True:
            loga = s
            logc = val
            if (logc > loga):
                tmp = logc
                logc = loga
                loga = tmp
            s = loga + np.log(1.0 + C3*np.exp(logc - loga)) 
    for i in range(2,b+1):
        s -= np.log(i)

    return s + lnC1

def tavare_structured_coal(Cder0,Cder1,Canc0,Canc1,Cmix0,Cmix1,Nnow,x,t,isTavare):
        '''
        Cder0: number of of lineages at start of the epoch (backwards in time)
        x: starting frequency (backwards in time)
        '''
        if isTavare:
                log_prob_coal_counts = tavare_log_prob_coal_counts
        else:
                log_prob_coal_counts = griffiths_log_prob_coal_counts
	
        logCondProb = 0
        if x == 0:
                #print('MIXED at time %d'%(t))
                #print(Cmix0,Cmix1)
                # assume mutation has occurred
                # just consider the Cmix process
                if Cder1 > 1:
                        #print(t,x,'-inf: mutation before coal!')
                        return -np.inf
                if Cmix0 == 1:
                        #print('0: everything has coaled.')
                        return 0
                b = Cmix1
                a = Cmix0

                logCondProb += log_prob_coal_counts(a,b,t,Nnow)
        else:
                #print('STRUCTURED  at time %d'%(t))
                if Cmix1 == Canc1 and Cder0 > 0:
                        #print(t,x,'-inf: structured after mut!')
                        return -np.inf
                # trying this hack to get rid of neutral traj bias
                #if w == 0:
                #       Canc0 = Cmix0
                #       Canc1 = Cmix1

                for (N,a,b) in zip([Nnow*x,Nnow*(1-x)],[Cder0,Canc0],[Cder1,Canc1]):
                        if a == 1 or a == 0:
                                #print(int(N),a,b,logCondProb)
                                continue
                        logCondProb += log_prob_coal_counts(a,b,t,N)
        if np.isnan(logCondProb) or np.isinf(logCondProb):
                logCondProb = -np.inf
        return logCondProb

def tavare_conditional_likelihood(b,a,Nnow,t,eps=50):
		## if popsize < eps, we assume mutation occurred during this epoch

        if a == 1:
        	return 0

        if Nnow < eps and b > 1:
        	return -np.inf

        k = np.arange(b,a+1)
        logSummands = -comb(k,2)/(2*Nnow) * t 
        logSummands += np.log(2*k-1) - np.log(k+b-1) \
                        - np.sum(np.log(np.arange(2,b+1))) - np.array([np.sum(np.log(np.arange(2,kprime-b+1))) for kprime in k]) \
                        + np.array([np.sum([np.log(b + y) + np.log(a - y) - np.log(a + y) for y in np.arange(0,kprime)]) for kprime in k])
        logCondProb = logsumexp( logSummands, b=(1-2*np.mod(k-b,2)) )

        return logCondProb


def derived_carriers_from_sites(sitesFile,posn,derivedAllele='G',ancientHap=None,invar=False):
	'''
	Takes the sitesFile
	Returns a list of individuals (labels in 
	the header of sitesFile) who carry derived allele
	'''

	f = open(sitesFile,'r')
	lines = f.readlines()

	headerLine = lines[0]
	inds = headerLine.split()[1:]

	if invar:
		if ancientHap == None:
			return [inds,[],[]]
		else:
			derInds = [ind for ind in inds if ind != ancientHap]
			return [derInds,[],[ancientHap]]	

	for line in lines:
		if line[0] == '#' or line[0] == 'N' or line[0] == 'R':
			continue
		cols = line.rstrip().split()
		thisPosn = int(cols[-2])
		alleles = cols[-1]
		if thisPosn < posn:
			continue
		elif thisPosn == posn:
			if ancientHap != None:
				idxsDerived = [i for (i,x) in enumerate(alleles) if x == derivedAllele and inds[i] != ancientHap]
				indsDerived = [inds[i] for i in idxsDerived]
				indsAnc = [ind for (i,ind) in enumerate(inds) if i not in idxsDerived and inds[i] != ancientHap]
				return [indsDerived,indsAnc,[ancientHap]]
			else:
				idxsDerived = [i for (i,x) in enumerate(alleles) if x == derivedAllele]
				indsDerived = [inds[i] for i in idxsDerived]
				indsAnc = [ind for (i,ind) in enumerate(inds) if i not in idxsDerived]

				return [indsDerived,indsAnc,[]]
		else:
			inds.remove(ancientHap)	
			return [[],inds,[ancientHap]]
			#raise ValueError('Specified posn not specified in sitesFile')

