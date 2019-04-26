import warnings
import argparse
import hmm
import tree_utils
import numpy as np
import scipy.stats as stats
import time
import glob
from Bio import Phylo
from io import StringIO

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
						we will use Tavares exact formula, rather than the Griffiths approx.',default=500)
###############################

args = parser.parse_args()

bedFile = args.treesFile
outFile = args.outFile

# time discretization
DISC = [[0,1],
		[1000,2],
		[2000,5],
		[5000,10],
		[10000,100],
		[10**5,-1]]

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
d=50
a=0.8
b=0.8
freqs = stats.beta.ppf(np.linspace(0,1,d),a,b)
dpfi = np.digitize(args.popFreq,freqs)

# read in global Phi(z) lookups
z_bins = np.genfromtxt('z_bins.txt')
z_logcdf = np.genfromtxt('z_logcdf.txt')
z_logsf = np.genfromtxt('z_logsf.txt')

# grid of S values
sMin=10**-9
sMax=0.5
deleterious=0
base = 1.2
steps = 20
w = (1+base**np.arange(steps))/np.sum(1+base**np.arange(steps))*(sMax - sMin)
if deleterious:
	S = np.concatenate((-1*(np.cumsum(w) + sMin)[::-1],np.array([sMin]),np.cumsum(w) + sMin))
else:
	S = np.concatenate((np.array([sMin]),np.cumsum(w) + sMin))

#pulse times to test for
#pulseTimes=np.array([0,200,400,600,800,1000,1200,1400,1600,1800,2000])
pulseTimes=np.array([])


# population size trajectory
N = 10**4*np.ones(T)

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
	C[0] = [n] + C[0]
	C[1] = [m] + C[1]
	C[2] = [m+1] + C[2]
	C = np.array(C)

	logL, mleSelOverTime = hmm.likelihood_grid0(S,pulseTimes,times,C,N,dpfi,freqs,z_bins,z_logcdf,z_logsf)


