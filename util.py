from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import str
from builtins import range
from past.utils import old_div
import argparse
import multiprocessing as mp
import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
from scipy.special import logit, expit
import logging
import resource


class MemoryFilter(logging.Filter):
    '''
    for use in debugging memory
    '''
    def filter(self, record):
        record.memusg = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss +
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        return True


def get_debug_func(debug_opt):
    if debug_opt:
        def debug(s):
            print('@ ' + s)
    else:
        def debug(s):
            return
    return debug

def length_parser_str(x):
    if x:
        return str(x)
    else:
        return None
        
def positive_int(val):
    val = int(val)
    if val <= 0:
        raise argparse.ArgumentError("invalid positive integer")
    return val

def positive_float(val):
    val = float(val)
    if val <= 0:
        raise argparse.ArgumentError("invalid positive float")
    return val

def probability(val):
    val = float(val)
    if val < 0 or val > 1:
        raise argparse.ArgumentError("invalid probability")
    return val

def nonneg_int(val):
    val = int(val)
    if val < 0:
        raise argparse.ArgumentTypeError(
                "invalid non-negative integer: {}".format(val))
    return val

def nonneg_float(val):
    val = float(val)
    if val < 0:
        raise argparse.ArgumentTypeError(
                "invalid non-negative integer: {}".format(val))
    return val
    
def mp_approx_fprime(x, inq, outq, eps = 1e-8):
    num_params = x.shape[0]
    xs = []
    for i in range(num_params):
        xp = x.copy()
        xp[i] += eps
        inq.put((i, xp))
    inq.put((-1, x))

    outputs = []
    for i in range(num_params+1):
        outputs.append(outq.get())
    outputs.sort()

    fxps = [fxp for idx, fxp in outputs[1:]]
    fx = outputs[0][1]

    fprimes = []
    for i in range(num_params):
        fprime = old_div((fxps[i]-fx), eps)
        fprimes.append(fprime)
    return np.array(fprimes)
