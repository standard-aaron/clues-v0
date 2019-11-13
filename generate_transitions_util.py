'''
Copyright (c) 2005-2017, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following disclaimer
       in the documentation and/or other materials provided with the
       distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
from numpy.core.numeric import binary_repr, asanyarray
from numpy.core.numerictypes import issubdtype


def logdot(a, b):
    max_a, max_b = np.max(a), np.max(b)
    exp_a, exp_b = a - max_a, b - max_b
    np.exp(exp_a, out=exp_a)
    np.exp(exp_b, out=exp_b)
    c = np.dot(exp_a, exp_b)
    with np.errstate(divide='ignore'):
        np.log(c, out=c)
    c += max_a + max_b
    return c


def log_matrix_power(lM, n):
    '''
    This function returns the logarithm of lM**n, where the ** denotes the
    matrix power operation. Matrix multiplication is done using logdot, a
    more numerically stable way to calculate np.log(np.dot(A,B)) for two
    matrices A and B

    It is modified from numpy.linalg's matrix_power function.
    '''
    
    lM = asanyarray(lM)
    if lM.ndim != 2 or lM.shape[0] != lM.shape[1]:
        raise ValueError("input must be a square array")
    if not issubdtype(type(n), int):
        raise TypeError("exponent must be an integer")

    result = lM
    if n <= 3:
        for _ in range(n-1):
            result=logdot(result, lM)
        return result

    # binary decomposition to reduce the number of Matrix
    # multiplications for n > 3.
    beta = binary_repr(n)
    Z, q, t = lM, 0, len(beta)
    while beta[t-q-1] == '0':
        Z = logdot(Z, Z)
        q += 1
    result = Z
    for k in range(q+1, t):
        Z = logdot(Z, Z)
        if beta[t-k-1] == '1':
            result = logdot(result, Z)
    return result

