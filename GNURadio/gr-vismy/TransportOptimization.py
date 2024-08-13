#!/usr/bin/env python
#
# SPDX-License-Identifier: GPL-3.0
#
# Project Title: Filter Coefficient Approximation using Sparse Assignment
# Author: Vijay Anavangot <a.vijay.2014@ieee.org>
#

from FirTapsToInteger import FIRfilterIntegerCoefficients
import numpy as np
import itertools as it


''' 
Sparse assignment of Filter coefficients aims to perform approximate filtering
operation using only shift operations and additions (or subtractions). Given an
integer sparse shift budget, we intent to find the filter coefficients and
their sparse shifts which "closely" approximate the actual filtering operation.
The degree of closeness of the filter coefficients is measured by the absolute
norm or l1 norm.

A vanila version for finding the filter coefficient and there shifting weights
is using OSAT. OSAT stands for One-Step-At-A-Time optimization and the idea is
to perform sparse assignment by approximating coefficients one at a time. At
each epoch the coefficient to be approximated is determined by fieldwise cost
which has the maximum absolute value.

The algorithm runs in as many steps as the sparsity budget.
───────────────────────────────────────────────────────────────────
             h[NTAPS-1:0]: Actual Filter                                                    
             hApp[NTAPS-1:0]: Approximate Filter
             hRep[NTAPS-1:0]: Shift operation representation of hApp                                            
                End Goal: minimize ||hApp - h||                                       
───────────────────────────────────────────────────────────────────
'''

class UniformSparseAssignment(object):
    '''!
            Individual coefficients have a sparsity constraint
    '''
    def __init__(self, id, hCoeff):
        ##
        self.name = f'coeff_{id}'
        ## Exact Filter Coefficient
        self.h = hCoeff
        ## Approximate Filter Coefficient
        self._hApp = None
        ## Sparse Representation of the Filter Coefficient
        self._hRep = []
        ## Number of shift assignements
        self.assgn = 0
        ## Absolute Cost after the approximation; it is hCoeff to start with
        self._cost = np.abs(hCoeff)
        ## Number of calls for optimization
        self._numCalls = 0

    def __call__(self, maxBitSetSize, numBits=6):
        self._numCalls += 1
        searchSet = [2**i for i in range(numBits + 1)] + [-(2**i) for i in range(numBits + 1)]
        self._hRep = MinSubsetNearTargetSum(searchSet, 2*(numBits+1), self.h, maxBitSetSize)
        self._hApp = np.sum(np.array(self._hRep))
        self.assgn = maxBitSetSize

    @property
    def cost(self):
        if self._hApp:
            self._cost = np.abs(self.h - self._hApp)
        return self._cost
    
    @property
    def hApp(self):
        return str(self._hApp) if self._hApp else 'None'

    @property
    def hRep(self):
        return ' '.join([f'{str(x):6s}' for x in self._hRep])

    @property
    def numCalls(self):
        return self._numCalls

    @staticmethod
    def acc(x, p):
        return x + p

    def __str__(self):
        return f'{self.name:8s}: {str(self.h):10s} --- Usage: {str(self.assgn):3s} | hApp: {self.hApp:6s} | Rep: {self.hRep:25s} | Cost: {self.cost}'


class NonUniformSparseAssignment(object):
    '''!
            Given sparsity budget is distributed across the filter taps to minimize the overall cost function
    '''
    def __init__(self, id, hCoeff):
        ##
        self.name = f'coeff_{id}'
        ## Exact Filter Coefficient
        self.h = hCoeff
        ## Approximate Filter Coefficient
        self._hApp = None
        ## Sparse Representation of the Filter Coefficient
        self._hRep = []
        ## Number of shift assignements
        self.assgn = 0
        ## Absolute Cost after the approximation; it is hCoeff to start with
        self._cost = np.abs(hCoeff)
        ## Number of calls for optimization
        self._numCalls = 0

    def __getitem__(self):
        pass

    def __getattr__(self):
        pass

    def __call__(self, numBits=6):
        self._numCalls += 1
        maxBitSetSize = self.assgn + 1
        searchSet = [2**i for i in range(numBits + 1)] + [-(2**i) for i in range(numBits + 1)]
        self._hRep = MinSubsetNearTargetSum(searchSet, 2*(numBits+1), self.h, maxBitSetSize)
        self._hApp = np.sum(np.array(self._hRep))
        self.assgn = maxBitSetSize

    @property
    def cost(self):
        if self._hApp:
            self._cost = np.abs(self.h - self._hApp)
        return self._cost

    @property
    def hApp(self):
        return str(self._hApp) if self._hApp else 'None'

    @property
    def hRep(self):
        return ' '.join([f'{str(x):6s}' for x in self._hRep])

    @property
    def numCalls(self):
        return self._numCalls

    @staticmethod
    def acc(x, p):
        return x + p


    def __contains__(self):
        pass

    def __array_finalize__(self):
        pass

    def __exit__(self):
        pass

    def __str__(self):
        return f'{self.name:8s}: {str(self.h):10s} --- Usage: {str(self.assgn):3s} | hApp: {self.hApp:6s} | Rep: {self.hRep:25s} | Cost: {self.cost}'

def MinSubsetNearTargetSum(set, n, val, maxBitSetSize):
    count = 0
    minSubSet = []
    # identify the shift combinations in the sparse form
    foundFlag = 0
    sparseSet = it.combinations(range(n), maxBitSetSize)
    costVal = np.Inf
    mIdx = []
    for bits in sparseSet:
        subSet = [set[i] for i in bits]
        if (costVal > np.abs(sum(subSet)-val)):
            costVal = np.abs(sum(subSet)-val)
            minSubSet = subSet
    return minSubSet

