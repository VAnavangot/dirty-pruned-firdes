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
        self.hApp = None
        ## Sparse Representation of the Filter Coefficient
        self.hRep = []
        ## Number of shift assignements
        self.assgn = 0
        ## Absolute Cost after the approximation; it is hCoeff to start with
        self._cost = np.abs(hCoeff)

    def __getitem__(self):
        pass

    def __getattr__(self):
        pass

    def __call__(self, numBits=6):
        maxBitSetSize = self.assgn + 1
        searchSet = [2**i for i in range(numBits + 1)] +[0] + [-(2**i) for i in range(numBits + 1)]
        self.hRep = self.MinSubsetOfTargetSum(searchSet, 2*(numBits+1), self.h, maxBitSetSize)
        self.hApp = np.sum(np.array(self.hRep))
        self.assgn = maxBitSetSize

    @property
    def cost(self):
        if self.hApp:
            self._cost = np.abs(self.h - self.hApp)
        return self._cost

    def __contains__(self):
        pass

    def __array_finalize__(self):
        pass

    def __exit__(self):
        pass

    def __str__(self):
        return f'|||{self.name} --- Usage: {self.assgn} | hApp: {self.hApp} | Rep: {self.hRep} | Cost: {self.cost}'

    @staticmethod
    def MinSubsetOfTargetSum(set, n, val, maxBitSetSize):
        count = 0
        minSubSet = []
        sizeSSET = n
        # identify the shift combinations in the sparse form
        foundFlag = 0
        sparseSet = it.combinations(range(n), maxBitSetSize)
        costVal = val
        mIdx = []
        for bits in sparseSet:
            subSet = [set[i] for i in bits]
            if (costVal > np.abs(sum(subSet)-val)):
                costVal = np.abs(sum(subSet)-val)
                minSubSet = subSet
        return minSubSet



def main():
    hCoeffs = [53, 21, 9]
    sparseBudget = 9
    nusa = [NonUniformSparseAssignment(i,el) for i,el in enumerate(hCoeffs)]
    # hh, nn =  convertTapsToInteger(hCoeffs/np.sum(np.array(hCoeffs)))
    # print(f'IntegerTaps: {hh}, requiredWidth: {nn}')

    for s in range(sparseBudget):
        getCost = [nusa[i].cost for i in range(len(nusa))]
        print(getCost)
        idx = np.argmax(np.array(getCost))
        print(f'index = {idx}')
        nusa[idx]()
        for i in range(len(hCoeffs)):
            print(str(nusa[i]))



if __name__ == "__main__":
    main()