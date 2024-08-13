#!/usr/bin/env python

from TransportOptimization import UniformSparseAssignment, NonUniformSparseAssignment, MinSubsetNearTargetSum
from FirTapsToInteger import FIRfilterIntegerCoefficients
import numpy as np
from gnuradio.filter import firdes as fir
from matplotlib import pyplot as plt

def main_nusa(hCoeffs, sparseBudget, reqBits):
    # hCoeffs = [51, 24, 11, 39, 1, 27, 17, 61, 49, 18]
    # sparseBudget = 27
    nusa = [NonUniformSparseAssignment(i,el) for i,el in enumerate(hCoeffs)]

    for s in range(sparseBudget):
        getCost = [nusa[i].cost for i in range(len(nusa))]
        idx = np.argmax(np.array(getCost))
        nusa[idx](numBits = reqBits)
    
    # for i in range(len(hCoeffs)):
    #     print(str(nusa[i]))
    p = 0
    for i in range(len(hCoeffs)):
            p = NonUniformSparseAssignment.acc(nusa[i].numCalls, p)

    print('-'*5)
    getCost = [nusa[i].cost for i in range(len(nusa))]
    print(f'Total Cost NUSA: {np.sum(np.array(getCost))} \t Num of Calls = {p}')
    print('-'*5)

def main_unsa(hCoeffs, sparseBudget, reqBits, boundary=None):
    # hCoeffs = [51, 24, 11, 39, 1, 27, 17, 61, 49, 18]
    # sparseBudget = 27
    b = len(hCoeffs) if boundary==None else int(boundary)
    hIdList = np.argsort(np.abs(hCoeffs))[-b:]
    unsa = [UniformSparseAssignment(i,el) for i,el in enumerate(hCoeffs)]

    q = sparseBudget//b
    r = sparseBudget%b

    if q:
        for i in hIdList:
            unsa[i](q, numBits=reqBits)
    
    getCost = [unsa[i].cost for i in hIdList]
    if r:
        idxList = np.argsort(np.array(getCost))[-r:]
        #
        for idx in idxList:
            unsa[idx](q+1)
    
    # for i in range(len(hCoeffs)):
    #         print(str(unsa[i]))
    #
    p = 0
    for i in range(len(hCoeffs)):
            p = UniformSparseAssignment.acc(unsa[i].numCalls, p)
    print('-'*5)
    getCost = [unsa[i].cost for i in range(len(unsa))]
    print(f'Total Cost UNSA: {np.sum(np.array(getCost))} \t Num of Calls = {p}')
    print('-'*5)

if __name__ == "__main__":
    g = 1
    t = 20
    fs = 250
    factor = 4
    hFIR = fir.low_pass(
        gain=g, sampling_freq=fs, cutoff_freq=fs / factor - t / 2, transition_width=t
    )
    hInteger, reqBits = FIRfilterIntegerCoefficients.convertTapsToInteger(hFIR)
    sparseBudget = 27
    print(f'Bit Requirement: {int(reqBits)}')
    # hInteger = [51, 21, 9, 39, 42, 17]
    # sparseBudget = 6
    # reqBits= 6
    plt.stairs(np.log2(np.sort(np.abs(hInteger))), fill=True)
    plt.hlines(np.mean(np.log2(np.abs(hInteger))), 0,32, color='brown')
    plt.hlines(np.median(np.log2(np.abs(hInteger))), 0,32, color='red')
    plt.show()
    for b in range(1,sparseBudget+1):
        main_unsa(hInteger, sparseBudget, int(reqBits), boundary=b)
    main_nusa(hInteger, sparseBudget, int(reqBits))
