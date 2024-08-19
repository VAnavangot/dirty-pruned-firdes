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
    
    for i in range(len(hCoeffs)):
        print(str(nusa[i]))
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

def main_hysa(hCoeffs, sparseBudget, reqBits, boundary=None):
    # hCoeffs = [51, 24, 11, 39, 1, 27, 17, 61, 49, 18]
    # sparseBudget = 27
    print(''*40 + f'BOUNDARY:{boundary}')
    b = len(hCoeffs) if boundary==None else int(boundary)
    hIdList = np.argsort(np.abs(hCoeffs))[-b:]
    hysa = [UniformSparseAssignment(i,el) for i,el in enumerate(hCoeffs)]

    q = sparseBudget//b
    r = sparseBudget%b

    if q:
        for i in hIdList:
            _r = hysa[i](q, numBits=reqBits)
            if _r:
                r+=_r

    for s in range(r):
        getCost = [hysa[i].cost for i in range(len(hysa))]
        idx = np.argmax(np.array(getCost))
        hysa[idx](maxBitSetSize = hysa[idx].assgn+1, numBits = reqBits)

    # THERE IS A NEED FOR EXCHANGE IN THIS STEP
        #   a. Find the coeff with the minimum absolute value
        #   b. Move the Sparse Weight to the Coeff with Highest Cost
        #   c. Criteria for Exchange Needs a Review
    prevCost = 0
    for i in range(len(hCoeffs)):
            prevCost += hysa[i].cost
    for _ in range(sparseBudget):
        minAbsCoeff = [np.min(np.abs(_h._hRep)) if _h._hRep else np.Inf for _h in hysa]
        getCost = [hysa[i].cost for i in range(len(hysa))]
        fromIdx = np.argmin(minAbsCoeff)
        toIdx = np.argmax(getCost)
        print(f'fid:{fromIdx} and tid:{toIdx}')
        if (hysa[fromIdx].cost < hysa[toIdx].cost) & (fromIdx!=toIdx) & (hysa[fromIdx].assgn!=0):
            # Dummy objects created
            dummyHYSA = [UniformSparseAssignment(fromIdx, hCoeffs[fromIdx]), UniformSparseAssignment(toIdx, hCoeffs[toIdx])]
            dummyHYSA[0](maxBitSetSize = hysa[fromIdx].assgn-1, numBits = reqBits)
            dummyHYSA[1](maxBitSetSize = hysa[toIdx].assgn+1, numBits = reqBits)
            dSum = dummyHYSA[0].cost + dummyHYSA[1].cost
            # Total Cost with the Dummy Cost
            #
            p = 0
            for i in np.delete(np.arange(len(hCoeffs)), [fromIdx, toIdx]):
                p += hysa[i].cost
            p += dSum
            print(f'Current TEST Cost: {p}')

            if p < prevCost:
            # if DELTA FROM IDX cost is less than DELTA TO IDX cost then move -- but this will require some effort
                print('EXCHANGE DONE')
                hysa[fromIdx](maxBitSetSize = hysa[fromIdx].assgn-1, numBits = reqBits)
                hysa[toIdx](maxBitSetSize = hysa[toIdx].assgn+1, numBits = reqBits)
                prevCost = p
            #
            else:
                print('NO EXCHANGE')
                break
        #
        else:
            print('NO EXCHANGE')
            break


    for i in range(len(hCoeffs)):
            print(str(hysa[i]))
    #
    p = 0
    for i in range(len(hCoeffs)):
            p = UniformSparseAssignment.acc(hysa[i].numCalls, p)
    print('-'*5)
    getCost = [hysa[i].cost for i in range(len(hysa))]
    print(f'Total Cost HYSA: {np.sum(np.array(getCost))} \t Num of Calls = {p}')
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
    hSorted = np.log2(np.sort(np.abs(hInteger)))
    print(f'All data -- {hSorted}')
    hMean = np.mean(np.log2(np.abs(hInteger)))
    print(f'Mean -- {hMean}')
    hMedian = np.median(np.log2(np.abs(hInteger)))
    print(f'Median -- {hMedian}')
    tol = np.std(hSorted)
    print(f'Tolerance -- {tol}')
    Index = next(i for i, _ in enumerate(hSorted) if np.isclose(_, hMedian, rtol=0.1*tol, atol=0.0*tol))
    print(f'Closest Index: {Index}')
    plt.stairs(np.log2(np.sort(np.abs(hInteger))), fill=True)
    plt.hlines(hMean, 0,32, color='brown')
    plt.hlines(hMedian, 0,32, color='red')
    plt.show()
    # for b in range(9,sparseBudget+1):
    #     main_hysa(hInteger, sparseBudget, int(reqBits), boundary=b)
    main_hysa(hInteger, sparseBudget, int(reqBits), boundary=Index)
    main_nusa(hInteger, sparseBudget, int(reqBits))
