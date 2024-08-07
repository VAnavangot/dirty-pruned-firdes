#!/usr/bin/env python

import itertools as it

import numpy as np
from gnuradio.filter import firdes as fir
from loguru import logger


class FIRfilterIntegerCoefficients(object):
    """!
    @brief The purpose of this class is to convert the real values in FIR filter to integer quantized representation with at most MAX_BIT_SET_SIZE bits. That is, every real-valued coefficient is converted to its sparse form with atmost MAX_BIT_SET_SIZE. For example, if  MAX_BIT_SET_SIZE=2 the integer representation is an addition (or subtraction) of two integers which are powers of 2.
    """

    def __init__(self, maxBitSetSize:int =2, maxBitWidth:int=20)-> None:
        """!
        @brief Filter converter initializer

        @param maxBitSetSize Number of elements in the sparse set
        @param maxBitWidth Total bit width of the coefficient representation
        """
        self.maxBitSetSize = maxBitSetSize
        self.maxBitWidth = maxBitWidth
    
    @property
    def maxBitSetSize(self):
        return self.maxBitSetSize
    
    @staticmethod
    def convertTapsToInteger(hFIR, maxBitWidth:int=20):
        requireWidth = -np.floor(np.log2(np.min(np.abs(hFIR))))
        #
        if requireWidth > maxBitWidth:
            raise ValueError("Required Fixed Point Width is greater than Max Width")
        #
        h = np.round(hFIR / 2 ** (-requireWidth))
        return (h, requireWidth)


    def __call__(self, hFIR):
        """!
        Convert the FIR filter coefficients to its integer representation, making the filter computations as left shift operations, followed by a final right shift operation

        The function returns the FIR coefficients as Numerator-> Vector and Denominator -> Scalar

        @param hFIR List/Array of FIR filter coefficients
        """
        h, requireWidth = self.convertTapsToInteger(hFIR, maxBitWidth=self.maxBitWidth) 
        numBits = int(requireWidth)

        self.searchSet = searchSet = [2**i for i in range(numBits + 1)] + [
            -(2**i) for i in range(numBits + 1)
        ]

        hTruncated = []

        for el in h:
            s = self.MinSubsetOfTargetSum(
                searchSet, 2 * (numBits + 1), el, self.maxBitSetSize
            )
            print(s)
            hTruncated.append(sum(s))  # sparse quantized coefficient

        denominator = np.sum(hTruncated)
        denominatorShift = np.round(np.log2(abs(denominator)))
        logger.info(f"Denominator Shift={denominatorShift}")
        # TODO: return an integer array with the index location of the sparse factors
        return [hTruncated, np.sign(denominator) * 2**denominatorShift]

    @staticmethod
    def ErrorVectorMetric(h:list[int], hHat:list[int]):
        assert len(h) == len(
            hHat
        ), "Filter lengths should be identical to compute error metric"
        return np.abs(h - hHat)

    @staticmethod
    def MinSubsetOfTargetSum(set, n, val, maxBitSetSize):
        """
        Core logic to find the best subset representing the value VAL

        @param set list which is the search space for the sparse representation
        @param n the size of the search space list
        @param val the value of the integer to be converted
        @param maxBitSetSize the sparsity requirement of the value
        """
        count = 0
        minSubSet = []
        sizeSSET = n
        # identify the shift combinations in the sparse form
        for i in range(1, n):
            foundFlag = 0
            sparseSet = it.combinations(range(n), i)
            for bits in sparseSet:
                subSet = [set[i] for i in bits]
                if sum(subSet) == val:
                    count += 1
                    foundFlag = 1
                    minSubSet = subSet
                    break
            if foundFlag == 1:
                break

        # it means no subset is found with given sum
        if count == 0:
            print("No subset is found")

        else:
            # print(np.sum(minSubSet), "-> \t", end="")
            if len(minSubSet) <= maxBitSetSize:
                return minSubSet
            else:
                restrictedSubSet = []
                # Restrict the MIN_SUBSET to size MAX_BIT_SET_SIZE
                for m in range(maxBitSetSize):
                    indexElement = np.argmax(np.abs(minSubSet))
                    maxElement = minSubSet[indexElement]
                    minSubSet.remove(maxElement)
                    restrictedSubSet.append(maxElement)
                return restrictedSubSet


def main():
    g = 1
    t = 20
    fs = 250
    factor = 4
    hFIR = fir.low_pass(
        gain=g, sampling_freq=fs, cutoff_freq=fs / factor - t / 2, transition_width=t
    )
    hInteger, reqBits = FIRfilterIntegerCoefficients.convertTapsToInteger(hFIR)
    convertor = FIRfilterIntegerCoefficients()
    [b, a] = convertor(hFIR)
    evm = FIRfilterIntegerCoefficients.ErrorVectorMetric(b,hInteger)
    logger.info(f'Error Vector: {evm}')
    for hEll in range(len(hFIR)):
        print(f"{hFIR[hEll]} -> {b[hEll]/a}")
    # print(f'Filter with integer coefficients:{b}')


if __name__ == "__main__":
    main()
