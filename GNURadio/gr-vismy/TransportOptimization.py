#!/usr/bin/env python

from .FirTapsToInteger import FIRfilterIntegerCoefficients
from enum import IntEnum
from loguru import logger

logger.add("out.log")


class TransportOptimization(object):
    """!
    Intention  of the block is to perform transport of sparse representation to the filter coefficients with larger absolute deviation from the filter coefficients with small or low absolute deviation

    IMPORTANT ASSUMPTION: We have a nearly uniform sparse representation accross all filter coefficients initially
    TODO: The implementation shall be ported to Rust for faster execution
    """

    class OptimizationType(IntEnum):
        UNIFORM_SPARSE = 0
        OVERALL_SPARSE = 1

    def __init__(
        self, sparseType: OptimizationType, sparsityBudget: int, nTaps: int, h
    ) -> None:
        ## Count of the iteration step
        self.iter = 0
        ## Flag to keep track of the end of optimization
        self.stop = False
        ## What is the kind of sparsity budget?
        self._constraintType = sparseType
        ## What are the filter taps?
        self.h, self.requireWidth  = FIRfilterIntegerCoefficients.convertTapsToInteger(h)

        bitSetSize = sparsityBudget // nTaps
        self._remainingBudget = sparsityBudget % nTaps
        if bitSetSize == 0:
            assert remainingBudget > 0, "Sparsity budget should be strictly positive."
            # Needs additional work to distribute filter representation
        else:
            self.convertor = FIRfilterIntegerCoefficients(maxBitSetSize=bitSetSize)

        """!
        if remainingBudget:
            # Allot the bits uniform accross highest k EVMs where k=remainingBudget and maxBitSetSize=1
        """

    def __call__(self, hFIR):
        [hHat, _denNorm] = self.convertor(
            hFIR
        )  # this will do uniform sparse distribution
        # STEPS TO DISTRIBUTE THE REMAINING BUDGET:
        # 1) Identify the absolute error of each coefficient
        _evm = ErrorVectorMetric(self.h, hHat)
        # 2) Add an extra representation value to the k coefficients having the largest EVM
        # Sorted list of filter coefficients are needed; but from where[???] and what is k?
        # k is same as self._remainingBudget
        remainingIndex = np.argsort(_evm)[-1::-1][:self._remainingBudget]
        # 3) Add the additional representation to the existing coefficient
        for i in remainingIndex:
            hHat[i] = FIRfilterIntegerCoefficients.MinSubsetOfTargetSum(self.h, 2*(int(self.requireWidth)+1), hFIR[i], self.convertor.maxBitSetSize)