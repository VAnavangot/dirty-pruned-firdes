#!/usr/bin/env python

from .FirTapsToInteger import FIRfilterIntegerCoefficients
from enum import IntEnum
from loguru import logger

logger.add("out.log")


class TransportOptimization(object):
    """!
    Intention  of the block is to perform transport of sparse representation to the filter coefficients with larger absolute deviation from the filter coefficients with small or low absolute deviation

    IMPORTANT ASSUMPTION: We have a nearly uniform sparse representation accross all filter coefficients initially
    """

    class OptimizationType(IntEnum):
        UNIFORM_SPARSE = 0
        OVERALL_SPARSE = 1

    def __init__(
        self, sparseType: OptimizationType, sparsityBudget: int, nTaps: int
    ) -> None:
        ## Count of the iteration step
        self.iter = 0
        ## Flag to keep track of the end of optimization
        self.stop = False
        ## What is the kind of sparsity budget?
        self._constraintType = sparseType

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
        [hHat, _normalization] = self.convertor(
            hFIR
        )  # this will do uniform sparse distribution
        # Steps to distribute the remaing budget
        # 1. Identify the absolute error of each coefficient
        _evm = ErrorVectorMetric(h, hHat)
        # 2. Add an extra representation value to the k coefficients having the largest EVM
        return [b, a]
