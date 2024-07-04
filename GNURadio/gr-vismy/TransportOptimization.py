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
        ##
        self._constraintType = sparseType

        if self._constaintType == OptimizationType.UNIFORM_SPARSE:
            bitSetSize = sparsityBudget // nTaps
            remainingBudget = sparsityBudget % nTaps
            if bitSetSize == 0:
                # Needs additional work to distribute filter representation
                pass
            else:
                self.convertor = FIRfilterIntegerCoefficients(maxBitSetSize=bitSetSize)

            if remainingBudget:
                # allot the bits uniform accross highest k EVMs
                pass
