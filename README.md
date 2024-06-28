## Dirty Filter Design

The goal of this repository is to explore the possibility of having a so-called "dirty" filter design which approximates the actual (floating point) filter with a fixed point filter given certain sparsity constraint. We will be focussing on 1-D FIR filters to begin while considering quantization with bit level sparsity.

Sparsity is widely used in signal representation space for indicating the number of __non-zero__ coefficients in the basis representation of a signal. A signal is $k$-sparse if that signal can be represented as a linear combination of $k$  or less coefficients. Signal operations performed over such $k$-sparse signals are efficient in terms of computation time and space complexity.

When it comes to filter design for communication, there are several blocks which desire sparse filter design. This might be due to computation time constraint or due to space constraint. So, in such a scenario we need filter taps designed so that atmost $k$ out of $m$ bits are set and remaining $(m-k)$ are unset. We have seen that such a $k$-sparse approximation is sufficient in applications such phase distortion correction or signal anti-folding. It will be handy for a DSP designer to use such approximations in order to see if the quantization criterion meets the desired filter response. 
