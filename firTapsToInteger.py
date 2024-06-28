#!/usr/bin/env python

import numpy as np
from gnuradio.filter import firdes as fir
from gnuradio.astrome.FirTapsToInteger import FIRfilterIntegerCoefficients
from matplotlib import pyplot as plt
from scipy.signal import iirdesign, lfilter


def main():
    fs = 250
    factor = 4
    PI = np.pi
    fo = fs/factor/2
    #
    t = np.arange(0,1,1/fs)
    x = np.cos(2*PI*fo*t) + 1j*np.sin(2*PI*fo*t) #np.sin(2*PI*fo/4*t)
    y = np.repeat(x,factor)
    #
    np.random.seed(1)
    [b,a] = iirdesign(0.95,0.99,1,60,ftype='butter')
    ty = lfilter(b, a, y)
    ny = ty+0.0*(np.random.randn(len(y))+1j*np.random.randn(len(y)))
    #
    g = 1
    tr = 20#/(factor/2)
    hFIR = fir.low_pass(gain=g, sampling_freq=fs, cutoff_freq=fs/factor-tr/factor, transition_width=tr)
    convertor = FIRfilterIntegerCoefficients(maxBitSetSize=2)
    [bb, aa] = convertor(hFIR)
    h = np.array(bb)/aa
    print(h)
    #
    fy = np.convolve(h, ny, mode='same')[:]
    FY = (np.abs(np.fft.fft(fy)))
    H = (np.abs(np.fft.fft(h)))
    freqfy = np.fft.fftfreq(len(FY), 1/fs)
    freqh = np.fft.fftfreq(len(H), 1/fs)
    # plt.plot(freqfy, FY)
    # plt.plot(freqh, H)
    # plt.title('Freq Response after LP filtering')
    # plt.xlabel('Frequency[Hz]')
    # plt.ylabel('Magnitude')

    # print(f'Filter with integer coefficients:{b}')
    ha = np.ones([factor])/factor
    ry = 1.0282951*np.convolve(ha,fy)[factor-1::factor]
    x = x[:]

    plt.figure()
    plt.plot(np.arange(len(ry)), ry.real, '--bo', alpha=0.3, label= 'Recovered Signal' )
    plt.plot(np.arange(len(x)), x.real, '--mo',alpha=0.3, label='Original Signal')
    plt.plot(np.arange(len(x)), abs(ry.real-x.real), '--rs', markersize=3, label= 'Error-Real')
    plt.title('Signal Averaging at Rx [REAL]')
    plt.text(x=40, y = -0.5, s=f'Mean Abs Error:{np.mean(abs(ry.real-x.real)):2.5f}')
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(np.arange(len(ry)), ry.imag, '--ro', alpha=0.3, label= 'Recovered Signal' )
    plt.plot(np.arange(len(x)), x.imag, '--mo',alpha=0.3, label='Original Signal')
    plt.plot(np.arange(len(x)), abs(ry.imag-x.imag), '--rs', markersize=3, label= 'Error-Imag')
    plt.title('Signal Averaging at Rx [IMAG]')
    plt.text(x=40, y = -0.5, s=f'Mean Abs Error:{np.mean(abs(ry.imag-x.imag)):2.5f}')
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
