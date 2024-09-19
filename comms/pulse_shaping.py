#! /usr/bin/env python3

# File: pulse_shaping.py - pulse_shaping and matched filtering;
#

r"""
# Pulse shaping and spectra

This module contains functions related to pulse shaping and the computation of spectra
of digitally modulated signals

## Pulse shapes

* `sine_squared_pulse`: generate sine-squared pulse $p(t) = A \sin^2(\pi t/T)$ for $0 \leq t < T$
* `rect_pulse`: generate rectangular pulse $p(t) = A$ for $0 \leq t < T$
* `half_sine_pulse`: generate half-sine pulse $p(t) = A \sin(\pi t/T)$ for $0 \leq t < T$
* `rc_pulse`: generate a raised cosine pulse
* `srrc_pulse`: generate a square-root raised cosine pulse

## Pulse shaping

* `pulse_shape`: function perform upsampling by factor `fsT` followed by filtering with specified pulse

## Bandwidth computation

* `bandwidth_3dB`: estimate 3dB bandwidth from samples of power spectra density
* `bandwidth_zz`: estimate zero-to-zero bandwidth from samples of power spectra density
* `bandwidth_containment`: estimate alpha-containment bandwidth from samples of power spectra density
"""


import numpy as np

#
# Various pulse shapes
#
# Note: pulses may be scaled to produce samples of a continuous-time pulse or
# like a discrete-time pulse.
# For a continuous-time pulse, the integral over the square of the pulse equals 1.
# For a discrete-time pulse,
# the sum over the square of the samples equals 1


def sine_squared_pulse(fsT, fs=1):
    r"""synthesize a sine squared pulse

    Inputs:
    -------
    fsT: samples per symbol period
    fs: if a sampling rate (other than fs=1) is specified, then pulses will be scaled like samples of a continuous-time pulse, i.e., such that \int_0^T p^2(t) dt = 1. Otherwise, scaling appropriate for a discrete-time pulse, i.e., \sum_n |p[n]|^2 = 1, is used. (default: fs=1)

    Returns:
    --------
    pulse of length fsT samples
    """
    nn = np.arange(fsT)
    pp = np.sqrt(8 * fs / (3 * fsT)) * np.sin(np.pi * nn / fsT) ** 2

    return pp


def rect_pulse(fsT, fs=1):
    r"""synthesize a rectangular pulse

    Inputs:
    -------
    fsT: samples per symbol period
    fs: if a sampling rate (other than fs=1) is specified, then pulses will be scaled like samples of a continuous-time pulse, i.e., such that \int_0^T p^2(t) dt = 1. Otherwise, scaling appropriate for a discrete-time pulse, i.e., \sum_n |p[n]|^2 = 1, is used. (default: fs=1)

    Returns:
    --------
    pulse of length fsT samples
    """
    pp = np.sqrt(fs / (fsT)) * np.ones(fsT)

    return pp


def half_sine_pulse(fsT, fs=1):
    r"""synthesize a half-sine pulse

    Inputs:
    -------
    fsT: samples per symbol period
    fs: if a sampling rate (other than fs=1) is specified, then pulses will be scaled like samples of a continuous-time pulse, i.e., such that \int_0^T p^2(t) dt = 1. Otherwise, scaling appropriate for a discrete-time pulse, i.e., \sum_n |p[n]|^2 = 1, is used. (default: fs=1)

    Returns:
    --------
    pulse of length fsT samples
    """
    nn = np.arange(fsT)
    pp = np.sqrt(2 * fs / (fsT)) * np.sin(np.pi * nn / fsT)

    return pp


def rc_pulse(a, fsT, N=5):
    r"""Construct a raised cosine pulse

    Inputs:
    -------
    a: roll-off factor
    fsT: number of samples per symbol period
    N: length of pulse in symbol periods; pulse ranges for -N \leq t/T \leq N (default: 5).

    Returns:
    --------
    Length 2*N*fsT+1 vector

    Note:
    -----
    This pulse is always scaled such that the amplitude at t=0 is equal to one.
    """
    # time axis with spacing 1/(fs*T)
    tt = np.linspace(-N, N, 2 * N * fsT + 1)

    if a == 0:
        return np.sinc(tt)

    if np.min(np.abs(tt - 1 / (2 * a))) > 1e-6:
        # fast if there is no divide by zero
        return np.sinc(tt) * np.cos(np.pi * a * tt) / (1 - (2 * a * tt) ** 2)
    else:
        # deal with the case when 1-(2*a*tt)**2 = 0
        ss = np.sinc(tt)
        ind_0 = np.where(np.abs(np.abs(tt) - 1 / (2 * a)) < 1e-6)
        tt[ind_0] = 0
        bb = np.cos(np.pi * a * tt) / (1 - (2 * a * tt) ** 2)
        bb[ind_0] = np.pi / 4

        return ss * bb


def srrc_pulse(a, fsT, fs=1, N=5):
    r"""Construct a raised cosine pulse

    Inputs:
    a: roll-off factor
    fsT: number of samples per symbol period
    fs: if a sampling rate (other than fs=1) is specified, then pulses will be scaled like samples of a continuous-time pulse, i.e., such that \int_0^T p^2(t) dt = 1. Otherwise, scaling appropriate for a discrete-time pulse, i.e., \sum_n |p[n]|^2 = 1, is used. (default: fs=1)
    N: length of pulse in symbol periods; pulse ranges for -N \leq t/T \leq N (default: 5).

    Returns:
    Length 2*N*fsT+1 vector
    """
    # time axis with spacing 1/fs
    tt = np.linspace(-N, N, 2 * N * fsT + 1)
    T = fsT / fs

    if a == 0:
        return np.sinc(tt)

    num = np.sin(np.pi * tt * (1 - a)) + 4 * a * tt * np.cos(np.pi * tt * (1 + a))
    den = np.pi * tt * (1 - (4 * a * tt) ** 2)

    # deal with divide-by-zeros: at zero location, place "L'Hospital value" in numerator
    # and 1 in denominator.
    # First divide-by-zero location is t=0; by L-Hospital, the value is (1 + a*(4/pi - 1))
    ind_0 = np.where(np.abs(tt) < 1e-6)
    num[ind_0] = 1 + a * (4 / np.pi - 1)
    den[ind_0] = 1
    # Second divide-by-zero location is t=+/-T/(4*a); by L-Hospital, the value is as shown below
    ind_0 = np.where(np.abs(np.abs(tt) - 1 / (4 * a)) < 1e-6)
    num[ind_0] = (
        a
        / np.sqrt(2)
        * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * a))
            + (1 - 2 / np.pi) * np.cos(np.pi / (4 * a))
        )
    )
    den[ind_0] = 1

    # scaling:
    hh = np.sqrt(fs / fsT) * num / den

    return hh


#
# pulse shaping
#
def pulse_shape(syms, pp, fsT):
    """perform pulse shaping for a sequence of symbols

    Inputs:
    -------
    syms: sequence of symbols
    pp: pulse shape (must have `fsT` samples per symbol period)
    fsT: samples per symbol period

    Returns:
    --------
    vector of signal samples; length is equal to (len(syms)-1)*fsT + len(pp)
    """
    # upsample the symbol sequence
    N_dd = (len(syms) - 1) * fsT + 1  # this avoids extra zeros at end
    dd = np.zeros(N_dd, dtype=syms.dtype)
    dd[0::fsT] = syms

    # convolve with pulse
    return np.convolve(dd, pp)


#
# Bandwidth computations
#
def bandwidth_3dB(ff, SS):
    """find the 3dB bandwidth of the spectrum SS

    Inputs:
    -------
    ff: frequency grid
    SS: samples of power spectral density taken at ff

    Returns:
    --------
    Estimate of the two-sided 3dB bandwidth
    """
    # find the peak of the PSD
    peak = np.max(SS)

    # find the location spectrum value closest to 0.5 * peak
    loc = np.argmin(np.abs(SS - 0.5 * peak))

    # ff[loc] is either a positive or negative frequency where S(f) = 0.5*peak
    # 3dB bandwidth is twice the absolute value of this quantity
    return 2 * abs(ff[loc])


def bandwidth_zz(ff, SS):
    """find the zero-to-zero bandwidth of the spectrum SS

    Inputs:
    -------
    ff: frequency grid
    SS: samples of power spectral density taken at ff

    Returns:
    --------
    Estimate of the two-sided bandwidth measured between the zeros that surround the main lobe

    **Note:** not all spectra have zeros.
    """
    # find the location od  the peak of the PSD
    loc = np.argmax(SS)
    peak = SS[loc]

    # search for the first time, we get close to zero
    while SS[loc] > 1e-4 * peak:
        loc += 1

    # ff[loc] is either a positive or negative frequency where S(f) = 0.*peak
    # 3dB bandwidth is twice the absolute value of this quantity
    return 2 * abs(ff[loc])


def bandwidth_containment(ff, SS, alpha):
    """find the alpha containment bandwidth bandwidth of the spectrum SS

    Inputs:
    -------
    ff: frequency grid
    SS: samples of power spectral density taken at ff
    alpha: fraction of power (0 < alpha < 1)

    Returns:
    --------
    Estimate of the two-sided bandwidth that contains fraction alpha of the total signal power
    """
    # total power
    P = np.sum(SS)

    # find the location od  the peak of the PSD
    loc = np.argmax(SS)
    peak = SS[loc]

    acc = peak / 2
    loc += 1
    # accumulate power untile we get to alpha*P
    while acc < alpha * P / 2:
        acc += SS[loc]
        loc += 1

    # ff[loc] is either a positive or negative frequency where S(f) = 0.*peak
    # 3dB bandwidth is twice the absolute value of this quantity
    return 2 * abs(ff[loc])


#
# Numerical computation of the Fourier transform
#
def numerical_FT(pp, fs, N):
    """compute Fourier transform numerically

    Input:
    ------
    pp: samples of time-domain signal to be transformed
    fs: sample rate
    N: number of frequency domain samples (must satisfy: N >= length(pulse))

    Returns:
    --------
    Length-N complex vector

    Note:
    -----
    * frequency range is from -fs/2 to fs/2
    * frequency resolution is df = fs/N
    * N should be a power of 2
    """

    assert len(pp) <= N, "pulse is too long"

    # zero-pad pp
    padded = np.zeros(N, dtype=complex)
    padded[: len(pp)] = pp

    # compute DFT (using FFT)
    PP = np.fft.fftshift(np.fft.fft(padded)) / fs

    return PP
