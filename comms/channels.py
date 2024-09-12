#! /usr/bin/env python3

# File: channels.py - models for channels;
#       additive Gaussian noise channels for various stages of the system

import numpy as np


def dgnc(symbols, sigma_sq):
    """discrete Gaussian noise channel

    This channel model is appropriate for modeling the transmission of complex symbols. If
    symbols are real-valued, only real-valued noise is added.

    Inputs:
    -------
    * symbols: information symbols to be transmitted
    * sigma_sq: noise variance; for complex symbols this variance is divided evenly between real and imaginary parts

    Returns:
    --------
    noisy symbols; a vector of the same length and type as symbols
    """
    n_syms = len(symbols)

    if symbols.dtype == complex:
        noise = np.sqrt(sigma_sq / 2) * (
            np.random.randn(n_syms) + 1j * np.random.randn(n_syms)
        )
    else:
        noise = np.sqrt(sigma_sq) * np.random.randn(n_syms)

    return symbols + noise
