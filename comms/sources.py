#! /usr/bin/env python3

# File: sources.py - sources and sinks;
#       various ways to produce or consume sequences of bits

"""
# Sources and Sinks

This module contains various functions that act as sources or sinks for bit sequences

## Sources:

* `string_source( string )`: convert a string to a sequence of bits
* `random_symbols( A, N )`: generate random symbols from the constellation A

## Sinks:

* `string_sink( bits )`: convert a sequence of bits into a string

"""

import numpy as np

from comms.utils import byte_to_bits, bits_to_byte


def random_symbols(A, N):
    """generate random symbols from the constellation A

    Inputs:
    -------
    * A - np.ndarray of symbols in constellation, e.g., A = np.array([1, -1]) for BPSK
    * N - number of random symbols to produce

    Returns:
    --------
    Numpy array of length N
    """
    return A[np.random.randint(len(A), size=N)]


def string_source(string):
    """convert a string to a vector of bits

    Inputs:
    -------
    * string - string to be converted into a bit-sequence; maybe ASCII or UTF-8

    Returns:
    --------
    Numpy vector of bits
    """
    # convert a string to a sequence of bytes
    bb = string.encode()
    Nb = len(bb)

    # allocate space
    bits = np.zeros(8 * Nb, dtype=np.uint8)

    for n in range(Nb):
        bits[8 * n : 8 * (n + 1)] = byte_to_bits(bb[n])

    return bits


def string_sink(bits):
    """convert a sequence of bits into a string

    Inputs:
    -------
    * bits - an iterable containing 0's and 1'; length must be a multiple of 8

    Returns:
    --------
    (string) the result of decoding sequence of bits
    """

    # check that number of bits is a multiple of 8
    if len(bits) % 8 != 0:
        raise ValueError(f"number of bits {len(bits)} is not divisible by 8.")

    # allocate storage
    n_bytes = len(bits) // 8
    bytes = np.zeros(n_bytes, dtype=np.uint8)

    for n in range(n_bytes):
        bytes[n] = bits_to_byte(bits[n * 8 : (n + 1) * 8])

    # decode the string (deal with unicode encoding)
    return bytes.tobytes().decode("utf-8", "replace")


if __name__ == "__main__":

    ## Round-trip test
    string = "Hi 😲"
    assert string == string_sink(string_source(string))

    # all good if we get here
    print("OK")
