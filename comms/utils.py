#! /usr/bin/env python3

# File: utils.py - utility functions used in several modules

"""
# comms.utils

Contains a set of utility function for use throughout the other modules.

* `Q(x)`: Gaussian error integral
* `byte_to_bits(b)`: convert a byte to a sequence of bits (in MSB first order)
* `bits_to_byte(bits)`: convert a sequence of 8 bits to a bytes
"""

import numpy as np
import numpy.typing as npt

from scipy import special


def Q(x):
    """Gaussian error integral

    Compute INT_x^infty p(x) dx, where p(x) is a standard Gaussian pdf

    Inputs:
    -------
    x: lower limit of integral

    Returns:
    --------
    (float) error integral
    """
    return 0.5 - 0.5 * special.erf(x / np.sqrt(2))


#
# bit-wrangling functions
#
def byte_to_bits(b):
    """convert a byte to a sequence of 8 bits (MSB first)

    Inputs:
    -------
    b: a single byte

    Returns:
    --------
    a NumPy vector of bits, stored as uint8
    """

    # allocate memory for bits
    bits = np.zeros(8, dtype=np.uint8)

    # define the mask
    mask = 128

    for n in range(8):
        # extract the MSB and store it
        bits[n] = (b & mask) >> 7
        # shift the bits by one position
        b = b << 1

    return bits


def bits_to_byte(bits):
    """convert a sequence of up to eight bits to a byte (MSB in first position)

    Inputs:
    -------
    bits: an iterable of eight 0's and 1's

    Returns:
    --------
    (uint8) a single byte

    """
    assert len(bits) <= 8, "Can only convert at most 8 bits at a time"

    res = np.uint8(0)

    for b in bits:
        res = res << 1
        res = res | b

    return res


def bits_to_int(bits: npt.ArrayLike) -> int:
    """convert a sequence of bits to an integer

    Inputs:
    -------
    bits: a sequence of 0's and 1's

    Returns:
    --------
    (int) the decimal represntation of the bit sequence; MSB is assumed to be the first bit.
    """
    res = 0

    for b in bits:
        res = res << 1
        res = res + b

    return res


def int_to_bits(nn: int, K: int):
    """convert an integer to a sequence of bits

    Inputs:
    -------
    nn: the integer to convert
    K: the length of the bit vector to return

    Returns:
    --------
    a vector of length `K` of 0's and 1's; the `dtype` of this vector is `uint8`.
    """
    # allocate memory for bits
    bits = np.zeros(K, dtype=np.uint8)

    # define the mask
    mask = 1 << (K - 1)

    for k in range(K):
        # extract the LSB and store it
        bits[k] = (nn & mask) >> (K - 1)
        # shift the bits by one position
        nn = nn << 1

    return bits


if __name__ == "__main__":
    # round-trip test of bits_to_byte and byte_to_bits
    for n in range(256):
        assert n == bits_to_byte(byte_to_bits(n))

    # round-trip test of bits_to_int and int_to_bits
    K = 10
    for n in range(2**K):
        assert bits_to_int(int_to_bits(n, K)) == n

    # all good if we get here
    print("OK")
