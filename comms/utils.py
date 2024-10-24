#! /usr/bin/env python3

# File: utils.py - utility functions used in several modules

"""
# comms.utils

Contains a set of utility function for use throughout the other modules.

* `Q(x)`: Gaussian error integral
* `bytes_to_bits(byte_seq)`: convert a sequence of bytes to a sequence of bits
* `bits_to_bytes(bit_seq)`: convert a sequence of bits into a byte sequence (MSB first)
* `byte_to_bits(b)`: convert a single byte to a sequence of bits (in MSB first order)
* `bits_to_byte(bits)`: convert a sequence of eight bits to a bytes
* `bits_to_int(bits)`: convert a sequence of bits to an integer
* `int_to_bits(nn, K): convert an integer to K bits
* `crc16(bytes, poly)`: CCITT CRC-16 checksum algorithm
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


def bytes_to_bits(byte_seq):
    """Convert a sequence of bytes into a sequence of bits (MSB first)

    Input:
    ------
    byte_seq: a sequence of bytes

    Returns:
    --------
    a vector of binary values, 8 times as long as the inout sequence (stored as uint8)
    """
    # make space for results
    Nb = len(byte_seq)
    bits = np.zeros(8 * Nb, dtype=np.uint8)

    for n in range(Nb):
        bits[8 * n : 8 * (n + 1)] = byte_to_bits(byte_seq[n])

    return bits


def bits_to_bytes(bit_seq):
    """Convert a sequence of bits into a byte sequence (MSB first)

    Input:
    ------
    * bit_seq: a sequence of binary values

    Returns:
    --------
    a vector of bytes (uint8), length is 1/8 of input length

    Exception:
    ----------
    A `ValueError` is raised if length of bit sequence is not divisible by 8.
    """
    Nb = len(bit_seq)
    if (Nb % 8) != 0:
        raise (ValueError, "number of bits must be a multiple of 8")

    byte_seq = np.zeros(Nb // 8, dtype=np.uint8)

    for n in range(Nb // 8):
        byte_seq[n] = bits_to_byte(bit_seq[8 * n : 8 * (n + 1)])

    return byte_seq


#
# bit-wrangling functions
#
def byte_to_bits(b):
    """convert a single byte to a sequence of 8 bits (MSB first)

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


def crc16(data: bytes, poly=0x8408):
    """
    CRC-16-CCITT Algorithm

    Args:
        data (bytes): byte sequence to be checked (type bytes)
        poly (int): 16-bit integer indicating the polynomial used by CRC (default 0x8408)

    Returns:
        16-bit CRC
    """
    data = bytearray(data)
    crc = 0xFFFF
    for b in data:
        cur_byte = 0xFF & b
        for _ in range(0, 8):
            if (crc & 0x0001) ^ (cur_byte & 0x0001):
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
            cur_byte >>= 1
    crc = ~crc & 0xFFFF
    crc = (crc << 8) | ((crc >> 8) & 0xFF)

    return crc & 0xFFFF


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
