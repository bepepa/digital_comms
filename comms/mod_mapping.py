#! /usr/bin/env python3

# File: mod_mapping.py - sources and sinks;
#       mapping groups of bits to symbols; includes several constellations

"""
# Modulation Mapping

This module provides tables (dicts) for mapping groups of bits to symbols and functions to perform this mapping.
Additionally, there are some functions to pretty-print or plot a constellation.

## Constellations:

The following constellations are provided:
* BPSK (`BPSK`)
* QPSK (`QPSK`)
* 16-QAM (`QAM16`)
* 64-QAM (`QAM64`)
* 4-PAM (`PAM4`)
* 8-PAM (`PAM8`)
* 8-PSK (`PSK8`)

## Modulation and Demodulation

The following functions are used to modulate bits to symbols and vice versa

* `mod_mapper( bits )`: map a sequence of bits to a sequence of symbols
* `demodulator( rx_symbols )`: Recover bit sequence from received symbols

## Helpers:

* `print_constellation(mod_table)`: print a table of bit patterns and symbols
* `plot_constellation(mod_table)`: plot the constellation

* `num_bits_per_symbol(mod_table)`: compute the number of bits per symbol
* `energy_per_symbol(mod_table)`: compute the energy per symbol
* `energy_per_bit(mod_table)`: compute the energy per bit
* `min_dist_sq(mod_table)`: compute the square of the smallest distance between symbols
* `energy_efficiency(mod_table)`: compute the energy efficiency of the constellation
* `N_min(mod_table)`: compute the average number of nearest neighbors
"""

import numpy as np
import matplotlib.pyplot as plt

from comms.utils import int_to_bits, bits_to_int, Q

#
# Constellations
#
# The following are from the 5G standard (TS 38.211, section 5.1)
# the scaling factor is omitted here
_bpsk_map = lambda b: (1 - 2 * b[0])  # different from TS 38.211 for simplicity
_qpsk_map = lambda b: (1 - 2 * b[0]) + 1j * (1 - 2 * b[1])
_16qam_map = lambda b: (1 - 2 * b[0]) * (2 - (1 - 2 * b[2])) + 1j * (1 - 2 * b[1]) * (
    2 - (1 - 2 * b[3])
)
_64qam_map = lambda b: (1 - 2 * b[0]) * (
    4 - (1 - 2 * b[2]) * (2 - (1 - 2 * b[4]))
) + 1j * (1 - 2 * b[1]) * (4 - (1 - 2 * b[3]) * (2 - (1 - 2 * b[5])))

# one-dimensional constellations derived from above
_4pam_map = lambda b: (1 - 2 * b[0]) * (2 - (1 - 2 * b[1]))
_8pam_map = lambda b: (1 - 2 * b[0]) * (4 - (1 - 2 * b[1]) * (2 - (1 - 2 * b[2])))

# 8PSK
_8psk_map = lambda b: np.exp(
    1j * np.pi / 8 * (1 - 2 * b[0]) * (4 - (1 - 2 * b[1]) * (2 - (1 - 2 * b[2])))
)

#
# use a *comprehension* to construct the corresponding dictionaries/tables
#
# the dictionaries are constructed with
# * keys equal to a sequence of K bits, represented by the corresponding decimal
# * values equal to the symbols
#
# length of the dictionary M = 2**K

BPSK = {n: _bpsk_map(int_to_bits(n, 1)) for n in range(2)}
QPSK = {n: _qpsk_map(int_to_bits(n, 2)) for n in range(4)}
QAM16 = {n: _16qam_map(int_to_bits(n, 4)) for n in range(16)}
QAM64 = {n: _64qam_map(int_to_bits(n, 6)) for n in range(64)}

PAM4 = {n: _4pam_map(int_to_bits(n, 2)) for n in range(4)}
PAM8 = {n: _8pam_map(int_to_bits(n, 3)) for n in range(8)}

PSK8 = {n: _8psk_map(int_to_bits(n, 3)) for n in range(8)}

# a list of all constellations
Constellations = [BPSK, QPSK, QAM16, QAM64, PAM4, PAM8, PSK8]


#
# Modulation mapper
#
def mod_mapper(bits, mod_table):
    """map a sequence of bits to a sequence of symbols

    Inputs:
    -------
    * bits: sequence of 0's and 1's
    * mod_table: dictionary containing the mapping from groups of bits to symbols

    Returns:
    --------
    a vector of symbols
    """

    # how many bits per symbol?
    K = int(np.log2(len(mod_table)))

    assert (
        len(bits) % K == 0
    ), "number of bits must be divisible by number of bit per symbol"

    # how many symbols will we get?
    N = len(bits) // K
    syms = np.zeros(N, dtype=complex)

    for n in range(N):
        key = bits_to_int(bits[K * n : K * (n + 1)])
        syms[n] = mod_table[key]

    return syms


def demodulator(syms, mod_table):
    """Recover bit sequence from received symbols

    Inputs:
    -------
    * syms: sequence of received (noisy) symbols
    * mod_table: dictionary containing the mapping from groups of bits to symbols

    Returns:
    --------
    a vector of bits
    """
    # how many bits per symbol?
    K = int(np.log2(len(mod_table)))

    # how many bits will we get?
    N = len(syms) * K
    bits = np.zeros(N, dtype=np.uint8)

    # put mod_table's symbol values and keys into a Numpy array
    alphabet = np.array(list(mod_table.values()))
    pattern = np.array(list(mod_table.keys()), dtype=int)

    # find the constellation point closest to received symb `s`
    for n in range(len(syms)):
        s = syms[n]
        ind = np.argmin(np.abs(alphabet - s))
        min_k = pattern[ind]

        # the index of the closest symbol is integer `min_k`
        # convert that to a sequence of K bits
        bits[n * K : (n + 1) * K] = int_to_bits(min_k, K)

    return bits


#
# Helper functions:
#
# the functions below take a constellation as input and produce useful outputs
#
def print_constellation(mod_table):
    """print a table listing bit patterns and associated symbols

    Inputs:
    -------
    * mod_table: dictionary associating bits and symbols

    Returns:
    --------
    nothing
    """

    M = len(mod_table)
    K = int(np.log2(M))

    print("|{:^20s}|{:^20s}|".format("Bits", "Symbol"))
    print("|{:^20s}|{:^20s}|".format("-" * 20, "-" * 20))

    for k, v in mod_table.items():
        bit_str = "{{:0{:d}b}}".format(K).format(k)
        symbol_str = "{:+4.3f}".format(v.real)
        if isinstance(v, complex):
            sign_str = "+"
            if v.imag < 0:
                sign_str = "-"
            symbol_str += " {:s} j {:4.3f}".format(sign_str, abs(v.imag))

        print("|{:^20s}|{:^20s}|".format(bit_str, symbol_str))


def plot_constellation(mod_table):
    """plot the constellation

    Inputs:
    -------
    * mod_table: dictionary associating bits and symbols

    Returns:
    --------
    nothing
    """
    M = len(mod_table)
    K = int(np.log2(M))
    max_real = max([x.real for x in mod_table.values()])

    for k, v in mod_table.items():
        bit_str = "{{:0{:d}b}}".format(K).format(k)

        plt.plot(v.real, v.imag, "ro", label=bit_str)
        plt.text(v.real, v.imag + 0.05, bit_str, ha="center", va="bottom")

    plt.xlabel("Real")
    plt.ylabel("Imag")

    # make squares square and give a bit of sapce for label to fit
    plt.axis("equal")
    plt.xlim(-1.2 * max_real, 1.2 * max_real)
    plt.ylim(-1.2 * max_real, 1.2 * max_real)

    plt.grid()

    # plt.show()


def num_bits_per_symbol(mod_table):
    """Compute number of bits per symbol

    Inputs:
    -------
    * mod_table: dictionary associating bits and symbols

    Returns:
    --------
    (int)
    """
    return int(np.log2(len(mod_table)))


def energy_per_symbol(mod_table):
    """compute average energy per symbol

    Inputs:
    -------
    * mod_table: dictionary associating bits and symbols

    Returns:
    --------
    (float)
    """
    Es = 0
    for v in mod_table.values():
        Es += np.abs(v) ** 2

    return Es / len(mod_table)


def energy_per_bit(mod_table):
    """compute average energy per bit

    Inputs:
    -------
    * mod_table: dictionary associating bits and symbols

    Returns:
    --------
    (float)
    """
    return energy_per_symbol(mod_table) / num_bits_per_symbol(mod_table)


def min_dist_sq(mod_table):
    """compute the square of the minimum distance of the constellation

    Inputs:
    -------
    * mod_table: dictionary associating bits and symbols

    Returns:
    --------
    (float)
    """
    d_min = np.infty

    for n, s in mod_table.items():
        for m, v in mod_table.items():
            # because of symmetry, we only need to check half of all pairs
            if m <= n:
                continue

            d = np.abs(s - v) ** 2
            d_min = d if (d < d_min) else d_min

    return d_min


def energy_efficiency(mod_table):
    """Energy efficiency of constellation

    Inputs:
    -------
    * mod_table: dictionary associating bits and symbols

    Returns:
    --------
    (float)
    """

    return min_dist_sq(mod_table) / energy_per_bit(mod_table)


def N_min(mod_table):
    """Compute the average number of nearest neighbors

    Inputs:
    -------
    * mod_table: dictionary associating bits and symbols

    Returns:
    --------
    (float)
    """

    d_min_sq = min_dist_sq(mod_table)

    N_min = 0
    for n, s in mod_table.items():
        for m, v in mod_table.items():
            # because of symmetry, we only need to check half of all pairs
            if m <= n:
                continue

            d_sq = np.abs(s - v) ** 2
            if np.abs(d_min_sq - d_sq) < 1e-8:
                N_min += 1

    # multiply by 2, since we only visited half of all pairs
    N_min *= 2 / len(mod_table)


def NN_Pr_symbol_error(mod_table, EbN0):
    """compute nearest-neighbor approximation for symbol error rate"""

    NN = N_min(mod_table)
    eta = energy_efficiency(mod_table)

    return Q(np.sqrt(eta / 2 * EbN0))


if __name__ == "__main__":
    N = 988
    EbN0 = 10
    bits = int_to_bits(N, 12)
    for mm in Constellations:
        K = int(np.log2(len(mm)))  # bits per symbol
        NN_Pr_symbol_error(mm, EbN0)

        syms = mod_mapper(bits, mm)
        rec_bits = demodulator(syms, mm)
        assert bits_to_int(rec_bits) == N

    # print_constellation(PAM8)
    # plot_constellation(QAM16)

    print("Ok")
