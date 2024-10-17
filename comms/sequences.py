# File: sequences.py
# generate sequences with good autocorrelation properties

import numpy as np
from typing import Tuple


def pop_count32(x: np.uint32) -> np.uint8:
    """count the number of 1s in 32-bit integer x using Wegner's method

    Inputs:
    x - a 32 bit integer

    Returns:
    an integer between 0 and 32 indicating the number of 1s in x
    """

    count: np.uint8 = 0

    while x > 0:
        x = x & (x - 1)
        count += 1

    return count


def lfsr_step(
    state: np.uint32, fb: np.uint32, N: int = 31
) -> Tuple[np.uint8, np.uint32]:
    """Single update step of LFSR with feedback connections fb

    Inputs:
    * state (uint32): initial state of the feedback shift register
    * fb (uint32): bit map indicating feedback connections; the LSB corresponds to bit 0 in the LFSR
    * N (int) order if the LFSR; must be less than 32 (default: 31)

    Returns:
    * elem - next element of the LFSR sequence
    * state - updated state

    Example: Generate the first 10 elements of a LFSR sequence with fb=0b1001 and initial state=0b1111
    >>> state = 0b1111
    >>> seq = np.empty(10, dtype=np.uint8)
    >>> for n in range(10):
    >>>     seq[n], state = lfsr31_step(state, 0b1001)
    >>> seq

    array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=uint8)
    """

    # save the LSB before updating state; it will be returned
    elem: np.uint8 = state & 0b1

    # update state
    fb_vec: np.uint32 = state & fb
    fb_bit: np.uint8 = pop_count32(fb_vec) & 0b1

    # insert feedback bit at position N, then shift
    state = (state | fb_bit << N) >> 1

    return elem, state


def lfsr(init: np.uint32, fb: np.uint32, M: int, N: int = 31, Nc: int = 0):
    """Compute M samples of a LFSR sequence

    Inputs:
    init - initial state of the LFSR
    fb - feedback connection for the LFSR
    M - number of samples to generate
    N - order of the LFSR (default: 31)
    Nc - statrting sample (default 0)

    Returns:
    length-M vector of bits (stored as np.uint8)
    """
    seq = np.empty(M, dtype=np.uint8)

    state = init

    for n in range(Nc):
        _, state = lfsr_step(state, fb, N)  # discard the first Nc

    for n in range(M):
        seq[n], state = lfsr_step(state, fb, N)  # keep the next Mc

    return seq
