# pll.py: feedback tracking loops for phase and amplitude

import numpy as np


class First_Order_Filter:
    """Class representing a first-order loop filter"""

    def __init__(self, alpha):
        """Initialize the gain of the filter

        Input:
        alpha - loop gain
        """
        self.alpha = alpha

    def __call__(self, d_n):
        """invoke the filter

        Input:
        d_n - single sample, representing error

        Returns:
        filtered sample; type is the same as the input
        """
        return self.alpha * d_n

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, new_val):
        if new_val < 2 and new_val > 0:
            self._alpha = new_val
        else:
            raise ValueError("alpha = {:} makes PLL unstable".format(new_val))


class Second_Order_Filter:
    """Class representing a second-order loop filter"""

    def __init__(self, alpha1, alpha2, state=0):
        """Initialize the gains and the state of the filter

        Input:
        alpha1 - linear gain
        alpha2 - integrator gain
        state - value held by integrator (optional, default: 0)
        """
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.state = state

    def __call__(self, d_n):
        """invoke the filter

        Input:
        d_n - single sample, representing error

        Returns:
        filtered sample; type is the same as the input
        """
        out = self.alpha1 * d_n + self.state
        self.state += self.alpha2 * d_n  # update state

        return out

    # below, we check that 0 <= alpha2 < alpha1 < 1
    @property
    def alpha1(self):
        return self._alpha1

    @alpha1.setter
    def alpha1(self, new_val):
        if new_val < 1 and new_val > 0:
            self._alpha1 = new_val
        else:
            raise ValueError("alpha1 = {:} can make PLL unstable".format(new_val))

    @property
    def alpha2(self):
        return self._alpha2

    @alpha2.setter
    def alpha2(self, new_val):
        if new_val < self._alpha1 and new_val > 0:
            self._alpha2 = new_val
        else:
            raise ValueError("alpha2 = {:} can make PLL unstable".format(new_val))

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_val):
        self._state = new_val

    @property
    def damping_factor(self):
        return self.alpha1 / 2 / np.sqrt(self.alpha2)


class Integrator:
    """Class representing an integrator"""

    def __init__(self, state=0):
        self.state = state

    def __call__(self, x_n):
        "Compute the output from the integrator"
        out = self.state
        self.state += x_n

        return out

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_val):
        self._state = new_val


def rotate_phase(Z_n, phi):
    """Rotate the phase of the input signal

    Inputs:
    Z_n - signal to be phase corrected
    phi - phase correction to apply

    Returns:
    complex sample
    """
    return Z_n * np.exp(-1j * phi)


def measure_phase(X_n, s_n=1):
    """measure the phase of modulated symbol

    It is assumed that X_n = s_n * exp(j dphi) + N_n, where s_n is an information symbol
    and N_n is noise. The goal is to estimate the phase (error) dphi

    Inputs:
    X_n - sample of phase-rotated signal
    s_n - information symbol (default: 1)

    Returns:
    phase error estimate dphi
    """

    return np.angle(X_n / s_n)


def scale_amplitude(Z_n, gamma):
    """Scale the amplitude of the input signal

    Inputs:
    Z_n - signal to be phase corrected
    gamma - amplitude correction to apply

    Returns:
    complex sample
    """
    return Z_n * gamma


def measure_amplitude_error(X_n, s_n=1):
    """measure the amplitude error of modulated symbol

    It is assumed that X_n = s_n * A + N_n, where s_n is an information symbol
    and N_n is noise. The goal is to estimate the gain gamma such that A * gamma = 1

    Inputs:
    X_n - sample of phase-rotated signal
    s_n - information symbol (default: 1)

    Returns:
    amplitude error estimate dA
    """

    return 1 - np.abs(X_n / s_n)
