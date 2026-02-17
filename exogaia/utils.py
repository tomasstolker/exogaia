"""
Module with utility functions.
"""

import numpy as np


def orbit_sky(nu, sma, ecc, inc, aop, pan):
    """
    Compute the projected sky-plane coordinates of a Keplerian orbit.

    Parameters
    ----------
    nu : float or ndarray
        True anomaly (radians).
    sma : float
        Semi-major axis.
    ecc : float
        Eccentricity (0 <= e < 1).
    inc : float
        Inclination (radians). i = 0 corresponds to a face-on orbit.
    aop : float
        Argument of pericenter (radians).
    pan : float
        Longitude of the ascending node (radians).

    Returns
    -------
    x : float or ndarray
        Projected x-coordinate on the sky plane.
    y : float or ndarray
        Projected y-coordinate on the sky plane.

    Notes
    -----
    The orbit is first computed in the orbital plane using the standard
    conic-section equation:

        r = a (1 - e^2) / (1 + e cos(nu))

    The position vector is then rotated by:
        1) argument of pericenter (aop),
        2) inclination (inc),
        3) longitude of ascending node (pan),

    to obtain the sky-plane projection.
    """

    r = sma * (1.0 - ecc**2.0) / (1.0 + ecc * np.cos(nu))

    # position in orbital plane
    x_op = r * np.sin(nu)
    y_op = r * np.cos(nu)

    # rotate by omega
    x1 = x_op * np.cos(aop) - y_op * np.sin(aop)
    y1 = x_op * np.sin(aop) + y_op * np.cos(aop)

    # incline
    x2 = x1 * np.cos(inc)

    # rotate by Omega to sky plane
    x = x2 * np.cos(-pan) - y1 * np.sin(-pan)
    y = x2 * np.sin(-pan) + y1 * np.cos(-pan)

    return x, y
