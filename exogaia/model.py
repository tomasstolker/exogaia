"""
Module with the ``BinaryModel`` class.
"""

from typing import Tuple

import kepler
import numpy as np

# from PyAstronomy import pyasl
from astropy.time import Time
from typeguard import typechecked

from exogaia.core import ExoGaia


class BinaryModel(ExoGaia):
    """
    Class for a binary model.
    """

    @typechecked
    def __init__(self, data_table, primary_mass: Tuple[float, float]) -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        self.data_table = data_table
        self.primary_mass = primary_mass

    def calc_model(self, params):
        """
        Binary model
        """

        # Gaia DR4 reference epoch
        ref_epoch = Time("2017.5", format="jyear", scale="tcb")

        obs_time = self.data_table["relative_time_day"]
        # obs_time_tcb = obs_time + ref_epoch.jyear

        # Design matrix for 5-param least-squares fit
        design = np.column_stack(
            [
                np.sin(self.data_table["scan_pos_angle"]),
                np.cos(self.data_table["scan_pos_angle"]),
                self.data_table["relative_time_year"]
                * np.sin(self.data_table["scan_pos_angle"]),
                self.data_table["relative_time_year"]
                * np.cos(self.data_table["scan_pos_angle"]),
                self.data_table["parallax_factor_al"],
            ]
        )

        sin_t = np.sin(self.data_table["scan_pos_angle"])
        cos_t = np.cos(self.data_table["scan_pos_angle"])

        star_param = params[0:5]
        star_model = design @ star_param

        sma, ecc, inc, aop, pan, tau, mtot = params[5:12]

        distance = 1.0 / (1e-3 * params[4])  # (pc)
        period = np.sqrt(sma**3 / mtot) * 365.25  # (days)
        t_per = period * tau  # (days)

        # Days relative to periastron
        delta_t = obs_time - t_per

        # Mean anomaly at observation epochs
        n = 2.0 * np.pi / period
        mean_anom_obs = (delta_t * n) % (2.0 * np.pi)

        # Secondary mass (Msun)
        secondary_mass = mtot - self.primary_mass[0]

        # Primary semi-major axis (au)
        sma1 = sma * secondary_mass / mtot

        # cos and sin of omega
        cos_aop = np.cos(aop)
        sin_aop = np.sin(aop)

        # cos and sin of Omega
        cos_pan = np.cos(pan)
        sin_pan = np.sin(pan)

        ecc_anom, cos_true_anom, sin_true_anom = kepler.kepler(mean_anom_obs, ecc)

        # (x, y) position in the orbital plane

        radius = sma1 * (1.0 - ecc * ecc_anom)

        cos_theta = cos_aop * cos_true_anom - sin_aop * sin_true_anom
        sin_theta = sin_aop * cos_true_anom - cos_aop * sin_true_anom

        x_orb = radius * cos_theta
        y_orb = radius * sin_theta

        # Rotate (x_orb, y_orb) into sky plane (x_sky, y_sky)

        # Rotate about z0 axis by aop
        x1 = cos_aop * x_orb - sin_aop * y_orb
        y1 = sin_aop * x_orb + cos_aop * y_orb

        # Rotate about x1 axis by -inc
        x2 = x1
        y2 = np.cos(inc) * y1

        # Rotate about z2 axis by Omega
        x_sky = cos_pan * x2 - sin_pan * y2
        y_sky = sin_pan * x2 + cos_pan * y2

        # Scale by from au to mas
        delta_ra = x_sky / distance * 3600.0  # (mas)
        delta_dec = y_sky / distance * 3600.0  # (mas)

        # kepl_orbit = pyasl.KeplerEllipse(
        #     a=a1,
        #     per=period,
        #     e=params[6],
        #     tau=t_per,
        #     Omega=np.degrees(params[9]),
        #     i=np.degrees(params[7]),
        #     w=np.degrees(params[8]),
        # )

        # xyz_pos = kepl_orbit.xyzPos(obs_time)

        # delta_ra = xyz_pos[:, 0] / distance * 3600.0  # (mas)
        # delta_dec = xyz_pos[:, 1] / distance * 3600.0  # (mas)

        # Calculate the 1D projected positions
        orbit_model = delta_ra * sin_t + delta_dec * cos_t

        return star_model + orbit_model
