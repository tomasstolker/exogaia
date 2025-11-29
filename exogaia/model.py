"""
Module with the ``BinaryModel`` class.
"""

from typing import Tuple

import numpy as np

from PyAstronomy import pyasl
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

        star_param = [params[0], params[1], params[2], params[3], params[4]]
        star_model = design @ star_param

        distance = 1.0 / (1e-3 * params[4])  # (pc)
        period = np.sqrt(params[5] ** 3 / params[11]) * 365.25  # (days)

        m2 = params[11] - self.primary_mass[0]
        a1 = params[5] * m2 / params[11]

        kepl_orbit = pyasl.KeplerEllipse(
            a=a1,
            per=period,
            e=params[6],
            tau=period * params[10],
            Omega=np.degrees(params[9]),
            i=np.degrees(params[7]),
            w=np.degrees(params[8]),
        )

        xyz_pos = kepl_orbit.xyzPos(self.data_table["relative_time_day"])

        # kepl_orbit = pyasl.BinaryOrbit(
        #     m2m1=m2 / self.primary_mass[0],
        #     mtot=params[11],
        #     per=period,
        #     e=params[6],
        #     tau=period * params[10],
        #     Omega=np.degrees(params[9]),
        #     w=np.degrees(params[8]),
        #     i=np.degrees(params[7]),
        # )

        # xyz_pos = kepl_orbit.xyzPos(self.data_table["relative_time_day"])[0]

        delta_ra = xyz_pos[:, 0] / distance * 3600.0  # (mas)
        delta_dec = xyz_pos[:, 1] / distance * 3600.0  # (mas)

        orbit_model = delta_ra * sin_t + delta_dec * cos_t

        return star_model + orbit_model
