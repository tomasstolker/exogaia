"""
Module for handling epoch astrometry data.
"""

from typing import Tuple

import pandas as pd

from exogaia.core import ExoGaia

# from astroquery.gaia import Gaia
# Gaia.ROW_LIMIT = -1


class EpochAstrometry(ExoGaia):
    """
    Class for handling epoch astrometry data.
    """

    def __init__(self, data_file: str, primary_mass: Tuple[float, float] = None):
        """
        Parameters
        ----------
        data_file : str
            Data file with the Gaia epoch astrometry.
        """

        self.print_section("Read astrometry data")

        self.data_file = data_file
        self.data_table = pd.read_csv(self.data_file)
        self.primary_mass = primary_mass

        print(f"Data file: {self.data_file}")
        print(f"Data shape: {self.data_table.shape}")

        if self.primary_mass is not None:
            print(
                f"\nPrimary mass (Msun): {self.primary_mass[0]:.2f} +/- {self.primary_mass[1]:.2f}"
            )

    def __repr__(self):
        return self.data_table.head().to_string()
