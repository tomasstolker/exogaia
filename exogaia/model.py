"""
Module with the ``BinaryModel`` class.
"""

import kepler

import matplotlib.pyplot as plt
import numpy as np

from typeguard import typechecked

from exogaia.core import ExoGaia


class BinaryModel(ExoGaia):
    """
    Class for a binary model.
    """

    @typechecked
    def __init__(self, epoch_astrometry, verbose=True) -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        self.data_table = epoch_astrometry.data_table
        # self.ref_epoch = epoch_astrometry.ref_epoch
        self.verbose = verbose

    def calc_orbit(self, model_params, obs_time=None):
        """
        Orbit model
        """

        if obs_time is None:
            obs_time = self.data_table["relative_time_day"]

        # obs_time_tcb = obs_time + self.ref_epoch.jyear

        sma, ecc, inc, aop, pan, tau, m1, m2 = model_params[5:13]
        parallax = model_params[2]  # (mas)

        period = np.sqrt(sma**3 / (m1 + m2)) * 365.25  # (days)
        t_per = period * tau  # (days)

        # Days relative to periastron
        delta_t = obs_time - t_per

        # Mean anomaly at observation epochs
        mean_anom_obs = delta_t * 2.0 * np.pi / period

        # Primary semi-major axis (au)
        sma1 = sma * m2 / (m1 + m2)

        # Solve Kepler's equation
        ecc_anom, _, _ = kepler.kepler(mean_anom_obs, ecc)

        # (x, y) position in the orbital plane
        x_orb = np.cos(ecc_anom) - ecc
        y_orb = np.sqrt(1.0 - ecc**2) * np.sin(ecc_anom)

        # Thiele-Innes constants
        thiele_innes_a = sma1 * (
            np.cos(aop) * np.cos(pan) - np.sin(aop) * np.sin(pan) * np.cos(inc)
        )
        thiele_innes_b = sma1 * (
            np.cos(aop) * np.sin(pan) + np.sin(aop) * np.cos(pan) * np.cos(inc)
        )
        thiele_innes_f = sma1 * (
            -np.sin(aop) * np.cos(pan) - np.cos(aop) * np.sin(pan) * np.cos(inc)
        )
        thiele_innes_g = sma1 * (
            -np.sin(aop) * np.sin(pan) + np.cos(aop) * np.cos(pan) * np.cos(inc)
        )

        # Rotate (x_orb, y_orb) into sky plane (x_sky, y_sky)
        x_sky = thiele_innes_b * x_orb + thiele_innes_g * y_orb
        y_sky = thiele_innes_a * x_orb + thiele_innes_f * y_orb

        # Scale from au to mas
        delta_ra = x_sky * parallax  # (mas)
        delta_dec = y_sky * parallax  # (mas)

        return delta_ra, delta_dec

    def calc_model(self, model_params):
        """
        Binary model
        """

        if self.verbose:
            self.print_section("Calculate binary model")

            print("Model paramers:")
            print(f"   - RA (deg) = {model_params[0]:.2f}")
            print(f"   - Dec (deg) = {model_params[1]:.2f}")
            print(f"   - Parallax (mas) = {model_params[2]:.2f}")
            print(f"   - Proper motion in RA (mas/yr) = {model_params[3]:.2f}")
            print(f"   - Proper motion in Dec (mas/yr) = {model_params[4]:.2f}")
            print(f"   - Semi-major axis (au) = {model_params[5]:.2f}")
            print(f"   - Eccentricity = {model_params[6]:.2f}")
            print(f"   - Inclination (deg) = {np.degrees(model_params[7]):.2f}")
            print(
                f"   - Argument of periastron (deg) = {np.degrees(model_params[8]):.2f}"
            )
            print(
                f"   - PA of ascending node (deg) = {np.degrees(model_params[9]):.2f}"
            )
            print(f"   - Epoch of periastron = {model_params[10]:.2f}")
            print(f"   - Primary mass (Msun) = {model_params[11]:.2f}")
            print(f"   - Secondary mass (Msun) = {model_params[12]:.2f}")

        # Design matrix for 5-param least-squares fit
        design = np.column_stack(
            [
                np.sin(self.data_table["scan_pos_angle"]),
                np.cos(self.data_table["scan_pos_angle"]),
                self.data_table["parallax_factor_al"],
                self.data_table["relative_time_year"]
                * np.sin(self.data_table["scan_pos_angle"]),
                self.data_table["relative_time_year"]
                * np.cos(self.data_table["scan_pos_angle"]),
            ]
        )

        star_param = model_params[0:5]
        star_model = design @ star_param

        delta_ra, delta_dec = self.calc_orbit(model_params)

        # Calculate 1D projected positions of orbit model
        sin_phi = np.sin(self.data_table["scan_pos_angle"].values)
        cos_phi = np.cos(self.data_table["scan_pos_angle"].values)
        orbit_model = delta_ra * sin_phi + delta_dec * cos_phi

        return star_model + orbit_model

    def calc_residuals(self, model_params):
        """
        Residuals
        """

        # Binary model
        bin_model = self.calc_model(model_params)

        if self.verbose:
            self.print_section("Calculate residuals")

        # Epoch astrometry data
        obs_pos = self.data_table["centroid_pos_al"]
        obs_err = self.data_table["centroid_pos_error_al"]

        # Residuals
        residuals = obs_pos - bin_model

        # Number of data points
        n_obs = len(obs_pos)

        # Number of model parameters
        n_param = len(model_params)

        # Number of degrees of freedom
        n_dof = n_obs - n_param

        # Reduced chi^2
        chi2_red = np.sum(residuals**2 / obs_err**2) / n_dof

        # RUWE
        ruwe = np.sqrt(chi2_red)

        if self.verbose:
            print(f"Reduced chi^2 = {chi2_red:.2f}")
            print(f"RUWE = {ruwe:.2f}")

        return residuals

    def plot_orbit(self, model_params, output_file: str = None):
        """
        Orbit plot
        """

        mtot = model_params[11] + model_params[12]
        period = np.sqrt(model_params[5] ** 3 / mtot) * 365.25
        obs_time = np.linspace(0.0, period, 1000)

        delta_ra_full, delta_dec_full = self.calc_orbit(model_params, obs_time=obs_time)
        delta_ra, delta_dec = self.calc_orbit(model_params, obs_time=None)
        residuals = self.calc_residuals(model_params)

        if self.verbose:
            self.print_section("Plot orbit")

        res_ra, res_dec = (
            np.sin(self.data_table["scan_pos_angle"]) * residuals,
            np.cos(self.data_table["scan_pos_angle"]) * residuals,
        )

        fig = plt.figure(figsize=(4, 4))

        plt.plot(
            delta_ra_full,
            delta_dec_full,
            ls="-",
            lw=1.0,
            marker="none",
            color="black",
        )

        for i in range(len(residuals)):
            x1 = delta_ra[i] + np.sin(self.data_table["scan_pos_angle"])[i] * (
                residuals[i] + self.data_table["centroid_pos_error_al"][i]
            )
            x2 = delta_ra[i] + np.sin(self.data_table["scan_pos_angle"])[i] * (
                residuals[i] - self.data_table["centroid_pos_error_al"][i]
            )
            y1 = delta_dec[i] + np.cos(self.data_table["scan_pos_angle"])[i] * (
                residuals[i] + self.data_table["centroid_pos_error_al"][i]
            )
            y2 = delta_dec[i] + np.cos(self.data_table["scan_pos_angle"])[i] * (
                residuals[i] - self.data_table["centroid_pos_error_al"][i]
            )
            plt.plot([x1, x2], [y1, y2], "-", lw=1, color="tab:gray")

        plt.plot(
            delta_ra + res_ra,
            delta_dec + res_dec,
            "o",
            ms=4.0,
            mew=1,
            mfc="tomato",
            mec="tab:gray",
        )

        plt.xlabel(r"$\Delta$RA (mas)")
        plt.ylabel(r"$\Delta$Dec (mas)")

        if output_file is not None:
            plt.savefig(output_file)

        return fig
