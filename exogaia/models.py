"""
Module with the ``BinaryModel`` class.
"""

from typing import List, Optional, Tuple, Union

import kepler
import matplotlib.pyplot as plt
import numpy as np

from astropy import constants as c
from astropy.coordinates import get_body_barycentric
from astropy.coordinates.representation.cartesian import CartesianRepresentation
from astropy.time import Time
from matplotlib.figure import Figure
from typeguard import typechecked

from exogaia.core import ExoGaia


class StarModel(ExoGaia):
    """
    Class for a star model.
    """

    @typechecked
    def __init__(
        self, star_param: Union[List[float], np.ndarray], epoch_astrometry
    ) -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        self.n_param = len(star_param)

        self.ra_coord = star_param[0]
        self.dec_coord = star_param[1]
        self.parallax = star_param[2]
        self.pm_ra = star_param[3]
        self.pm_dec = star_param[4]

        if len(star_param) > 5:
            self.pmdot_ra = star_param[5]
            self.pmdot_dec = star_param[6]

        if len(star_param) > 7:
            self.pmdotdot_ra = star_param[7]
            self.pmdotdot_dec = star_param[8]

        self.data_table = epoch_astrometry.data_table
        self.ref_epoch = epoch_astrometry.ref_epoch

    @typechecked
    def barycentric_position(self, obs_time: np.ndarray) -> CartesianRepresentation:
        """
        Returns
        -------
        NoneType
            None
        """

        bar_pos = get_body_barycentric(
            body="earth", time=Time(obs_time, format="jyear"), ephemeris=None
        )

        # Gaia orbits in Lagrangian L2 of the Earth-Sun-Moon system
        # https://en.wikipedia.org/wiki/Lagrange_point#L2

        mu = c.M_earth.value / (c.M_sun.value + c.M_earth.value)

        return bar_pos + bar_pos * (mu / 3) ** (1 / 3)

    @typechecked
    def calc_model(
        self, obs_time: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        NoneType
            None
        """

        if obs_time is None:
            obs_time = self.data_table["obs_time_tcb"].to_numpy()

        gaia_pos = self.barycentric_position(obs_time)
        gaia_pos = gaia_pos.xyz.to_value()

        alpha_hat = np.array(
            [-np.sin(np.radians(self.ra_coord)), np.cos(np.radians(self.ra_coord)), 0.0]
        )

        delta_hat = np.array(
            [
                -np.cos(np.radians(self.ra_coord)) * np.sin(np.radians(self.dec_coord)),
                -np.sin(np.radians(self.ra_coord)) * np.sin(np.radians(self.dec_coord)),
                np.cos(np.radians(self.dec_coord)),
            ]
        )

        design_ra = np.column_stack(
            [
                np.full(obs_time.size, 1.0),
                -(alpha_hat @ gaia_pos),
                obs_time - self.ref_epoch.value,
            ]
        )

        design_dec = np.column_stack(
            [
                np.full(obs_time.size, 1.0),
                -(delta_hat @ gaia_pos),
                obs_time - self.ref_epoch.value,
            ]
        )

        params_ra = [self.ra_coord, self.parallax, self.pm_ra]
        params_dec = [self.dec_coord, self.parallax, self.pm_dec]

        delta_ra = design_ra @ params_ra
        delta_dec = design_dec @ params_dec

        if self.n_param in [7, 9]:
            delta_ra += 0.5 * (obs_time - self.ref_epoch.value) ** 2 * self.pmdot_ra

            delta_dec += 0.5 * (obs_time - self.ref_epoch.value) ** 2 * self.pmdot_dec

        if self.n_param == 9:
            delta_ra += (
                (1.0 / 6.0) * (obs_time - self.ref_epoch.value) ** 3 * self.pmdotdot_ra
            )

            delta_dec += (
                (1.0 / 6.0) * (obs_time - self.ref_epoch.value) ** 3 * self.pmdotdot_dec
            )

        return delta_ra - self.ra_coord, delta_dec - self.dec_coord


class BinaryModel(ExoGaia):
    """
    Class for a binary model.
    """

    @typechecked
    def __init__(self, epoch_astrometry, verbose: bool = True) -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        self.data_table = epoch_astrometry.data_table
        self.verbose = verbose

    @typechecked
    def solve_kepler(
        self,
        period: float,
        ecc: float,
        tau: float,
        obs_time: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the Kepler equation
        """

        if obs_time is None:
            obs_time = self.data_table["relative_time_day"].to_numpy()

        # Time of periastron (days)
        t_per = period * tau

        # Days relative to periastron
        delta_t = obs_time - t_per

        # Mean anomaly at observation epochs
        mean_anom_obs = delta_t * 2.0 * np.pi / period

        # Solve Kepler's equation
        ecc_anom, _, _ = kepler.kepler(mean_anom_obs, ecc)

        # (x, y) position in the orbital plane
        x_orb = np.cos(ecc_anom) - ecc
        y_orb = np.sqrt(1.0 - ecc**2) * np.sin(ecc_anom)

        return x_orb, y_orb

    @typechecked
    def thiele_innes(
        self, sma: float, inc: float, aop: float, pan: float
    ) -> Tuple[float, float, float, float]:
        """
        Thiele-Innes constants
        """

        thiele_innes_a = sma * (
            np.cos(aop) * np.cos(pan) - np.sin(aop) * np.sin(pan) * np.cos(inc)
        )

        thiele_innes_b = sma * (
            np.cos(aop) * np.sin(pan) + np.sin(aop) * np.cos(pan) * np.cos(inc)
        )

        thiele_innes_f = -sma * (
            np.sin(aop) * np.cos(pan) + np.cos(aop) * np.sin(pan) * np.cos(inc)
        )

        thiele_innes_g = -sma * (
            np.sin(aop) * np.sin(pan) - np.cos(aop) * np.cos(pan) * np.cos(inc)
        )

        return thiele_innes_a, thiele_innes_b, thiele_innes_f, thiele_innes_g

    @typechecked
    def calc_orbit(
        self,
        model_param: Union[List[float], np.ndarray],
        obs_time: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Orbit model
        """

        sma, ecc, inc, aop, pan, tau, m1, m2 = model_param[5:13]

        # Parallax (mas)
        parallax = model_param[2]  # (mas)

        # Orbital period (days)
        period = np.sqrt(sma**3 / (m1 + m2)) * 365.25

        # Solve the Kepler equation
        x_orb, y_orb = self.solve_kepler(period, ecc, tau, obs_time)

        # Primary semi-major axis (au)
        sma1 = sma * m2 / (m1 + m2)

        # Add 180 deg to convert from secondary to primary
        aop += np.pi

        # Thiele-Innes constants
        thiele_innes_a, thiele_innes_b, thiele_innes_f, thiele_innes_g = (
            self.thiele_innes(sma1, inc, aop, pan)
        )

        # Rotate (x_orb, y_orb) into sky plane (x_sky, y_sky)
        x_sky = thiele_innes_b * x_orb + thiele_innes_g * y_orb
        y_sky = thiele_innes_a * x_orb + thiele_innes_f * y_orb

        # Scale from au to mas
        x_sky *= parallax
        y_sky *= parallax

        return x_sky, y_sky

    @typechecked
    def calc_model(self, model_param: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Binary model
        """

        # Epoch astrometry data
        rel_yr = self.data_table["relative_time_year"].to_numpy()
        scan_ang = self.data_table["scan_pos_angle"].to_numpy()
        par_fac = self.data_table["parallax_factor_al"].to_numpy()

        if self.verbose:
            self.print_section("Calculate binary model")

            print("Stellar parameters:")
            print(f"   - RA (deg) = {model_param[0]:.2f}")
            print(f"   - Dec (deg) = {model_param[1]:.2f}")
            print(f"   - Parallax (mas) = {model_param[2]:.2f}")
            print(f"   - Proper motion in RA (mas/yr) = {model_param[3]:.2f}")
            print(f"   - Proper motion in Dec (mas/yr) = {model_param[4]:.2f}")

            print("\nOrbit parameters:")
            print(f"   - Primary mass (Msun) = {model_param[11]:.2f}")
            print(f"   - Secondary mass (Msun) = {model_param[12]:.2f}")
            print(f"   - Semi-major axis (au) = {model_param[5]:.2f}")
            print(f"   - Eccentricity = {model_param[6]:.2f}")
            print(f"   - Inclination (deg) = {np.degrees(model_param[7]):.2f}")
            print(
                f"   - Argument of periastron (deg) = {np.degrees(model_param[8]):.2f}"
            )
            print(f"   - PA of ascending node (deg) = {np.degrees(model_param[9]):.2f}")
            print(f"   - Relative time of periastron = {model_param[10]:.2f}")

        # Design matrix for 5-param linear projection
        design = np.column_stack(
            [
                np.sin(scan_ang),
                np.cos(scan_ang),
                par_fac,
                rel_yr * np.sin(scan_ang),
                rel_yr * np.cos(scan_ang),
            ]
        )

        star_param = model_param[0:5]
        star_model = design @ star_param

        delta_ra, delta_dec = self.calc_orbit(model_param)

        # Calculate 1D projected positions of orbit model
        orbit_model = delta_ra * np.sin(scan_ang) + delta_dec * np.cos(scan_ang)

        return star_model + orbit_model

    @typechecked
    def calc_residuals(self, model_param: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Residuals
        """

        # Binary model
        bin_model = self.calc_model(model_param)

        if self.verbose:
            self.print_section("Calculate residuals")

        # Epoch astrometry data
        obs_pos = self.data_table["centroid_pos_al"].to_numpy()
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()

        # Residuals
        residuals = obs_pos - bin_model

        # Number of data points
        n_obs = len(obs_pos)

        # Number of model parameters
        n_param = len(model_param)

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

    @typechecked
    def plot_orbit(
        self, model_param: Union[List[float], np.ndarray], output_file: str = None
    ) -> Figure:
        """
        Orbit plot
        """

        # Epoch astrometry data
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()
        scan_ang = self.data_table["scan_pos_angle"].to_numpy()

        mtot = model_param[11] + model_param[12]
        period = np.sqrt(model_param[5] ** 3 / mtot) * 365.25
        obs_time = np.linspace(0.0, period, 1000)

        delta_ra_full, delta_dec_full = self.calc_orbit(model_param, obs_time=obs_time)
        delta_ra, delta_dec = self.calc_orbit(model_param, obs_time=None)
        residuals = self.calc_residuals(model_param)

        if self.verbose:
            self.print_section("Plot orbit")

        res_ra, res_dec = (
            np.sin(scan_ang) * residuals,
            np.cos(scan_ang) * residuals,
        )

        fig = plt.figure(figsize=(4, 4))
        ax = plt.gca()

        plt.plot(
            delta_ra_full,
            delta_dec_full,
            ls="-",
            lw=1.0,
            marker="none",
            color="black",
            zorder=1,
        )

        for i, res_item in enumerate(residuals):
            x1 = delta_ra[i] + np.sin(scan_ang[i]) * (res_item + obs_err)
            x2 = delta_ra[i] + np.sin(scan_ang[i]) * (res_item - obs_err)
            y1 = delta_dec[i] + np.cos(scan_ang[i]) * (res_item + obs_err)
            y2 = delta_dec[i] + np.cos(scan_ang[i]) * (res_item - obs_err)
            plt.plot([x1, x2], [y1, y2], "-", lw=1, color="black")

        plt.plot(
            delta_ra[0] + res_ra[0],
            delta_dec[0] + res_dec[0],
            "o",
            ms=4.0,
            mew=1,
            mfc="tab:green",
            mec="black",
            zorder=3,
        )

        plt.plot(
            delta_ra[1:-1] + res_ra[1:-1],
            delta_dec[1:-1] + res_dec[1:-1],
            "o",
            ms=4.0,
            mew=1,
            mfc="tab:purple",
            mec="black",
            zorder=2,
        )

        plt.plot(
            delta_ra[-1] + res_ra[-1],
            delta_dec[-1] + res_dec[-1],
            "o",
            ms=4.0,
            mew=1,
            mfc="tab:red",
            mec="black",
            zorder=3,
        )

        plt.xlabel(r"$\Delta$RA (mas)")
        plt.ylabel(r"$\Delta$Dec (mas)")

        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        lim_list = np.array([x_lim[0], x_lim[1], y_lim[0], y_lim[1]])
        lim_max = np.amax(np.abs(lim_list))

        plt.xlim(lim_max, -lim_max)
        plt.ylim(-lim_max, lim_max)

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)

        return fig
