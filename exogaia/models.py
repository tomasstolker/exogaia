"""
Module with the ``StarModel`` and ``BinaryModel`` classes.
"""

import kepler
import matplotlib.pyplot as plt
import numpy as np

from astropy import constants as c
from astropy.coordinates import get_body_barycentric
from astropy.coordinates.representation.cartesian import CartesianRepresentation
from astropy.time import Time
from beartype import beartype, typing
from matplotlib.figure import Figure

from exogaia.core import ExoGaia


class StarModel(ExoGaia):
    """
    Class with an astrometric model of a stellar track.
    """

    @beartype
    def __init__(self, epoch_astrometry) -> None:
        """
        Parameters
        ----------
        epoch_astrometry : EpochAstrometry
            ``EpochAstrometry`` object that contains the data.

        Returns
        -------
        NoneType
            None
        """

        self.data_table = epoch_astrometry.data_table
        self.ref_epoch = epoch_astrometry.ref_epoch

    @beartype
    def barycentric_position(self, obs_time: np.ndarray) -> CartesianRepresentation:
        """
        Method for calculating the Cartesian position of the Gaia
        satellite relative to the barycenter of the Solar System.

        Parameters
        ----------
        obs_time : np.ndarray
            Array with the observing epochs in Julian years
            on the TCB scale.

        Returns
        -------
        CartesianRepresentation
            Cartesian coordinates of the Gaia satellite at ``obs_time``.
        """

        time = Time(obs_time, format="jyear", scale="tcb")

        bar_pos = get_body_barycentric(body="earth", time=time, ephemeris=None)

        # Gaia orbits at Lagrangian L2 of the Earth-Sun-Moon system
        # https://en.wikipedia.org/wiki/Lagrange_point#L2

        mu = c.M_earth.value / (c.M_sun.value + c.M_earth.value)

        return bar_pos + bar_pos * (mu / 3) ** (1 / 3)

    @beartype
    def calc_model(
        self,
        model_param: typing.Union[typing.List[float], np.ndarray],
        obs_time: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray, typing.Optional[np.ndarray]]:
        """
        Method for calculating the stellar track.

        Parameters
        ----------
        model_param : list(float), np.ndarray
            List or array with the model parameters, in the following order:
            RA (deg), Dec (deg), parallax (mas), RA proper motion (mas/yr),
            Dec proper motion (mas/yr).
        obs_time : np.ndarray, None
            Array with the observing epochs in Julian years on the TCB
            scale. The epochs are selected from the ``EpochAstrometry``
            if the argument is set to ``None``.

        Returns
        -------
        np.ndarray
            Array with the RA coordinates (mas) relative to the
            RA coordinate at ``ref_epoch``.
        np.ndarray
            Array with the Dec coordinates (mas) relative to the
            Dec coordinate at ``ref_epoch``.
        np.ndarray
            Array with the 1D projected positions (mas). Will only
            be returned if the size of ``obs_time`` is equal to the
            size of ``self.data_table["scan_pos_angle"]``.
        """

        if obs_time is None:
            obs_time = self.data_table["obs_time_tcb"].to_numpy()

        scan_ang = self.data_table["scan_pos_angle"].to_numpy()

        n_param = len(model_param)

        ra_coord = model_param[0]
        dec_coord = model_param[1]
        parallax = model_param[2]
        pm_ra = model_param[3]
        pm_dec = model_param[4]

        if len(model_param) > 5:
            pmdot_ra = model_param[5]
            pmdot_dec = model_param[6]
        else:
            pmdot_ra = None
            pmdot_dec = None

        if len(model_param) > 7:
            pmdotdot_ra = model_param[7]
            pmdotdot_dec = model_param[8]

        else:
            pmdotdot_ra = None
            pmdotdot_dec = None

        gaia_pos = self.barycentric_position(obs_time)
        gaia_pos = gaia_pos.xyz.to_value()

        alpha_hat = np.array(
            [-np.sin(np.radians(ra_coord)), np.cos(np.radians(ra_coord)), 0.0]
        )

        delta_hat = np.array(
            [
                -np.cos(np.radians(ra_coord)) * np.sin(np.radians(dec_coord)),
                -np.sin(np.radians(ra_coord)) * np.sin(np.radians(dec_coord)),
                np.cos(np.radians(dec_coord)),
            ]
        )

        design_ra = np.column_stack(
            [
                np.full(obs_time.size, 1.0),
                -(alpha_hat @ gaia_pos),
                obs_time - self.ref_epoch.tcb.jyear,
            ]
        )

        design_dec = np.column_stack(
            [
                np.full(obs_time.size, 1.0),
                -(delta_hat @ gaia_pos),
                obs_time - self.ref_epoch.tcb.jyear,
            ]
        )

        params_ra = [ra_coord, parallax, pm_ra]
        params_dec = [dec_coord, parallax, pm_dec]

        delta_ra = design_ra @ params_ra
        delta_dec = design_dec @ params_dec

        if n_param in [7, 9]:
            delta_ra += 0.5 * (obs_time - self.ref_epoch.tcb.jyear) ** 2 * pmdot_ra

            delta_dec += 0.5 * (obs_time - self.ref_epoch.tcb.jyear) ** 2 * pmdot_dec

        if n_param == 9:
            delta_ra += (
                (1.0 / 6.0) * (obs_time - self.ref_epoch.tcb.jyear) ** 3 * pmdotdot_ra
            )

            delta_dec += (
                (1.0 / 6.0) * (obs_time - self.ref_epoch.tcb.jyear) ** 3 * pmdotdot_dec
            )

        # Calculate 1D projected positions
        if scan_ang.size == delta_ra.size and scan_ang.size == delta_dec.size:
            delta_pos = delta_ra * np.sin(scan_ang) + delta_dec * np.cos(scan_ang)
        else:
            delta_pos = None

        return delta_ra - ra_coord, delta_dec - dec_coord, delta_pos


class BinaryModel(ExoGaia):
    """
    Class with an astrometric model of a binary system
    with a dark companion.
    """

    @beartype
    def __init__(self, epoch_astrometry, verbose: bool = True) -> None:
        """
        Parameters
        ----------
        epoch_astrometry : EpochAstrometry
            ``EpochAstrometry`` object that contains the data.
        verbose : bool
            Print some information.

        Returns
        -------
        NoneType
            None
        """

        self.data_table = epoch_astrometry.data_table
        self.verbose = verbose

    @beartype
    def solve_kepler(
        self,
        period: float,
        ecc: float,
        tau: float,
        rel_time_day: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Method for solving the Kepler equation.

        Parameters
        ----------
        period : float
            Orbital period (days).
        ecc : float
            Eccentricity.
        tau : float
            Time of periastron, as fraction of the period,
            relative to ``ref_epoch``.
        rel_time_day : np.ndarray, None
            Array with the observing epochs in Julian days relative
            to the ``ref_epoch``. The epochs are selected from the
            ``EpochAstrometry`` if the argument is set to ``None``.

        Returns
        -------
        np.ndarray
            Array with :math:`x` coordinates (au) in the orbital plane.
        np.ndarray
            Array with :math:`y` coordinates (au) in the orbital plane.
        """

        if rel_time_day is None:
            rel_time_day = self.data_table["relative_time_day"].to_numpy()

        # t_per: time of periastron, relative to ref_epoch (days)
        # tau: fractional time of periastron, relative to ref_epoch
        t_per = period * tau

        # rel_time_day: observation times in days relative to ref_epoch
        # delta_t: observation times relative to time of periastron
        delta_t = rel_time_day - t_per

        # Mean anomaly at observation epochs
        mean_anom_obs = delta_t * 2.0 * np.pi / period

        # Solve Kepler's equation
        ecc_anom, _, _ = kepler.kepler(mean_anom_obs, ecc)

        # (x, y) position in the orbital plane
        x_orb = np.cos(ecc_anom) - ecc
        y_orb = np.sqrt(1.0 - ecc**2) * np.sin(ecc_anom)

        return x_orb, y_orb

    @beartype
    def thiele_innes(
        self, sma: float, inc: float, aop: float, pan: float
    ) -> typing.Tuple[float, float, float, float]:
        """
        Method for calculating the Thiele-Innes constants.

        Parameters
        ----------
        sma : float
            Semi-major axis (au).
        inc : float
            Inclination (rad).
        aop : float
            Argument of periastron (rad).
        pan : float
            Position angle of the ascending node (rad).

        Returns
        -------
        float
            Thiele-Innes A constant.
        float
            Thiele-Innes B constant.
        float
            Thiele-Innes F constant.
        float
            Thiele-Innes G constant.
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

    @beartype
    def calc_orbit(
        self,
        model_param: typing.Union[typing.List[float], np.ndarray],
        rel_time_day: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Method for calculating the orbital model.

        Parameters
        ----------
        model_param : list(float), np.ndarray
            List or array with the model parameters, in the following
            order:  RA (deg), Dec (deg), parallax (mas), RA proper
            motion (mas/yr), Dec proper motion (mas/yr), semi-major
            axis (au), eccentricity, inclination (rad), argument of
            periastron (rad), position angle of ascending node (rad),
            relative time of periastron, primary mass (Msun),
            secondary mass (Msun).
        rel_time_day : np.ndarray, None
            Array with the observing times in Julian days relative
            to the ``ref_epoch``. The observing times are selected
            from the ``EpochAstrometry`` if the argument is set to
            ``None``.

        Returns
        -------
        np.ndarray
            Array with the RA coordinates (mas) relative to the
            system's barycenter.
        np.ndarray
            Array with the Dec coordinates (mas) relative to the
            system's barycenter.
        """

        sma, ecc, inc, aop, pan, tau, m1, m2 = model_param[5:13]

        # Parallax (mas)
        parallax = model_param[2]  # (mas)

        # Orbital period (days)
        period = np.sqrt(sma**3 / (m1 + m2)) * 365.25

        # Solve the Kepler equation
        x_orb, y_orb = self.solve_kepler(period, ecc, tau, rel_time_day)

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

    @beartype
    def calc_model(
        self, model_param: typing.Union[typing.List[float], np.ndarray]
    ) -> np.ndarray:
        """
        Method for calculating the astrometry of the combined
        stellar track and Kepler orbit.

        Parameters
        ----------
        model_param : list(float), np.ndarray
            List or array with the model parameters, in the following
            order:  RA (deg), Dec (deg), parallax (mas), RA proper
            motion (mas/yr), Dec proper motion (mas/yr), semi-major
            axis (au), eccentricity, inclination (rad), argument of
            periastron (rad), position angle of ascending node (rad),
            relative time of periastron, primary mass (Msun),
            secondary mass (Msun).

        Returns
        -------
        np.ndarray
            Array with the 1D projected astrometry at the
            observation epoch of ``EpochAstrometry``.
        """

        # Epoch astrometry data
        rel_yr = self.data_table["relative_time_year"].to_numpy()
        scan_ang = self.data_table["scan_pos_angle"].to_numpy()
        par_fac = self.data_table["parallax_factor_al"].to_numpy()

        # Orbital period (days)
        # Use sma in au and masses in Msun
        period = (
            np.sqrt(model_param[5] ** 3 / (model_param[11] + model_param[12])) * 365.25
        )

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
            print(f"   - Period (days) = {period:.2f}")

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

    @beartype
    def calc_residuals(
        self, model_param: typing.Union[typing.List[float], np.ndarray]
    ) -> np.ndarray:
        """
        Method for calculating the residuals between model
        astrometry and the epoch astrometry.

        Parameters
        ----------
        model_param : list(float), np.ndarray
            List or array with the model parameters, in the following
            order:  RA (deg), Dec (deg), parallax (mas), RA proper
            motion (mas/yr), Dec proper motion (mas/yr), semi-major
            axis (au), eccentricity, inclination (rad), argument of
            periastron (rad), position angle of ascending node (rad),
            relative time of periastron, primary mass (Msun),
            secondary mass (Msun).

        Returns
        -------
        np.ndarray
            Array with the residuals, as data minus model.
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

    @beartype
    def plot_orbit(
        self,
        model_param: typing.Union[typing.List[float], np.ndarray],
        plot_file: typing.Optional[str] = None,
    ) -> Figure:
        """
        Method for plotting the on-sky orbit.

        Parameters
        ----------
        model_param : list(float), np.ndarray
            List or array with the model parameters, in the following
            order:  RA (deg), Dec (deg), parallax (mas), RA proper
            motion (mas/yr), Dec proper motion (mas/yr), semi-major
            axis (au), eccentricity, inclination (rad), argument of
            periastron (rad), position angle of ascending node (rad),
            relative time of periastron, primary mass (Msun),
            secondary mass (Msun).
        plot_file : str, None
            File name for the output plot. The plot is shown
            instead of stored if the arguments is set to ``None``.

        Returns
        -------
        Figure
            The Matplotlib ``Figure`` object that can be used
            for further adjustments of the plot.
        """

        # Epoch astrometry data
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()
        scan_ang = self.data_table["scan_pos_angle"].to_numpy()

        mtot = model_param[11] + model_param[12]
        period = np.sqrt(model_param[5] ** 3 / mtot) * 365.25
        rel_time_day = np.linspace(0.0, period, 1000)

        delta_ra_full, delta_dec_full = self.calc_orbit(
            model_param, rel_time_day=rel_time_day
        )
        delta_ra, delta_dec = self.calc_orbit(model_param, rel_time_day=None)
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

        if plot_file is None:
            plt.show()
        else:
            plt.savefig(plot_file)

        return fig
