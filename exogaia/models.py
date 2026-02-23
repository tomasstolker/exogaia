"""
Module with the ``StarModel`` and ``KeplerModel`` classes.
"""

from numbers import Real

import kepler
import matplotlib.pyplot as plt
import numpy as np

from astropy import constants as c
from astropy import units as u
from astropy.coordinates import get_body_barycentric
from astropy.coordinates.representation.cartesian import CartesianRepresentation
from astropy.time import Time
from beartype import beartype, typing
from matplotlib.figure import Figure

from exogaia.core import ExoGaia
from exogaia.utils import orbit_sky


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
        self.ra = epoch_astrometry.ra
        self.dec = epoch_astrometry.dec

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
    def calc_2d_model(
        self,
        model_param: typing.Union[typing.List[Real], np.ndarray],
        obs_time: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray, typing.Optional[np.ndarray]]:
        """
        Method for calculating the stellar track, returning separately
        the RA and Dec components. This function calculates the motion
        due to parallax, which might be less precise than using the
        parallax factors provided by Gaia, but these are not provided
        for RA and Dec separately. Typically these are not needed,
        but for creating a 2D plot of the stellar track, we need
        to calculate the effect in RA and Dec separately.

        Parameters
        ----------
        model_param : list(float), np.ndarray
            List or array with the model parameters, in the following
            order: RA offset (mas), Dec offset (mas), parallax (mas),
            RA proper motion (mas/yr), Dec proper motion (mas/yr).
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
            size of ``self.data_table["sin_scan_ang"]`` and
            ``self.data_table["cos_scan_ang"]``.
        """

        if obs_time is None:
            obs_time = self.data_table["obs_time_tcb"].to_numpy()
            sin_scan_ang = self.data_table["sin_scan_ang"].to_numpy()
            cos_scan_ang = self.data_table["cos_scan_ang"].to_numpy()

        else:
            sin_scan_ang = None
            cos_scan_ang = None

        rel_year = obs_time - self.ref_epoch.tcb.jyear

        n_param = len(model_param)

        # Model parameters

        ra_offset = model_param[0]
        dec_offset = model_param[1]
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

        # RA and Dec coordinates

        ra_coord = self.ra + ra_offset
        dec_coord = self.dec + ra_offset

        # Position of Gaia relative to the Solar System
        # barycenter at each observation time
        # TODO See Wright & Howard (2009)

        gaia_pos = self.barycentric_position(obs_time)
        gaia_pos = gaia_pos.xyz.to_value()

        # Tangent plane unit vectors on the sky
        # Local directions of increasing RA and Dec
        # at the source position, projected onto the sky

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

        # Design matrix for RA and Dec offsets

        design_ra = np.column_stack(
            [
                np.full(obs_time.size, 1.0),
                -(alpha_hat @ gaia_pos),
                rel_year,
            ]
        )

        design_dec = np.column_stack(
            [
                np.full(obs_time.size, 1.0),
                -(delta_hat @ gaia_pos),
                rel_year,
            ]
        )

        # RA and Dec parameter vectors

        params_ra = [ra_offset, parallax, pm_ra]
        params_dec = [dec_offset, parallax, pm_dec]

        # RA and Dec offsets

        delta_ra = design_ra @ params_ra
        delta_dec = design_dec @ params_dec

        # Add accelerations components

        if n_param in [7, 9]:
            delta_ra += 0.5 * rel_year**2 * pmdot_ra
            delta_dec += 0.5 * rel_year**2 * pmdot_dec

        if n_param == 9:
            delta_ra += (1.0 / 6.0) * rel_year**3 * pmdotdot_ra
            delta_dec += (1.0 / 6.0) * rel_year**3 * pmdotdot_dec

        # Calculate 1D projected positions

        if sin_scan_ang is not None and cos_scan_ang is not None:
            delta_pos = delta_ra * sin_scan_ang + delta_dec * cos_scan_ang
        else:
            delta_pos = None

        return delta_ra, delta_dec, delta_pos

    @beartype
    def calc_1d_model(
        self,
        model_param: typing.Union[typing.List[Real], np.ndarray],
        calc_parallax: bool = False,
    ) -> np.ndarray:
        """
        Method for calculating the 1D stellar track, including the
        effect from proper motion and parallax. The function
        will use the observation epochs and scan angles that are
        stored in the ``data_table`` of ``epoch_astrometry``,
        so it can't be used for calculating epoch astrometry
        of arbitrary observation epochs. For that purpose,
        the :class:`~exogaia.models.StarModel.calc_2d_model`
        method should be used.

        Parameters
        ----------
        model_param : list(float), np.ndarray
            List or array with the model parameters, in the following
            order: RA offset (mas), Dec offset (mas), parallax (mas),
            RA proper motion (mas/yr), Dec proper motion (mas/yr).
        calc_parallax : bool
            Calculate the parallax effect or adopt the parallax
            factors from Gaia (default: False). The latter will
            be slightly more accurate. This parameter was mainly
            included for testing purposes.

        Returns
        -------
        np.ndarray
            Array with the 1D projected positions (mas).
        """

        rel_year = self.data_table["relative_time_year"].to_numpy()
        sin_scan_ang = self.data_table["sin_scan_ang"].to_numpy()
        cos_scan_ang = self.data_table["cos_scan_ang"].to_numpy()
        par_fac = self.data_table["parallax_factor_al"].to_numpy()

        n_param = len(model_param)

        # Model parameters

        if calc_parallax:
            # This function calculates the effect from the parallax,
            # presumably in a more simplistic approach than the
            # 1D parallax factor provided with the Gaia epoch
            # astrometry.

            _, _, delta_pos = self.calc_2d_model(model_param, obs_time=None)

        else:
            # Design matrix for 5-param linear projection

            design = np.column_stack(
                [
                    sin_scan_ang,
                    cos_scan_ang,
                    par_fac,
                    rel_year * sin_scan_ang,
                    rel_year * cos_scan_ang,
                ]
            )

            delta_pos = design @ model_param[0:5]

            # Add accelerations components

            if n_param in [7, 9]:
                delta_pos += 0.5 * rel_year**2 * model_param[5]
                delta_pos += 0.5 * rel_year**2 * model_param[6]

            if n_param == 9:
                delta_pos += (1.0 / 6.0) * rel_year**3 * model_param[7]
                delta_pos += (1.0 / 6.0) * rel_year**3 * model_param[8]

        return delta_pos


class KeplerModel(ExoGaia):
    """
    Class with a Kepler model for simulating 1D astrometry.
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

        self.epoch_astrometry = epoch_astrometry
        self.data_table = epoch_astrometry.data_table
        self.ref_epoch = self.epoch_astrometry.ref_epoch
        self.verbose = verbose

    @beartype
    def solve_kepler(
        self,
        period: Real,
        ecc: Real,
        tau: Real,
        obs_time: typing.Optional[np.ndarray] = None,
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
        obs_time : np.ndarray, None
            Array with the observing epochs in Julian years on the TCB
            scale. The epochs are selected from the ``EpochAstrometry``
            if the argument is set to ``None``.

        Returns
        -------
        np.ndarray
            Array with :math:`x` coordinates in the orbital plane,
            in units of the semi-major axis.
        np.ndarray
            Array with :math:`y` coordinates in the orbital plane,
            in units of the semi-major axis.
        """

        if obs_time is None:
            rel_time_day = self.data_table["relative_time_day"].to_numpy()

        else:
            rel_time_year = obs_time - self.ref_epoch.tcb.jyear
            rel_time_day = rel_time_year * u.year.to(u.day)

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
        # in units of the semi-major axis.
        x_orb = np.cos(ecc_anom) - ecc
        y_orb = np.sqrt(1.0 - ecc**2) * np.sin(ecc_anom)

        return x_orb, y_orb

    @beartype
    def thiele_innes(
        self, sma: Real, inc: Real, aop: Real, pan: Real
    ) -> typing.Tuple[Real, Real, Real, Real]:
        """
        Method for calculating the Thiele-Innes elements.

        Parameters
        ----------
        sma : float
            Semi-major axis (mas).
        inc : float
            Inclination (rad).
        aop : float
            Argument of periastron (rad).
        pan : float
            Position angle of the ascending node (rad).

        Returns
        -------
        float
            Thiele-Innes A element (mas).
        float
            Thiele-Innes B element (mas).
        float
            Thiele-Innes F element (mas).
        float
            Thiele-Innes G element (mas).
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
        model_param: typing.Union[typing.List[Real], np.ndarray],
        obs_time: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Method for calculating the orbital model, using the 5
        parameters for the stellar track and the

        Parameters
        ----------
        model_param : list(float), np.ndarray
            List or array with the model parameters, in the following
            order:  RA offset (mas), Dec offset (mas), parallax (mas),
            RA proper motion (mas/yr), Dec proper motion (mas/yr),
            period (days), eccentricity, relative time of periastron,
            semi-major axis (mas), inclination (rad), argument of
            periastron (rad), position angle of ascending node (rad).
        obs_time : np.ndarray, None
            Array with the observing epochs in Julian years on the TCB
            scale. The epochs are selected from the ``EpochAstrometry``
            if the argument is set to ``None``.

        Returns
        -------
        np.ndarray
            Array with the RA coordinates (mas) relative to the
            system's barycenter.
        np.ndarray
            Array with the Dec coordinates (mas) relative to the
            system's barycenter.
        """

        per, ecc, tau, sma, inc, aop, pan = model_param[5:]

        # Parallax (mas)
        # parallax = model_param[2]  # (mas)

        # Orbital period (days)
        # period = np.sqrt(sma**3 / (m1 + m2)) * 365.25

        # Primary semi-major axis (au)
        # sma1 = sma * m2 / (m1 + m2)

        # Add 180 deg to convert from secondary to primary
        # aop += np.pi

        # Thiele-Innes elements
        thiele_innes_a, thiele_innes_b, thiele_innes_f, thiele_innes_g = (
            self.thiele_innes(sma, inc, aop, pan)
        )

        # Solve the Kepler equation
        x_orb, y_orb = self.solve_kepler(per, ecc, tau, obs_time)

        # Rotate (x_orb, y_orb) into sky plane (x_sky, y_sky)
        # See equation 9 in Holl et al. (2023)
        # The units of x_sky and y_sky are mas
        # because the units of sma is mas
        x_sky = thiele_innes_b * x_orb + thiele_innes_g * y_orb
        y_sky = thiele_innes_a * x_orb + thiele_innes_f * y_orb

        return x_sky, y_sky

    @beartype
    def calc_2d_model(
        self,
        model_param: typing.Union[typing.List[Real], np.ndarray],
        obs_time: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Method for calculating the astrometry of the combined
        stellar track and Kepler orbit.

        Parameters
        ----------
        model_param : list(float), np.ndarray
            List or array with the model parameters, in the following
            order:  RA offset (mas), Dec offset (mas), parallax (mas),
            RA proper motion (mas/yr), Dec proper motion (mas/yr),
            period (days), eccentricity, relative time of periastron,
            semi-major axis (mas), inclination (rad), argument of
            periastron (rad), position angle of ascending node (rad).
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
        """

        # Epoch astrometry data
        rel_year = self.data_table["relative_time_year"].to_numpy()
        sin_scan_ang = self.data_table["sin_scan_ang"].to_numpy()
        cos_scan_ang = self.data_table["cos_scan_ang"].to_numpy()
        par_fac = self.data_table["parallax_factor_al"].to_numpy()

        # Orbital period (days)
        # Use sma in au and masses in Msun
        # period = (
        #     np.sqrt(model_param[5] ** 3 / (model_param[11] + model_param[12])) * 365.25
        # )

        if self.verbose:
            self.print_section("Calculate orbit model")

            print("Stellar parameters:")
            print(f"   - RA (deg) = {model_param[0]:.3f}")
            print(f"   - Dec (deg) = {model_param[1]:.3f}")
            print(f"   - Parallax (mas) = {model_param[2]:.3f}")
            print(f"   - Proper motion in RA (mas/yr) = {model_param[3]:.3f}")
            print(f"   - Proper motion in Dec (mas/yr) = {model_param[4]:.3f}")

            print("\nOrbit parameters:")
            print(f"   - Period (days) = {model_param[5]:.3f}")
            print(f"   - Eccentricity = {model_param[6]:.3f}")
            print(f"   - Relative time of periastron = {model_param[7]:.3f}")
            print(f"   - Semi-major axis (mas) = {model_param[8]:.3f}")
            print(f"   - Inclination (deg) = {np.degrees(model_param[9]):.3f}")
            print(
                f"   - Argument of periastron (deg) = {np.degrees(model_param[10]):.3f}"
            )
            print(
                f"   - PA of ascending node (deg) = {np.degrees(model_param[11]):.3f}"
            )

        # Stellar track

        star_model = StarModel(self.epoch_astrometry)
        delta_ra_star, delta_dec_star, _ = star_model.calc_2d_model(
            model_param, obs_time
        )

        # Orbit

        delta_ra_orbit, delta_dec_orbit = self.calc_orbit(model_param, obs_time)

        # Total RA/Dec offset

        delta_ra = delta_ra_star + delta_ra_orbit
        delta_dec = delta_dec_star + delta_dec_orbit

        return delta_ra, delta_dec

    @beartype
    def calc_1d_model(
        self,
        model_param: typing.Union[typing.List[Real], np.ndarray],
        obs_time: typing.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Method for calculating the astrometry of the combined
        stellar track and Kepler orbit.

        Parameters
        ----------
        model_param : list(float), np.ndarray
            List or array with the model parameters, in the following
            order:  RA offset (mas), Dec offset (mas), parallax (mas),
            RA proper motion (mas/yr), Dec proper motion (mas/yr),
            period (days), eccentricity, relative time of periastron,
            semi-major axis (mas), inclination (rad), argument of
            periastron (rad), position angle of ascending node (rad).
        obs_time : np.ndarray, None
            Array with the observing epochs in Julian years on the TCB
            scale. The epochs are selected from the ``EpochAstrometry``
            if the argument is set to ``None``.

        Returns
        -------
        np.ndarray
            Array with the 1D projected astrometry at the
            observation epoch of ``EpochAstrometry``.
        """

        # Epoch astrometry data
        rel_year = self.data_table["relative_time_year"].to_numpy()
        sin_scan_ang = self.data_table["sin_scan_ang"].to_numpy()
        cos_scan_ang = self.data_table["cos_scan_ang"].to_numpy()
        par_fac = self.data_table["parallax_factor_al"].to_numpy()

        # Orbital period (days)
        # Use sma in au and masses in Msun
        # period = (
        #     np.sqrt(model_param[5] ** 3 / (model_param[11] + model_param[12])) * 365.25
        # )

        if self.verbose:
            self.print_section("Calculate orbit model")

            print("Stellar parameters:")
            print(f"   - RA (deg) = {model_param[0]:.3f}")
            print(f"   - Dec (deg) = {model_param[1]:.3f}")
            print(f"   - Parallax (mas) = {model_param[2]:.3f}")
            print(f"   - Proper motion in RA (mas/yr) = {model_param[3]:.3f}")
            print(f"   - Proper motion in Dec (mas/yr) = {model_param[4]:.3f}")

            print("\nOrbit parameters:")
            print(f"   - Period (days) = {model_param[5]:.3f}")
            print(f"   - Eccentricity = {model_param[6]:.3f}")
            print(f"   - Relative time of periastron = {model_param[7]:.3f}")
            print(f"   - Semi-major axis (mas) = {model_param[8]:.3f}")
            print(f"   - Inclination (deg) = {np.degrees(model_param[9]):.3f}")
            print(
                f"   - Argument of periastron (deg) = {np.degrees(model_param[10]):.3f}"
            )
            print(
                f"   - PA of ascending node (deg) = {np.degrees(model_param[11]):.3f}"
            )

        # Design matrix for 5-param linear projection

        design = np.column_stack(
            [
                sin_scan_ang,
                cos_scan_ang,
                par_fac,
                rel_year * sin_scan_ang,
                rel_year * cos_scan_ang,
            ]
        )

        # 1D stellar track from linear projection on design matrix

        star_model = design @ model_param[0:5]

        # star_comp = StarModel(self.epoch_astrometry)
        # star_test = star_comp.calc_1d_model(model_param, calc_parallax=True)
        # plt.plot(rel_year, star_model-star_test, "o")
        # plt.show()

        delta_ra, delta_dec = self.calc_orbit(model_param, obs_time)

        # Calculate 1D projected positions of orbit model
        orbit_model = delta_ra * sin_scan_ang + delta_dec * cos_scan_ang

        return star_model + orbit_model

    @beartype
    def calc_residuals(
        self, model_param: typing.Union[typing.List[Real], np.ndarray]
    ) -> np.ndarray:
        """
        Method for calculating the residuals between the model
        astrometry and the Gaia epoch astrometry.

        Parameters
        ----------
        model_param : list(float), np.ndarray
            List or array with the model parameters, in the following
            order:  RA offset (mas), Dec offset (mas), parallax (mas),
            RA proper motion (mas/yr), Dec proper motion (mas/yr),
            period (days), eccentricity, relative time of periastron,
            semi-major axis (mas), inclination (rad), argument of
            periastron (rad), position angle of ascending node (rad).

        Returns
        -------
        np.ndarray
            Array with the residuals, as data minus model.
        """

        if self.verbose:
            self.print_section("Calculate residuals")

        # Epoch astrometry data
        obs_pos = self.data_table["centroid_pos_al"].to_numpy()
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()

        # Residuals
        residuals = obs_pos - self.calc_1d_model(model_param)

        # Number of data points
        n_obs = len(obs_pos)

        # Number of model parameters
        n_param = len(model_param)

        # Number of degrees of freedom
        n_dof = n_obs - n_param

        # Reduced chi^2
        chi2_red = np.sum(residuals**2 / obs_err**2) / n_dof

        # RUWE
        if self.epoch_astrometry.sim_data:
            ruwe = np.sqrt(chi2_red)
        else:
            ruwe = np.sqrt(chi2_red) / self.epoch_astrometry.u0_norm

        if self.verbose:
            print(f"\nReduced chi^2 = {chi2_red:.2f}")
            print(f"RUWE = {ruwe:.2f}")

        return residuals

    @beartype
    def plot_orbit(
        self,
        model_param: typing.Union[typing.List[Real], np.ndarray],
        plot_file: typing.Optional[str] = None,
    ) -> Figure:
        """
        Method for plotting the on-sky orbit.

        Parameters
        ----------
        model_param : list(float), np.ndarray
            List or array with the model parameters, in the following
            order:  RA offset (mas), Dec offset (mas), parallax (mas),
            RA proper motion (mas/yr), Dec proper motion (mas/yr),
            period (days), eccentricity, relative time of periastron,
            semi-major axis (mas), inclination (rad), argument of
            periastron (rad), position angle of ascending node (rad).
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
        sin_scan_ang = self.data_table["sin_scan_ang"].to_numpy()
        cos_scan_ang = self.data_table["cos_scan_ang"].to_numpy()

        yr_start = self.ref_epoch
        yr_end = self.ref_epoch + (model_param[5] / 365.25) * u.yr

        obs_time_full = np.linspace(yr_start, yr_end, 1000)
        obs_time_full = obs_time_full.tcb.jyear

        # mtot = model_param[11] + model_param[12]
        # period = np.sqrt(model_param[5] ** 3 / mtot) * 365.25

        delta_ra_full, delta_dec_full = self.calc_orbit(
            model_param, obs_time=obs_time_full
        )

        delta_ra, delta_dec = self.calc_orbit(model_param, obs_time=None)

        residuals = self.calc_residuals(model_param)

        if self.verbose:
            self.print_section("Plot orbit")

        res_ra, res_dec = (
            sin_scan_ang * residuals,
            cos_scan_ang * residuals,
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
            x1 = delta_ra[i] + sin_scan_ang[i] * (res_item + obs_err)
            x2 = delta_ra[i] + sin_scan_ang[i] * (res_item - obs_err)
            y1 = delta_dec[i] + cos_scan_ang[i] * (res_item + obs_err)
            y2 = delta_dec[i] + cos_scan_ang[i] * (res_item - obs_err)
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

        plt.plot(
            0.0,
            0.0,
            marker="x",
            ms=5.0,
            mew=1.5,
            ls="none",
            color="tab:gray",
            mec="tab:gray",
            zorder=3,
            label="Barycenter",
        )

        # Time of periastron in Julian years
        t_per = self.ref_epoch.value + (model_param[5] * model_param[7]) / 365.25

        delta_ra_per, delta_dec_per = self.calc_orbit(
            model_param, obs_time=np.array([t_per])
        )

        plt.plot(
            delta_ra_per,
            delta_dec_per,
            marker="+",
            ms=5.0,
            mew=1.5,
            ls="none",
            color="tab:olive",
            zorder=3,
            label=rf"Periastron ($t_\mathrm{{per}} = {t_per:.2f}$)",
        )

        x_nodes, y_nodes = orbit_sky(
            nu=np.array([model_param[10], np.pi + model_param[10]]),
            sma=model_param[8],
            ecc=model_param[6],
            inc=model_param[9],
            aop=model_param[10],
            pan=model_param[11],
        )

        plt.plot(
            x_nodes,
            y_nodes,
            ls=":",
            lw=1,
            marker="none",
            color="tab:gray",
            label="Line of nodes",
        )

        plt.xlabel(r"$\Delta$RA (mas)")
        plt.ylabel(r"$\Delta$Dec (mas)")

        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        lim_list = np.array([x_lim[0], x_lim[1], y_lim[0], y_lim[1]])
        lim_max = np.amax(np.abs(lim_list))

        plt.xlim(lim_max, -lim_max)
        plt.ylim(-lim_max, lim_max)

        plt.legend(loc="upper left", frameon=False, fontsize=8)

        if plot_file is None:
            plt.show()
        else:
            plt.savefig(plot_file)

        return fig
