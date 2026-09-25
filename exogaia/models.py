"""
Module with the ``StarModel`` and ``KeplerModel`` classes.
"""

from numbers import Real

import kepler
import matplotlib.pyplot as plt
import numpy as np

from astropy import constants as c
from astropy import units as u
from astropy.coordinates import get_body_barycentric, SkyCoord
from astropy.coordinates.representation.cartesian import CartesianRepresentation
from astropy.time import Time
from beartype import beartype, typing
from matplotlib.figure import Figure

from exogaia.utils import param_dict_to_list, print_section


class StarModel:
    """
    Class with an astrometric model of a stellar track.
    """

    @beartype
    def __init__(self, epoch_astrometry, verbose: bool = True) -> None:
        """
        Parameters
        ----------
        epoch_astrometry : EpochAstrometry
            ``EpochAstrometry`` object that contains the data.
        verbose : bool
            Print information.

        Returns
        -------
        NoneType
            None
        """

        self.epoch_astrometry = epoch_astrometry
        self.data_table = epoch_astrometry.data_table
        self.ref_epoch = epoch_astrometry.ref_epoch
        self.ra_ref = epoch_astrometry.ra_ref
        self.dec_ref = epoch_astrometry.dec_ref
        self.verbose = verbose

        if self.verbose:
            print_section("Star model", bound_char="=")
            print(f"Epoch astrometry: {self.epoch_astrometry.__class__.__name__}")

    @beartype
    def barycentric_position(
        self,
        sat_loc: typing.Literal["Earth", "L2"],
        obs_time: typing.Optional[np.ndarray] = None,
    ) -> CartesianRepresentation:
        """
        Calculate the barycentric position of an observer.

        The observer can be placed either at the geocenter or at the
        approximate Sun-Earth L2 point. The L2 position is calculated
        along the Sun-Earth direction using the leading-order restricted
        three-body approximation. The Lissajous orbit of Gaia around L2
        is not included.

        Parameters
        ----------
        sat_loc : str
            Observer location. Supported values are ``"L2"`` and
            ``"Earth"``.
        obs_time : np.ndarray, optional
            Observing epochs expressed as Julian years on the TCB
            scale. If ``None``, the epochs are read from
            ``self.data_table["obs_time_tcb"]``.

        Returns
        -------
        CartesianRepresentation
            Barycentric ICRS Cartesian coordinates of the observer
            at ``obs_time``.
        """

        if obs_time is None:
            obs_time = self.data_table["obs_time_tcb"].to_numpy()

        time_tcb = Time(obs_time, format="jyear", scale="tcb")

        earth_pos = get_body_barycentric(body="earth", time=time_tcb)

        if sat_loc == "L2":
            # sat_loc = "L2"
            sun_pos = get_body_barycentric(body="sun", time=time_tcb)

            # Vector from the Sun to Earth
            sun_earth = earth_pos - sun_pos
            sun_earth_dist = sun_earth.norm()
            sun_earth_unit = sun_earth / sun_earth_dist

            # Leading-order distance from Earth to the Sun-Earth L2 point
            mu = (c.M_earth / (c.M_sun + c.M_earth)).decompose().value

            l2_dist = sun_earth_dist * (mu / 3.0) ** (1.0 / 3.0)

            return earth_pos + l2_dist * sun_earth_unit

        # sat_loc = "Earth"
        return earth_pos

    @beartype
    def calc_2d_model(
        self,
        model_param: typing.Dict[str, Real],
        obs_time: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray, typing.Optional[np.ndarray]]:
        """
        Method for calculating the stellar track, returning separately
        the RA and Dec components. This function calculates the motion
        due to parallax, which might be less precise than using the
        parallax factors, but these are not provided for RA and Dec
        separately. Typically these are not needed, but for creating
        a 2D plot of the stellar track, we need to calculate the
        effect in RA and Dec separately.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters. The mandatory
            parameters are ``ra_offset`` (mas), ``dec_offset`` (mas),
            ``parallax`` (mas), ``pm_ra`` (mas/yr), and ``pm_dec``
            (mas/yr). The optional parameters are the acceleration
            parameters ``pm_dot_ra`` (mas/yr^2) and ``pm_dot_dec``
            (mas/yr^2). In that case, also the derivative on the
            acceleration, ``pm_dotdot_ra`` (mas/yr^3) and
            ``pm_dotdot_dec`` (mas/yr^3), can be included.
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

        if self.verbose:
            print_section("Calculate stellar track")

        if self.verbose:
            print("Stellar parameters:")
            print(f"   - RA offset (mas) = {model_param['ra_offset']:.3f}")
            print(f"   - Dec offset (mas) = {model_param['dec_offset']:.3f}")
            print(f"   - Parallax (mas) = {model_param['parallax']:.3f}")
            print(f"   - Proper motion in RA (mas/yr) = {model_param['pm_ra']:.3f}")
            print(f"   - Proper motion in Dec (mas/yr) = {model_param['pm_dec']:.3f}")

        if obs_time is None:
            obs_time = self.data_table["obs_time_tcb"].to_numpy()
            sin_scan_ang = self.data_table["sin_scan_ang"].to_numpy()
            cos_scan_ang = self.data_table["cos_scan_ang"].to_numpy()

        else:
            sin_scan_ang = None
            cos_scan_ang = None

        rel_year = obs_time - self.ref_epoch.tcb.jyear

        # RA and Dec coordinates

        ra_coord = self.ra_ref + model_param["ra_offset"]
        dec_coord = self.dec_ref + model_param["dec_offset"]

        # Position of the satellite relative to the Solar System
        # barycenter at each observation time.
        # TODO See Wright & Howard (2009)

        data_type = self.epoch_astrometry.__class__.__name__

        if data_type == "GaiaAstrometry":
            sat_pos = self.barycentric_position("L2", obs_time)

        elif data_type == "HipparcosAstrometry":
            sat_pos = self.barycentric_position("Earth", obs_time)

        else:
            raise ValueError(
                f"The data type of epoch astrometry is not supported: {data_type}"
            )

        sat_pos = sat_pos.xyz.to_value()

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
                -(alpha_hat @ sat_pos),
                rel_year,
            ]
        )

        design_dec = np.column_stack(
            [
                np.full(obs_time.size, 1.0),
                -(delta_hat @ sat_pos),
                rel_year,
            ]
        )

        # RA and Dec parameter vectors

        params_ra = [
            model_param["ra_offset"],
            model_param["parallax"],
            model_param["pm_ra"],
        ]
        params_dec = [
            model_param["dec_offset"],
            model_param["parallax"],
            model_param["pm_dec"],
        ]

        # RA and Dec offsets

        delta_ra = design_ra @ params_ra
        delta_dec = design_dec @ params_dec

        # Add accelerations components

        if "pm_dot_ra" in model_param and "pm_dot_dec" in model_param:
            if self.verbose:
                print(
                    f"   - PM acceleration in RA (mas/yr^2) = {model_param['pm_dot_ra']:.3f}"
                )
                print(
                    f"   - PM acceleration in Dec (mas/yr^2) = {model_param['pm_dot_dec']:.3f}"
                )

            delta_ra += 0.5 * rel_year**2 * model_param["pm_dot_ra"]
            delta_dec += 0.5 * rel_year**2 * model_param["pm_dot_dec"]

        if "pm_dotdot_ra" in model_param and "pm_dotdot_dec" in model_param:
            if self.verbose:
                print(
                    "   - PM acceleration derivative in RA "
                    f"(mas/yr^3) = {model_param['pm_dotdot_ra']:.3f}"
                )
                print(
                    "   - PM acceleration derivative in Dec "
                    f"(mas/yr^3) = {model_param['pm_dotdot_dec']:.3f}"
                )

            delta_ra += (1.0 / 6.0) * rel_year**3 * model_param["pm_dotdot_ra"]
            delta_dec += (1.0 / 6.0) * rel_year**3 * model_param["pm_dotdot_dec"]

        # Calculate 1D projected positions

        if sin_scan_ang is not None and cos_scan_ang is not None:
            delta_pos = delta_ra * sin_scan_ang + delta_dec * cos_scan_ang
        else:
            delta_pos = None

        return delta_ra, delta_dec, delta_pos

    @beartype
    def calc_1d_model(
        self,
        model_param: typing.Dict[str, Real],
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
        model_param : dict
            Dictionary with the model parameters. The mandatory
            parameters are ``ra_offset`` (mas), ``dec_offset`` (mas),
            ``parallax`` (mas), ``pm_ra`` (mas/yr), and ``pm_dec``
            (mas/yr). The optional parameters are the acceleration
            parameters ``pm_dot_ra`` (mas/yr^2) and ``pm_dot_dec``
            (mas/yr^2). In that case, also the derivative on the
            acceleration, ``pm_dotdot_ra`` (mas/yr^3) and
        calc_parallax : bool
            Calculate the parallax effect or use the parallax
            factors from Gaia/Hipparcos (default: False). The latter
            will be somewhat more accurate. This parameter was mainly
            included for testing purposes.

        Returns
        -------
        np.ndarray
            Array with the 1D projected positions (mas).
        """

        if self.verbose:
            print_section("Calculate stellar track")

        param_list = param_dict_to_list(model_param)

        if self.verbose:
            print("Stellar parameters:")
            print(f"   - RA offset (mas) = {model_param['ra_offset']:.3f}")
            print(f"   - Dec offset (mas) = {model_param['dec_offset']:.3f}")
            print(f"   - Parallax (mas) = {model_param['parallax']:.3f}")
            print(f"   - Proper motion in RA (mas/yr) = {model_param['pm_ra']:.3f}")
            print(f"   - Proper motion in Dec (mas/yr) = {model_param['pm_dec']:.3f}")

        rel_year = self.data_table["relative_time_year"].to_numpy()
        sin_scan_ang = self.data_table["sin_scan_ang"].to_numpy()
        cos_scan_ang = self.data_table["cos_scan_ang"].to_numpy()
        par_fac = self.data_table["parallax_factor_al"].to_numpy()

        # Model parameters

        if calc_parallax:
            # This function calculates the effect from the parallax,
            # but ignores higher order effects.

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

            delta_pos = design @ param_list[0:5]

            # Add accelerations components

            if "pm_dot_ra" in model_param and "pm_dot_dec" in model_param:
                delta_pos += 0.5 * rel_year**2 * model_param["pm_dot_ra"]
                delta_pos += 0.5 * rel_year**2 * model_param["pm_dot_dec"]

                if self.verbose:
                    print(
                        f"   - PM acceleration in RA (mas/yr^2) = {model_param['pm_dot_ra']:.3f}"
                    )
                    print(
                        f"   - PM acceleration in Dec (mas/yr^2) = {model_param['pm_dot_dec']:.3f}"
                    )

            if "pm_dotdot_ra" in model_param and "pm_dotdot_dec" in model_param:
                delta_pos += (1.0 / 6.0) * rel_year**3 * model_param["pm_dotdot_ra"]
                delta_pos += (1.0 / 6.0) * rel_year**3 * model_param["pm_dotdot_dec"]

                if self.verbose:
                    print(
                        "   - PM acceleration derivative in RA "
                        f"(mas/yr^3) = {model_param['pm_dotdot_ra']:.3f}"
                    )
                    print(
                        "   - PM acceleration derivative in Dec "
                        f"(mas/yr^3) = {model_param['pm_dotdot_dec']:.3f}"
                    )

        return delta_pos


class KeplerModel:
    """
    Class with a Kepler model for simulating 1D
    and 2D astrometry.
    """

    @beartype
    def __init__(
        self,
        epoch_astrometry,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        epoch_astrometry : EpochAstrometry
            ``EpochAstrometry`` object that contains the data.
        verbose : bool
            Print information.

        Returns
        -------
        NoneType
            None
        """

        self.epoch_astrometry = epoch_astrometry
        self.data_table = epoch_astrometry.data_table
        self.ref_epoch = self.epoch_astrometry.ref_epoch
        self.verbose = verbose

        if self.verbose:
            print_section("Kepler model", bound_char="=")
            print(f"Epoch astrometry: {self.epoch_astrometry.__class__.__name__}")

    @beartype
    def solve_kepler(
        self,
        per: Real,
        ecc: Real,
        tau: Real,
        obs_time: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Method for solving the Kepler equation.

        Parameters
        ----------
        per : float
            Orbital period (days).
        ecc : float
            Eccentricity.
        tau : float
            Time of periastron passage relative to ``ref_epoch``, expressed as
            a fraction of the orbital period. Thus ``tau=0`` means periastron
            occurs at ``ref_epoch``, ``tau=0.5`` means periastron occurs half
            a period after ``ref_epoch``.
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
        t_per = per * tau

        # rel_time_day: observation times in days relative to ref_epoch
        # delta_t: observation times relative to time of periastron
        delta_t = rel_time_day - t_per

        # Mean anomaly at observation epochs
        mean_anom_obs = delta_t * 2.0 * np.pi / per

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
        model_param: typing.Dict[str, Real],
        obs_time: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Method for calculating the astrometric orbit on the sky.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters. The mandatory
            parameters are the period (days), ``per``, eccentricity,
            ``ecc``, relative time of periastron, ``tau``, semi-major
            axis (mas), ``sma``, inclination (rad), ``inc``,
            argument of periastron (rad), ``aop``, and position angle
            of ascending node (rad), ``pan``. Optionally, the
            dictionary may also include ``ra_offset`` (mas),
            ``dec_offset`` (mas), ``parallax`` (mas), ``pm_ra``
            (mas/yr), and ``pm_dec`` (mas/yr), but these parameters
            are not used.
        obs_time : np.ndarray, None
            Array with the observing epochs in Julian years on the TCB
            scale. The epochs are selected from the ``EpochAstrometry``
            if the argument is set to ``None``.

        Returns
        -------
        np.ndarray
            Array with the RA coordinates (mas) relative
            to the system's barycenter.
        np.ndarray
            Array with the Dec coordinates (mas) relative
            to the system's barycenter.
        """

        if self.verbose:
            print_section("Calculate orbit model")

        if self.verbose:
            print("Orbit parameters:")
            print(f"   - Period (days) = {model_param['per']:.3f}")
            print(f"   - Eccentricity = {model_param['ecc']:.3f}")
            print(f"   - Relative time of periastron = {model_param['tau']:.3f}")
            print(f"   - Semi-major axis (mas) = {model_param['sma']:.3f}")

            inc_deg = np.degrees(model_param["inc"])
            aop_deg = np.degrees(model_param["aop"])
            pan_deg = np.degrees(model_param["pan"])

            print(f"   - Inclination (deg) = {inc_deg:.3f}")
            print(f"   - Argument of periastron (deg) = {aop_deg:.3f}")
            print(f"   - PA of ascending node (deg) = {pan_deg:.3f}")

        # Calculate the Thiele-Innes elements
        # sma is in mas, so the TI elements are also in mas

        thiele_innes_a, thiele_innes_b, thiele_innes_f, thiele_innes_g = (
            self.thiele_innes(
                model_param["sma"],
                model_param["inc"],
                model_param["aop"],
                model_param["pan"],
            )
        )

        # Solve the Kepler equation
        # x_orb and y_orb are a dimensionless ellipse
        # so should be scaled by the semi-major axis

        x_orb, y_orb = self.solve_kepler(
            model_param["per"],
            model_param["ecc"],
            model_param["tau"],
            obs_time,
        )

        # Rotate (x_orb, y_orb) into sky plane (x_sky, y_sky)
        # See Eq. 9 in Holl et al. (2023)
        # The units of x_sky and y_sky are mas
        # because the units of sma in thiele_innes are mas

        x_sky = thiele_innes_b * x_orb + thiele_innes_g * y_orb
        y_sky = thiele_innes_a * x_orb + thiele_innes_f * y_orb

        # if apply_bias:
        #     if self.flux_ratio is None or self.mass_ratio is None:
        #         warnings.warn(
        #             "The binary bias can't be applied because "
        #             "the arguments of  'flux_ratio' and/or "
        #             "'mass_ratio' have not been set."
        #         )
        #
        #     else:
        #         pos_scaling = (self.flux_ratio - self.mass_ratio) / (
        #             (1.0 + self.flux_ratio) * (1.0 + self.mass_ratio)
        #         )
        #
        #         x_sky *= pos_scaling
        #         y_sky *= pos_scaling

        # if mas_units:
        #     # Convert by default from (au) to (mas)
        #     # unless parallax is not part of model_param
        #     if "parallax" in model_param:
        #         x_sky *= model_param["parallax"]
        #         y_sky *= model_param["parallax"]
        #
        #     else:
        #         raise ValueError(
        #             "To return the orbit in mas, 'parallax' "
        #             "should be part of 'model_param'. You can "
        #             "set 'mas_units=False' to return the orbit "
        #             "astrometry in au."
        #         )

        return x_sky, y_sky

    @beartype
    def calc_2d_model(
        self,
        model_param: typing.Dict[str, Real],
        obs_time: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Method for calculating the astrometry of the combined
        stellar track and Kepler orbit.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters: The dictionary
            should include the stellar parameters ``ra_offset`` (mas),
            ``dec_offset`` (mas), ``parallax`` (mas), ``pm_ra``
            (mas/yr), and ``pm_dec`` (mas/yr), and also the
            orbital parameters, so period (days), ``per``, eccentricity,
            ``ecc``, relative time of periastron, ``tau``, semi-major
            axis (mas), ``sma``, inclination (rad), ``inc``, argument of
            periastron (rad), ``aop``, and position angle of ascending
            node (rad), ``pan``.
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

        # Stellar track

        star_model = StarModel(self.epoch_astrometry, verbose=self.verbose)

        delta_ra_star, delta_dec_star, _ = star_model.calc_2d_model(
            model_param, obs_time
        )

        # Orbit

        delta_ra_orbit, delta_dec_orbit = self.calc_orbit(
            model_param,
            obs_time,
        )

        # Total RA/Dec offset

        delta_ra = delta_ra_star + delta_ra_orbit
        delta_dec = delta_dec_star + delta_dec_orbit

        return delta_ra, delta_dec

    @beartype
    def calc_1d_model(
        self,
        model_param: typing.Dict[str, Real],
    ) -> np.ndarray:
        """
        Calculate the astrometry of the combined stellar track and
        Kepler orbit. The observation epochs and scan angles that
        are stored in the ``data_table`` of ``epoch_astrometry``
        will be used, so it is not possible to calculate epoch
        astrometry of arbitrary observation epochs. For that
        purpose, :class:`~exogaia.models.StarModel.calc_2d_model`
        should be used.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters: The dictionary
            should include the stellar parameters ``ra_offset`` (mas),
            ``dec_offset`` (mas), ``parallax`` (mas), ``pm_ra``
            (mas/yr), and ``pm_dec`` (mas/yr), and also the
            orbital parameters, so period (days), ``per``, eccentricity,
            ``ecc``, relative time of periastron, ``tau``, semi-major
            axis (mas), ``sma``, inclination (rad), ``inc``, argument of
            periastron (rad), ``aop``, and position angle of ascending
            node (rad), ``pan``.

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

        if self.verbose:
            print_section("Calculate stellar track")

        param_list = param_dict_to_list(model_param)

        if self.verbose:
            print("Stellar parameters:")
            print(f"   - RA offset (mas) = {model_param['ra_offset']:.3f}")
            print(f"   - Dec offset (mas) = {model_param['dec_offset']:.3f}")
            print(f"   - Parallax (mas) = {model_param['parallax']:.3f}")
            print(f"   - Proper motion in RA (mas/yr) = {model_param['pm_ra']:.3f}")
            print(f"   - Proper motion in Dec (mas/yr) = {model_param['pm_dec']:.3f}")

        star_model = design @ param_list[0:5]

        # Orbit component

        delta_ra, delta_dec = self.calc_orbit(
            model_param,
            obs_time=None,
        )

        # Calculate 1D projected positions of orbit model

        orbit_model = delta_ra * sin_scan_ang + delta_dec * cos_scan_ang

        return star_model + orbit_model

    @beartype
    def calc_residuals(
        self,
        model_param: typing.Dict[str, Real],
    ) -> np.ndarray:
        """
        Calculate the residuals of the 1D model astrometry
        with respect to the observed epoch astrometry.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters: The dictionary
            should include the stellar parameters ``ra_offset`` (mas),
            ``dec_offset`` (mas), ``parallax`` (mas), ``pm_ra``
            (mas/yr), and ``pm_dec`` (mas/yr), and also the
            orbital parameters, so period (days), ``per``, eccentricity,
            ``ecc``, relative time of periastron, ``tau``, semi-major
            axis (mas), ``sma``, inclination (rad), ``inc``, argument of
            periastron (rad), ``aop``, and position angle of ascending
            node (rad), ``pan``.

        Returns
        -------
        np.ndarray
            Array with the residuals, as data minus model.
        """

        # Epoch astrometry data

        obs_pos = self.data_table["centroid_pos_al"].to_numpy()
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()

        # Calculate 1D residuals

        residuals = obs_pos - self.calc_1d_model(model_param)

        if self.verbose:
            print_section("Calculate residuals")

            print(f"Residual range: {residuals.min():.3f} to {residuals.max():.3f} mas")
            print(f"Residual RMS: {np.sqrt(np.mean(residuals**2)):.3f} mas")
            print(f"Normalized RMS: {np.sqrt(np.mean((residuals / obs_err) ** 2)):.3f}")

        # Number of data points
        n_obs = len(obs_pos)

        # Number of model parameters
        n_param = len(model_param)

        if self.verbose:
            print(f"\nNumber of epochs = {n_obs}")
            print(f"Number of parameters = {n_param}")

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
        model_param: typing.Dict[str, Real],
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

        # Do not print output with calc_orbit

        verbose_check = bool(self.verbose)

        if verbose_check:
            self.verbose = False

        # Full orbit model with 1000 steps

        yr_start = self.ref_epoch
        yr_end = self.ref_epoch + (model_param["per"] / 365.25) * u.yr

        obs_time_full = np.linspace(yr_start, yr_end, 1000)
        obs_time_full = obs_time_full.tcb.jyear

        # 2D orbit of the photocenter

        delta_ra_full, delta_dec_full = self.calc_orbit(
            model_param,
            obs_time=obs_time_full,
        )

        # Position at time of periastron (in Julian years)

        t_per = (
            self.ref_epoch.tcb.jyear
            + (model_param["per"] * model_param["tau"]) / 365.25
        )

        delta_ra_per, delta_dec_per = self.calc_orbit(
            model_param,
            obs_time=np.array([t_per]),
        )

        # Position at time of apastron (in Julian years)

        period_years = model_param["per"] / 365.25

        delta_ra_ap, delta_dec_ap = self.calc_orbit(
            model_param,
            obs_time=np.array([t_per + 0.5 * period_years]),
        )

        # Orbit model at observation epochs

        delta_ra, delta_dec = self.calc_orbit(
            model_param,
            obs_time=None,
        )

        # Activate verbose again

        if verbose_check:
            self.verbose = True

        # Calculate residuals

        residuals = self.calc_residuals(model_param)

        res_ra = sin_scan_ang * residuals
        res_dec = cos_scan_ang * residuals

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
            label="Photocenter",
        )

        for i, res_item in enumerate(residuals):
            x1 = delta_ra[i] + sin_scan_ang[i] * (res_item + obs_err[i])
            x2 = delta_ra[i] + sin_scan_ang[i] * (res_item - obs_err[i])
            y1 = delta_dec[i] + cos_scan_ang[i] * (res_item + obs_err[i])
            y2 = delta_dec[i] + cos_scan_ang[i] * (res_item - obs_err[i])

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

        # Plot periastron and line of nodes

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

        plt.plot(
            [delta_ra_per, delta_ra_ap],
            [delta_dec_per, delta_dec_ap],
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

        plt.legend(loc="lower left", frameon=False, fontsize=8)

        if plot_file is None:
            plt.show()
        else:
            plt.savefig(plot_file)

        if self.verbose:
            print_section("Plot orbit")

            print(f"Output file: {plot_file}")

        return fig
