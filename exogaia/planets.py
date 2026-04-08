"""
Module for the ``OccurrenceRate`` class.
"""

import inspect
import warnings

from functools import wraps
from numbers import Real

import numpy as np

from astropy import units as u
from beartype import beartype, typing

from exogaia.core import ExoGaia


class OccurrenceRate(ExoGaia):
    """
    Planet occurrence-rate model and population generator.

    The occurrence-rate density is defined as:

        d^2N / (d ln(a) d ln(M_p))

    That is, the number of planets per star per logarithmic
    interval in semi-major axis and planet mass.
    """

    @beartype
    def __init__(
        self,
        primary_mass: typing.Union[Real, np.ndarray],
        occ_rate: typing.Union[str, typing.Callable] = "cls_fulton2021",
        sma_range: typing.Tuple[Real, Real] = None,
        mass_range: typing.Tuple[Real, Real] = None,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        primary_mass : float, np.ndarray
            Primary mass (Msun) as float or array.
        occ_rate : str, Callable
            Occurrence-rate prescription. Several implementations
            are available ("cls_fulton2021", "gpi_nielsen2019").
            Or, a function can be provided as argument, which should
            calculate the occurrence rate as function of semi-major
            axis (au), companion mass (Mjup), and primary mass (Msun),
            so in the from ``occ_rate(sma, mass_planet, mass_star)``
        sma_range : tuple(float, float), None
            Allowed semi-major axis range (au) in case the argument
            of ``occ_rate`` is a callable. Otherwise, the argument
            can be left to ``None``.
        mass_range : tuple(float, float), optional
            Allowed companion mass range (Mjup) in case the argument
            of ``occ_rate`` is a callable. Otherwise, the argument
            can be left to ``None``.
        verbose : bool
            Print some information.

        Returns
        -------
        NoneType
            None
        """

        self.verbose = verbose

        if self.verbose:
            self.print_section("Occurrence rate")

        if isinstance(primary_mass, np.ndarray):
            self.primary_mass = primary_mass
        else:
            self.primary_mass = np.array([primary_mass])

        occ_list = ["cls_fulton2021", "gpi_nielsen2019"]

        if isinstance(occ_rate, str):
            if occ_rate == "cls_fulton2021":
                # See Fulton et al. (2021)

                self.sma_range = (0.03, 30.0)  # (au)
                mass_min = (30.0 * u.M_earth).to(u.M_jup)  # (Mearth) -> (Mjup)
                mass_max = (6000.0 * u.M_earth).to(u.M_jup)  # (Mearth) -> (Mjup)
                self.mass_range = (mass_min.value, mass_max.value)  # (Mjup)

                self.occ_rate = self.cls_fulton2021

            elif occ_rate == "gpi_nielsen2019":
                # See Nielsen et al. (2019)

                self.sma_range = (10.0, 100.0)  # (au)
                self.mass_range = (5.0, 13.0)  # (Mjup)

                self.occ_rate = self.gpi_nielsen2019

            else:
                raise ValueError(
                    "The prescription for the occurrence rate, "
                    f"'occ_rate={occ_rate}', is not "
                    "recognized. Please select 'occ_rate' from "
                    f"the following list: {occ_list}"
                )

        else:
            self.sma_range = sma_range
            self.mass_range = mass_range
            self.occ_rate = occ_rate

        if self.verbose:
            print(f"Occurrence rate: {self.occ_rate.__name__}")
            print("\nValid ranges:")
            print(
                "   - Semi-major axis (au): "
                f"{self.sma_range[0]:.2f} - {self.sma_range[1]:.2f}"
            )
            print(
                "   - Companion mass (Mjup): "
                f"{self.mass_range[0]:.2f} - {self.mass_range[1]:.2f}"
            )

            if isinstance(self.primary_mass, Real):
                print(f"\nPrimary mass (Msun): {primary_mass:.2f}")

            else:
                print(
                    "\nPrimary mass range (Msun): "
                    f"{np.min(primary_mass):.2f} - "
                    f"{np.max(primary_mass):.2f}"
                )

        self.log_sma_edges = None
        self.log_mass_edges = None
        self.log_sma_mid = None
        self.log_mass_mid = None
        self.occ_grid = None

    def __repr__(self) -> str:
        """
        Representation of the ``OccurrenceRate`` object that includes
        details on the occurrence rate prescription.
        """

        cls_name = self.__class__.__name__

        if self.primary_mass is None:
            mass_str = "None"
        else:
            if self.primary_mass.size == 1:
                mass_str = f"{self.primary_mass[0]:.3f}"
            else:
                mass_str = (
                    f"array[{self.primary_mass.size}]"
                    f"[{np.nanmin(self.primary_mass):.2f}–"
                    f"{np.nanmax(self.primary_mass):.2f}]"
                )

        occ_name = (
            self.occ_rate.__name__ if callable(self.occ_rate) else str(self.occ_rate)
        )

        sma_str = (
            f"{self.sma_range[0]:.2g}-{self.sma_range[1]:.2g} au"
            if self.sma_range is not None
            else "None"
        )

        mass_str_range = (
            f"{self.mass_range[0]:.2g}-{self.mass_range[1]:.2g} Mjup"
            if self.mass_range is not None
            else "None"
        )

        return (
            f"{cls_name}(M⋆={mass_str} Msun, "
            f"occ='{occ_name}', "
            f"a={sma_str}, "
            f"Mp={mass_str_range})"
        )

    @staticmethod
    def validate_ranges(sma_range, mass_range):
        """
        Decorator for validating parameter ranges.
        """

        def decorator(func):
            sig = inspect.signature(func)

            @wraps(func)
            def wrapper(*args):
                bound = sig.bind(*args)
                bound.apply_defaults()

                def check_range(name, rng):
                    if name in bound.arguments:
                        arr = np.asarray(bound.arguments[name])
                        if not np.all((arr >= rng[0]) & (arr <= rng[1])):
                            warnings.warn(
                                f"{name} is outside the " f"supported range {rng}"
                            )

                check_range("sma", sma_range)
                check_range("mass_planet", mass_range)

                return func(*args)

            return wrapper

        return decorator

    @beartype
    def integrate_occ_rate(self, mass_star: Real) -> Real:
        """
        Integrate the planet occurrence rate over the allowed
        semi-major axis and companion mass ranges for a given
        primary mass. The integration is performed in ln(m)
        and ln(a), so `self.occ_rate` should return
        ``d^2N / (d ln(m) d ln(a))``

        Parameters
        ----------
        mass_star : float
            Primary mass (Msun).

        Returns
        -------
        float
            Total expected number of planets per star.
        """

        # Bin edges

        self.log_sma_edges = np.linspace(
            np.log(self.sma_range[0]),
            np.log(self.sma_range[1]),
            101,
        )

        self.log_mass_edges = np.linspace(
            np.log(self.mass_range[0]),
            np.log(self.mass_range[1]),
            201,
        )

        # Bin mid points

        self.log_mass_mid = 0.5 * (self.log_mass_edges[:-1] + self.log_mass_edges[1:])
        self.log_sma_mid = 0.5 * (self.log_sma_edges[:-1] + self.log_sma_edges[1:])

        # Shape of the grids: (len(log_sma), len(log_mass))

        sma_grid, mass_grid = np.meshgrid(
            np.exp(self.log_sma_mid), np.exp(self.log_mass_mid), indexing="ij"
        )

        # Shape of the grid: (len(log_sma), len(log_mass))

        self.occ_grid = self.occ_rate(sma_grid, mass_grid, mass_star)

        # Integrate occurrence grid over log(Mp) and log(sma)

        int_mass = np.trapezoid(self.occ_grid, self.log_mass_mid, axis=1)
        occ_int = np.trapezoid(int_mass, self.log_sma_mid, axis=0)

        return occ_int

    @beartype
    def sample_planets(
        self,
        allow_reject: bool = True,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Draw a synthetic planet population for the current stellar sample.

        For each star in ``self.primary_mass``, this method:

        1. Computes the integrated planet occurrence rate over the defined
           semi-major axis and mass grid using ``integrate_occ_rate``.
        2. Performs a Bernoulli trial with success probability equal to the
           integrated occurrence rate (if ``allow_reject=True``).
        3. If the star is assigned a planet, samples its semi-major axis and
           mass from the discretized occurrence surface
           ``self.occ_grid``, which represents

               d²N / (d ln a d ln M),

           defined on a logarithmic (a, M) grid.

        The 2D occurrence surface is converted into a normalized
        discrete probability distribution by multiplying by the
        bin widths in ln(a) and ln(M). Sampling proceeds as:

            - First, a semi-major axis bin is drawn from the marginalized
              distribution over ln(a).
            - Then, a mass bin is drawn from the conditional distribution
              over ln(M) at fixed ln(a).
            - Finally, values are drawn uniformly within the selected
              logarithmic bins.

        At most one planet is assigned per star, even if the integrated
        occurrence rate exceeds unity.

        Parameters
        ----------
        allow_reject : bool
            If ``True`` (default), each star hosts a planet with
            probability equal to its integrated occurrence rate.
            If ``False``, every star is forced to host exactly
            one planet.

        Returns
        -------
        np.ndarray
            Semi-major axes (au). Entries are NaN for stars without
            an assigned planet in case ``allow_reject=True``. The
            array has the same length as ``self.primary_mass``.
        np.ndarray
            Planet masses (Msun). Entries are NaN for stars without
            an assigned planet in case ``allow_reject=True``. The
            array has the same length as ``self.primary_mass``.
        """

        if self.verbose:
            self.print_section("Sample planets")

            print(f"Number of stars: {self.primary_mass.size}")

        sma_list = np.full(self.primary_mass.size, np.nan)
        mass_list = np.full(self.primary_mass.size, np.nan)

        for star_idx, star_mass in enumerate(self.primary_mass):
            # Only considering systems with 1 planet, even though
            # occ_int can be larger than one, so on average more
            # than one planet per star, for high stellar masses.

            occ_int = self.integrate_occ_rate(star_mass)

            if allow_reject:
                ran_num = np.random.rand()
            else:
                ran_num = -np.inf

            if ran_num < occ_int:
                # bin widths

                dlog_mass = np.diff(self.log_mass_edges)
                dlog_sma = np.diff(self.log_sma_edges)

                # Convert differential rate to integrated probability per bin

                occ_pdf = self.occ_grid * dlog_sma[:, None] * dlog_mass[None, :]

                # Normalize the PDF

                occ_pdf /= occ_pdf.sum()

                # Sample semi-major axis marginal

                p_sma = occ_pdf.sum(axis=1)
                cdf_sma = np.cumsum(p_sma)
                i_sma = np.searchsorted(cdf_sma, np.random.rand())

                # Sample mass conditional on semi-major aixs

                p_mass_given_sma = occ_pdf[i_sma] / p_sma[i_sma]
                cdf_mass = np.cumsum(p_mass_given_sma)
                i_mass = np.searchsorted(cdf_mass, np.random.rand())

                # Sample inside log(sma) and log(mass) bin

                log_sma = np.random.uniform(
                    self.log_sma_edges[i_sma], self.log_sma_edges[i_sma + 1]
                )

                log_mass = np.random.uniform(
                    self.log_mass_edges[i_mass], self.log_mass_edges[i_mass + 1]
                )

                sma_list[star_idx] = np.exp(log_sma)
                mass_list[star_idx] = np.exp(log_mass)

        # Convert from Mjup to Msun

        mass_list = (mass_list * u.M_jup).to(u.M_sun).value

        if self.verbose:
            print(f"Number of planets: {np.sum(~np.isnan(sma_list))}")

            if len(sma_list) == 1:
                print(f"\nSemi-major axis (au) = {sma_list[0]:.2f}")
                print(f"Companion mass (Msun) = {mass_list[0]:.2e}")

            elif len(sma_list) > 1:
                print(
                    "\nSemi-major axis range (au) = "
                    f"{np.min(sma_list):.2f} - {np.max(sma_list):.2f}"
                )
                print(
                    "Companion mass range (Msun) = "
                    f"{np.min(mass_list):.2e} - {np.max(mass_list):.2e}"
                )

        return sma_list, mass_list

    @beartype
    @staticmethod
    @validate_ranges(
        sma_range=(0.03, 30.0),
        mass_range=(
            ((30.0 * u.M_earth).to(u.M_jup)).value,
            ((6000.0 * u.M_earth).to(u.M_jup)).value,
        ),
    )
    def cls_fulton2021(
        sma: typing.Union[Real, np.ndarray],
        mass_planet: typing.Union[Real, np.ndarray],
        mass_star: typing.Union[Real, np.ndarray],
    ) -> typing.Union[Real, np.ndarray]:
        """
        Giant-planet occurrence-rate density following Fulton et al. (2021).

        This method implements the semi-major axis distribution from
        Fulton et al. (2021, Eq. 5), converted into a two-dimensional
        occurrence-rate density of the form

            d^2N / (d ln(a) d ln(M_p)),

        i.e. the expected number of planets per star per logarithmic
        interval in semi-major axis and planet mass.

        The original Fulton et al. (2021) relation is reported as the
        number of planets per 100 stars per Δln(a)=0.63 bin over the
        planet-mass range 30–6000 Earth masses.

        The occurrence density is log-flat in planet mass within
        30–6000 M⊕, so it is independent of `mass_planet` and only
        depends on the semi-major axis and stellar mass.

        Additionally, the normalization is scaled linearly with stellar mass
        following Lammers & Winn (2025, Eq. 24), such that

            N_giant ∝ M_star.

        Parameters
        ----------
        sma : float, np.ndarray
            Semi-major axis (au).
        mass_planet : float, np.ndarray
            Planet mass (Mjup). This parameter is not used but
            is required to comply with the common format of the
            occurrence rate functions.
        mass_star : float, np.ndarray
            Primary mass (Msun).

        Returns
        -------
        float, np.ndarray
            Occurrence-rate density in units of planets per star
            per unit ln(a) per unit ln(M_p).
        """

        # Broken powerlaw parameters
        # See Fulton et al. (2021)

        norm_fulton = 350.0
        beta = -0.86
        a_break = 3.6
        gamma = 1.59

        # Number of giant planets scales approximately
        # linearly with stellar mass.
        # See Equation 24 in Lammers & Winn (2025)

        norm_lammers = norm_fulton * (mass_star / 0.9)

        occ_rate = (
            norm_lammers * (sma**beta) * (1.0 - np.exp(-((sma / a_break) ** gamma)))
        )

        # This is shown in Figure 3 of Fulton et al. (2021)
        # That is, the number of observed planets
        # per 100 stars per ln(a)=0.63 bin
        # occ_rate *= 100.0/n_stars

        # Normalize by the number of stars, and ln(a) and ln(Mp) range
        # See Equation 23 in Lammers & Winn (2025)

        n_stars = 719
        delta_log_mass = np.log(6000 / 30)  # log(M/Mearth) - log(M/Mearth)
        delta_log_sma = 0.63  # log(a/au)

        occ_rate /= n_stars * delta_log_mass * delta_log_sma

        return occ_rate

    @beartype
    @staticmethod
    @validate_ranges(sma_range=(10.0, 100.0), mass_range=(5.0, 13.0))
    def gpi_nielsen2019(
        sma: typing.Union[Real, np.ndarray],
        mass_planet: typing.Union[Real, np.ndarray],
        mass_star: typing.Union[Real, np.ndarray],
    ) -> typing.Union[Real, np.ndarray]:
        """
        Giant-planet occurrence-rate density from the GPIES
        (Nielsen et al. 2019, Table 3; planets around all stars).

        This method implements the double power-law model (see
        Equation 7) for 5-13 Mjup companions within 10–100 au.
        The occurrence model is defined in linear variables
        as d^2N / (dm da), but the implementation returns the
        logarithmic occurrence-rate density:

            d^2N / (d ln m d ln a),

        obtained via the transformation

            d^2N / (d ln m d ln a) = m a · d^2N / (dm da).

        The stellar-mass scaling is normalized at 1.75 Msun.

        Parameters
        ----------
        sma : float, np.ndarray
            Semi-major axis (au).
        mass_planet : float, np.ndarray
            Planet mass (Mjup).
        mass_star : float, np.ndarray
            Primary mass (Msun).

        Returns
        -------
        float, np.ndarray
            Occurrence-rate density in units of planets per star
            per unit ln(a) per unit ln(M_p).
        """

        # Add the ranges here, to make it a static method

        sma_range = (10.0, 100.0)  # (au)
        mass_range = (5.0, 13.0)  # (Mjup)

        # Median GPIES parameters (all stars)
        # See Table 3 in Nielsen et al. (2019)

        f = 0.035
        alpha = -2.277
        beta = -1.68
        gamma = 2.03

        # Compute the normalization C1 (see Equation 3)

        if alpha == -1.0:
            mass_term = np.log(mass_range[1] / mass_range[0])

        else:
            mass_term = (
                mass_range[1] ** (alpha + 1) - mass_range[0] ** (alpha + 1)
            ) / (alpha + 1)

        if beta == -1.0:
            sma_term = np.log(sma_range[1] / sma_range[0])

        else:
            sma_term = (sma_range[1] ** (beta + 1) - sma_range[0] ** (beta + 1)) / (
                beta + 1
            )

        c1_norm = 1.0 / (mass_term * sma_term)

        # Occurrence rate as (d^2 N) / (dm da)

        occ_rate_lin = (
            f * c1_norm * mass_planet**alpha * sma**beta * (mass_star / 1.75) ** gamma
        )

        # Occurrence rate as (d^2 N) / (dlog(m) dlog(a))

        occ_rate_log = occ_rate_lin * mass_planet * sma

        return occ_rate_log
