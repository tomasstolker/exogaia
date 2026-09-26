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
from scipy.stats import poisson

from exogaia.utils import print_section


class OccurrenceRate:
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
        sma_range: typing.Optional[typing.Tuple[Real, Real]] = None,
        mass_range: typing.Optional[typing.Tuple[Real, Real]] = None,
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
            so in the form ``occ_rate(sma, mass_planet, mass_star)``
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
            print_section("Occurrence rate", bound_char="=")

        self.primary_mass = np.atleast_1d(primary_mass).astype(float)

        if self.primary_mass.ndim != 1:
            raise ValueError(
                "'primary_mass' should be a scalar or one-dimensional array"
            )

        if np.any(~np.isfinite(self.primary_mass)) or np.any(self.primary_mass <= 0.0):
            raise ValueError("'primary_mass' should contain finite positive values")

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
            if sma_range is None or mass_range is None:
                raise ValueError(
                    "'sma_range' and 'mass_range' should be provided "
                    "when 'occ_rate' is a callable."
                )

            self.sma_range = sma_range
            self.mass_range = mass_range
            self.occ_rate = occ_rate

        if (
            not np.all(np.isfinite(self.sma_range))
            or self.sma_range[0] <= 0.0
            or self.sma_range[0] >= self.sma_range[1]
        ):
            raise ValueError(
                "'sma_range' should contain two increasing finite " "positive values"
            )

        if (
            not np.all(np.isfinite(self.mass_range))
            or self.mass_range[0] <= 0.0
            or self.mass_range[0] >= self.mass_range[1]
        ):
            raise ValueError(
                "'mass_range' should contain two increasing finite " "positive values"
            )

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

            if self.primary_mass.size == 1:
                print(f"\nPrimary mass (Msun): {self.primary_mass[0]:.2f}")

            else:
                print(
                    "\nPrimary mass range (Msun): "
                    f"{np.min(self.primary_mass):.2f} - "
                    f"{np.max(self.primary_mass):.2f}"
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
            def wrapper(*args, **kwargs):
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                def check_range(name, rng):
                    if name in bound.arguments:
                        arr = np.asarray(bound.arguments[name])
                        if not np.all((arr >= rng[0]) & (arr <= rng[1])):
                            warnings.warn(
                                f"{name} is outside the supported range {rng}",
                                stacklevel=2,
                            )

                check_range("sma", sma_range)
                check_range("mass_planet", mass_range)

                return func(*args, **kwargs)

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

        if not np.isfinite(mass_star) or mass_star <= 0.0:
            raise ValueError("'mass_star' should be finite and positive")

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

        dlog_sma = np.diff(self.log_sma_edges)
        dlog_mass = np.diff(self.log_mass_edges)

        occ_int = np.sum(self.occ_grid * dlog_sma[:, None] * dlog_mass[None, :])

        return occ_int

    @beartype
    def sample_planets(
        self,
        require_planet: bool = False,
        seed: typing.Optional[int] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Draw a synthetic planet population for the current stellar sample.

        For each star, the occurrence-rate density is integrated over the
        adopted semi-major axis and companion-mass range to obtain the
        expected number of planets. The actual number of planets is drawn
        from a Poisson distribution with this expectation value.

        Planet properties are sampled from the occurrence-rate density

            d^2N / (d ln(a) d ln(M_p)).

        Sampling is performed by first selecting a cell from the
        discretized occurrence-rate grid according to its integrated
        probability and then drawing uniformly in ln(a) and ln(M_p)
        within that cell.

        Parameters
        ----------
        require_planet : bool
            If ``False`` (default), the number of planets for each star
            is drawn from the full Poisson distribution and can therefore
            be zero. If ``True``, the number of planets is drawn from the
            corresponding zero-truncated Poisson distribution, so every
            star hosts at least one planet.
        seed : int, None
            Seed for the random number generator. If ``None`` (default),
            a fresh random generator is used.

        Returns
        -------
        np.ndarray
            Semi-major axes of the sampled planets (au).
        np.ndarray
            Masses of the sampled planets (Msun).
        np.ndarray
            Integer indices of the host stars in ``self.primary_mass``.
            Multiple planets can therefore have the same host index.
        """

        if self.verbose:
            print_section("Sample planets")
            print(f"Number of stars: {self.primary_mass.size}")

        rng = np.random.default_rng(seed)

        sma_samples = []
        mass_samples = []
        host_indices = []

        for star_idx, star_mass in enumerate(self.primary_mass):
            # Calculate the occurrence grid for this stellar mass
            occ_int = float(self.integrate_occ_rate(star_mass))

            if not np.isfinite(occ_int):
                raise ValueError(
                    f"The integrated occurrence rate is not finite "
                    f"for primary mass {star_mass:.3f} Msun."
                )

            if occ_int < 0.0:
                raise ValueError(
                    f"The integrated occurrence rate is negative "
                    f"({occ_int:.3e}) for primary mass "
                    f"{star_mass:.3f} Msun."
                )

            if self.occ_grid is None:
                raise RuntimeError(
                    "The occurrence-rate grid was not initialized by "
                    "'integrate_occ_rate'."
                )

            if not np.all(np.isfinite(self.occ_grid)):
                raise ValueError(
                    f"The occurrence-rate grid contains non-finite values "
                    f"for primary mass {star_mass:.3f} Msun."
                )

            if np.any(self.occ_grid < 0.0):
                raise ValueError(
                    f"The occurrence-rate grid contains negative values "
                    f"for primary mass {star_mass:.3f} Msun."
                )

            # Integrated occurrence rate in each grid cell
            dlog_sma = np.diff(self.log_sma_edges)
            dlog_mass = np.diff(self.log_mass_edges)

            cell_rate = self.occ_grid * dlog_sma[:, None] * dlog_mass[None, :]

            rate_sum = float(np.sum(cell_rate))

            if not np.isfinite(rate_sum) or rate_sum < 0.0:
                raise ValueError(
                    f"The integrated occurrence grid is invalid "
                    f"for primary mass {star_mass:.3f} Msun."
                )

            # No planets can be drawn from a zero occurrence rate
            if rate_sum == 0.0:
                if require_planet:
                    raise ValueError(
                        "Cannot require a planet when the integrated "
                        f"occurrence rate is zero for primary mass "
                        f"{star_mass:.3f} Msun."
                    )

                continue

            # Draw planet multiplicity
            if require_planet:
                # Draw from the Poisson distribution conditional on N >= 1
                p_zero = np.exp(-rate_sum)
                p_nonzero = -np.expm1(-rate_sum)

                quantile = p_zero + p_nonzero * rng.random()
                n_planets = int(poisson.ppf(quantile, mu=rate_sum))

                # Guard against the exact lower numerical boundary
                n_planets = max(1, n_planets)

            else:
                n_planets = int(rng.poisson(rate_sum))

            if n_planets == 0:
                continue

            # Normalize the integrated cell rates to obtain sampling
            # probabilities for the 2D occurrence-rate grid
            cell_prob = cell_rate.ravel() / rate_sum

            # Protect against tiny floating-point normalization errors
            cell_prob /= cell_prob.sum()

            # Draw grid cells for all planets around this star
            flat_idx = rng.choice(
                cell_prob.size,
                size=n_planets,
                replace=True,
                p=cell_prob,
            )

            if self.occ_grid.ndim != 2:
                raise RuntimeError(
                    "Expected a 2D occurrence-rate grid, "
                    f"got {self.occ_grid.ndim} dimensions."
                )

            grid_idx = np.unravel_index(
                flat_idx,
                self.occ_grid.shape,
            )

            sma_idx = grid_idx[0]
            mass_idx = grid_idx[1]

            # Draw uniformly within the selected logarithmic bins
            log_sma = rng.uniform(
                self.log_sma_edges[sma_idx],
                self.log_sma_edges[sma_idx + 1],
            )

            log_mass = rng.uniform(
                self.log_mass_edges[mass_idx],
                self.log_mass_edges[mass_idx + 1],
            )

            sma_samples.extend(np.exp(log_sma))
            mass_samples.extend(np.exp(log_mass))
            host_indices.extend([star_idx] * n_planets)

        sma_samples = np.asarray(sma_samples, dtype=float)
        mass_samples = np.asarray(mass_samples, dtype=float)
        host_indices = np.asarray(host_indices, dtype=int)

        # Convert companion masses from Mjup to Msun
        mass_samples = (mass_samples * u.M_jup).to(u.M_sun).value

        if self.verbose:
            n_planets = sma_samples.size
            n_systems = np.unique(host_indices).size

            print(f"Number of planets: {n_planets}")
            print(f"Number of planetary systems: {n_systems}")

            if n_planets > 0:
                planet_counts = np.bincount(
                    host_indices,
                    minlength=self.primary_mass.size,
                )

                print("Maximum planet multiplicity: " f"{np.max(planet_counts)}")
                print(
                    "\nSemi-major axis range (au) = "
                    f"{np.min(sma_samples):.2f} - "
                    f"{np.max(sma_samples):.2f}"
                )
                print(
                    "Companion mass range (Msun) = "
                    f"{np.min(mass_samples):.2e} - "
                    f"{np.max(mass_samples):.2e}"
                )

        return sma_samples, mass_samples, host_indices

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
