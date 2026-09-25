"""
Module for Gaia epoch astrometry data.
"""

import warnings

from abc import ABC, abstractmethod
from numbers import Real
from pathlib import Path

import h5py
import healpy
import numpy as np
import pandas as pd
import pooch

from astropy import constants as c
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from beartype import beartype, typing
from imf.imf import make_cluster
from scipy.interpolate import RegularGridInterpolator

from exogaia.models import KeplerModel, StarModel
from exogaia.planets import OccurrenceRate
from exogaia.utils import binary_bias, print_section, read_hipparcos_header

Gaia.ROW_LIMIT = -1


class EpochAstrometry(ABC):
    """
    Abstract base class for epoch astrometry datasets.
    """

    @beartype
    def __init__(
        self,
        primary_mass: typing.Optional[typing.Tuple[Real, Real]] = None,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        primary_mass : tuple(float, float), None
            Stellar mass and uncertainty (M$_\\odot$), provided as
            ``(mass, mass_error)``.
        verbose : bool
            Print information.

        Returns
        -------
        NoneType
            None.
        """

        self.primary_mass = primary_mass
        self.verbose = verbose

        self.data_table: typing.Optional[pd.DataFrame] = None
        self.ref_epoch: typing.Optional[Time] = None
        self.time_start: typing.Optional[Time] = None
        self.time_end: typing.Optional[Time] = None
        self.sim_data: bool = False
        self.u0_norm: typing.Optional[float] = None

        self.ra_ref: typing.Optional[Time] = None
        self.dec_ref: typing.Optional[Time] = None
        self.parallax: typing.Optional[Time] = None
        self.pm_ra: typing.Optional[Time] = None
        self.pm_dec: typing.Optional[Time] = None

    def __repr__(self) -> str:
        """
        Return a string representation of the stored data table.

        Returns
        -------
        str
            First rows of the data table if available, otherwise a
            message indicating that no data are loaded.
        """

        if self.data_table is None:
            return "Data table is empty"

        return self.data_table.head().to_string()

    @abstractmethod
    def retrieve_data(self, *args, **kwargs) -> dict:
        """
        Retrieve and store epoch astrometry.

        Returns
        -------
        dict
            Reference astrometric parameters.
        """


class GaiaAstrometry(EpochAstrometry):
    """
    Class for reading, querying, and simulating Gaia
    epoch astrometry data.
    """

    @beartype
    def __init__(
        self,
        primary_mass: typing.Optional[typing.Tuple[Real, Real]] = None,
        gaia_release: typing.Literal["DR1", "DR2", "DR3", "DR4", "DR5"] = "DR4",
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        primary_mass : tuple(float, float), None
            Primary mass and uncertainty (Msun). The primary mass is
            not stored if the argument is set to ``None``. When
            simulating data with
            :func:`~exogaia.data.EpochAstrometry.simulate_data`,
            a mass will be drawn from an initial mass function if
            the argument of ``primary_mass`` is set to ``None``.
        gaia_release : str
            Gaia release (DR3, DR4, DR5) of the epoch astrometry.
            Simulating data is possible for all releases, but querying
            data will only be possible for the upcoming DR4 and DR5.
        verbose : bool
            Print some information.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(primary_mass, verbose)

        self.verbose = verbose

        if self.verbose:
            print_section("Gaia astrometry", bound_char="=")

        self.data_folder = Path.home() / ".exogaia"

        if not self.data_folder.exists():
            if self.verbose:
                print(f"Creating folder: {str(self.data_folder)}")

            self.data_folder.mkdir(parents=True, exist_ok=False)

        self.gaia_release = gaia_release
        self.source_id = None
        self.g_mag = None

        # Start of the Gaia mission
        self.time_start = Time("2014-07-25 10:30:00", scale="utc")

        if self.gaia_release == "DR1":
            # https://esdcdoi.esac.esa.int/doi/html/data/astronomy/gaia/DR1.html
            # Only the end day and not the end time is known?
            self.ref_epoch = Time("2015.0", format="jyear", scale="tcb")
            self.time_end = Time("2015-09-16 00:00:00", scale="utc")

        elif self.gaia_release == "DR2":
            # https://www.cosmos.esa.int/web/gaia/dr2
            self.ref_epoch = Time("2015.5", format="jyear", scale="tcb")
            self.time_end = Time("2016-05-23 11:35:00", scale="utc")

        elif self.gaia_release == "DR3":
            # https://www.cosmos.esa.int/web/gaia/dr3
            self.ref_epoch = Time("2016.0", format="jyear", scale="tcb")
            self.time_end = Time("2017-05-28 08:44:00", scale="utc")

        elif self.gaia_release == "DR4":
            # https://www.cosmos.esa.int/web/gaia/dr4
            self.ref_epoch = Time("2017.5", format="jyear", scale="tcb")
            self.time_end = Time("2020-01-20 22:00:00", scale="utc")

        elif self.gaia_release == "DR5":
            self.ref_epoch = Time("2020.0", format="jyear", scale="tcb")
            self.time_end = Time("2025-01-15 00:00:00", scale="utc")

        else:
            raise ValueError("The Gaia release {self.gaia_release} is not supported.")

        if self.verbose:
            print(f"Gaia release: {self.gaia_release}")
            print(f"Reference epoch: {self.ref_epoch.tcb.jyear_str}")

            if primary_mass is not None:
                print(
                    f"\nPrimary mass (Msun): {primary_mass[0]:.2f} "
                    f"+/- {primary_mass[1]:.2f}"
                )

    @beartype
    def read_file(
        self,
        data_file: str,
    ) -> None:
        """
        Method for reading in epoch astrometry from a file.

        Parameters
        ----------
        data_file : str
            Data file with the Gaia epoch astrometry.

        Returns
        -------
        NoneType
            None
        """

        if self.verbose:
            print_section("Read data file")

        self.data_table = pd.read_csv(data_file)

        if self.verbose:
            print(f"Data file: {data_file}")
            print(f"Data shape: {self.data_table.shape}")

        if "relative_time_year" not in self.data_table:
            self.data_table["relative_time_year"] = (
                self.data_table["obs_time_tcb"] - self.ref_epoch.tcb.jyear
            )

        if "relative_time_day" not in self.data_table:
            self.data_table["relative_time_day"] = self.data_table[
                "relative_time_year"
            ] * u.year.to(u.day)

    @staticmethod
    @beartype
    def _interpolate_u0(g_mag: Real, nu_eff: Real) -> Real:
        """
        Interpolate the normalization factor u0 that is
        used to convert UWE into RUWE. This factor
        depends on the stellar magnitude and effective
        wavenumber.

        Determined from the re-normalised Unit Weight Error (RUWE)
        tables of u0(g,c) by L. Lindegren (2023 Sep 13)
        https://www.cosmos.esa.int/web/gaia/dr3-auxiliary-data

        Parameters
        ----------
        g_mag : float
            Gaia G magnitude.
        nu_eff : float
            Effective wavenumber (µm⁻¹).

        Returns
        -------
        float
            The u0 normalization.
        """

        file_folder = Path(__file__).resolve().parent.parent
        data_file = file_folder / "data" / "table_u0_g_c_p5.txt"

        norm_g_mag, norm_nu_eff, norm_u0 = np.loadtxt(
            data_file,
            skiprows=1,
            delimiter=",",
            unpack=True,
        )

        g_vals = np.unique(norm_g_mag)
        c_vals = np.unique(norm_nu_eff)

        u0_grid = norm_u0.reshape(g_vals.size, c_vals.size)

        # Should look the same as plot_u0_g_c_p5.pdf
        # plt.pcolormesh(g_vals, c_vals, u0_grid.T, cmap='rainbow')
        # plt.colorbar()
        # plt.show()

        u0_interp = RegularGridInterpolator(
            (g_vals, c_vals),
            u0_grid,
            method="linear",
            bounds_error=True,
        )

        return float(u0_interp((g_mag, nu_eff)))

    @beartype
    def simulate_data(
        self,
        model_param: typing.Optional[typing.Dict[str, Real]] = None,
        mass_2: typing.Optional[Real] = None,
        flux_ratio: typing.Optional[Real] = None,
        occ_rate: typing.Union[typing.Optional[str], OccurrenceRate] = None,
        sigma_per_ccd: typing.Optional[Real] = None,
        csv_out: typing.Optional[str] = None,
        reject_fraction: typing.Optional[Real] = None,
        seed: typing.Optional[int] = None,
        allow_reject: bool = True,
    ) -> typing.Dict[str, Real]:
        """
        Simulate the epoch astrometry for a single star or binary
        system. The primary mass stored with the ``primary_mass``
        attribute will be used, or, it will get sampled from
        and initial mass function if the argument of
        ``primary_mass`` is set to ``None``.

        Parameters
        ----------
        model_param : dict
            Dictionary containing the model parameters for the
            stellar track and optionally the binary orbit.

            For a **single star**, the following parameters
            are needed. However, this is optional and not
            required if
            :func:`~exogaia.data.EpochAstrometry.query_source`
            has already been used to retrieve these values:

                - 'ra' : float
                    Right ascension at the reference epoch
                    ``ref_epoch`` of the Gaia release (deg).
                - 'dec' : float
                    Declination at the reference epoch
                    ``ref_epoch`` of the Gaia release (deg).
                - 'parallax' : float
                    Parallax (mas).
                - 'pm_ra' : float
                    Proper motion in right ascension (mas/yr).
                - 'pm_dec' : float
                    Proper motion in declination (mas/yr).
                - 'g_mag' : float, optional
                    Apparent Gaia G-band magnitude.

            Optional offset relative to the RA/Dec coordinates
            at the reference epoch of the Gaia release:

                - 'ra_offset' : float, optional
                    Offset in right ascension (mas).
                - 'dec_offset' : float, optional
                    Offset in declination (mas).

            For a **binary system**, also the full set of orbital
            parameters can be provided. Random parameters will
            be sampled for any of the parameters that are missing
            in the dictionary:

                - 'ecc' : float
                    Orbital eccentricity. A random value will be
                    drawn from a beta distribution as defined in
                    `Kipping 2013 <https://ui.adsabs.harvard.edu
                    /abs/2013MNRAS.434L..51K>`_
                    if the parameter is not included.
                - 'tau' : float
                    Time of periastron as fraction of the orbital
                    period, relative to the reference epoch of
                    ``gaia_release``. A random value will be drawn
                    from a uniform distribution between 0.0 and 1.0
                    if the parameter is not included.
                - 'inc' : float
                    Inclination (rad). A random value will be drawn
                    from a sin(theta) distribution, with theta between
                    0.0 and pi, if the parameter is not included.
                - 'aop' : float
                    Argument of periastron (rad). A random value will
                    be drawn from a uniform distribution between 0.0
                    and 2pi if the parameter is not included.
                - 'pan' : float
                    Position angle of the ascending node (rad). A
                    random value will be drawn from a uniform
                    distribution between 0.0 and 2pi if the parameter
                    is not included.
                - 'sma' : float
                    Relative semi-major axis (au). This should be
                    the relative semi-major axis of the primary and
                    secondary, so sma = a1 + a2. A random value will
                    be drawn from a log-uniform distribution between
                    0.1 and 30 au if the argument is set to ``None``.
        mass_2 : float, None
            Companion mass (Msun). A single star is simulated
            by setting the argument of both ``mass_2`` and
            ``occ_rate`` to ``None``.
        flux_ratio : float, None
            Flux ratio in the G band between the companion
            and primary star. For substellar companions,
            the argument can be set to zero. Setting the
            argument to ``None`` will calculate the flux
            ratio as :math:`f = q^{3.5}` (see e.g.
            `Penoyre et al. 2022 <https://ui.adsabs.
            harvard.edu/abs/2022MNRAS.513.2437P>`_), which
            approximates the flux ratio for main-sequence
            binaries, with :math:`q` the binary mass ratio.
        occ_rate : str, OccurrenceRate, None
            Occurrence-rate prescription ("fls_fulton2021",
            "gpi_nielsen2019") for sampling the companion
            mass, ``mass_2``, and the semi-major. Or, an
            ``OccurrenceRate`` object, which also allows
            for a manually provided occurrence rate
            function. The arguments of ``mass_2`` and
            ``sma_rel`` will be ignored if the argument
            of ``occ_rate`` is not set to ``None``.
        sigma_per_ccd : float, None
            The AL uncertainty per CCD (mas). Setting the
            argument to ``None`` will adopt the G magnitude
            dependent uncertainty from Holl et al. (2023).
        csv_out : str
            Output CSV file to store the simulated data.
        reject_fraction : float, None
            The fraction of data to reject. About 10 percent of
            FOV transits has an issue (see Lindegren et al. 2021).
            The default is ``None``, in which case no data
            is rejected.
        seed : int, None
            Seed for the random number generator. Random seed is
            used if set to ``None``. Set the argument to a
            positive integer for reproducibility.
        allow_reject : bool
            If ``True`` (default) and ``occ_rate`` is not ``None``,
            each star hosts a planet with probability equal to its
            integrated occurrence rate. If ``False``, every star
            is forced to host exactly one planet.

        Returns
        -------
        dict
            Dictionary with the model parameters. Important!
            The ``sma`` in the dictionary is the semi-major
            axis of the photocenter (mas), assuming an
            unresolved binary orbit.
        """

        rng = np.random.default_rng(seed=seed)

        # For simulated data, UWE == RUWE when fitting models

        self.sim_data = True

        # Create empty model_param dictionary if needed

        if model_param is None:
            model_param = {}

        # Adopt stellar parameters from class attributes
        # or use the parameters from the model_param dictionary

        if "ra_ref" in model_param:
            self.ra_ref = model_param["ra_ref"]
            del model_param["ra_ref"]

        elif self.ra_ref is None:
            raise ValueError(
                "Please either provide the 'ra_ref' in the "
                "model_param  dictionary or run 'query_source()'  "
                "to adopt the values from a Gaia source."
            )

        if "dec_ref" in model_param:
            self.dec_ref = model_param["dec_ref"]
            del model_param["dec_ref"]

        elif self.dec_ref is None:
            raise ValueError(
                "Please either provide the 'dec_ref' in the "
                "model_param dictionary or run 'query_source()' "
                "to adopt the values from a Gaia source."
            )

        if "parallax" in model_param:
            self.parallax = model_param["parallax"]

        elif self.parallax is not None:
            model_param["parallax"] = self.parallax

        else:
            raise ValueError(
                "Please either provide the 'parallax' in the "
                "model_param dictionary or run 'query_source()' "
                "to adopt the values from a Gaia source."
            )

        if "pm_ra" in model_param:
            self.pm_ra = model_param["pm_ra"]

        elif self.pm_ra is not None:
            model_param["pm_ra"] = self.pm_ra

        else:
            raise ValueError(
                "Please either provide the 'pm_ra' in the "
                "model_param dictionary or run 'query_source()' "
                "to adopt the values from a Gaia source."
            )

        if "pm_dec" in model_param:
            self.pm_dec = model_param["pm_dec"]

        elif self.pm_dec is not None:
            model_param["pm_dec"] = self.pm_dec

        else:
            raise ValueError(
                "Please either provide the 'pm_dec' in the "
                "model_param dictionary or run 'query_source()' "
                "to adopt the values from a Gaia source."
            )

        if "g_mag" in model_param:
            self.g_mag = model_param["g_mag"]
            del model_param["g_mag"]

        elif self.g_mag is None:
            raise ValueError(
                "Please either provide the 'g_mag' in the "
                "model_param dictionary or run 'query_source()' "
                "to adopt the values from a Gaia source."
            )

        if "ra_offset" not in model_param:
            model_param["ra_offset"] = 0.0

        if "dec_offset" not in model_param:
            model_param["dec_offset"] = 0.0

        # Sample primary mass from IMF if needed

        if self.primary_mass is None:
            # Cluster of 1000 Msun, which creates
            # a sample of about 1700 stars
            star_masses = make_cluster(1000, verbose=False, silent=True)

            # Select only stars above the hydrogen-burning limit
            star_masses = star_masses[star_masses > 0.08]

            # Select a random star from the sample
            prim_mass = rng.choice(star_masses, size=1)

            # Set an arbitrary uncertainty of 0.1 Msun
            self.primary_mass = (prim_mass[0], 0.1)  # (Msun)

        # Simulating single star or binary system?

        if (mass_2 is None or np.isnan(mass_2)) and occ_rate is None:
            binary = False

        else:
            binary = True

            # Sample random orbit parameters if not provided

            if occ_rate is None:
                if "sma" not in model_param:
                    # Sample log-uniform between 0.1 and 30 au
                    log_sma = rng.uniform(np.log10(0.1), np.log10(30.0))
                    model_param["sma"] = 10.0**log_sma

                # if mass_2 is None:
                #     # This should never happen
                #     # Sample uniform between 0.001 and 0.02 Msun
                #     model_param["mass_2"] = rng.uniform(0.001, 0.02)

            else:
                if isinstance(occ_rate, str):
                    occ_rate = OccurrenceRate(
                        primary_mass=self.primary_mass[0],
                        occ_rate=occ_rate,
                        verbose=self.verbose,
                    )

                if "sma" not in model_param:
                    if mass_2 is None:
                        sma, mass_2 = occ_rate.sample_planets(allow_reject=allow_reject)
                        mass_2 = mass_2[0]

                    else:
                        sma, _ = occ_rate.sample_planets(allow_reject=allow_reject)

                    sma = sma[0]

                    if np.isnan(sma):
                        binary = False

                    else:
                        model_param["sma"] = sma

            if binary:
                if "ecc" not in model_param:
                    # Beta distribution (Kipping 2013)
                    model_param["ecc"] = rng.beta(a=0.867, b=3.03)

                if "inc" not in model_param:
                    model_param["inc"] = np.arccos(rng.uniform(-1.0, 1.0))

                if "aop" not in model_param:
                    model_param["aop"] = rng.uniform(0.0, 2.0 * np.pi)

                if "pan" not in model_param:
                    model_param["pan"] = rng.uniform(0.0, 2.0 * np.pi)

                if "tau" not in model_param:
                    model_param["tau"] = rng.uniform(0.0, 1.0)

        if self.verbose:
            print_section("Simulate data")

            if binary:
                print("System type: binary")
            else:
                print("System type: single")

            if self.primary_mass is not None:
                print(
                    f"\nPrimary mass (Msun): {self.primary_mass[0]:.2f} "
                    f"+/- {self.primary_mass[1]:.2f}"
                )

        # Scan angles, parallax factors, and observation
        # times have been retrieved with GOST (see
        # https://gaia.esac.esa.int/gost/index.jsp) for the
        # RA/DEC of all HEALPix indices with NSIDE=64
        # (i.e. 49152 indices). The data for each index is
        # stored in a separate group of the HDF5 file.

        file_name = "healpix_data.hdf5"
        healpix_file = self.data_folder / file_name
        url = "https://home.strw.leidenuniv.nl/~stolker/exogaia/healpix_data.hdf5"

        if not healpix_file.exists():
            if self.verbose:
                print()

            pooch.retrieve(
                url=url,
                known_hash="54c786f345b891849f43134c9587b03acb9626d935594c2f62469963981ba798",
                fname=file_name,
                path=self.data_folder,
                progressbar=True,
            )

        # Read the scan data for a given sky position by
        # selecting the nearest HEALPix position

        nside = 64

        # theta and phi are in degrees when lonlat=True
        pix_num = healpy.ang2pix(
            nside=nside, theta=self.ra_ref, phi=self.dec_ref, lonlat=True
        )

        with h5py.File(healpix_file, "r") as hdf5_file:
            healpix_table = Table(hdf5_file[f"healpix_{nside}_{pix_num:05d}"])

        # Observation times in Julian days on TCB scale
        obs_time_full = healpix_table[
            "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"
        ]

        # https://www.cosmos.esa.int/web/gaia/dr3
        # Gaia DR3 data (both Gaia EDR3 and the full Gaia DR3) is based on
        # data collected between 25 July 2014 (10:30 UTC) and 28 May 2017
        # (08:44 UTC) spanning a period of 34 months of data collection.
        # The reference epoch for Gaia DR3 is 2016.0.

        # https://www.cosmos.esa.int/web/gaia/dr4
        # Gaia DR4 data is based on data collected between 25 July 2014
        # (10:30 UTC) and 20 January 2020 (22:00 UTC) spanning a period
        # of 66 months of data collection.
        # The reference epoch for Gaia DR4 is J2017.5.

        # Positions and proper motions are referred to the ICRS, to which
        # the optical reference frame defined by Gaia DR4 (Gaia-CRF4) is
        # aligned. The time coordinate for Gaia DR4 results is the
        # barycentric coordinate time (TCB).

        time_select = (obs_time_full > self.time_start.tcb.jd) & (
            obs_time_full < self.time_end.tcb.jd
        )

        table_select = healpix_table[time_select]

        # Reject a fraction of the data. About 10% of FOV
        # transits has an issue (see Lindegren et al. 2021)

        if reject_fraction is not None:
            rand_unif = rng.uniform(low=0.0, high=1.0, size=len(table_select))
            table_select = table_select[rand_unif > reject_fraction]

        psi, plx_factor, obs_time_tcb = (
            table_select["scanAngle[rad]"],
            table_select["parallaxFactorAlongScan"],
            table_select["ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"],
        )

        t_ast = Time(obs_time_tcb, format="jd", scale="tcb") - self.ref_epoch
        t_ast_day = t_ast.to_value("day")
        t_ast_yr = t_ast.to_value("yr")

        # Median CCD AL-scan abscissa uncertainty persource.
        # Digitized from Fig. 3 in Holl et al. (2023)

        n_ccd_avg = 8

        if sigma_per_ccd is None:
            file_folder = Path(__file__).resolve().parent.parent
            data_file = file_folder / "data/holl2023_sigma.csv"

            df = pd.read_csv(data_file)

            sigma_per_ccd = np.interp(
                self.g_mag,
                df["Gaia G mag"],
                df["CCD AL scan uncertainty (mas)"],
            )

        sigma_per_transit = sigma_per_ccd / np.sqrt(n_ccd_avg)

        if self.verbose:
            print(f"\nAL scan uncertainty (per CCD) = {1e3*sigma_per_ccd:.2f} uas")

            print(
                "AL scan uncertainty (per transit) = "
                f"{1e3*sigma_per_transit:.2f} uas"
            )

        sim_astrom = {
            "obs_time_tcb": t_ast_yr + self.ref_epoch.tcb.jyear,
            "relative_time_year": t_ast_yr,
            "relative_time_day": t_ast_day,
            "scan_pos_angle": np.degrees(psi),
            "sin_scan_ang": np.sin(psi),
            "cos_scan_ang": np.cos(psi),
            "parallax_factor_al": plx_factor,
        }

        self.data_table = pd.DataFrame(sim_astrom)

        if binary:
            # Calculate mass ratio

            mass_ratio = mass_2 / self.primary_mass[0]

            # Calculate flux ratio if needed

            if flux_ratio is None:
                flux_ratio = (mass_2 / self.primary_mass[0]) ** 3.5

            # Orbital period (days)
            # Use semi-major axis in au and masses in Msun

            period = (
                np.sqrt(model_param["sma"] ** 3 / (self.primary_mass[0] + mass_2))
                * 365.25
            )

            model_param["per"] = float(period)

            if self.verbose:
                print("\nAdditional parameters:")
                print(f"   - Primary mass (Msun) = {self.primary_mass[0]:.2f}")
                print(f"   - Companion mass (Msun) = {mass_2:.2e}")
                print(f"   - Mass ratio = {mass_2/self.primary_mass[0]:.2e}")
                print(f"   - Flux ratio = {flux_ratio:.2e}")
                print(f"   - Relative semi-major axis (au) = {model_param['sma']:.2f}")

        # Calculate the 1D astrometry of the stellar track
        # self is the current EpochAstrometry object

        star_model = StarModel(epoch_astrometry=self, verbose=self.verbose)

        cen_pos = star_model.calc_1d_model(model_param=model_param)

        sin_scan_ang = self.data_table["sin_scan_ang"].to_numpy()
        cos_scan_ang = self.data_table["cos_scan_ang"].to_numpy()

        if binary:
            # Convert relative semi-major axis from (au) to (mas)

            model_param["sma"] *= model_param["parallax"]

            # Add the component from the binary orbit, using the
            # relative semi-major axis (mas) in model_param

            kepler_model = KeplerModel(
                epoch_astrometry=self,
                verbose=self.verbose,
            )

            delta_ra, delta_dec = kepler_model.calc_orbit(
                model_param=model_param,
                obs_time=None,
            )

            delta_eta_rel = delta_ra * sin_scan_ang + delta_dec * cos_scan_ang

            # Apply the AL binary bias and add to the stellar track

            cen_pos += binary_bias(
                delta_eta_rel, mass_ratio, flux_ratio, verbose=self.verbose
            )

            # Convert semi-major axis from relative to photocenter (mas)
            # This assumes an unresolved binary

            f_term = flux_ratio / (1.0 + flux_ratio)
            m_term = mass_ratio / (1.0 + mass_ratio)

            model_param["sma"] *= f_term - m_term
            model_param["sma"] = abs(model_param["sma"])

            # Convert argument of periastron from relative to photocenter (mas)

            model_param["aop"] = (model_param["aop"] + np.pi) % (2.0 * np.pi)

            if self.verbose:
                print(f"\nPhotocenter semi-major axis (mas) = {model_param['sma']:.2f}")

                print(
                    "Photocenter argument of periastron "
                    f"(rad) = {model_param['aop']:.2f}"
                )

        cen_pos += rng.normal(loc=0.0, scale=sigma_per_transit, size=len(psi))

        self.data_table["centroid_pos_al"] = cen_pos

        self.data_table["centroid_pos_error_al"] = np.full(
            cen_pos.size, sigma_per_transit
        )

        if csv_out is not None:
            self.data_table.to_csv(csv_out, index=False)

        return model_param

    @beartype
    def get_nss_tables(self, gaia_release="DR3") -> None:
        """
        Method for downloading and storing the Gaia non-single star
        (NSS) tables. The output will be stored in `ECSV files
        <https://docs.astropy.org/en/stable/io/ascii/ecsv.html>`_.
        Currently, only ``gaia_release="DR3"`` is supported.

        Parameters
        ----------
        gaia_release : str
            Gaia release of which the NSS tables get downloaded.
            Currently, the only possible argument is "DR3".

        Returns
        -------
        NoneType
            None
        """

        if self.verbose:
            print_section("Retrieve Gaia non-single star tables")

        if gaia_release != "DR3":
            raise ValueError(
                "The get_nss_table() only supports Gaia DR3. "
                "Please set the 'gaia_release' argument to 'DR3'."
            )

        # List all Gaia tables
        # for table_item in Gaia.load_tables(only_names=True):
        #     print (table_item.get_qualified_name())

        if gaia_release == "DR3":
            # Gaia DR3 NSS tables
            gaia_tables = [
                "gaiadr3.nss_acceleration_astro",
                "gaiadr3.nss_non_linear_spectro",
                "gaiadr3.nss_two_body_orbit",
                "gaiadr3.nss_vim_fl",
            ]

        elif gaia_release == "DR4":
            # Gaia DR4 NSS tables

            gaia_tables = [
                "gaiadr4.nss_acceleration_astro",
                "gaiadr4.nss_non_linear_spectro",
                "gaiadr4.nss_two_body_orbit",
                "gaiadr4.nss_vim_fl",
                "gaiadr4.nss_epoch_flags",
                "gaiadr4.nss_masses",
                "gaiadr4.nss_multiple_orbits",
                "gaiadr4.nss_multiplicity",
                "gaiadr4.nss_resolved_pair",
            ]

        else:
            raise ValueError("The '{self.gaia_release}' is not supported.")

        for table_item in gaia_tables:
            # Query Gaia NSS tables

            gaia_query = f"""
            SELECT *
            FROM {table_item}
            """

            # Launch the Gaia job and get the results

            gaia_job = Gaia.launch_job_async(
                gaia_query, dump_to_file=False, verbose=False
            )
            gaia_result = gaia_job.get_results()

            gaia_result.write(
                f"{table_item}.ecsv",
                format="ascii.ecsv",
                overwrite=True,
            )

    @beartype
    def query_source(
        self,
        source_id: typing.Optional[typing.Union[int, np.int64, str]] = None,
        gaia_release: str = "DR3",
    ) -> typing.Dict[str, Real]:
        """
        Method for retrieving stellar parameter from the Gaia catalog,
        specifically the RA/Dec, parallax, proper motion, and G-band
        magnitude.

        Parameters
        ----------
        source_id : int, str
            Gaia source ID for the selected ``gaia_release``.
        gaia_release : str
            Gaia data release (default: DR3). The release can be set
            both here and with the class initialization such that the
            stellar parameters can be retrieved from DR3, but epoch
            astrometry can be simulated for DR4/DR5.

        Returns
        -------
        dict
            Dictionary with the retrieved stellar parameters:
            RA, Dec, parallax, RA proper motion, Dec proper
            motion, G-band magnitude.
        """

        if self.verbose:
            print_section("Querying source")

        self.source_id = source_id

        if gaia_release != "DR3":
            raise ValueError(
                "The 'query_source' method supports "
                "currently only gaia_release='DR3'."
            )

        if self.verbose:
            print(f"Gaia release: {gaia_release}")
            print(f"Source ID: {self.source_id}\n")

        # Retrieve RA, Dec, parallax, proper motion, and G magnitude

        gaia_query = f"""
        SELECT ra, ra_error, dec, dec_error, parallax, parallax_error,
               pmra, pmra_error, pmdec, pmdec_error, phot_g_mean_mag,
               astrometric_chi2_al, astrometric_n_good_obs_al, ruwe,
               astrometric_excess_noise, astrometric_excess_noise_sig,
               non_single_star, nu_eff_used_in_astrometry
        FROM gaia{gaia_release.lower()}.gaia_source
        WHERE source_id = {self.source_id}
        """

        gaia_job = Gaia.launch_job(gaia_query, dump_to_file=False, verbose=False)

        gaia_result = gaia_job.get_results()[0]

        ra_ref = float(gaia_result["ra"])
        ra_ref_error = float(gaia_result["ra_error"])
        dec_ref = float(gaia_result["dec"])
        dec_ref_error = float(gaia_result["dec_error"])
        parallax = float(gaia_result["parallax"])
        parallax_error = float(gaia_result["parallax_error"])
        pmra = float(gaia_result["pmra"])
        pmra_error = float(gaia_result["pmra_error"])
        pmdec = float(gaia_result["pmdec"])
        pmdec_error = float(gaia_result["pmdec_error"])
        phot_g_mean_mag = float(gaia_result["phot_g_mean_mag"])

        if self.verbose:
            print(f"\nRA = {ra_ref:.3f} deg +/- {ra_ref_error:.3f} mas")
            print(f"Dec = {dec_ref:.3f} deg +/- {dec_ref_error:.3f} mas")
            print(f"Parallax = {parallax:.3f} +/- {parallax_error:.3f} mas")
            print(f"Proper motion in RA = {pmra:.3f} +/- {pmra_error:.3f} mas/yr")
            print(f"Proper motion in Dec = {pmdec:.3f} +/- {pmdec_error:.3f} mas/yr")
            print(f"G-band magnitude = {phot_g_mean_mag:.3f}")

        # Effective wavenumber (i.e. pseudocolor) of the source
        # used in the astrometric solution (um-1)
        if not np.ma.is_masked(gaia_result["nu_eff_used_in_astrometry"]):
            nu_eff = float(gaia_result["nu_eff_used_in_astrometry"])
            if self.verbose:
                print(f"Pseudocolor = {nu_eff:.2f} um-1")

        else:
            nu_eff = None
            if self.verbose:
                print("Pseudocolor = None")

        self.ra_ref = ra_ref
        self.dec_ref = dec_ref
        self.parallax = parallax
        self.pm_ra = pmra
        self.pm_dec = pmdec
        self.g_mag = phot_g_mean_mag

        # Gaia (E)DR3: Re-normalised Unit Weight Error (RUWE)
        # tables of u0(g,c) by L. Lindegren (2023 Sep 13)
        # https://www.cosmos.esa.int/web/gaia/dr3-auxiliary-data

        if nu_eff is None:
            self.u0_norm = None

        else:
            self.u0_norm = self._interpolate_u0(self.g_mag, nu_eff)

        n_param = 5

        uwe = np.sqrt(
            gaia_result["astrometric_chi2_al"]
            / (gaia_result["astrometric_n_good_obs_al"] - n_param)
        )

        if self.verbose:
            print(f"\nUWE = {uwe:.2f}")

        if "ruwe" in gaia_result.columns:
            if not np.ma.is_masked(gaia_result["ruwe"]) and self.u0_norm is not None:
                ruwe = gaia_result["ruwe"]
                if self.verbose:
                    print(f"RUWE = {ruwe:.2f}")

                if not np.isclose(self.u0_norm, uwe / ruwe, rtol=1e-2, atol=0.0):
                    warnings.warn(
                        f"The renormalization value is {self.u0_norm:.4f} "
                        f"whereas the ratio of uwe/ruwe is {uwe/ruwe:.4f}."
                    )

        if "astrometric_excess_noise" in gaia_result.columns:
            if (
                not np.ma.is_masked(gaia_result["astrometric_excess_noise"])
                and gaia_result["astrometric_excess_noise"] != 0.0
            ):
                aen = gaia_result["astrometric_excess_noise"]
                aen_sig = gaia_result["astrometric_excess_noise_sig"]

                if self.verbose:
                    print(
                        f"Astrometric excess noise (mas) = {aen:.4f} +/- {aen/aen_sig:.4f}"
                    )

        if "non_single_star" in gaia_result.columns:
            if not np.ma.is_masked(gaia_result["non_single_star"]):
                if self.verbose:
                    print(f"Non single star = {gaia_result['non_single_star']}")

        model_param = {
            "ra_ref": self.ra_ref,
            "dec_ref": self.dec_ref,
            "parallax": self.parallax,
            "pm_ra": self.pm_ra,
            "pm_dec": self.pm_dec,
            "g_mag": self.g_mag,
        }

        return model_param

    @beartype
    def retrieve_data(
        self,
        source_id: typing.Optional[typing.Union[int, np.int64, str]] = None,
        exclude_outliers: bool = True,
        combine_ccds: bool = False,
    ) -> None:
        """
        Method for retrieving the epoch astrometry for the selected
        Gaia source. This will only be possible for the future DR4
        and DR5 data releases.

        Parameters
        ----------
        source_id : int, str
            Gaia source ID for the selected ``gaia_release`` of the
            class initialization.
        exclude_outliers : bool
            Exclude astrometry points that are flagged in the table
            as outlier (default: True). To be implemented.
        combine_ccds : bool
            Combine/average the measurements of the 9 CCDs per
            transit ID (default: False). The weighted combination
            of the positions and uncertainties is calculated,
            assuming uncorrelated uncertainties between CCDs.
            To be implemented.

        Returns
        -------
        NoneType
            None
        """

        self.query_source(source_id, gaia_release=self.gaia_release)

        if self.verbose:
            print_section("Retrieving epoch astrometry")

        self.source_id = source_id

        if self.verbose:
            print(f"Gaia release: {self.gaia_release}")
            print(f"Source ID: {self.source_id}")

        raise NotImplementedError(
            "Retrieval of Gaia DR4 and DR5 epoch astrometry is not yet implemented."
        )

        # gaia_tables = [
        #     "epoch_astrometry",
        #     "bright_source_astrometry",
        #     "nss_acceleration_astro",
        #     "nss_two_body_orbit",
        # ]
        #
        # for table_item in gaia_tables:
        #     if self.verbose:
        #         print(f"\nTable: gaia{self.gaia_release.lower()}.{table_item}")
        #
        #     # Query Gaia source ID in NSS tables for selected Gaia source ID
        #
        #     gaia_query = f"""
        #     SELECT *
        #     FROM gaia{self.gaia_release.lower()}.{table_item}
        #     WHERE source_id = {self.source_id}
        #     """
        #
        #     # Launch the Gaia job and get the results
        #
        #     gaia_job = Gaia.launch_job_async(
        #         gaia_query, dump_to_file=False, verbose=False
        #     )
        #
        #     gaia_result = gaia_job.get_results()
        #
        #     if self.verbose:
        #         if len(gaia_result) > 0:
        #             print("\nTable parameters:")
        #             for param_item in gaia_result[0].columns:
        #                 print(f"   - {param_item} = {gaia_result[0][param_item]}")
        #
        #     else:
        #         if self.verbose:
        #             print(f"\nSource not found in {table_item}")
        #
        #     if exclude_outliers:
        #         pass
        #
        #     if combine_ccds:
        #         pass

    @beartype
    def retrieve_gaia_bh3(
        self,
        exclude_outliers: bool = True,
        combine_ccds: bool = False,
    ) -> None:
        """
        Method for storing the Gaia DR4 epoch astrometry of
        the black hole Gaia BH3 in the ``data_table``. The
        data file can be found `here <https://github.com/
        tomasstolker/exogaia/blob/main/data/
        gaiabh3_epochast.dat>`_, which is from
        `here <https://github.com/esa/gaia-bhthree>`_.

        Parameters
        ----------
        exclude_outliers : bool
            Exclude astrometry points that are flagged in the table
            as outlier (default: True).
        combine_ccds : bool
            Combine/average the measurements of the 9 CCDs per
            transit ID (default: False). The weighted combination
            of the positions and uncertainties is calculated,
            assuming uncorrelated uncertainties between CCDs.

        Returns
        -------
        NoneType
            None
        """

        self.source_id = 4318465066420528000

        if self.gaia_release != "DR4":
            raise ValueError(
                "Please set 'gaia_release' to 'DR4' when using "
                "the DR4 epoch astrometry data of Gaia BH3."
            )

        _ = self.query_source(source_id=self.source_id, gaia_release="DR3")

        if self.verbose:
            print_section("Gaia BH3 epoch data")

        file_folder = Path(__file__).resolve().parent.parent
        data_file = file_folder / "data/gaiabh3_epochast.dat"

        if self.verbose:
            print(f"Gaia release: {self.gaia_release}")
            print(f"Reference epoch: {self.ref_epoch.tcb.jyear_str}")
            print(f"Source ID: {self.source_id}")

        if self.primary_mass is None:
            self.primary_mass = (0.76, 0.05)  # (Msun)

        if self.verbose:
            print(
                f"\nPrimary mass (Msun): {self.primary_mass[0]:.2f} "
                f"+/- {self.primary_mass[1]:.2f}"
            )

        self.data_table = pd.read_csv(
            data_file, sep=r"\s+", header="infer", comment="#", skip_blank_lines=True
        )

        self.data_table["centroid_pos_al"] = self.data_table["centroid_pos_al"]
        self.data_table["centroid_pos_error_al"] = self.data_table[
            "centroid_pos_error_al"
        ]

        if exclude_outliers:
            self.data_table = self.data_table[self.data_table["outlier_flag"] == 0]

        if combine_ccds:
            cols_to_keep = [
                "transit_id",
                "obs_time_tcb",
                "parallax_factor_al",
                "scan_pos_angle",
                "centroid_pos_al",
                "centroid_pos_error_al",
            ]

            # Select the central row/CCD for each transit
            agg_dict = {col: lambda x: x.iloc[len(x) // 2] for col in cols_to_keep}

            def weighted_mean(al_pos, al_err):
                weight = 1.0 / al_err**2
                return np.sum(weight * al_pos) / np.sum(weight)

            def weighted_error(al_err):
                return 1.0 / np.sqrt(np.sum(1.0 / al_err**2))

            agg_dict["centroid_pos_al"] = lambda x: weighted_mean(
                x.values, self.data_table.loc[x.index, "centroid_pos_error_al"].values
            )

            agg_dict["centroid_pos_error_al"] = weighted_error

            df_transit = (
                self.data_table[cols_to_keep]
                .groupby("transit_id", sort=False)
                .agg(agg_dict)
                .reset_index(drop=True)
            )

            self.data_table = df_transit

        # Convert scan angles from degrees to radians
        # Store the sin and cos since only these are needed
        scan_ang = np.radians(self.data_table["scan_pos_angle"])
        self.data_table["sin_scan_ang"] = np.sin(scan_ang)
        self.data_table["cos_scan_ang"] = np.cos(scan_ang)

        # Convert from Julian days to Julian years
        self.data_table["obs_time_tcb"] = Time(
            self.data_table["obs_time_tcb"], format="jd", scale="tcb"
        ).tcb.jyear

        self.data_table["relative_time_year"] = (
            self.data_table["obs_time_tcb"] - self.ref_epoch.tcb.jyear
        )

        self.data_table["relative_time_day"] = self.data_table[
            "relative_time_year"
        ] * u.year.to(u.day)

        if self.verbose:
            print(f"\nData file: {data_file}")
            print(f"Data shape: {self.data_table.shape}")

    @beartype
    def retrieve_dr4_prelease(
        self,
        source_id: typing.Union[int, np.int64],
        exclude_outliers: bool = True,
        combine_ccds: bool = False,
    ) -> None:
        """
        Method for retrieving the epoch astrometry for the selected
        Gaia source. This will only be possible for the future DR4
        and DR5 data releases. Important: when setting,
        ``gaia_release='DR3'``, the DR4 epoch astrometry will be
        cropped to ``time_end`` of DR3. This is useful for testing
        purposes, for example to calculate the RUWE and orbit
        parameters for comparison with the actual DR3 RUWE and
        NSS orbit parameters.

        Parameters
        ----------
        source_id : int
            Gaia DR4 source ID. Should be any of the sources that are
            part of the `pre-release <https://www.cosmos.esa.int/web/
            gaia/dr4-prerelease>`_ list.
        exclude_outliers : bool
            Exclude astrometry points that are flagged in the table
            as outlier (default: True). To be implemented.
        combine_ccds : bool
            Combine/average the measurements of the 9 CCDs per
            transit ID (default: False). The weighted combination
            of the positions and uncertainties is calculated,
            assuming uncorrelated uncertainties between CCDs.
            To be implemented.

        Returns
        -------
        NoneType
            None
        """

        if self.gaia_release not in ["DR3", "DR4"]:
            raise ValueError(
                "Please set 'gaia_release' to 'DR3' or "
                "'DR4' when using the DR4 pre-release "
                "epoch astrometry data."
            )

        if self.gaia_release == "DR3":
            warnings.warn(
                "By setting 'gaia_release' to 'DR3', the DR4 "
                "epoch astrometry will be cropped to an end "
                "date of {self.time_end}."
            )

        self.source_id = source_id

        self.query_source(self.source_id, gaia_release="DR3")

        if self.verbose:
            print_section("Retrieving epoch astrometry")

        if self.verbose:
            print(f"Gaia release: {self.gaia_release}")
            print(f"Reference epoch: {self.ref_epoch.tcb.jyear_str}")
            print(f"Source ID: {self.source_id}")

            print(
                f"\nPrimary mass (Msun): {self.primary_mass[0]:.2f} "
                f"+/- {self.primary_mass[1]:.2f}"
            )

        file_name = "GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml"
        data_file = self.data_folder / file_name
        url = (
            "https://home.strw.leidenuniv.nl/~stolker/exogaia/"
            "GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml"
        )

        if not data_file.exists():
            if self.verbose:
                print()

            pooch.retrieve(
                url=url,
                known_hash="f81f4dc11064b72d99f536e3d34365694b839629b424d8247a4b5501016f3ce3",
                fname=file_name,
                path=self.data_folder,
                progressbar=True,
            )

        table = Table.read(data_file, format="votable")
        df_full = table.to_pandas()

        if self.verbose:
            print(f"\nData file: {data_file}")
            print(f"Data shape: {df_full.shape}")

        # Extract data of selected source_id
        # Should be part of https://www.cosmos.esa.int/web/gaia/dr4-prerelease

        self.data_table = df_full[df_full["source_id"] == self.source_id]

        # Remove column 'centroid_pos_ac' or any column with all rows set to NaN

        self.data_table.dropna(axis="columns", how="all", inplace=True)

        # Explode the columns that contains lists
        # This transforms list-like values in a column into multiple rows, while
        # duplicating the values in the other columns that contain single values

        cols_explode = [
            "obs_time_tcb",
            "scan_pos_angle",
            "colour_factor_al",
            "colour_factor_ac",
            "centroid_pos_al",
            "centroid_pos_error_al",
            "calculated_pos_ac",
            "used_by_agis_al",
            "used_by_agis_ac",
            "ccd_proc_flags",
            "ipd_error_al",
            "ipd_error_ac",
            "gates",
            "source_dist_to_last_ci",
            "sub_pixel_coord",
            "mu",
        ]

        self.data_table = self.data_table.explode(cols_explode).reset_index(drop=True)

        # Total number of rows for selected source

        n_total = self.data_table.shape[0]

        # Find rows containing masked values and remove those rows

        mask = self.data_table.apply(lambda col: col.map(np.ma.is_masked)).any(axis=1)
        self.data_table = self.data_table.loc[~mask].reset_index(drop=True)

        # Convert object dtype to the actual dtype

        self.data_table = self.data_table.infer_objects(copy=False).convert_dtypes()

        # Exclude outliers based on ccd_proc_flags

        if exclude_outliers:
            # self.data_table = self.data_table[
            #     self.data_table["ccd_proc_flags"] == 0
            # ].reset_index(drop=True)

            self.data_table = self.data_table[self.data_table["used_by_agis_al"]]

        # Remaining number of rows after removing rows with a masked value
        # and optionally removing outliers based on used_by_agis_al

        n_selected = self.data_table.shape[0]

        print("\nNumber of rows for selected source:")
        print(f"   - Total = {n_total}")
        print(f"   - Selected = {n_selected}")

        # Sort data chronologically

        self.data_table.sort_values("obs_time_tcb", inplace=True, ignore_index=True)

        if self.data_table["nu_eff_used_in_astrometry"].nunique() != 1:
            raise ValueError(
                "There should be only one unique value of "
                "nu_eff_used_in_astrometry for each source."
            )

        # Gaia (E)DR3: Re-normalised Unit Weight Error (RUWE)
        # tables of u0(g,c) by L. Lindegren (2023 Sep 13)
        # https://www.cosmos.esa.int/web/gaia/dr3-auxiliary-data

        nu_eff = self.data_table["nu_eff_used_in_astrometry"].unique()[0]
        self.u0_norm = self._interpolate_u0(self.g_mag, nu_eff)

        # Select relevant columns needed for exogaia

        # cols_select = [
        #     "transit_id",
        #     "obs_time_tcb",
        #     "scan_pos_angle",
        #     "centroid_pos_al",
        #     "centroid_pos_error_al",
        #     "parallax_factor_al",
        # ]
        #
        # self.data_table = self.data_table[cols_select]

        # Convert selected columns from object dtype to numeric

        cols_convert = ["scan_pos_angle", "centroid_pos_al", "centroid_pos_error_al"]

        for item in cols_convert:
            self.data_table[item] = pd.to_numeric(
                self.data_table[item], errors="coerce"
            )

        # Combine CCDs per transit

        if combine_ccds:
            cols_to_keep = list(self.data_table.columns)

            # Select the central row/CCD for each transit
            agg_dict = {col: lambda x: x.iloc[len(x) // 2] for col in cols_to_keep}

            def weighted_mean(al_pos, al_err):
                weight = 1.0 / al_err**2
                return np.sum(weight * al_pos) / np.sum(weight)

            def weighted_error(al_err):
                return 1.0 / np.sqrt(np.sum(1.0 / al_err**2))

            agg_dict["centroid_pos_al"] = lambda x: weighted_mean(
                x.values, self.data_table.loc[x.index, "centroid_pos_error_al"].values
            )

            agg_dict["centroid_pos_error_al"] = weighted_error

            df_transit = (
                self.data_table[cols_to_keep]
                .groupby("transit_id", sort=False)
                .agg(agg_dict)
                .reset_index(drop=True)
            )

            self.data_table = df_transit

        # Convert scan angles from degrees to radians
        # Store the sin and cos since only these are needed

        scan_ang = np.radians(self.data_table["scan_pos_angle"].astype(float))
        self.data_table["sin_scan_ang"] = np.sin(scan_ang)
        self.data_table["cos_scan_ang"] = np.cos(scan_ang)

        # obs_time_tcb : effective observing time as TCB (Long[10] array, Time[ns])
        # Effective observation time for each CCD in this FoV transit.
        # Centre of the actual exposure timefor the given window/CCD, depending on
        # readout time and gate. In units of nanoseconds sinceJ2010.0(TCB). The
        # presentation as long integer (rather than as double float) is chosen for
        # the sake of keepingthe full resolution of the time coordinates used in
        # Gaia/DPAC, and for the sake of invertability oftransformations from OBMT
        # to TCB and vice versa.Such transformations are done using HATT
        # (High-Accuracy Time Transformations), which havethe resolution of
        # 1 nanosecond. But users should be aware that the absolute accuracy of
        # them isin the order of 150 nanoseconds

        # Store obs_time_tcb in Julian years

        obs_time = (
            Time("J2010.0", scale="tcb")
            + self.data_table["obs_time_tcb"].astype(np.int64).to_numpy() * u.ns
        )

        self.data_table["obs_time_tcb"] = obs_time.tcb.jyear

        # Exclude observations after the requested end time of DR3

        if self.gaia_release == "DR3":
            self.data_table = self.data_table[
                self.data_table["obs_time_tcb"] <= self.time_end.tcb.jyear
            ].reset_index(drop=True)

        # Store times as Julian years and days relative to ref_epoch

        time_from_ref = self.data_table["obs_time_tcb"] - self.ref_epoch.tcb.jyear

        self.data_table["relative_time_year"] = time_from_ref.to_numpy()

        self.data_table["relative_time_day"] = self.data_table[
            "relative_time_year"
        ] * u.year.to(u.day)


class HipparcosAstrometry(EpochAstrometry):
    """
    Class for querying Hipparcos epoch astrometry data.
    """

    @beartype
    def __init__(
        self,
        hip_id: int,
        primary_mass: typing.Tuple[Real, Real],
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        primary_mass : tuple(float, float)
            Primary mass and uncertainty (Msun). The primary mass is
            not stored if the argument is set to ``None``. When
            simulating data with
            :func:`~exogaia.data.EpochAstrometry.simulate_data`,
            a mass will be drawn from an initial mass function if
            the argument of ``primary_mass`` is set to ``None``.
        verbose : bool
            Print some information.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(primary_mass, verbose)

        self.verbose = verbose

        if self.verbose:
            print_section("Hipparcos astrometry", bound_char="=")

        self.data_folder = Path.home() / ".exogaia"

        if not self.data_folder.exists():
            if self.verbose:
                print(f"Creating folder: {str(self.data_folder)}")

            self.data_folder.mkdir(parents=True, exist_ok=False)

        self.hip_id = hip_id
        self.gaia_dr3_id = None
        self.hp_mag = None
        self.solution_type = None
        self.u0_norm = 1.0  # Needed for the RUWE calculation

        self.hgca_pm_hip = None
        self.hgca_pm_gaia = None
        self.hgca_pm_hg = None

        self.hgca_cov_hip = None
        self.hgca_cov_gaia = None
        self.hgca_cov_hg = None

        self.hgca_dpm_gaia_hg = None
        self.hgca_dv_gaia_hg = None

        self.hgca_epoch_gaia = None
        self.hgca_epoch_hip = None

        self.hgca_chisq = None

        self.ref_epoch = Time(1991.25, format="jyear", scale="tcb")
        self.time_start = Time("1989-11-26", scale="utc")
        self.time_end = Time("1993-08-15", scale="utc")

        if self.verbose:
            print(f"Reference epoch: {self.ref_epoch.tcb.jyear_str}")

            print(
                f"Primary mass (Msun): {primary_mass[0]:.2f} "
                f"+/- {primary_mass[1]:.2f}"
            )

    @beartype
    def retrieve_data(
        self,
    ) -> dict:
        """
        Retrieve Hipparcos-2 epoch astrometry.

        Query SIMBAD for the Gaia DR3 counterpart of the Hipparcos
        source, download the source-specific Intermediate Astrometric
        Data file when it is not already available locally, and read
        the reference astrometric solution from its header. Rejected
        or invalid observations are removed, and the accepted
        along-scan measurements are stored in ``self.data_table``.

        Returns
        -------
        dict
            Reference astrometric parameters used for the
            Hipparcos-2 solution.
        """

        if self.verbose:
            print_section("Retrieve Hipparcos data")

        # Query SIMBAD for the corresponding Gaia DR3 source ID

        Simbad.add_votable_fields("ids")
        simbad_result = Simbad.query_object(f"HIP {self.hip_id}")

        if simbad_result is None or len(simbad_result) == 0:
            raise ValueError(f"HIP {self.hip_id} was not found in SIMBAD.")

        gaia_matches = [
            identifier.strip()
            for identifier in simbad_result["ids"][0].split("|")
            if identifier.strip().startswith("Gaia DR3 ")
        ]

        if not gaia_matches:
            raise ValueError(
                f"HIP {self.hip_id} does not have a Gaia DR3 ID in SIMBAD."
            )

        self.gaia_dr3_id = np.int64(gaia_matches[0].removeprefix("Gaia DR3 ").strip())

        # Hipparcos epochs
        # The Hipparcos IAD epochs are provided as offsets in Julian years (?)
        # relative to J1991.25. These epochs are represented on the TCB scale
        # for consistency with the Gaia epoch astrometry. The adopted time
        # scale has a negligible effect (?) at Hipparcos precision.

        if self.verbose:
            print(f"Hipparcos ID: {self.hip_id}")
            print(f"Gaia DR3 ID: {self.gaia_dr3_id}")
            print(f"Reference epoch: {self.ref_epoch.tcb.jyear_str}")

        # Download the source-specific Hipparcos-2 IAD file

        hip_str = f"{self.hip_id:06d}"  # e.g. 25486 -> "025486"
        hip_folder = f"H{hip_str[:3]}"  # "H025"
        file_name = f"H{hip_str}.d"  # "H025486.d"

        data_file = self.data_folder / file_name

        url = (
            "https://home.strw.leidenuniv.nl/"
            "~stolker/exogaia/ResRec_JavaTool_2014/"
            f"{hip_folder}/{file_name}"
        )

        if not data_file.exists():
            if self.verbose:
                print()

            pooch.retrieve(
                url=url,
                known_hash=None,
                fname=file_name,
                path=self.data_folder,
                progressbar=True,
            )

        # Read Hipparcos-2 header

        header = read_hipparcos_header(data_file)

        ra_ref = header["ra_deg"]
        ra_ref_error = header["ra_error"]
        dec_ref = header["dec_deg"]
        dec_ref_error = header["dec_error"]
        parallax = header["parallax"]
        parallax_error = header["parallax_error"]
        pmra = header["pm_ra"]
        pmra_error = header["pm_ra_error"]
        pmdec = header["pm_dec"]
        pmdec_error = header["pm_dec_error"]
        hp_mag = header["hp_mag"]

        dpm_ra = None
        dpm_ra_error = None
        dpm_dec = None
        dpm_dec_error = None
        ddpm_ra = None
        ddpm_ra_error = None
        ddpm_dec = None
        ddpm_dec_error = None

        self.solution_type = header["solution_type"]

        if self.solution_type in (7, 9):
            dpm_ra = header["dpmRA"]
            dpm_ra_error = header["e_dpmRA"]
            dpm_dec = header["dpmDE"]
            dpm_dec_error = header["e_dpmDE"]

            if self.solution_type == 9:
                ddpm_ra = header["ddpmRA"]
                ddpm_ra_error = header["e_ddpmRA"]
                ddpm_dec = header["ddpmDE"]
                ddpm_dec_error = header["e_ddpmDE"]

        if self.verbose:
            solution_labels = {
                1: "stochastic",
                5: "5-parameters",
                7: "7-parameters",
                9: "9-parameters",
            }

            sol_type = solution_labels.get(self.solution_type, "other")

            print(f"\nSolution type: {header['solution_type']} ({sol_type})")
            print(f"F2: {header['f2']:.2f}")

            print(f"\nRA = {ra_ref:.3f} deg +/- {ra_ref_error:.3f} mas")
            print(f"Dec = {dec_ref:.3f} deg +/- {dec_ref_error:.3f} mas")
            print(f"Parallax = {parallax:.3f} +/- {parallax_error:.3f} mas")
            print(f"Proper motion in RA = {pmra:.3f} +/- {pmra_error:.3f} mas/yr")
            print(f"Proper motion in Dec = {pmdec:.3f} +/- {pmdec_error:.3f} mas/yr")
            print(f"Variance inflation: {header['var']:.2f} mas")

            if self.solution_type in (7, 9):
                print(f"\ndmu/dt in RA = {dpm_ra:.3f} +/- {dpm_ra_error:.3f} mas/yr^2")
                print(f"dmu/dt in Dec = {dpm_dec:.3f} +/- {dpm_dec_error:.3f} mas/yr^2")

                if self.solution_type == 9:
                    print(
                        f"\nd^2mu/d^2t in RA = {ddpm_ra:.3f} +/- {ddpm_ra_error:.3f} mas/yr^3"
                    )
                    print(
                        f"d^2mu/d^2t in Dec = {ddpm_dec:.3f} +/- {ddpm_dec_error:.3f} mas/yr^3"
                    )

            print(f"\nHp-band magnitude = {hp_mag:.3f}")

        self.ra_ref = ra_ref
        self.dec_ref = dec_ref
        self.parallax = parallax
        self.pm_ra = pmra
        self.pm_dec = pmdec
        self.hp_mag = hp_mag

        model_param = {
            "ra_ref": self.ra_ref,
            "dec_ref": self.dec_ref,
            "parallax": self.parallax,
            "pm_ra": self.pm_ra,
            "pm_dec": self.pm_dec,
            "hp_mag": self.hp_mag,
        }

        # Read Hipparcos-2 data
        # https://www.cosmos.esa.int/web/hipparcos/hipparcos-2

        #   3 - 6   I4                 IORB     Orbit Number
        #   8 - 14  F7.4   yr-1991.25  EPOCH    Epoch
        #  16 - 22  F7.4               PARF     Parallax factor
        #  24 - 30  F7.4               CPSI     cos(psi) (1)
        #  32 - 38  F7.4               SPSI     sin(psi) (1)
        #  40 - 46  F7.2   mas         RES      Abscissa residual
        #  48 - 53  F6.2   mas         SRES     Formal error on abscissa residual (2)
        # --------------------------------------------------------------------------------
        # Note (1): The Hipparcos-2 angle psi is related to the Gaia scan angle theta as
        # theta = pi/2 - psi. See Brandt et al. (2021), Section 2.
        # Note (2): Rejected observations are marked as negative or zero (0.00) values.

        dtype = np.dtype(
            [
                ("iorb", np.int32),
                ("epoch", np.float64),
                ("parf", np.float64),
                ("cpsi", np.float64),
                ("spsi", np.float64),
                ("res", np.float64),
                ("sres", np.float64),
            ]
        )

        iad_data = np.loadtxt(data_file, dtype=dtype)

        # In the Hipparcos-2 IAD, SRES <= 0 marks observations that
        # were rejected from the published astrometric solution

        accepted = (
            np.isfinite(iad_data["epoch"])
            & np.isfinite(iad_data["parf"])
            & np.isfinite(iad_data["cpsi"])
            & np.isfinite(iad_data["spsi"])
            & np.isfinite(iad_data["res"])
            & np.isfinite(iad_data["sres"])
            & (iad_data["sres"] > 0.0)
        )

        n_rejected = np.count_nonzero(~accepted)

        iad_data = iad_data[accepted]

        # Hipparcos uses theta = pi/2 - psi with psi and theta
        # the Gaia and Hipparcos scan angles, respectively.
        # header['var'] is nonzero only when self.solution_type == 1

        self.data_table = pd.DataFrame(
            {
                "orbit_number": iad_data["iorb"],
                "parallax_factor_al": iad_data["parf"],
                "sin_scan_ang": iad_data["cpsi"],
                "cos_scan_ang": iad_data["spsi"],
            }
        )

        # Check if the dataframe contains data

        if self.data_table.empty:
            raise ValueError(
                f"No accepted Hipparcos scans remain for HIP {self.hip_id}."
            )

        # Store times as Julian years and days relative to hip_ref_epoch

        self.data_table["obs_time_tcb"] = self.ref_epoch.tcb.jyear + iad_data["epoch"]

        self.data_table["relative_time_year"] = iad_data["epoch"]

        self.data_table["relative_time_day"] = self.data_table[
            "relative_time_year"
        ] * u.year.to(u.day)

        # Star track, relative to RA/Dec at hip_ref_epoch

        obs_pos_al = (
            self.parallax * self.data_table["parallax_factor_al"]
            + self.pm_ra
            * self.data_table["relative_time_year"]
            * self.data_table["sin_scan_ang"]
            + self.pm_dec
            * self.data_table["relative_time_year"]
            * self.data_table["cos_scan_ang"]
        )

        if self.solution_type in [7, 9]:
            obs_pos_al += (
                0.5
                * dpm_ra
                * self.data_table["relative_time_year"] ** 2
                * self.data_table["sin_scan_ang"]
                + 0.5
                * dpm_dec
                * self.data_table["relative_time_year"] ** 2
                * self.data_table["cos_scan_ang"]
            )

        if self.solution_type == 9:
            obs_pos_al += (
                ddpm_ra
                / 6.0
                * self.data_table["relative_time_year"] ** 3
                * self.data_table["sin_scan_ang"]
                + ddpm_dec
                / 6.0
                * self.data_table["relative_time_year"] ** 3
                * self.data_table["cos_scan_ang"]
            )

        self.data_table["centroid_pos_al"] = obs_pos_al + iad_data["res"]
        self.data_table["centroid_pos_error_al"] = iad_data["sres"]

        # Subtract the jitter which is nonzero when solution_type = 1

        # self.data_table["centroid_pos_error_al"] = np.sqrt(
        #     iad_data["sres"] ** 2 - header["var"] ** 2
        # )

        if self.verbose:
            print(f"\nAccepted scans: {len(self.data_table)}")
            print(f"Rejected scans: {n_rejected}")
            print("\nReference: van Leeuwen F. (2007), ASSL, 350")

        return model_param

    @beartype
    def add_hgca(
        self,
    ) -> pd.DataFrame:
        """
        Retrieve Hipparcos–Gaia Catalog of Accelerations data.

        The Gaia EDR3 version of the Hipparcos–Gaia Catalog of
        Accelerations (HGCA) is downloaded when it is not already
        available locally. The catalog is searched for the selected
        Hipparcos source, identified by `self.hip_id`.

        For the matching source, the Hipparcos, Gaia, and long-term
        Hipparcos–Gaia proper-motion vectors and covariance matrices
        are stored as class attributes. The Gaia proper-motion anomaly
        is calculated as

        .. math::

        ```
        \\Delta\boldsymbol{\\mu}_{\\mathrm{Gaia-HG}}
        =
        \\boldsymbol{\\mu}_{\\mathrm{Gaia}}
        -
        \\boldsymbol{\\mu}_{\\mathrm{HG}},
        ```

        where :math:`\\boldsymbol{\\mu}_{\\mathrm{HG}}` is the long-term
        proper motion derived from the Hipparcos and Gaia positions.
        The anomaly amplitude and its equivalent tangential velocity
        are also calculated and stored.

        Returns
        -------
        DataFrame
            One-row table containing the matching HGCA source.
        """

        if self.verbose:
            print_section("Retrieve Hipparcos–Gaia Catalog of Accelerations")

        def proper_motion_cov(
            pmra_error: float,
            pmdec_error: float,
            correlation: float,
        ) -> np.ndarray:
            """
            Construct a proper-motion covariance matrix.

            Parameters
            ----------
            pmra_error : float
                Uncertainty on proper motion in RA* (mas/yr).
            pmdec_error : float
                Uncertainty on proper motion in Dec (mas/yr).
            correlation : float
                Correlation coefficient between the RA* and
                Dec proper motions.

            Returns
            -------
            np.ndarray
                Proper-motion covariance matrix with shape (2, 2),
                in (mas/yr)^2.
            """

            covariance = correlation * pmra_error * pmdec_error

            cov_matrix = np.array(
                [
                    [pmra_error**2, covariance],
                    [covariance, pmdec_error**2],
                ],
            )

            return cov_matrix

        # Download the HGCA

        file_name = "HGCA_vEDR3.fits"
        fits_file = self.data_folder / file_name
        url = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/254/42/HGCA_vEDR3.fits"

        if not fits_file.exists():
            pooch.retrieve(
                url=url,
                known_hash="23684d583baaa236775108b360c650e79770a695e16914b1201f290c1826065c",
                fname=file_name,
                path=self.data_folder,
                progressbar=True,
            )

            if self.verbose:
                print()

        # Read HGCA and select source

        hgca_table = Table.read(fits_file)
        hgca_sources = hgca_table["hip_id"]

        hgca_match = hgca_table[hgca_sources == self.hip_id]
        n_match = len(hgca_match)

        if len(hgca_match) == 0:
            raise ValueError(
                f"HIP {self.hip_id} is not present in the "
                "Hipparcos-Gaia Catalog of Accelerations."
            )

        if n_match > 1:
            raise RuntimeError(f"Found {n_match} HGCA entries for HIP {self.hip_id}.")

        row = hgca_match[0]

        # Proper-motion vectors (in mas/yr)

        self.hgca_pm_hip = np.array(
            [
                row["pmra_hip"],
                row["pmdec_hip"],
            ],
        )

        self.hgca_pm_gaia = np.array(
            [
                row["pmra_gaia"],
                row["pmdec_gaia"],
            ],
        )

        self.hgca_pm_hg = np.array(
            [
                row["pmra_hg"],
                row["pmdec_hg"],
            ],
        )

        # Construct the 2 x 2 covariance matrices

        self.hgca_cov_hip = proper_motion_cov(
            pmra_error=row["pmra_hip_error"],
            pmdec_error=row["pmdec_hip_error"],
            correlation=row["pmra_pmdec_hip"],
        )

        self.hgca_cov_gaia = proper_motion_cov(
            pmra_error=row["pmra_gaia_error"],
            pmdec_error=row["pmdec_gaia_error"],
            correlation=row["pmra_pmdec_gaia"],
        )

        self.hgca_cov_hg = proper_motion_cov(
            pmra_error=row["pmra_hg_error"],
            pmdec_error=row["pmdec_hg_error"],
            correlation=row["pmra_pmdec_hg"],
        )

        # Gaia minus Hipparcos-Gaia proper-motion anomaly (mas/yr)
        self.hgca_dpm_gaia_hg = self.hgca_pm_gaia - self.hgca_pm_hg
        self.hgca_cov_dpm_gaia_hg = self.hgca_cov_gaia + self.hgca_cov_hg

        # The Gaia and Hipparcos central epochs differ
        # slightly between RA and Dec in the HGCA.

        self.hgca_epoch_hip = np.array(
            [
                row["epoch_ra_hip"],
                row["epoch_dec_hip"],
            ],
        )

        self.hgca_epoch_gaia = np.array(
            [
                row["epoch_ra_gaia"],
                row["epoch_dec_gaia"],
            ],
        )

        # HGCA chi-square

        self.hgca_chisq = row["chisq"]

        # Store the Hipparcos proper motion. This will overwrite
        # the proper motion attribute stored by query_data().

        self.pm_ra = row["pmra_hip"]
        self.pm_dec = row["pmdec_hip"]
        self.pm_ra_error = row["pmra_hip_error"]
        self.pm_dec_error = row["pmdec_hip_error"]

        # Conversion factor between angular proper motion and
        # tangential velocity. An object with a proper motion
        # of 1 arcsec/yr at a distance of 1 pc moves tangentially
        # by 1 au each year. Therefore,
        #
        #     v_tan (km/s) = (1 au / yr) × μ / ϖ,
        #
        # where μ and ϖ are expressed in the same angular units
        # (e.g. both in mas). The quantity 1 au/yr equals
        # approximately 4.74047 km/s.
        # Use the parallax from HGCA.

        auyear_kms = (c.au / u.year).to("km/s").value
        self.hgca_dv_gaia_hg = auyear_kms * self.hgca_dpm_gaia_hg / self.parallax

        if self.verbose:
            print(f"Hipparcos ID: {row['hip_id']}")

            if self.gaia_dr3_id is not None:
                print(f"Gaia DR3 ID: {self.gaia_dr3_id}")

            print(f"HGCA chi-square: {self.hgca_chisq:.2f}")

            # Proper-motion uncertainties from the covariance matrices

            pm_hip_error = np.sqrt(np.diag(self.hgca_cov_hip))
            pm_gaia_error = np.sqrt(np.diag(self.hgca_cov_gaia))
            pm_hg_error = np.sqrt(np.diag(self.hgca_cov_hg))

            # Assuming the Gaia and HG proper-motion measurements are independent
            self.hgca_cov_dpm_gaia_hg = self.hgca_cov_gaia + self.hgca_cov_hg

            # Uncertainties of the RA and Dec components (mas/yr)
            self.hgca_dpm_gaia_hg_error = np.sqrt(np.diag(self.hgca_cov_dpm_gaia_hg))

            # Covariance of the tangential velocity anomaly (km/s)^2
            self.hgca_cov_dv_gaia_hg = (
                auyear_kms / self.parallax
            ) ** 2 * self.hgca_cov_dpm_gaia_hg

            # 1-sigma uncertainties
            dv_error = np.sqrt(np.diag(self.hgca_cov_dv_gaia_hg))

            print(
                "\nHipparcos proper motion:"
                f"\n   PM RA  = {self.hgca_pm_hip[0]:.4f}"
                f" +/- {pm_hip_error[0]:.4f} mas/yr"
                f"\n   PM Dec = {self.hgca_pm_hip[1]:.4f}"
                f" +/- {pm_hip_error[1]:.4f} mas/yr"
            )

            print(
                "\nGaia proper motion:"
                f"\n   PM RA  = {self.hgca_pm_gaia[0]:.4f}"
                f" +/- {pm_gaia_error[0]:.4f} mas/yr"
                f"\n   PM Dec = {self.hgca_pm_gaia[1]:.4f}"
                f" +/- {pm_gaia_error[1]:.4f} mas/yr"
            )

            print(
                "\nHipparcos-Gaia proper motion:"
                f"\n   PM RA  = {self.hgca_pm_hg[0]:.4f}"
                f" +/- {pm_hg_error[0]:.4f} mas/yr"
                f"\n   PM Dec = {self.hgca_pm_hg[1]:.4f}"
                f" +/- {pm_hg_error[1]:.4f} mas/yr"
            )

            print(
                "\nGaia-HG proper motion anomaly:"
                f"\n   Delta PM RA  = {self.hgca_dpm_gaia_hg[0]:.4f}"
                f" +/- {self.hgca_dpm_gaia_hg_error[0]:.4f} mas/yr"
                f"\n   Delta PM Dec = {self.hgca_dpm_gaia_hg[1]:.4f}"
                f" +/- {self.hgca_dpm_gaia_hg_error[1]:.4f} mas/yr"
            )

            print(
                "\nTangential velocity anomaly:"
                f"\n   Delta v RA   = {self.hgca_dv_gaia_hg[0]:.4f}"
                f" +/- {dv_error[0]:.4f} km/s"
                f"\n   Delta v Dec  = {self.hgca_dv_gaia_hg[1]:.4f}"
                f" +/- {dv_error[1]:.4f} km/s"
            )

            print("\nReference: Brandt T. D. (2021), ApJS, 254, 42")

        return hgca_match.to_pandas()
