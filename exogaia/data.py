"""
Module for Gaia epoch astrometry data.
"""

import warnings

from numbers import Real
from pathlib import Path

import h5py
import healpy
import numpy as np
import pandas as pd
import pooch

from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from astroquery.gaia import Gaia
from beartype import beartype, typing
from scipy.interpolate import RegularGridInterpolator

from exogaia.core import ExoGaia
from exogaia.models import KeplerModel, StarModel
from exogaia.planets import OccurrenceRate

Gaia.ROW_LIMIT = -1


class EpochAstrometry(ExoGaia):
    """
    Class for reading, querying, and simulating Gaia
    epoch astrometry data.
    """

    @beartype
    def __init__(
        self,
        primary_mass: typing.Optional[typing.Tuple[Real, Real]] = None,
        gaia_release: str = "DR4",
    ) -> None:
        """
        Parameters
        ----------
        primary_mass : tuple(float, float), None
            Primary mass and uncertainty (Msun). The primary mass is
            not stored if the argument is set to ``None``. Not all
            functionalities of ``exogaia`` can be used in that case.
        gaia_release : str
            Gaia release (DR3, DR4, DR5) of the epoch astrometry.
            Simulating data is possible for all releases, but querying
            data will only be possible for the upcoming DR4 and DR5.

        Returns
        -------
        NoneType
            None
        """

        self.print_section("Epoch astrometry")

        self.gaia_release = gaia_release
        self.primary_mass = primary_mass
        self.data_table = None

        self.source_id = None
        self.ra = None
        self.dec = None
        self.parallax = None
        self.pmra = None
        self.pmdec = None
        self.phot_g_mean_mag = None
        self.u0_norm = None
        self.sim_data = False

        # Start of the Gaia mission
        self.time_start = Time("2014-07-25 10:30:00", scale="utc")

        if self.gaia_release == "DR3":
            self.ref_epoch = Time("2016.0", format="jyear", scale="tcb")
            self.time_end = Time("2017-05-28 08:44:00", scale="utc")

        elif self.gaia_release == "DR4":
            self.ref_epoch = Time("2017.5", format="jyear", scale="tcb")
            self.time_end = Time("2020-01-20 22:00:00", scale="utc")

        elif self.gaia_release == "DR5":
            self.ref_epoch = Time("2020.0", format="jyear", scale="tcb")
            self.time_end = Time("2025-01-15 00:00:00", scale="utc")

        print(f"Gaia release: {self.gaia_release}")
        print(f"Reference epoch: {self.ref_epoch}")

        if primary_mass is not None:
            print(
                f"\nPrimary mass (Msun): {primary_mass[0]:.2f} "
                f"+/- {primary_mass[1]:.2f}"
            )

    def __repr__(self):
        """
        String representation of the data table of the class.

        Returns
        -------
        str
            Header of the data table if available.
        """

        if self.data_table is None:
            data_str = "Data table is empty"
        else:
            data_str = self.data_table.head().to_string()

        return data_str

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

        self.print_section("Read data file")

        self.data_table = pd.read_csv(data_file)

        print(f"Data file: {data_file}")
        print(f"Data shape: {self.data_table.shape}")

        if "relative_time_year" not in self.data_table:
            self.data_table["relative_time_year"] = (
                self.data_table["obs_time_tcb"] - self.ref_epoch.tcb.jyear
            )

        if "relative_time_day" not in self.data_table:
            self.data_table["relative_time_year"] = self.data_table[
                "relative_time_year"
            ] * u.year.to(u.day)

    @beartype
    def simulate_data(
        self,
        mass_2: typing.Optional[Real] = None,
        sma: typing.Optional[Real] = None,
        ecc: typing.Optional[Real] = None,
        inc: typing.Optional[Real] = None,
        aop: typing.Optional[Real] = None,
        pan: typing.Optional[Real] = None,
        tau: typing.Optional[Real] = None,
        sigma_per_ccd: typing.Optional[Real] = None,
        csv_out: typing.Optional[str] = None,
        ra: typing.Optional[Real] = None,
        dec: typing.Optional[Real] = None,
        parallax: typing.Optional[Real] = None,
        pmra: typing.Optional[Real] = None,
        pmdec: typing.Optional[Real] = None,
        phot_g_mean_mag: typing.Optional[Real] = None,
        reject_fraction: typing.Optional[Real] = None,
        occ_rate: typing.Union[typing.Optional[str], OccurrenceRate] = None,
        verbose: bool = True,
    ) -> typing.List[Real]:
        """
        Method to simulate the epoch astrometry for a
        single star or binary system. The primary mass
        from the ``primary_mass`` attribute is used.

        Parameters
        ----------
        mass_2 : float, None
            Companion mass (Mjup). A single star is simulated
            by setting the argument of ``mass_2`` to ``None``.
        sma : float, None
            Semi-major axis (au). This should be the relative
            semi-major axis of the primary and secondary,
            so sma = a1+a2. A random value is drawn from a
            log-uniform distribution between 0.1 and 30 au
            if the argument is set to ``None``.
        ecc : float, None
            Eccentricity. A random value is drawn from a beta
            distribution as defined in Kipping 2013 if the
            argument is set to ``None``.
        inc : float, None
            Inclination (rad). A random value is drawn from a
            sin(theta) distribution, with theta between 0.0 and pi,
            if the arguments is set to ``None``.
        aop : float, None
            Argument of periastron (rad). A random value is drawn from
            a uniform distribution between 0.0 and 2pi if the arguments
            is set to ``None``.
        pan : float, None
            Position angle of the ascending node (rad). A random value
            is drawn from a uniform distribution between 0.0 and 2pi
            if the arguments is set to ``None``.
        tau : float, None
            Time of periastron as fraction of the orbital period,
            relative to the reference epoch of ``gaia_release``.
            A random value is drawn from a uniform distribution
            between 0.0 and 1.0 if the arguments is set to
            ``None``.
        sigma_per_ccd : float, None
            The AL uncertainty per CCD (mas). Setting the argument
            to ``None`` will adopt the G magnitude dependent
            uncertainty from Holl et al. (2023).
        csv_out : str
            Output CSV file to store the simulated epoch astrometry.
        ra : float, None
            RA coordinate (deg) at the reference epoch of the
            ``gaia_release``, relative to the RA at the
            reference epoch. Typically, this value can be
            set to zero.,
        dec : float, None
            Dec coordinate (deg) at the reference epoch of the
            ``gaia_release``, relative to the Dec at the
            reference epoch. Typically, this value can be
            set to zero.,
        parallax : float
            Parallax (mas).
        pmra : float, None
            Proper motion in RA (mas/yr).
        pmdec : float, None
            Proper motion in Dec (mas/yr).
        phot_g_mean_mag : float, None
            Gaia G-band magnitude.
        reject_fraction : float, None
            The fraction of data to reject. About 10 percent of
            FOV transits has an issue (see Lindegren et al. 2021).
            The default is ``None``, in which case no data
            is rejected.
        occ_rate : str, OccurrenceRate, None
            Occurrence-rate prescription ("fls_fulton2021",
            "gpi_nielsen2019") for sampling the companion
            mass, ``mass_2``, and the semi-major axis,
            ``sma``. Or, an ``OccurrenceRate`` object, which
            also allows for a manually provided occurrence
            rate function. The arguments of ``mass_2`` and
            ``sma`` are ignored if the argument of
            ``occ_rate`` is not set to ``None``.
        verbose : bool
            Print some information.

        Returns
        -------
        list(float)
            List with model parameters, as RA (deg), Dec (deg),
            parallax (mas), RA proper motion (mas/yr), Dec proper
            motion (mas/yr). For a binary system, followed by
            semi-major axis (au), eccentricity, inclination (rad),
            argument of periastron (rad), position angle of
            ascending node (rad), relative time of periastron,
            primary mass (Msun), companion mass (Mjup).
        """

        # For simulated data, UWE == RUWE when fitting models

        self.sim_data = True

        # Adopt stellar parameters from class attributes

        if (
            ra is not None
            and dec is not None
            and parallax is not None
            and pmra is not None
            and pmdec is not None
            and phot_g_mean_mag is not None
        ):
            self.ra = ra
            self.dec = dec
            self.parallax = parallax
            self.pmra = pmra
            self.pmdec = pmdec
            self.phot_g_mean_mag = phot_g_mean_mag

        elif (
            self.ra is None
            or self.dec is None
            or self.parallax is None
            or self.pmra is None
            or self.pmdec is None
            or self.phot_g_mean_mag is None
        ):
            raise ValueError(
                "Please either provide the stellar parameters "
                "as arguments of 'ra', 'dec', 'parallax', "
                "'pmra', 'pmdec', and 'phot_g_mean_mag', or "
                "use the 'query_source()' method to adopt "
                "the values from Gaia."
            )

        # Simulating single star or binary system?

        mass_1 = self.primary_mass[0]

        if mass_2 is None and occ_rate is None:
            binary = False

            mass_2 = 0.0
            sma = 0.0
            ecc = 0.0
            inc = 0.0
            aop = 0.0
            pan = 0.0
            tau = 0.0

        else:
            binary = True

            # Sample random orbit parameters if not provided

            rng = np.random.default_rng()

            if sma is None:
                if occ_rate is None:
                    # Sample log-uniform between 0.1 and 30 au

                    log_sma = rng.uniform(np.log10(0.1), np.log10(30.0))
                    sma = 10.0**log_sma

                else:
                    if isinstance(occ_rate, str):
                        occ_rate = OccurrenceRate(
                            primary_mass=self.primary_mass[0], occ_rate=occ_rate
                        )

                    sma, mass_2 = occ_rate.sample_planets(allow_reject=False)
                    sma, mass_2 = sma[0], mass_2[0]

            if ecc is None:
                # Beta distribution (Kipping 2013)
                ecc = rng.beta(a=0.867, b=3.03)

            if inc is None:
                inc = np.arccos(rng.uniform(-1.0, 1.0))

            if aop is None:
                aop = rng.uniform(0.0, 2.0 * np.pi)

            if pan is None:
                pan = rng.uniform(0.0, 2.0 * np.pi)

            if tau is None:
                tau = rng.uniform(0.0, 1.0)

        if verbose:
            self.print_section("Simulate data")

            if binary:
                print("System type: binary")
            else:
                print("System type: single")

        # Scan angles, parallax factors, and observation
        # times have been retrieved with GOST (see
        # https://gaia.esac.esa.int/gost/index.jsp) for the
        # RA/DEC of all HEALPix indices with NSIDE=64
        # (i.e. 49152 indices). The data for each index is
        # stored in a separate group of the HDF5 file.

        data_folder = Path.home() / ".exogaia"

        if not data_folder.exists():
            print(f"Creating folder: {str(data_folder)}")
            data_folder.mkdir(parents=True, exist_ok=False)

        file_name = "healpix_data.hdf5"
        healpix_file = data_folder / file_name
        url = "https://home.strw.leidenuniv.nl/~stolker/exogaia/healpix_data.hdf5"

        if not healpix_file.exists():
            print()

            pooch.retrieve(
                url=url,
                known_hash="54c786f345b891849f43134c9587b03acb9626d935594c2f62469963981ba798",
                fname=file_name,
                path=data_folder,
                progressbar=True,
            )

        # Read the scan data for a given sky position by
        # selecting the nearest HEALPix position

        nside = 64

        # theta and phi are in degrees when lonlat=True
        pix_num = healpy.ang2pix(nside=nside, theta=self.ra, phi=self.dec, lonlat=True)

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
            rand_gen = np.random.default_rng()
            rand_unif = rand_gen.uniform(low=0.0, high=1.0, size=len(table_select))
            table_select = table_select[rand_unif > reject_fraction]

        psi, plx_factor, obs_time_tcb = (
            table_select["scanAngle[rad]"],
            table_select["parallaxFactorAlongScan"],
            table_select["ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"],
        )

        t_ast = Time(obs_time_tcb, format="jd", scale="tcb") - self.ref_epoch.tcb
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
                self.phot_g_mean_mag,
                df["Gaia G mag"],
                df["CCD AL scan uncertainty (mas)"],
            )

        sigma_per_transit = sigma_per_ccd / np.sqrt(n_ccd_avg)
        if verbose:
            print(f"\nAL scan uncertainty (per CCD) = {1e3*sigma_per_ccd:.2f} uas")
            print(
                f"AL scan uncertainty (per transit) = {1e3*sigma_per_transit:.2f} uas"
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

        if verbose:
            print("\nStellar parameters:")
            print(f"   - RA (deg) = {self.ra:.2f}")
            print(f"   - Dec (deg) = {self.dec:.2f}")
            print(f"   - Parallax (mas) = {self.parallax:.2f}")
            print(f"   - Proper motion in RA (mas/yr) = {self.pmra:.2f}")
            print(f"   - Proper motion in Dec (mas/yr) = {self.pmdec:.2f}")
            print(f"   - G-band magnitude = {self.phot_g_mean_mag:.2f}")

        # Setting the RA/Dec offsets, which are relative to the
        # RA/Dec coordinates at the gaia_release epoch, to zero

        ra_offset = 0.0  # (mas)
        dec_offset = 0.0  # (mas)

        if binary:
            # Orbital period (days)
            # Use semi-major axis in au and masses in Msun
            mass_2_sun = ((mass_2 * u.M_jup).to(u.M_sun)).value
            period = np.sqrt(sma**3 / (mass_1 + mass_2_sun)) * 365.25

            # Semi-major axis of the photocenter (=primary) (au)
            sma_0 = sma * mass_2_sun / (mass_1 + mass_2_sun)
            sma_0 *= self.parallax  # (mas)

            if verbose:
                print("\nOrbit parameters:")
                print(f"   - Primary mass (Msun) = {mass_1:.2f}")
                print(f"   - Companion mass (Mjup) = {mass_2:.2f}")
                print(f"   - Relative semi-major axis (au) = {sma:.2f}")
                print(f"   - Eccentricity = {ecc:.2f}")
                print(f"   - Inclination (deg) = {np.degrees(inc):.2f}")
                print(f"   - Argument of periastron (deg) = {np.degrees(aop):.2f}")
                print(f"   - PA of ascending node (deg) = {np.degrees(pan):.2f}")
                print(f"   - Relative time of periastron = {tau:.2f}")
                print(f"   - Period (days) = {period:.2f}")
                print(f"   - Semi-major axis of photocenter (mas) = {sma_0:.2f}")

            model_param = [
                ra_offset,
                dec_offset,
                self.parallax,
                self.pmra,
                self.pmdec,
                period,
                ecc,
                tau,
                sma_0,
                inc,
                aop,
                pan,
            ]

            # self is the current EpochAstrometry object
            kepler_model = KeplerModel(epoch_astrometry=self, verbose=False)
            cen_pos = kepler_model.calc_1d_model(model_param=model_param)

        else:
            model_param = [
                ra_offset,
                dec_offset,
                self.parallax,
                self.pmra,
                self.pmdec,
            ]

            star_model = StarModel(epoch_astrometry=self)
            cen_pos = star_model.calc_1d_model(model_param=model_param)

        rng = np.random.default_rng()
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

        self.print_section("Retrieve Gaia non-single star tables")

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
                f"gaia{gaia_release}_{table_item}.ecsv",
                format="ascii.ecsv",
                overwrite=True,
            )

    @beartype
    def query_source(
        self,
        source_id: typing.Optional[typing.Union[int, str]] = None,
        gaia_release: str = "DR3",
    ) -> typing.List[Real]:
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
        list(float)
            List with retrieved stellar parameters, as RA, Dec, parallax,
            RA proper motion, Dec proper motion, G-band magnitude.
        """

        self.print_section("Querying source")

        self.source_id = source_id

        if gaia_release in ["DR4", "DR5"]:
            raise ValueError(
                "The 'query_source' method supports "
                "currently only gaia_release='DR3'."
            )

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

        gaia_job = Gaia.launch_job_async(gaia_query, dump_to_file=False, verbose=False)

        gaia_result = gaia_job.get_results()[0]

        ra = float(gaia_result["ra"])
        ra_error = float(gaia_result["ra_error"])
        dec = float(gaia_result["dec"])
        dec_error = float(gaia_result["dec_error"])
        parallax = float(gaia_result["parallax"])
        parallax_error = float(gaia_result["parallax_error"])
        pmra = float(gaia_result["pmra"])
        pmra_error = float(gaia_result["pmra_error"])
        pmdec = float(gaia_result["pmdec"])
        pmdec_error = float(gaia_result["pmdec_error"])
        phot_g_mean_mag = float(gaia_result["phot_g_mean_mag"])

        # Effective wavenumber (i.e. pseudocolor) of the source
        # used in the astrometric solution (um-1)
        nu_eff = float(gaia_result["nu_eff_used_in_astrometry"])

        print(f"\nRA = {ra:.3f} deg +/- {ra_error:.3f} mas")
        print(f"Dec = {dec:.3f} deg +/- {dec_error:.3f} mas")
        print(f"Parallax = {parallax:.3f} +/- {parallax_error:.3f} mas")
        print(f"Proper motion in RA = {pmra:.3f} +/- {pmra_error:.3f} mas/yr")
        print(f"Proper motion in Dec = {pmdec:.3f} +/- {pmdec_error:.3f} mas/yr")
        print(f"G-band magnitude = {phot_g_mean_mag:.3f}")
        print(f"Pseudocolor = {nu_eff:.2f} um-1")

        self.ra = ra
        self.dec = dec
        self.parallax = parallax
        self.pmra = pmra
        self.pmdec = pmdec
        self.phot_g_mean_mag = phot_g_mean_mag

        # Gaia (E)DR3: Re-normalised Unit Weight Error (RUWE)
        # tables of u0(g,c) by L. Lindegren (2023 Sep 13)
        # https://www.cosmos.esa.int/web/gaia/dr3-auxiliary-data

        file_folder = Path(__file__).resolve().parent.parent
        data_file = file_folder / "data/table_u0_g_c_p5.txt"

        norm_g_mag, norm_nu_eff, norm_u0 = np.loadtxt(
            data_file, skiprows=1, delimiter=",", unpack=True
        )

        g_vals = np.unique(norm_g_mag)
        c_vals = np.unique(norm_nu_eff)

        u0_grid = np.reshape(norm_u0, (g_vals.size, c_vals.size))

        # Should look the same as plot_u0_g_c_p5.pdf
        # plt.pcolormesh(g_vals, c_vals, u0_grid.T, cmap='rainbow')
        # plt.colorbar()
        # plt.show()

        u0_interp = RegularGridInterpolator(
            (g_vals, c_vals), u0_grid, method="linear", bounds_error=True
        )

        self.u0_norm = u0_interp((self.phot_g_mean_mag, nu_eff))

        n_param = 5

        uwe = np.sqrt(
            gaia_result["astrometric_chi2_al"]
            / (gaia_result["astrometric_n_good_obs_al"] - n_param)
        )

        print(f"\nUWE = {uwe:.2f}")

        if "ruwe" in gaia_result.columns:
            if not np.ma.is_masked(gaia_result["ruwe"]):
                ruwe = gaia_result["ruwe"]
                print(f"RUWE = {ruwe:.2f}")

                if not np.isclose(self.u0_norm, uwe / ruwe, rtol=1e-2, atol=0.0):
                    warnings.warn(
                        f"The renormalization value is {self.u0_norm:.6f} "
                        f"whereas the ratio of uwe/ruwe is {uwe/ruwe:.6f}."
                    )

        if "astrometric_excess_noise" in gaia_result.columns:
            if not np.ma.is_masked(gaia_result["astrometric_excess_noise"]):
                aen = gaia_result["astrometric_excess_noise"]
                aen_sig = gaia_result["astrometric_excess_noise_sig"]
                print(
                    f"Astrometric excess noise (mas) = {aen:.4f} +/- {aen/aen_sig:.4f}"
                )

        if "non_single_star" in gaia_result.columns:
            if not np.ma.is_masked(gaia_result["non_single_star"]):
                print(f"Non single star = {gaia_result['non_single_star']}")

        return [ra, dec, parallax, pmra, pmdec, phot_g_mean_mag]

    @beartype
    def retrieve_data(
        self,
        source_id: typing.Optional[typing.Union[int, str]] = None,
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

        self.print_section("Retrieving epoch astrometry")

        self.source_id = source_id

        if self.gaia_release in ["DR4", "DR5"]:
            raise ValueError(
                "The 'retrieve_data' method will only support "
                "the future DR4 and DR5 data releases."
            )

        print(f"Gaia release: {self.gaia_release}")
        print(f"Source ID: {self.source_id}")

        gaia_tables = [
            "epoch_astrometry",
            "bright_source_astrometry",
            "nss_acceleration_astro",
            "nss_two_body_orbit",
        ]

        for table_item in gaia_tables:
            print(f"\nTable: gaia{self.gaia_release.lower()}.{table_item}")

            # Query Gaia source ID in NSS tables for selected Gaia source ID

            gaia_query = f"""
            SELECT *
            FROM gaia{self.gaia_release.lower()}.{table_item}
            WHERE source_id = {self.source_id}
            """

            # Launch the Gaia job and get the results

            gaia_job = Gaia.launch_job_async(
                gaia_query, dump_to_file=False, verbose=False
            )

            gaia_result = gaia_job.get_results()

            if len(gaia_result) > 0:
                print("\nTable parameters:")
                for param_item in gaia_result[0].columns:
                    print(f"   - {param_item} = {gaia_result[0][param_item]}")

            else:
                print(f"\nSource not found in {table_item}")

            if exclude_outliers:
                pass

            if combine_ccds:
                pass

    @beartype
    def gaia_bh3(
        self,
        exclude_outliers: bool = True,
        combine_ccds: bool = False,
    ) -> None:
        """
        Method for storing the Gaia DR3 epoch astrometry of
        the black hole Gaia BH3 in the ``data_table``. The
        data file can be found `here <https://github.com/
        tomasstolker/exogaia/blob/main/data/
        gaiabh3_epochast.dat>`_.

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

        _ = self.query_source(source_id=self.source_id, gaia_release="DR3")

        self.print_section("Gaia BH3 epoch data")

        file_folder = Path(__file__).resolve().parent.parent
        data_file = file_folder / "data/gaiabh3_epochast.dat"

        if self.gaia_release == "DR3":
            self.ref_epoch = Time("2016.0", format="jyear", scale="tcb")
            self.time_end = Time("2017-05-28 08:44:00", scale="utc")

        print(f"Gaia release: {self.gaia_release}")
        print(f"Reference epoch: {self.ref_epoch}")
        print(f"Source ID: {self.source_id}")

        if self.primary_mass is None:
            self.primary_mass = (0.76, 0.05)  # (Msun)

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

        if "relative_time_year" not in self.data_table:
            self.data_table["relative_time_year"] = (
                self.data_table["obs_time_tcb"] - self.ref_epoch.tcb.jyear
            )

        if "relative_time_day" not in self.data_table:
            self.data_table["relative_time_day"] = self.data_table[
                "relative_time_year"
            ] * u.year.to(u.day)

        print(f"\nData file: {data_file}")
        print(f"Data shape: {self.data_table.shape}")
