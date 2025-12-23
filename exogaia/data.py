"""
Module for Gaia epoch astrometry data.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import healpy
import numpy as np
import pandas as pd
import pooch

from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from astroquery.gaia import Gaia
from typeguard import typechecked

from exogaia.core import ExoGaia
from exogaia.models import BinaryModel, StarModel

Gaia.ROW_LIMIT = -1


class EpochAstrometry(ExoGaia):
    """
    Class for reading, querying, and simulating Gaia
    epoch astrometry data.
    """

    @typechecked
    def __init__(
        self,
        primary_mass: Tuple[float, float] = None,
        gaia_release: str = "DR3",
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

        self.ra = None
        self.dec = None
        self.parallax = None
        self.pmra = None
        self.pmdec = None
        self.phot_g_mean_mag = None

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
                f"\nPrimary mass (Msun): {primary_mass[0]:.2f} +/- {primary_mass[1]:.2f}"
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

    @typechecked
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
                self.data_table["obs_time_tcb"] - self.ref_epoch.jyear
            )

        if "relative_time_day" not in self.data_table:
            self.data_table["relative_time_year"] = self.data_table[
                "relative_time_year"
            ] * u.year.to(u.day)

    @typechecked
    def simulate_data(
        self,
        mass_1: Optional[float] = None,
        mass_2: Optional[float] = None,
        sma: Optional[float] = None,
        ecc: Optional[float] = None,
        inc: Optional[float] = None,
        aop: Optional[float] = None,
        pan: Optional[float] = None,
        tau: Optional[float] = None,
        sigma_per_ccd: Optional[float] = None,
        csv_out: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        parallax: Optional[float] = None,
        pmra: Optional[float] = None,
        pmdec: Optional[float] = None,
        phot_g_mean_mag: Optional[float] = None,
    ) -> List[float]:
        """
        Method to simulate the epoch astrometry for a single star
        or binary system. The HEALPix data has been adopted from
        ``gaiamock`` by El-Badry et al. (2025).

        MIT License

        Copyright (c) 2024 kareemelbadry

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        Parameters
        ----------
        mass_1 : float, None
            Primary mass (Msun). A single star is simulated by setting
            the argument to ``None``.
        mass_2 : float, None
            Secondary mass (Msun). A single star is simulated by setting
            the argument to ``None``.
        sma : float, None
            Semi-major axis (au). A single star is simulated by setting
            the argument to ``None``.
        ecc : float, None
            Eccentricity. A single star is simulated by setting
            the argument to ``None``.
        inc : float, None
            Inclination (rad). A single star is simulated by setting
            the argument to ``None``.
        aop : float, None
            Argument of periastron (rad). A single star is simulated by
            setting the argument to ``None``.
        pan : float, None
            Position angle of the ascending nodes (rad). A single star
            is simulated by setting the argument to ``None``.
        tau : float, None
            Periastron time, relative to the reference epoch
            of ``gaia_release``. A single star is simulated by setting
        sigma_per_ccd : float, None
            The AL uncertainty per CCD (mas). Setting the argument
            to ``None`` will adopt the G magnitude dependent
            uncertainty from Holl et al. (2023).
        csv_out : str
            Output CSV file to store the simulated epoch astrometry.
        ra : float, None
            RA coordinate (deg) at the reference epoch of
            the ``gaia_release``.
        dec : float, None
            Dec coordinate (deg) at the reference epoch of
            the ``gaia_release``.
        parallax : float
            Parallax (mas).
        pmra : float, None
            Proper motion in RA (mas/yr).
        pmdec : float, None
            Proper motion in Dec (mas/yr).
        phot_g_mean_mag : float, None
            Gaia G-band magnitude.

        Returns
        -------
        list(float)
            List with model parameters, as RA (deg), Dec (deg),
            parallax (mas), RA proper motion (mas/yr), Dec proper
            motion (mas/yr). For a binary system, followed by
            semi-major axis (au), eccentricity, inclination (rad),
            argument of periastron (rad), position angle of
            ascending node (rad), relative time of periastron,
            primary mass (Msun), secondary mass (Msun).
        """

        self.print_section("Simulate data")

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

        # Simulating single star or binary system?

        if (
            mass_1 is None
            or mass_2 is None
            or sma is None
            or ecc is None
            or inc is None
            or aop is None
            or pan is None
            or tau is None
        ):
            binary = False

            mass_1 = 0.0
            mass_2 = 0.0
            sma = 0.0
            ecc = 0.0
            inc = 0.0
            aop = 0.0
            pan = 0.0
            tau = 0.0

            print("System type: single")

        else:
            binary = True
            print("System type: binary")

        # HEALPix data

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
                known_hash="a470eb369d50cb14587e9d0c4450d5e4264e93b6c81f909d157dc0e3e9fbab43",
                fname=file_name,
                path=data_folder,
                progressbar=True,
            )

        # Find the scan times and angles for a given sky position
        # by searching for the nearest position in a set of pre-
        # downloaded 49152 sky positions (healpix level 64)

        healp_num = healpy.ang2pix(64, self.ra, self.dec, lonlat=True)

        with h5py.File(healpix_file, "r") as hdf5_file:
            healp_table = Table(hdf5_file[f"healpix_64_{healp_num}"][:])

        obs_time_full = healp_table[
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

        table_select = healp_table[time_select]

        # Reject 10% of the data
        # See Sect. 3.3 in El-Badry et al. (2024)
        # rand_gen = np.random.default_rng()
        # rand_unif = rand_gen.uniform(low=0.0, high=1.0, size=len(table_select))
        # table_select = table_select[rand_unif > 0.1]

        psi, plx_factor, obs_time_tcb = (
            table_select["scanAngle[rad]"],
            table_select["parallaxFactorAlongScan"],
            table_select["ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"],
        )

        t_ast_day = obs_time_tcb - self.ref_epoch.jd
        t_ast_yr = t_ast_day / 365.25

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
        print(f"\nAL scan uncertainty (per CCD) = {1e3*sigma_per_ccd:.2f} uas")
        print(f"AL scan uncertainty (per transit) = {1e3*sigma_per_transit:.2f} uas")

        sim_astrom = {
            "obs_time_tcb": t_ast_yr + self.ref_epoch.jyear,
            "relative_time_year": t_ast_yr,
            "relative_time_day": t_ast_yr * u.year.to(u.day),
            "scan_pos_angle": psi,
            "parallax_factor_al": plx_factor,
        }

        self.data_table = pd.DataFrame(sim_astrom)

        print("\nStellar parameters:")
        print(f"   - RA (deg) = {self.ra:.2f}")
        print(f"   - Dec (deg) = {self.dec:.2f}")
        print(f"   - Parallax (mas) = {self.parallax:.2f}")
        print(f"   - Proper motion in RA (mas/yr) = {self.pmra:.2f}")
        print(f"   - Proper motion in Dec (mas/yr) = {self.pmdec:.2f}")
        print(f"   - G-band magnitude = {self.phot_g_mean_mag:.2f}")

        if binary:
            print("\nOrbit parameters:")
            print(f"   - Primary mass (Msun) = {mass_1:.2f}")
            print(f"   - Secondary mass (Msun) = {mass_2:.2f}")
            print(f"   - Semi-major axis (au) = {sma:.2f}")
            print(f"   - Eccentricity = {ecc:.2f}")
            print(f"   - Inclination (deg) = {np.degrees(inc):.2f}")
            print(f"   - Argument of periastron (deg) = {np.degrees(aop):.2f}")
            print(f"   - PA of ascending node (deg) = {np.degrees(pan):.2f}")
            print(f"   - Relative time of periastron = {tau:.2f}")

            model_param = [
                self.ra,
                self.dec,
                self.parallax,
                self.pmra,
                self.pmdec,
                sma,
                ecc,
                inc,
                aop,
                pan,
                tau,
                mass_1,
                mass_2,
            ]

            # self is the current EpochAstrometry object
            bin_model = BinaryModel(epoch_astrometry=self, verbose=False)
            cen_pos = bin_model.calc_model(model_param=model_param)

        else:
            model_param = [
                self.ra,
                self.dec,
                self.parallax,
                self.pmra,
                self.pmdec,
            ]

            star_model = StarModel(epoch_astrometry=self)
            _, _, cen_pos = star_model.calc_model(model_param=model_param)

        rng = np.random.default_rng()
        cen_pos += rng.normal(loc=0.0, scale=sigma_per_transit, size=len(psi))

        self.data_table["centroid_pos_al"] = cen_pos

        self.data_table["centroid_pos_error_al"] = np.full(
            cen_pos.size, sigma_per_transit
        )

        if csv_out is not None:
            self.data_table.to_csv(csv_out, index=False)

        return model_param

    @typechecked
    def get_nss_tables(self) -> None:
        """
        Method for downloading and storing the Gaia non-single star
        (NSS) tables. The output will be stored in `ECSV files
        <https://docs.astropy.org/en/stable/io/ascii/ecsv.html>`_.
        Currently, only ``gaia_release="DR3"`` is supported.

        Returns
        -------
        NoneType
            None
        """

        self.print_section("Retrieve Gaia non-single star tables")

        if self.gaia_release in ["DR4", "DR5"]:
            raise ValueError(
                "The get_nss_table() only supports Gaia DR3. "
                "Please set the 'gaia_release' argument to 'DR3'."
            )

        # List all Gaia tables
        # for table_item in Gaia.load_tables(only_names=True):
        #     print (table_item.get_qualified_name())

        if self.gaia_release == "DR4":
            # Gaia DR3 NSS tables
            gaia_tables = [
                "gaiadr3.nss_acceleration_astro",
                "gaiadr3.nss_non_linear_spectro",
                "gaiadr3.nss_two_body_orbit",
                "gaiadr3.nss_vim_fl",
            ]

        elif self.gaia_release == "DR4":
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
                f"gaia{self.gaia_release}_{table_item}.ecsv",
                format="ascii.ecsv",
                overwrite=True,
            )

    @typechecked
    def query_source(
        self, source_id: Optional[Union[int, str]] = None, gaia_release: str = "DR3"
    ) -> List[float]:
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

        if gaia_release in ["DR4", "DR5"]:
            raise ValueError(
                "The 'query_source' method supports "
                "currently only gaia_release='DR3'."
            )

        print(f"Gaia release: {gaia_release}")
        print(f"Source ID: {source_id}\n")

        # Retrieve RA, Dec, parallax, proper motion, and G magnitude

        gaia_query = f"""
        SELECT ra, ra_error, dec, dec_error, parallax, parallax_error,
               pmra, pmra_error, pmdec, pmdec_error, phot_g_mean_mag
        FROM gaia{gaia_release.lower()}.gaia_source
        WHERE source_id = {source_id}
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

        print(f"\nRA: {ra:.3f} deg +/- {ra_error:.3f} mas")
        print(f"Dec: {dec:.3f} deg +/- {dec_error:.3f} mas")
        print(f"Parallax: {parallax:.3f} +/- {parallax_error:.3f} mas")
        print(f"Proper motion in RA: {pmra:.3f} +/- {pmra_error:.3f} mas/yr")
        print(f"Proper motion in Dec: {pmdec:.3f} +/- {pmdec_error:.3f} mas/yr")
        print(f"G-band magnitude: {phot_g_mean_mag:.3f}")

        self.ra = ra
        self.dec = dec
        self.parallax = parallax
        self.pmra = pmra
        self.pmdec = pmdec
        self.phot_g_mean_mag = phot_g_mean_mag

        return [ra, dec, parallax, pmra, pmdec, phot_g_mean_mag]

    @typechecked
    def retrieve_data(self, source_id: Optional[Union[int, str]] = None) -> None:
        """
        Method for retrieving the epoch astrometry for the selected
        Gaia source. This will only be possible for the future DR4
        and DR5 data releases.

        Parameters
        ----------
        source_id : int, str
            Gaia source ID for the selected ``gaia_release`` of the
            class initialization.

        Returns
        -------
        NoneType
            None
        """

        self.query_source(source_id, gaia_release=self.gaia_release)

        self.print_section("Retrieving epoch astrometry")

        # Gaia DR4 epoch astrometry tables
        # gaiadr4.epoch_astrometry
        # gaiadr4.bright_source_astrometry

        if self.gaia_release in ["DR4", "DR5"]:
            raise ValueError(
                "The 'retrieve_data' method will only support "
                "the future DR4 and DR5 data releases."
            )

        print(f"Gaia release: {self.gaia_release}")
        print(f"Source ID: {source_id}")

        for table_item in ["nss_acceleration_astro", "nss_two_body_orbit"]:
            print(f"\nTable: gaia{self.gaia_release.lower()}.{table_item}")

            # Query Gaia source ID in NSS tables for selected Gaia source ID

            gaia_query = f"""
            SELECT *
            FROM gaia{self.gaia_release.lower()}.{table_item}
            WHERE source_id = {source_id}
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

    @typechecked
    def gaia_bh3(self) -> None:
        """
        Method for storing the Gaia DR3 epoch astrometry of
        the black hole Gaia BH3 in the ``data_table``. The
        data file can be found `here <https://github.com/
        tomasstolker/exogaia/blob/main/data/
        gaiabh3_epochast.dat>`_.

        Returns
        -------
        NoneType
            None
        """

        self.print_section("Gaia BH3 epoch data")

        file_folder = Path(__file__).resolve().parent.parent
        data_file = file_folder / "data/gaiabh3_epochast.dat"

        if self.gaia_release == "DR3":
            self.ref_epoch = Time("2016.0", format="jyear", scale="tcb")
            self.time_end = Time("2017-05-28 08:44:00", scale="utc")

        print(f"Gaia release: {self.gaia_release}")
        print(f"Reference epoch: {self.ref_epoch}")
        print("Source ID: 4318465066420528000")

        self.data_table = pd.read_csv(
            data_file, sep=r"\s+", header="infer", comment="#", skip_blank_lines=True
        )

        if "relative_time_year" not in self.data_table:
            self.data_table["relative_time_year"] = (
                self.data_table["obs_time_tcb"] - self.ref_epoch.jyear
            )

        if "relative_time_day" not in self.data_table:
            self.data_table["relative_time_year"] = self.data_table[
                "relative_time_year"
            ] * u.year.to(u.day)

        print(f"\nData file: {data_file}")
        print(f"Data shape: {self.data_table.shape}")
