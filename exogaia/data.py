"""
Module for handling epoch astrometry data.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

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
from exogaia.models import BinaryModel

Gaia.ROW_LIMIT = -1


class EpochAstrometry(ExoGaia):
    """
    Class for handling epoch astrometry data.
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
        data_file : str
            Data file with the Gaia epoch astrometry.
        """

        self.print_section("Epoch astrometry")

        self.gaia_release = gaia_release
        self.primary_mass = primary_mass
        self.data_table = None

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
        print(f"\nPrimary mass (Msun): {primary_mass[0]:.2f} +/- {primary_mass[1]:.2f}")

    def __repr__(self):
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
        Parameters
        ----------
        data_file : str
            Data file with the Gaia epoch astrometry.
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
        ra: float,
        dec: float,
        parallax: float,
        pmra: float,
        pmdec: float,
        mass_1: float,
        mass_2: float,
        sma: float,
        ecc: float,
        inc: float,
        aop: float,
        pan: float,
        tau: float,
        phot_g_mean_mag: float,
        sigma_per_ccd: Optional[float] = None,
        csv_out: Optional[str] = None,
    ) -> None:
        """
        Method to predict the epoch astrometry for a binary
        for a given Gaia data release. The function and
        ``healpix`` data has been adopted from ``gaiamock``
        by El-Badry et al. (2025)

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
        ra : float
            RA coordinate (deg) at the reference epoch of the ``gaia_release``.
        dec : float
            Dec coordinate (deg) at the reference epoch of the ``gaia_release``.
        parallax : float
            Parallax (mas).
        pmra : float
            Proper motion in RA (mas/yr).
        pmdec : float
            Proper motion in Dec (mas/yr).
        mass_1 : float
            Primary mass (Msun).
        mass_2 : float
            Secondary mass (Msun).
        sma : float
            Semi-major axis (au)
        ecc : float
            Eccentricity.
        inc : float
            Inclination (rad).
        aop : float
            Argument of periastron (rad).
        pan : float
            Position angle of the ascending nodes (rad).
        tau : float
            Periastron time, relative to the reference epoch of ``gaia_release``.
        phot_g_mean_mag : float
            Gaia G-band magnitude.
        sigma_per_ccd : float, None
            The AL uncertainty per CCD (mas). Setting the argument
            to ``None`` will adopt the G magnitude dependent
            uncertainty from Holl et al. (2023).
        csv_out : str
            Output CSV file to store the simulated epoch astrometry.
        """

        self.print_section("Simulate data")

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

        healp_num = healpy.ang2pix(64, ra, dec, lonlat=True)

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
                phot_g_mean_mag, df["Gaia G mag"], df["CCD AL scan uncertainty (mas)"]
            )

        sigma_per_transit = sigma_per_ccd / np.sqrt(n_ccd_avg)
        print(f"AL scan uncertainty (per CCD) = {1e3*sigma_per_ccd:.2f} uas")
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
        print(f"   - RA (deg) = {ra:.2f}")
        print(f"   - Dec (deg) = {dec:.2f}")
        print(f"   - Parallax (mas) = {parallax:.2f}")
        print(f"   - Proper motion in RA (mas/yr) = {pmra:.2f}")
        print(f"   - Proper motion in Dec (mas/yr) = {pmdec:.2f}")
        print(f"   - G-band magnitude (au) = {phot_g_mean_mag:.2f}")

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
            ra,
            dec,
            parallax,
            pmra,
            pmdec,
            sma,
            ecc,
            inc,
            aop,
            pan,
            tau,
            mass_1,
            mass_2,
        ]

        bin_model = BinaryModel(self, verbose=False)
        cen_pos = bin_model.calc_model(model_param=model_param)

        rng = np.random.default_rng()
        cen_pos += rng.normal(loc=0.0, scale=sigma_per_transit, size=len(psi))

        self.data_table["centroid_pos_al"] = cen_pos

        self.data_table["centroid_pos_error_al"] = np.full(
            cen_pos.size, sigma_per_transit
        )

        if csv_out is not None:
            self.data_table.to_csv(csv_out, index=False)

    @typechecked
    def get_nss_tables(self) -> None:
        """
        Parameters
        ----------
        data_file : str
            Data file with the Gaia epoch astrometry.
        """

        self.print_section("Retrieve Gaia non-single star tables")

        # List all Gaia tables
        # for table_item in Gaia.load_tables(only_names=True):
        #     print (table_item.get_qualified_name())

        # Gaia DR3 NSS tables
        # gaiadr3.nss_acceleration_astro
        # gaiadr3.nss_non_linear_spectro
        # gaiadr3.nss_two_body_orbit
        # gaiadr3.nss_vim_fl

        if self.gaia_release in ["DR4", "DR5"]:
            raise ValueError(
                "The get_nss_table() only supports Gaia DR3. "
                "Please set the 'gaia_release' argument to 'DR3'."
            )

        for table_item in ["nss_acceleration_astro", "nss_two_body_orbit"]:
            # Query Gaia NSS tables

            gaia_query = f"""
            SELECT *
            FROM gaia{self.gaia_release.lower()}.{table_item}
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
    def query_source(self, source_id: Optional[Union[int, str]] = None) -> None:
        """
        Parameters
        ----------
        data_file : str
            Data file with the Gaia epoch astrometry.
        """

        self.print_section(f"Querying source in GAIA {self.gaia_release}")

        if self.gaia_release in ["DR4", "DR5"]:
            raise ValueError("TODO")

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
        Parameters
        ----------
        data_file : str
            Data file with the Gaia epoch astrometry.
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
