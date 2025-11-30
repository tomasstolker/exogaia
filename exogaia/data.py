"""
Module for handling epoch astrometry data.
"""

from pathlib import Path
from typing import Tuple

import h5py
import healpy
import kepler
import numpy as np
import pandas as pd
import pooch

from astropy import units as u
from astropy.table import Table
from astropy.time import Time

from exogaia.core import ExoGaia

# from astroquery.gaia import Gaia
# Gaia.ROW_LIMIT = -1


class EpochAstrometry(ExoGaia):
    """
    Class for handling epoch astrometry data.
    """

    def __init__(
        self,
        primary_mass: Tuple[float, float] = None,
        gaia_release="DR3",
    ):
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

        if self.gaia_release == "DR3":
            self.ref_epoch = Time("2016.0", format="jyear", scale="tcb")

        elif self.gaia_release == "DR4":
            self.ref_epoch = Time("2017.5", format="jyear", scale="tcb")

        elif self.gaia_release == "DR5":
            self.ref_epoch = Time("2020.0", format="jyear", scale="tcb")

        else:
            raise ValueError(
                f"The 'gaia_release={self.gaia_release}' is not "
                "supported. Please select 'DR3', 'DR4', or 'DR5'."
            )

        print(f"Gaia release: {self.gaia_release}")
        print(f"Reference epoch: {self.ref_epoch}")
        print(f"\nPrimary mass (Msun): {primary_mass[0]:.2f} +/- {primary_mass[1]:.2f}")

    def __repr__(self):
        if self.data_table is None:
            data_str = "Data table is empty"
        else:
            data_str = self.data_table.head().to_string()

        return data_str

    def read_file(
        self,
        data_file: str,
    ):
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

    def simulate_data(
        self,
        ra,
        dec,
        parallax,
        pmra,
        pmdec,
        m1,
        m2,
        period,
        Tp,
        ecc,
        pan,
        inc,
        aop,
        phot_g_mean_mag,
        f=0.0,
    ):
        """
        Method to predict the epoch astrometry for a binary
        for a given Gaia data release. The function and
        ``healpix`` data has been adopted from ``gaiamock``
        by El-Badry et al. (2025)

        Parameters
        ----------
        ra : float
            The coordinates of the source at the reference
            time (which is different for dr3/dr4/dr5) (deg)
        dec : float
            The coordinates of the source at the reference
            time (which is different for dr3/dr4/dr5) (deg)
        parallax : float
            The true parallax (i.e., 1/d) (mas)
        pmra : float
            The true proper motions in mas/yr
        pmdec : float
            The true proper motions in mas/yr
        m1 : float
            Mass of the star more luminous in the G-band, in Msun
        m2 : float
            Mass of the other star, in Msun
        period : float
            Orbital period in days
        Tp : float
            Periastron time in days
        ecc : float
            Eccentricity
        pan : float
            "big Omega" in radians
        inc : float
            inclination in radians, defined so that
            0 or pi is face-on, and pi/2 is edge-on.
        aop : float
            "little omega" in radians
        phot_g_mean_mag : float
            G-band magnitude
        f : float
            flux ratio, F2/F1, in the G-band (default: 0.0)

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
        healp_num = healpy.ang2pix(
            64, np.radians(90.0 - np.array(dec)), np.radians(np.array(ra)), nest=False
        )

        with h5py.File(healpix_file, "r") as hdf5_file:
            healp_table = Table(hdf5_file[f"healpix_64_{healp_num}"][:])

        obs_time_full = healp_table[
            "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"
        ]

        if self.gaia_release == "DR3":
            time_select = (obs_time_full > 2456891.5) & (obs_time_full < 2457902)

        elif self.gaia_release == "DR4":
            time_select = (obs_time_full > 2456891.5) & (obs_time_full < 2458868.5)

        elif self.gaia_release == "DR5":
            time_select = np.ones(len(healp_table), dtype=bool)

        else:
            raise ValueError(
                f"The 'gaia_release={self.gaia_release}' is not "
                "supported. Please select 'DR3', 'DR4', or 'DR5'."
            )

        table_select = healp_table[time_select]

        # Reject 10% of the data
        # See Sect. 3.3 in El-Badry et al. (2024)
        rand_gen = np.random.default_rng()
        rand_unif = rand_gen.uniform(low=0.0, high=1.0, size=len(table_select))
        table_select = table_select[rand_unif > 0.1]

        psi, plx_factor, obs_time_tcb = (
            table_select["scanAngle[rad]"],
            table_select["parallaxFactorAlongScan"],
            table_select["ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"],
        )

        t_ast_day = obs_time_tcb - self.ref_epoch.jd
        t_ast_yr = t_ast_day / 365.25

        # Uncertainty per CCD -- so not per FoV trnasit.
        # This gives the uncertainty *per CCD* (not per FOV transit),
        # taken from Fig. 3 in https://arxiv.org/abs/2206.05439
        # This is the "EDR3 adjusted" line from that figure, which
        # is already inflated compared to the formal uncertainties.

        n_ccd_avg = 8

        g_vals = [4, 5, 6, 7, 8.2, 8.4, 10, 11, 12]
        g_vals += [13, 14, 15, 16, 17, 18, 19, 20]

        sigma_eta = [0.4, 0.35, 0.15, 0.17, 0.23, 0.13]
        sigma_eta += [0.13, 0.135, 0.125, 0.13, 0.15, 0.23]
        sigma_eta += [0.36, 0.63, 1.05, 2.05, 4.1]

        sigma_per_ccd = np.interp(phot_g_mean_mag, g_vals, sigma_eta)
        epoch_err_per_transit = sigma_per_ccd / np.sqrt(n_ccd_avg)

        if phot_g_mean_mag < 13:
            extra_noise = rand_gen.uniform(low=0, high=0.04, size=1)
        else:
            extra_noise = 0

        mean_anom_obs = 2.0 * np.pi / period * (t_ast_day - Tp)
        EE, _, _ = kepler.kepler(mean_anom_obs, ecc)

        a_au = ((m1 + m2) * (period / 365.25) ** 2) ** (1 / 3)
        a_mas = a_au * parallax

        A_pred = a_mas * (
            np.cos(aop) * np.cos(pan) - np.sin(aop) * np.sin(pan) * np.cos(inc)
        )
        B_pred = a_mas * (
            np.cos(aop) * np.sin(pan) + np.sin(aop) * np.cos(pan) * np.cos(inc)
        )
        F_pred = -a_mas * (
            np.sin(aop) * np.cos(pan) + np.cos(aop) * np.sin(pan) * np.cos(inc)
        )
        G_pred = -a_mas * (
            np.sin(aop) * np.sin(pan) - np.cos(aop) * np.cos(pan) * np.cos(inc)
        )
        cpsi, spsi = np.cos(psi), np.sin(psi)

        X = np.cos(EE) - ecc
        Y = np.sqrt(1 - ecc**2) * np.sin(EE)

        x, y = B_pred * X + G_pred * Y, A_pred * X + F_pred * Y
        delta_eta = -y * cpsi - x * spsi

        def al_bias_binary(delta_eta, q, f, ang_res=90):
            """
            This function predicts the epoch astrometry for a
            binary assuming that the 1D centroid is at the peak
            of the combined AL flux profile, following the model
            from Lindegren+2022
            q = m2/m1 is the flux ratio
            f = F2/F1 is the light ratio
            u is the effective angular resolution in mas.
            delta_eta = rho*cos(psi-theta) = (-y*cos(psi) - x*sin(psi))
                where rho is the angular separation between the
                two stars, psi is the scan angle, and theta is
                position angle.
            """

            def solve_for_x(ff, xi, tol=1e-6, niter_max=100):
                """
                ff is flux ratio, xi is angular separation in units of angular resolution.
                tol is a tolerance to monitor convergence.
                niter_max is the maximum number of iterations
                """
                x = 0.0
                for _ in range(niter_max):
                    thisx = ff * xi / (ff + np.exp(0.5 * xi**2 - xi * x))
                    if abs(thisx - x) < tol:
                        break
                    x = thisx
                return x

            # the first two cases reduce to the same thing, but
            # it's better to separate them for numerical stability.
            if np.abs(delta_eta / ang_res) <= 0.1:
                deta = (f / (1 + f) - q / (1 + q)) * delta_eta

            elif (
                np.abs(delta_eta / ang_res) > 0.1
                and np.abs(delta_eta / ang_res) <= 3 - f
            ):
                B = solve_for_x(ff=f, xi=delta_eta / ang_res)
                deta = ang_res * B - q / (1 + q) * delta_eta

            elif np.abs(delta_eta / ang_res) > 3 - f:
                deta = -q / (1 + q) * delta_eta

            else:
                raise ValueError("TODO")

            return deta

        bias = np.array(
            [
                al_bias_binary(delta_eta=delta_eta[i], q=m2 / m1, f=f)
                for i in range(len(psi))
            ]
        )

        cen_pos = (
            pmra * t_ast_yr * spsi + pmdec * t_ast_yr * cpsi + parallax * plx_factor
        )

        cen_pos += bias
        cen_pos += epoch_err_per_transit * np.random.randn(len(psi))
        cen_pos += extra_noise * np.random.randn(len(psi))

        sim_astrom = {
            "obs_time_tcb": t_ast_yr + self.ref_epoch.jyear,
            "relative_time_year": t_ast_yr,
            "relative_time_day": t_ast_yr * u.year.to(u.day),
            "centroid_pos_al": cen_pos,
            "centroid_pos_error_al": np.full(cen_pos.size, epoch_err_per_transit),
            "scan_pos_angle": psi,
            "parallax_factor_al": plx_factor,
        }

        self.data_table = pd.DataFrame(sim_astrom)
