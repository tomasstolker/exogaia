"""
Module with the ``NestedSampler`` class.
"""

import os
import pickle
import warnings

from typing import Optional

import numpy as np

from typeguard import typechecked

try:
    import pymultinest

except:
    warnings.warn(
        "PyMultiNest could not be imported. "
        "Perhaps because MultiNest was not built "
        "and/or found at the LD_LIBRARY_PATH "
        "(Linux) or DYLD_LIBRARY_PATH (Mac)?"
    )

from exogaia.core import ExoGaia
from exogaia.data import EpochAstrometry
from exogaia.model import BinaryModel
from exogaia.priors import LogUniformPrior, SinPrior, UniformPrior


class NestedSampler(ExoGaia):
    """
    Class with the nested sampler.
    """

    @typechecked
    def __init__(self, epoch_astrometry: EpochAstrometry = None) -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        self.data_table = epoch_astrometry.data_table
        self.primary_mass = epoch_astrometry.primary_mass

        self.priors = {}
        self.priors["sma"] = LogUniformPrior(1e-3, 1.0)
        self.priors["ecc"] = UniformPrior(0.0, 1.0)
        self.priors["inc"] = SinPrior()
        self.priors["aop"] = UniformPrior(0.0, 2.0 * np.pi)
        self.priors["pan"] = UniformPrior(0.0, 2.0 * np.pi)
        self.priors["tau"] = UniformPrior(0.0, 1.0)

        if epoch_astrometry.primary_mass is None:
            self.priors["mtot"] = UniformPrior(0.0, 5.0)
        else:
            self.priors["mtot"] = UniformPrior(0.0, 5.0)

        self.output_folder = None
        self.ln_z = None
        self.ln_z_error = None

    @typechecked
    def run_multinest(
        self,
        pickle_file: str = "exogaia.pkl",
        n_live_points: int = 200,
        resume: bool = False,
        output_folder: str = "multinest/",
        kwargs_multinest: Optional[dict] = None,
    ):
        """
        Run MultiNest
        """

        self.print_section("Orbit fit with MultiNest")

        self.output_folder = output_folder

        # Create empty dictionary if needed

        if kwargs_multinest is None:
            kwargs_multinest = {}

        # Check kwargs_multinest keywords

        if "n_live_points" in kwargs_multinest:
            warnings.warn(
                "Please specify the number of live points "
                "as argument of 'n_live_points' instead "
                "of using 'kwargs_multinest'."
            )

            del kwargs_multinest["n_live_points"]

        if "resume" in kwargs_multinest:
            warnings.warn(
                "Please use the 'resume' parameter "
                "instead of setting the value with "
                "'kwargs_multinest'."
            )

            del kwargs_multinest["resume"]

        if "outputfiles_basename" in kwargs_multinest:
            warnings.warn(
                "Please use the 'output_folder' parameter "
                "instead of setting the value of "
                "'outputfiles_basename' in "
                "'kwargs_multinest'."
            )

            del kwargs_multinest["outputfiles_basename"]

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()
            MPI.COMM_WORLD.Barrier()

        except ImportError:
            mpi_rank = 0

        # Create the output folder if required

        if mpi_rank == 0 and not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

        # Number of model parameters
        n_params = 5 + 7

        binary_model = BinaryModel(
            data_table=self.data_table, primary_mass=self.primary_mass
        )

        def prior_transform(cube, n_dim: int, n_param: int):
            """
            Prior transform.
            """

            R = {
                "delta_ra0_halfspan": 20.0,  # mas (so delta_RA0 in [-span, +span])
                "delta_dec0_halfspan": 20.0,  # mas
                "mu_max": 50.0,  # mas/yr (so mu in [-mu_max, +mu_max])
                "parallax_min": 0.0,
                "parallax_max": 20.0,  # mas
                "sma_min": 0.000001,
                "sma_max": 0.1,  # a (uniform)
                "log_sma_min": -10,
                "log_sma_max": 0,  # log10(a/au) (log-uniform)
                "ecc_min": 0.0,
                "ecc_max": 1.0,  # eccentricity
                "tau_min": 0.0,
                "tau_max": 1.0,  # time units (e.g. fractional year or MJD span)
                "mtot_min": self.primary_mass[0],  # System mass (Msun)
                "mtot_max": 10.0,
            }

            # --- 5-parameter astrometry ---
            # delta_RA0, delta_DEC0 : uniform in [-halfspan, +halfspan] (mas)
            dra0_span = R["delta_ra0_halfspan"]
            ddec0_span = R["delta_dec0_halfspan"]
            cube[0] = -dra0_span + cube[0] * (2.0 * dra0_span)
            cube[1] = -ddec0_span + cube[1] * (2.0 * ddec0_span)

            # proper motions: uniform in [-mu_max, +mu_max] (mas/yr)
            mu_max = R["mu_max"]
            cube[2] = -mu_max + cube[2] * (2.0 * mu_max)
            cube[3] = -mu_max + cube[3] * (2.0 * mu_max)

            # parallax: uniform between parallax_min and parallax_max (mas)
            cube[4] = R["parallax_min"] + cube[4] * (
                R["parallax_max"] - R["parallax_min"]
            )

            # Semimajor axis a: log-uniform between a_min and a_max
            # cube[5] = R["log_sma_min"] + cube[5] * (R["log_sma_max"] - R["log_sma_min"])
            cube[5] = R["sma_min"] + cube[5] * (R["sma_max"] - R["sma_min"])

            # Eccentricity: uniform [0, 1)
            cube[6] = R["ecc_min"] + cube[6] * (R["ecc_max"] - R["ecc_min"])

            # Inclination: isotropic -> i = arccos(1 - 2u)
            cube[7] = np.arccos(1.0 - 2.0 * cube[7])

            # Angles: uniform [0, 2π)
            cube[8] = 2.0 * np.pi * cube[8]  # argument of periastron

            # Angles: uniform [0, 2π)
            cube[9] = 2.0 * np.pi * cube[9]  # longitude of ascending node

            # Epoch of periastron: uniform between T0_min and T0_max
            cube[10] = R["tau_min"] + cube[10] * (R["tau_max"] - R["tau_min"])

            # System mass
            cube[11] = R["mtot_min"] + cube[11] * (R["mtot_max"] - R["mtot_min"])

            return cube

        def log_likelihood(params, n_dim: int, n_param: int):
            """
            Log-likelihood function.
            """

            res = self.data_table["centroid_pos_al"] - binary_model.calc_model(params)
            var = self.data_table["centroid_pos_error_al"] ** 2

            return -0.5 * np.sum(res**2 / var)

        pymultinest.run(
            log_likelihood,
            prior_transform,
            n_params,
            outputfiles_basename=self.output_folder,
            resume=resume,
            n_live_points=n_live_points,
            **kwargs_multinest,
        )

        # MultiNest analyzer
        analyzer = pymultinest.analyse.Analyzer(
            n_params=n_params,
            outputfiles_basename=self.output_folder,
            verbose=False,
        )

        # Get a dictionary with the ln(Z) and its errors, the
        # individual modes and their parameters quantiles of
        # the parameter posteriors
        sampling_stats = analyzer.get_stats()

        # Posterior samples
        samples = analyzer.get_equal_weighted_posterior()

        # Last column is the log-likelihood
        ln_like = samples[:, -1]
        samples = samples[:, :-1]

        # Nested sampling log-evidence
        self.ln_z = sampling_stats["nested sampling global log-evidence"]
        self.ln_z_error = sampling_stats["nested sampling global log-evidence error"]
        print(f"ln(Z) = {self.ln_z:.2f} +/- {self.ln_z_error:.2f}")

        # Nested importance sampling log-evidence
        imp_ln_z = sampling_stats["nested importance sampling global log-evidence"]
        imp_ln_z_error = sampling_stats[
            "nested importance sampling global log-evidence error"
        ]
        print(f"Ln(Z) (importance sampling) = {imp_ln_z:.2f} +/- {imp_ln_z_error:.2f}")

        # Save results to pickle

        if mpi_rank == 0:
            pickle_data = {
                "samples": samples,
                "ln_like": ln_like,
                "ln_z": (self.ln_z, self.ln_z_error),
                "data_table": self.data_table,
                "primary_mass": self.primary_mass,
            }

            with open(pickle_file, "wb") as open_file:
                pickle.dump(pickle_data, open_file, protocol=pickle.HIGHEST_PROTOCOL)
