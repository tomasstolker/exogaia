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
from exogaia.priors import LogUniformPrior, NormalPrior, SinPrior, UniformPrior


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

        self.epoch_astrometry = epoch_astrometry
        self.data_table = epoch_astrometry.data_table
        self.primary_mass = epoch_astrometry.primary_mass

        # Set default priors
        self.priors = {}
        self.priors["ra"] = UniformPrior(-10.0, 10.0)
        self.priors["dec"] = UniformPrior(-10.0, 10.0)
        self.priors["parallax"] = UniformPrior(0.0, 100.0)
        self.priors["pmra"] = UniformPrior(-50.0, 50.0)
        self.priors["pmdec"] = UniformPrior(-50.0, 50.0)
        self.priors["sma"] = LogUniformPrior(1e-3, 2.0)
        self.priors["ecc"] = UniformPrior(0.0, 1.0)
        self.priors["inc"] = SinPrior()
        self.priors["aop"] = UniformPrior(0.0, 2.0 * np.pi)
        self.priors["pan"] = UniformPrior(0.0, 2.0 * np.pi)
        self.priors["tau"] = UniformPrior(0.0, 1.0)
        self.priors["m1"] = NormalPrior(self.primary_mass[0], self.primary_mass[1])
        self.priors["m2"] = UniformPrior(0.0, 1)

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
        self.m2_min = self.priors["m2"].min_val

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
        n_params = 5 + 8

        binary_model = BinaryModel(
            epoch_astrometry=self.epoch_astrometry, verbose=False
        )

        def prior_transform(cube, n_dim: int, n_param: int):
            """
            Prior transform
            """

            # delta_RA, delta_Dec (mas)
            # Default: uniform [-10, 10]
            cube[0] = self.priors["ra"].draw_samples(1)
            cube[1] = self.priors["dec"].draw_samples(1)

            # Parallax (mas)
            # Default: uniform [0, 100]
            cube[2] = self.priors["parallax"].draw_samples(1)

            # Proper motion (mas/yr)
            # Default: uniform [-50, 50]
            cube[3] = self.priors["pmra"].draw_samples(1)
            cube[4] = self.priors["pmdec"].draw_samples(1)

            # Semi-major axis (au)
            # Default: log-uniform [log10(1e-3), log10(2)]
            cube[5] = self.priors["sma"].draw_samples(1)

            # Eccentricity
            # Default: uniform [0, 1]
            cube[6] = self.priors["ecc"].draw_samples(1)

            # Inclination (rad)
            # Default: isotropic -> i = arccos(1 - 2u)
            cube[7] = self.priors["inc"].draw_samples(1)

            # Argument of periastron (rad)
            # Default: uniform [0, 2π]
            cube[8] = self.priors["aop"].draw_samples(1)

            # Position angle of ascending node (rad)
            # Default: uniform [0, 2π]
            cube[9] = self.priors["pan"].draw_samples(1)

            # Epoch of periastron
            # Default: uniform [0, 1]
            cube[10] = self.priors["tau"].draw_samples(1)

            # Primary mass (Msun)
            # Default: normal(primary_mass[0], primary_mass[1])
            cube[11] = self.priors["m1"].draw_samples(1)

            # Secondary mass (Msun)
            # Default: uniform(0, 1) with m2 < m1
            m2_prior = UniformPrior(self.m2_min, cube[11])
            cube[12] = m2_prior.draw_samples(1)

            return cube

        def log_likelihood(params, n_dim: int, n_param: int):
            """
            Log-likelihood function
            """

            bin_model = binary_model.calc_model(params)
            res = self.data_table["centroid_pos_al"] - bin_model
            var = self.data_table["centroid_pos_error_al"] ** 2

            return -0.5 * np.sum(res**2 / var)

        # Run sampling with MultiNest
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

        # Get a dictionary with ln(Z), the individual modes and
        # the parameters quantiles of the parameter posteriors
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
                "epoch_astrometry": self.epoch_astrometry,
            }

            with open(pickle_file, "wb") as open_file:
                pickle.dump(pickle_data, open_file, protocol=pickle.HIGHEST_PROTOCOL)
