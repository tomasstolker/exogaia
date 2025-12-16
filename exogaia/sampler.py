"""
Module with the ``NestedSampler`` class.
"""

import os
import pickle
import sys
import warnings

from typing import Optional, Union

import dynesty
import numpy as np

from schwimmbad import MPIPool
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
from exogaia.leastsq import LeastSquares
from exogaia.models import BinaryModel
from exogaia.priors import (
    FixedPrior,
    LogUniformPrior,
    NormalPrior,
    SinPrior,
    UniformPrior,
)


class NestedSampler(ExoGaia):
    """
    Class with the nested sampler.
    """

    @typechecked
    def __init__(self, epoch_astrometry: EpochAstrometry) -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        self.epoch_astrometry = epoch_astrometry
        self.data_table = epoch_astrometry.data_table
        self.primary_mass = epoch_astrometry.primary_mass

        self.output_folder = None
        self.ln_z = None
        self.ln_z_error = None

        self.priors = {}
        self.set_priors()

        self.binary_model = BinaryModel(
            epoch_astrometry=self.epoch_astrometry, verbose=False
        )

        # Number of model parameters
        self.n_params = 5 + 8

    @typechecked
    def set_priors(self) -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        # Set default priors

        least_sq = LeastSquares(epoch_astrometry=self.epoch_astrometry)

        _, best_param, _, ruwe = least_sq.singl_5param()

        if ruwe > 1.1:
            _, best_param, _, ruwe = least_sq.accel_7param()

        if ruwe > 1.1:
            _, best_param, _, ruwe = least_sq.accel_9param()

        self.priors["ra"] = NormalPrior(best_param[0], 0.1)
        self.priors["dec"] = NormalPrior(best_param[1], 0.1)
        self.priors["parallax"] = NormalPrior(best_param[2], 0.1)
        self.priors["pmra"] = NormalPrior(best_param[3], 0.1)
        self.priors["pmdec"] = NormalPrior(best_param[4], 0.1)
        self.priors["sma"] = LogUniformPrior(1e-3, 100.0)
        self.priors["ecc"] = UniformPrior(0.0, 1.0)
        self.priors["inc"] = SinPrior()
        self.priors["aop"] = UniformPrior(0.0, 2.0 * np.pi)
        self.priors["pan"] = UniformPrior(0.0, 2.0 * np.pi)
        self.priors["tau"] = UniformPrior(0.0, 1.0)
        self.priors["mass_1"] = NormalPrior(
            self.primary_mass[0], self.primary_mass[1], truncate_zero=True
        )
        self.priors["mass_2"] = UniformPrior(0.0, 1.0)

    @typechecked
    def _prior_transform(self, cube):
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
        cube[11] = self.priors["mass_1"].draw_samples(1)

        # Secondary mass (Msun)
        if isinstance(self.priors["mass_2"], FixedPrior):
            cube[12] = self.priors["mass_2"].fix_val
        else:
            # Default: uniform [min_m2, m1] with mass_2 < mass_1
            if isinstance(self.priors["mass_2"], UniformPrior):
                m2_prior = UniformPrior(self.priors["mass_2"].min_val, cube[11])
                cube[12] = m2_prior.draw_samples(1)

            else:
                m2_sample = np.inf
                m2_prior = NormalPrior(
                    self.priors["mass_2"].mu,
                    self.priors["mass_2"].sigma,
                    truncate_zero=True,
                )

                while m2_sample > cube[11]:
                    m2_sample = m2_prior.draw_samples(1)

                cube[12] = m2_sample

        return cube

    @typechecked
    def _ln_likelihood(self, params) -> float:
        """
        Log-likelihood function
        """

        bin_model = self.binary_model.calc_model(params)
        # self.binary_model.plot_orbit(params, 'test.png')

        if np.any(np.isnan(bin_model)):
            print("NAN", params)
            return -np.inf

        if np.any(np.isinf(bin_model)):
            print("INF", params)
            return -np.inf

        res = self.data_table["centroid_pos_al"] - bin_model
        var = self.data_table["centroid_pos_error_al"] ** 2

        return -0.5 * np.sum(res**2 / var)

    @typechecked
    def run_multinest(
        self,
        pickle_file: str = "exogaia.pkl",
        n_live_points: int = 500,
        resume: bool = False,
        output_folder: str = "multinest/",
        kwargs_multinest: Optional[dict] = None,
    ) -> None:
        """
        Run MultiNest

        Returns
        -------
        NoneType
            None
        """

        self.print_section("Orbit fit with MultiNest")

        self.output_folder = output_folder

        # Priors

        print("Priors:")
        for param_name, param_prior in self.priors.items():
            print(f"   - {param_name} = {param_prior}")
        print()

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

        @typechecked
        def _lnprior_multinest(cube, n_dim: int, n_param: int) -> None:
            """
            Function to transform the unit cube into the parameter
            cube. It is not clear how to pass additional arguments
            to the function, therefore it is placed here.

            Parameters
            ----------
            cube : LP_c_double
                Unit cube.
            n_dim : int
                Number of dimensions.
            n_param : int
                Number of parameters.

            Returns
            -------
            NoneType
                None
            """

            self._prior_transform(cube)

        @typechecked
        def _lnlike_multinest(
            params, n_dim: int, n_param: int
        ) -> Union[float, np.float64]:
            """
            Function for return the log-likelihood for the
            sampled parameter cube.

            Parameters
            ----------
            params : LP_c_double
                Cube with sampled model parameters.
            n_dim : int
                Number of dimensions. This parameter is mandatory
                but not used by the function.
            n_param : int
                Number of parameters. This parameter is mandatory
                but not used by the function.

            Returns
            -------
            float
                Log-likelihood.
            """

            param_list = []
            for i in range(self.n_params):
                param_list.append(params[i])

            return self._ln_likelihood(param_list)

        # Run sampling with MultiNest
        pymultinest.run(
            _lnlike_multinest,
            _lnprior_multinest,
            self.n_params,
            outputfiles_basename=self.output_folder,
            resume=resume,
            n_live_points=n_live_points,
            **kwargs_multinest,
        )

        # MultiNest analyzer
        analyzer = pymultinest.analyse.Analyzer(
            n_params=self.n_params,
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
        print(f"\nNested Sampling ln(Z) = {self.ln_z:.2f} +/- {self.ln_z_error:.2f}")

        # Nested importance sampling log-evidence
        imp_ln_z = sampling_stats["nested importance sampling global log-evidence"]
        imp_ln_z_error = sampling_stats[
            "nested importance sampling global log-evidence error"
        ]
        print(
            f"Importance Nested Sampling ln(Z) = {imp_ln_z:.2f} +/- {imp_ln_z_error:.2f}"
        )

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

    @typechecked
    def run_dynesty(
        self,
        n_live_points: int = 500,
        resume: bool = False,
        output_folder: str = "dynesty/",
        evidence_tolerance: float = 0.5,
        dynamic: bool = False,
        sample_method: str = "auto",
        bound: str = "multi",
        n_pool: Optional[int] = None,
        mpi_pool: bool = False,
    ) -> None:
        """
        Function for running the fit with a grid of model spectra.
        The parameter estimation and computation of the marginalized
        likelihood (i.e. model evidence), are done with ``Dynesty``.

        When using MPI, it is also required to install ``mpi4py`` (e.g.
        ``pip install mpi4py``), otherwise an error may occur when the
        ``output_folder`` is created by multiple processes.

        Parameters
        ----------
        n_live_points : int
            Number of live points used by the nested sampling
            with ``Dynesty``.
        resume : bool
            Resume the posterior sampling from a previous run.
        output_folder : str
            Path that is used for the output files from ``Dynesty``.
        evidence_tolerance : float
            The dlogZ value used to terminate a nested sampling run,
            or the initial dlogZ value passed to a dynamic nested
            sampling run.
        dynamic : bool
            Whether to use static or dynamic nested sampling (see
            `Dynesty documentation <https://dynesty.readthedocs.io/
            en/stable/dynamic.html>`_).
        sample_method : str
            The sampling method that should be used ('auto', 'unif',
            'rwalk', 'slice', 'rslice' (see `sampling documentation
            <https://dynesty.readthedocs.io/en/stable/
            quickstart.html#nested-sampling-with-dynesty>`_).
        bound : str
            Method used to approximately bound the prior using the
            current set of live points ('none', 'single', 'multi',
            'balls', 'cubes'). `Conditions the sampling methods
            <https://dynesty.readthedocs.io/en/stable/
            quickstart.html#nested-sampling-with-dynesty>`_ used
            to propose new live points
        n_pool : int
            The number of processes for the local multiprocessing. The
            parameter is not used when the argument is set to ``None``.
        mpi_pool : bool
            Distribute the workers to an ``MPIPool`` on a cluster,
            using ``schwimmbad``.

        Returns
        -------
        NoneType
            None
        """

        self.print_section("Orbit fit with Dynesty")

        self.output_folder = output_folder

        # Priors

        print("Priors:")
        for param_name, param_prior in self.priors.items():
            print(f"   - {param_name} = {param_prior}")
        print()

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()
            MPI.COMM_WORLD.Barrier()

        except ImportError:
            mpi_rank = 0

        # Create the output folder if required

        if mpi_rank == 0 and not os.path.exists(output_folder):
            print(f"Creating output folder: {output_folder}")
            os.mkdir(output_folder)

        else:
            print(f"Output folder: {output_folder}")

        print()

        out_basename = os.path.join(output_folder, "")

        if not mpi_pool:
            if n_pool is not None:
                with dynesty.pool.Pool(
                    n_pool,
                    self._ln_likelihood,
                    self._prior_transform,
                    ptform_args=None,
                ) as pool:
                    print(f"Initialized a Dynesty.pool with {n_pool} workers")

                    if dynamic:
                        if resume:
                            dsampler = dynesty.DynamicNestedSampler.restore(
                                fname=out_basename + "dynesty.save",
                                pool=pool,
                            )

                            print(
                                "Resumed a Dynesty run from "
                                f"{out_basename}dynesty.save"
                            )

                        else:
                            dsampler = dynesty.DynamicNestedSampler(
                                loglikelihood=pool.loglike,
                                prior_transform=pool.prior_transform,
                                ndim=self.n_params,
                                pool=pool,
                                sample=sample_method,
                                bound=bound,
                            )

                        dsampler.run_nested(
                            dlogz_init=evidence_tolerance,
                            nlive_init=n_live_points,
                            checkpoint_file=out_basename + "dynesty.save",
                            resume=resume,
                        )

                    else:
                        if resume:
                            dsampler = dynesty.NestedSampler.restore(
                                fname=out_basename + "dynesty.save",
                                pool=pool,
                            )

                            print(
                                "Resumed a Dynesty run from "
                                f"{out_basename}dynesty.save"
                            )

                        else:
                            dsampler = dynesty.NestedSampler(
                                loglikelihood=pool.loglike,
                                prior_transform=pool.prior_transform,
                                ndim=self.n_params,
                                pool=pool,
                                nlive=n_live_points,
                                sample=sample_method,
                                bound=bound,
                            )

                        dsampler.run_nested(
                            dlogz=evidence_tolerance,
                            checkpoint_file=out_basename + "dynesty.save",
                            resume=resume,
                        )
            else:
                if dynamic:
                    if resume:
                        dsampler = dynesty.DynamicNestedSampler.restore(
                            fname=out_basename + "dynesty.save"
                        )

                        print(f"Resumed a Dynesty run from {out_basename}dynesty.save")

                    else:
                        dsampler = dynesty.DynamicNestedSampler(
                            loglikelihood=self._ln_likelihood,
                            prior_transform=self._prior_transform,
                            ndim=self.n_params,
                            ptform_args=None,
                            sample=sample_method,
                            bound=bound,
                        )

                    dsampler.run_nested(
                        dlogz_init=evidence_tolerance,
                        nlive_init=n_live_points,
                        checkpoint_file=out_basename + "dynesty.save",
                        resume=resume,
                    )

                else:
                    if resume:
                        dsampler = dynesty.NestedSampler.restore(
                            fname=out_basename + "dynesty.save"
                        )

                        print(f"Resumed a Dynesty run from {out_basename}dynesty.save")

                    else:
                        dsampler = dynesty.NestedSampler(
                            loglikelihood=self._ln_likelihood,
                            prior_transform=self._prior_transform,
                            ndim=self.n_params,
                            ptform_args=None,
                            sample=sample_method,
                            bound=bound,
                        )

                    dsampler.run_nested(
                        dlogz=evidence_tolerance,
                        checkpoint_file=out_basename + "dynesty.save",
                        resume=resume,
                    )

        else:
            pool = MPIPool()

            if not pool.is_master():
                pool.wait()
                sys.exit(0)

            print("Created an MPIPool object.")

            if dynamic:
                if resume:
                    dsampler = dynesty.DynamicNestedSampler.restore(
                        fname=out_basename + "dynesty.save",
                        pool=pool,
                    )

                else:
                    dsampler = dynesty.DynamicNestedSampler(
                        loglikelihood=self._ln_likelihood,
                        prior_transform=self._prior_transform,
                        ndim=self.n_params,
                        ptform_args=None,
                        pool=pool,
                        sample=sample_method,
                        bound=bound,
                    )

                dsampler.run_nested(
                    dlogz_init=evidence_tolerance,
                    nlive_init=n_live_points,
                    checkpoint_file=out_basename + "dynesty.save",
                    resume=resume,
                )

            else:
                if resume:
                    dsampler = dynesty.NestedSampler.restore(
                        fname=out_basename + "dynesty.save",
                        pool=pool,
                    )

                else:
                    dsampler = dynesty.NestedSampler(
                        loglikelihood=self._ln_likelihood,
                        prior_transform=self._prior_transform,
                        ndim=self.n_params,
                        ptform_args=None,
                        pool=pool,
                        nlive=n_live_points,
                        sample=sample_method,
                        bound=bound,
                    )

                dsampler.run_nested(
                    dlogz=evidence_tolerance,
                    checkpoint_file=out_basename + "dynesty.save",
                    resume=resume,
                )

        # Samples and ln(L)

        results = dsampler.results
        samples = results.samples_equal()
        ln_like = results.logl

        print(f"\nSamples shape: {samples.shape}")
        print(f"Number of iterations: {results.niter}")

        out_file = out_basename + "post_equal_weights.dat"
        print(f"Storing samples: {out_file}")
        np.savetxt(out_file, np.c_[samples, ln_like])

        # Nested sampling log-evidence

        self.ln_z = results.logz[-1]
        self.ln_z_error = results.logzerr[-1]
        print(f"\nln(Z) = {self.ln_z:.2f} +/- {self.ln_z_error:.2f}")

        # Get the sample with the maximum likelihood

        max_idx = np.argmax(ln_like)
        max_lnlike = ln_like[max_idx]
        best_params = samples[max_idx]

        print("\nSample with the maximum likelihood:")
        print(f"   - ln(L) = {max_lnlike:.2f}")

        # param_check = {}
        # for param_idx, param_item in enumerate(best_params):
        #     param_check[self.modelpar[param_idx]] = param_item
        #     if -0.1 < param_item < 0.1:
        #         print(f"   - {self.modelpar[param_idx]} = {param_item:.2e}")
        #     else:
        #         print(f"   - {self.modelpar[param_idx]} = {param_item:.2f}")

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()

        except ImportError:
            mpi_rank = 0
