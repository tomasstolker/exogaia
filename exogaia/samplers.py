"""
Module with the ``NestedSampler`` and ``MCMCSampler`` classes.
"""

import os
import pickle
import sys
import warnings

import dynesty
import emcee
import numpy as np
import reddemcee

from beartype import beartype, typing
from schwimmbad import MPIPool

from exogaia.core import ExoGaia
from exogaia.data import EpochAstrometry
from exogaia.leastsq import LeastSquares
from exogaia.models import KeplerModel
from exogaia.priors import (
    LogUniformPrior,
    NormalPrior,
    SinPrior,
    UniformPrior,
)


class NestedSampler(ExoGaia):
    """
    Class for nested sampling with ``MultiNest`` and ``Dynesty``.
    """

    @beartype
    def __init__(
        self, epoch_astrometry: EpochAstrometry, least_squares: LeastSquares
    ) -> None:
        """
        Parameters
        ----------
        epoch_astrometry : EpochAstrometry
            ``EpochAstrometry`` object that contains the data.
        least_squares : LeastSquares
            ``LeastSquares`` object after running
            ``:func:`~exogaia.leastsq.LeastSquares.orbit_fit```
            such that the ``best_param`` attribute contains
            the best-fit parameters from the least-squares fit.

        Returns
        -------
        NoneType
            None
        """

        self.print_section("Nested sampler")

        self.epoch_astrometry = epoch_astrometry
        self.data_table = epoch_astrometry.data_table
        self.primary_mass = epoch_astrometry.primary_mass
        self.least_squares = least_squares

        self.output_folder = None
        self.ln_z = None
        self.ln_z_error = None

        self.priors = {}
        self.set_priors()

        self.kepler_model = KeplerModel(
            epoch_astrometry=self.epoch_astrometry, verbose=False
        )

        # Parameter index numbers
        self.param_indices = {
            "ra_offset": 0,
            "dec_offset": 1,
            "parallax": 2,
            "pmra": 3,
            "pmdec": 4,
            "per": 5,
            "ecc": 6,
            "tau": 7,
            "sma_0": 8,
            "inc": 9,
            "aop": 10,
            "pan": 11,
        }

        # Number of model parameters
        self.n_params = len(self.param_indices)

    @beartype
    def set_priors(self) -> None:
        """
        Method for setting the default parameter priors.

        Returns
        -------
        NoneType
            None
        """

        self.priors["ra_offset"] = NormalPrior(self.least_squares.best_param[0], 0.1)
        self.priors["dec_offset"] = NormalPrior(self.least_squares.best_param[1], 0.1)

        self.priors["parallax"] = NormalPrior(
            self.least_squares.best_param[2], 0.1, truncate_zero=True
        )

        self.priors["pmra"] = NormalPrior(self.least_squares.best_param[3], 0.1)
        self.priors["pmdec"] = NormalPrior(self.least_squares.best_param[4], 0.1)

        if (
            len(self.least_squares.best_param) == 12
            and self.least_squares.param_cov is not None
        ):
            param_sig = np.sqrt(np.diag(self.least_squares.param_cov))

            self.priors["per"] = NormalPrior(
                self.least_squares.best_param[5], param_sig[5], truncate_zero=True
            )
            self.priors["ecc"] = NormalPrior(
                self.least_squares.best_param[6],
                param_sig[6],
                truncate_zero=True,
                truncate_upper=1.0,
            )
            self.priors["tau"] = NormalPrior(
                self.least_squares.best_param[7],
                param_sig[7],
                truncate_zero=True,
                truncate_upper=1.0,
            )
            self.priors["sma_0"] = NormalPrior(
                self.least_squares.best_param[8], param_sig[8], truncate_zero=True
            )
            self.priors["inc"] = NormalPrior(
                self.least_squares.best_param[9],
                param_sig[9],
                truncate_zero=True,
                truncate_upper=np.pi,
            )
            self.priors["aop"] = NormalPrior(
                self.least_squares.best_param[10],
                param_sig[10],
                truncate_zero=True,
                truncate_upper=2.0 * np.pi,
            )
            self.priors["pan"] = NormalPrior(
                self.least_squares.best_param[11],
                param_sig[11],
                truncate_zero=True,
                truncate_upper=2.0 * np.pi,
            )

        else:
            self.priors["per"] = LogUniformPrior(1e1, 1e5)
            self.priors["ecc"] = UniformPrior(0.0, 1.0)
            self.priors["tau"] = UniformPrior(0.0, 1.0)
            self.priors["sma_0"] = LogUniformPrior(1e-3, 100.0)
            self.priors["inc"] = SinPrior()
            self.priors["aop"] = UniformPrior(0.0, 2.0 * np.pi)
            self.priors["pan"] = UniformPrior(0.0, 2.0 * np.pi)
            # self.priors["sma"] = LogUniformPrior(1e-3, 100.0)
            # self.priors["mass_1"] = NormalPrior(
            #     self.primary_mass[0], self.primary_mass[1], truncate_zero=True
            # )
            # self.priors["mass_2"] = UniformPrior(0.0, 1.0)

    @beartype
    def prior_transform(self, cube):
        """
        Method for transforming the unit cube into a cube
        with parameter samples.

        Parameters
        ----------
        cube : LP_c_double
            Input unit cube.

        Returns
        -------
        LP_c_double
            Output cube with sampled model parameters.
        """

        # RA/Dec offset at Gaia reference epoch
        # relative to the Gaia coordinates of the
        # source at the reference epoch (mas)
        # Default: uniform [-10, 10]
        cube[0] = self.priors["ra_offset"].draw_samples(1)[0]
        cube[1] = self.priors["dec_offset"].draw_samples(1)[0]

        # Parallax (mas)
        # Default: uniform [0, 100]
        cube[2] = self.priors["parallax"].draw_samples(1)[0]

        # Proper motion (mas/yr)
        # Default: uniform [-50, 50]
        cube[3] = self.priors["pmra"].draw_samples(1)[0]
        cube[4] = self.priors["pmdec"].draw_samples(1)[0]

        # Period (days)
        # Default: log-uniform [log10(1e1), log10(5)]
        cube[5] = self.priors["per"].draw_samples(1)[0]

        # Eccentricity
        # Default: uniform [0, 1]
        cube[6] = self.priors["ecc"].draw_samples(1)[0]

        # Epoch of periastron
        # Default: uniform [0, 1]
        cube[7] = self.priors["tau"].draw_samples(1)[0]

        # Semi-major axis of photocenter (mas)
        # Default: log-uniform [log10(1e-3), log10(2)]
        cube[8] = self.priors["sma_0"].draw_samples(1)[0]

        # Inclination (rad)
        # Default: isotropic -> i = arccos(1 - 2u)
        cube[9] = self.priors["inc"].draw_samples(1)[0]

        # Argument of periastron (rad)
        # Default: uniform [0, 2π]
        cube[10] = self.priors["aop"].draw_samples(1)[0]

        # Position angle of ascending node (rad)
        # Default: uniform [0, 2π]
        cube[11] = self.priors["pan"].draw_samples(1)[0]

        # # Primary mass (Msun)
        # # Default: normal(primary_mass[0], primary_mass[1])
        # cube[12] = self.priors["mass_1"].draw_samples(1)[0]
        #
        # # Secondary mass (Msun)
        # if isinstance(self.priors["mass_2"], FixedPrior):
        #     cube[12] = self.priors["mass_2"].fix_val
        # else:
        #     # Default: uniform [min_m2, m1] with mass_2 < mass_1
        #     if isinstance(self.priors["mass_2"], UniformPrior):
        #         m2_prior = UniformPrior(self.priors["mass_2"].min_val, cube[11])
        #         cube[12] = m2_prior.draw_samples(1)[0]
        #
        #     else:
        #         m2_sample = np.inf
        #         m2_prior = NormalPrior(
        #             self.priors["mass_2"].mu,
        #             self.priors["mass_2"].sigma,
        #             truncate_zero=True,
        #         )
        #
        #         while m2_sample > cube[11]:
        #             m2_sample = m2_prior.draw_samples(1)[0]
        #
        #         cube[12] = m2_sample

        return cube

    @beartype
    def log_likelihood(self, params) -> typing.Union[np.float64, float]:
        """
        Method for calculating the log-likelihood for the
        sampled parameter cube.

        Parameters
        ----------
        params : LP_c_double
            Cube with sampled model parameters.

        Returns
        -------
        float
            Log-likelihood.
        """

        delta_eta = self.kepler_model.calc_1d_model(params)
        # self.kepler_model.plot_orbit(params, 'test.png')

        if np.any(np.isnan(delta_eta)):
            print("NAN", params)
            return -np.inf

        if np.any(np.isinf(delta_eta)):
            print("INF", params)
            return -np.inf

        res = self.data_table["centroid_pos_al"] - delta_eta
        var = self.data_table["centroid_pos_error_al"] ** 2

        return -0.5 * np.sum(res**2 / var)

    @beartype
    def run_multinest(
        self,
        pickle_file: str = "exogaia.pkl",
        n_live_points: int = 500,
        resume: bool = False,
        output_folder: str = "multinest/",
        kwargs_multinest: typing.Optional[dict] = None,
    ) -> None:
        """
        Function to run the ``PyMultiNest`` wrapper of the
        ``MultiNest`` sampler. While ``PyMultiNest`` can be
        installed with ``pip`` from the PyPI repository,
        ``MultiNest`` has to be built manually. See the
        `PyMultiNest documentation <http://johannesbuchner.
        github.io/PyMultiNest/install.html>`_. The library
        path of ``MultiNest`` should be set to the
        environmental variable ``LD_LIBRARY_PATH`` on a
        Linux machine and ``DYLD_LIBRARY_PATH`` on a Mac.
        Alternatively, the variable can be set before
        importing the ``species`` package, for example:

        .. code-block:: python

            >>> import os
            >>> os.environ['DYLD_LIBRARY_PATH'] = '/path/to/MultiNest/lib'
            >>> import species

        When using MPI, it is also required to install ``mpi4py`` (e.g.
        ``pip install mpi4py``), otherwise an error may occur when the
        ``output_folder`` is created by multiple processes.

        Parameters
        ----------
        pickle_file : str
            Output file name in which the results will be stored.
            The output file is a Pickle file.
        n_live_points : int
            Number of live points used for the nested sampling.
        resume : bool
            Resume the posterior sampling from a previous run.
        output_folder : str
            Path that is used for the output files from ``MultiNest``.
        kwargs_multinest : dict, None
            Dictionary with keyword arguments that can be used to
            adjust the parameters of the `run() function
            <https://github.com/JohannesBuchner/PyMultiNest/blob/
            master/pymultinest/run.py>`_ of the ``PyMultiNest``
            sampler. See also the `documentation of MultiNest
            <https://github.com/JohannesBuchner/MultiNest>`_.

        Returns
        -------
        NoneType
            None
        """

        import pymultinest

        self.print_section("Run MultiNest")

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

        @beartype
        def log_prior_multinest(cube, n_dim: int, n_param: int) -> None:
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

            self.prior_transform(cube)

        @beartype
        def log_like_multinest(
            params, n_dim: int, n_param: int
        ) -> typing.Union[float, np.float64]:
            """
            Method for calculating the log-likelihood for the
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

            return self.log_likelihood(param_list)

        # Run sampling with MultiNest
        pymultinest.run(
            log_like_multinest,
            log_prior_multinest,
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

    @beartype
    def run_dynesty(
        self,
        n_live_points: int = 500,
        resume: bool = False,
        output_folder: str = "dynesty/",
        evidence_tolerance: float = 0.5,
        dynamic: bool = False,
        sample_method: str = "auto",
        bound: str = "multi",
        n_pool: typing.Optional[int] = None,
        mpi_pool: bool = False,
    ) -> None:
        """
        Method for running the nested sampling with ``Dynesty``, which
        computes both the Bayesian evidence (i.e. :math:`\\log{Z}`) and
        the parameter posterior distributions.

        When using MPI, it is also required to install ``mpi4py`` (e.g.
        ``pip install mpi4py``), otherwise an error may occur when the
        ``output_folder`` is created by multiple processes.

        Parameters
        ----------
        n_live_points : int
            Number of live points used for the nested sampling.
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

        self.print_section("Run Dynesty")

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
                    self.log_likelihood,
                    self.prior_transform,
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
                            loglikelihood=self.log_likelihood,
                            prior_transform=self.prior_transform,
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
                            loglikelihood=self.log_likelihood,
                            prior_transform=self.prior_transform,
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
                        loglikelihood=self.log_likelihood,
                        prior_transform=self.prior_transform,
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
                        loglikelihood=self.log_likelihood,
                        prior_transform=self.prior_transform,
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
        # best_params = samples[max_idx]

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


class MCMCSampler(ExoGaia):
    """
    Class for MCMC sampling with ``emcee`` and ``reddemcee``.
    """

    @beartype
    def __init__(
        self, epoch_astrometry: EpochAstrometry, least_squares: LeastSquares
    ) -> None:
        """
        Parameters
        ----------
        epoch_astrometry : EpochAstrometry
            ``EpochAstrometry`` object that contains the data.
        least_squares : LeastSquares
            ``LeastSquares`` object after running
            ``:func:`~exogaia.leastsq.LeastSquares.orbit_fit```
            such that the ``best_param`` attribute contains
            the best-fit parameters from the least-squares fit.

        Returns
        -------
        NoneType
            None
        """

        self.print_section("MCMC sampler")

        self.epoch_astrometry = epoch_astrometry
        self.data_table = epoch_astrometry.data_table
        self.primary_mass = epoch_astrometry.primary_mass
        self.least_squares = least_squares

        self.output_folder = None

        self.priors = {}
        self.set_priors()

        self.kepler_model = KeplerModel(
            epoch_astrometry=self.epoch_astrometry, verbose=False
        )

        # Parameter index numbers
        self.param_indices = {
            "ra_offset": 0,
            "dec_offset": 1,
            "parallax": 2,
            "pmra": 3,
            "pmdec": 4,
            "per": 5,
            "ecc": 6,
            "tau": 7,
            "sma_0": 8,
            "inc": 9,
            "aop": 10,
            "pan": 11,
        }

        # Number of model parameters
        self.n_params = len(self.param_indices)

    @beartype
    def set_priors(self) -> None:
        """
        Method for setting the default parameter priors.

        Returns
        -------
        NoneType
            None
        """

        self.priors["ra_offset"] = NormalPrior(self.least_squares.best_param[0], 0.1)
        self.priors["dec_offset"] = NormalPrior(self.least_squares.best_param[1], 0.1)

        self.priors["parallax"] = NormalPrior(
            self.least_squares.best_param[2], 0.1, truncate_zero=True
        )

        self.priors["pmra"] = NormalPrior(self.least_squares.best_param[3], 0.1)
        self.priors["pmdec"] = NormalPrior(self.least_squares.best_param[4], 0.1)

        if (
            len(self.least_squares.best_param) == 12
            and self.least_squares.param_cov is not None
        ):
            param_sig = np.sqrt(np.diag(self.least_squares.param_cov))

            self.priors["per"] = NormalPrior(
                self.least_squares.best_param[5], param_sig[5], truncate_zero=True
            )
            self.priors["ecc"] = NormalPrior(
                self.least_squares.best_param[6],
                param_sig[6],
                truncate_zero=True,
                truncate_upper=1.0,
            )
            self.priors["tau"] = NormalPrior(
                self.least_squares.best_param[7],
                param_sig[7],
                truncate_zero=True,
                truncate_upper=1.0,
            )
            self.priors["sma_0"] = NormalPrior(
                self.least_squares.best_param[8], param_sig[8], truncate_zero=True
            )
            self.priors["inc"] = NormalPrior(
                self.least_squares.best_param[9],
                param_sig[9],
                truncate_zero=True,
                truncate_upper=np.pi,
            )
            self.priors["aop"] = NormalPrior(
                self.least_squares.best_param[10],
                param_sig[10],
                truncate_zero=True,
                truncate_upper=2.0 * np.pi,
            )
            self.priors["pan"] = NormalPrior(
                self.least_squares.best_param[11],
                param_sig[11],
                truncate_zero=True,
                truncate_upper=2.0 * np.pi,
            )

        else:
            self.priors["per"] = LogUniformPrior(1e1, 1e5)
            self.priors["ecc"] = UniformPrior(0.0, 1.0)
            self.priors["tau"] = UniformPrior(0.0, 1.0)
            self.priors["sma_0"] = LogUniformPrior(1e-3, 100.0)
            self.priors["inc"] = SinPrior()
            self.priors["aop"] = UniformPrior(0.0, 2.0 * np.pi)
            self.priors["pan"] = UniformPrior(0.0, 2.0 * np.pi)
            # self.priors["sma"] = LogUniformPrior(1e-3, 100.0)
            # self.priors["mass_1"] = NormalPrior(
            #     self.primary_mass[0], self.primary_mass[1], truncate_zero=True
            # )
            # self.priors["mass_2"] = UniformPrior(0.0, 1.0)

    @beartype
    def log_prior(self, params: np.ndarray) -> float:
        """
        Method for the log-prior used by the MCMC.

        Parameters
        ----------
        params : np.ndarray
            Array with the model parameters. The order of the parameter
            values should be the same as in the ``calc_model`` method
            of :class:`~exogaia.models.KeplerModel`.

        Returns
        -------
        float
            Log-prior of the model evaluation.
        """

        log_prior = 0.0

        for param_item, param_idx in self.param_indices.items():
            if isinstance(self.priors[param_item], UniformPrior):
                if (
                    params[param_idx] < self.priors[param_item].min_val
                    or params[param_idx] > self.priors[param_item].max_val
                ):
                    log_prior += -np.inf

            elif isinstance(self.priors[param_item], LogUniformPrior):
                if (
                    params[param_idx] < 10.0 ** self.priors[param_item].log_min
                    or params[param_idx] > 10.0 ** self.priors[param_item].log_max
                ):
                    log_prior += -np.inf

            elif isinstance(self.priors[param_item], NormalPrior):
                if self.priors[param_item].truncate_zero and params[param_idx] < 0.0:
                    log_prior += -np.inf

                elif (
                    self.priors[param_item].truncate_upper is not None
                    and params[param_idx] > self.priors[param_item].truncate_upper
                ):
                    log_prior += -np.inf

                else:
                    log_prior += (
                        -0.5
                        * (
                            (params[param_idx] - self.priors[param_item].mu)
                            / self.priors[param_item].sigma
                        )
                        ** 2
                    )

            elif isinstance(self.priors[param_item], SinPrior):
                if params[param_idx] <= 0.0 or params[param_idx] >= np.pi:
                    log_prior += -np.inf

                else:
                    log_prior += np.log(np.sin(params[param_idx]))

            else:
                raise ValueError(self.priors[param_item])

        return log_prior

    @beartype
    def log_likelihood(self, params: np.ndarray) -> float:
        """
        Method for the log-likelihood used by the MCMC.

        Parameters
        ----------
        params : np.ndarray
            Array with the model parameters. The order of the parameter
            values should be the same as in the ``calc_model`` method
            of :class:`~exogaia.models.KeplerModel`.

        Returns
        -------
        float
            Log-likelihood of the model evaluation.
        """

        delta_eta = self.kepler_model.calc_1d_model(params)

        if np.any(np.isnan(delta_eta)):
            print("NAN", params)
            return -np.inf

        if np.any(np.isinf(delta_eta)):
            print("INF", params)
            return -np.inf

        res = self.data_table["centroid_pos_al"] - delta_eta
        var = self.data_table["centroid_pos_error_al"] ** 2

        return -0.5 * np.sum(res**2 / var)

    @beartype
    def log_probability(self, params: np.ndarray) -> typing.Tuple[float, float]:
        """
        Method for the log-probability used by the MCMC.

        Parameters
        ----------
        params : np.ndarray
            Array with the model parameters. The order of the parameter
            values should be the same as in the ``calc_model`` method
            of :class:`~exogaia.models.KeplerModel`.

        Returns
        -------
        float
            Log-probability (i.e. log-prior + log-likelihood)
            of the model evaluation.
        float
            Log-prior of the model evaluation.
        """

        log_prior = self.log_prior(params)

        if np.isfinite(log_prior):
            log_prob = log_prior + self.log_likelihood(params)
        else:
            log_prob = -np.inf

        return log_prob, log_prior

    @beartype
    def run_mcmc(
        self,
        pickle_file: str = "exogaia.pkl",
        n_walkers: int = 200,
        n_steps: int = 1000,
        progress: bool = True,
    ) -> None:
        """
        Method for running the MCMC ensemble sampler of ``emcee``.

        Parameters
        ----------
        pickle_file : str
            Output file name in which the results will be stored.
            The output file is a Pickle file.
        n_walkers : int
            Number of walkers that will explore the posterior landscape.
        n_steps : int
            Number of steps that each walker will make.
        progress : bool
            Display progress bar (default: False).

        Returns
        -------
        NoneType
            None
        """

        self.print_section("Run emcee")

        sampler = emcee.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=self.n_params,
            log_prob_fn=self.log_probability,
        )

        init_pos = np.zeros((n_walkers, self.n_params))
        for param_item, param_idx in self.param_indices.items():
            init_pos[:, param_idx] = self.priors[param_item].draw_samples(n_walkers)

        sampler.run_mcmc(initial_state=init_pos, nsteps=n_steps, progress=progress)

        sampler.get_autocorr_time(quiet=True)

        samples = sampler.get_chain(flat=False, thin=1, discard=0)
        log_prob = sampler.get_log_prob(flat=False, thin=1, discard=0)
        log_prior = sampler.get_blobs(flat=False, thin=1, discard=0)

        # Save results to pickle

        pickle_data = {
            "samples": samples,
            "ln_like": log_prob - log_prior,
            "ln_z": None,
            "data_table": self.data_table,
            "primary_mass": self.primary_mass,
            "epoch_astrometry": self.epoch_astrometry,
        }

        with open(pickle_file, "wb") as open_file:
            pickle.dump(pickle_data, open_file, protocol=pickle.HIGHEST_PROTOCOL)

    @beartype
    def run_ptmcmc(
        self,
        pickle_file: str = "exogaia.pkl",
        n_temps: int = 20,
        n_walkers: int = 200,
        n_steps: int = 1000,
        n_sweeps: int = 10,
        progress: bool = True,
    ) -> None:
        """
        Method for running the adaptive parallel tempering tempered
        MCMC ensemble sampler of ``reddemcee``.

        Parameters
        ----------
        pickle_file : str
            Output file name in which the results will be stored.
            The output file is a Pickle file.
        n_temps : int
            Number of temperatures.
        n_walkers : int
            Number of walkers that will explore the posterior landscape.
        n_steps : int
            Number of steps that each walker will make.
        n_sweeps : int
            Number of sweeps to run.
        progress : bool
            Display progress bar (default: False).

        Returns
        -------
        NoneType
            None
        """

        self.print_section("Run reddemcee")

        sampler = reddemcee.PTSampler(
            nwalkers=n_walkers,
            ndim=self.n_params,
            log_like=self.log_likelihood,
            log_prior=self.log_prior,
            ntemps=n_temps,
        )

        init_pos = np.zeros((n_temps, n_walkers, self.n_params))
        for param_item, param_idx in self.param_indices.items():
            init_pos[:, :, param_idx] = self.priors[param_item].draw_samples(n_walkers)

        sampler.run_mcmc(
            initial_state=init_pos, nsteps=n_steps, nsweeps=n_sweeps, progress=progress
        )

        sampler.get_autocorr_time(quiet=True)

        samples = sampler.get_chain(flat=False, thin=1, discard=0)
        # log_prob = sampler.get_log_prob(flat=False, thin=1, discard=0)
        log_like = sampler.get_log_like(flat=False, thin=1, discard=0)

        # Save results to pickle

        pickle_data = {
            "samples": samples,
            "ln_like": log_like,
            "ln_z": None,
            "data_table": self.data_table,
            "primary_mass": self.primary_mass,
            "epoch_astrometry": self.epoch_astrometry,
        }

        with open(pickle_file, "wb") as open_file:
            pickle.dump(pickle_data, open_file, protocol=pickle.HIGHEST_PROTOCOL)
