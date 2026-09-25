import os
import pytest

from exogaia.data import GaiaAstrometry
from exogaia.leastsq import LeastSquares
from exogaia.samplers import MCMCSampler
from exogaia.results import SamplingResults


class TestExoGaia:
    def setup_class(self) -> None:
        self.limit = 1e-8

        self.test_dir = os.path.dirname(__file__) + "/"

        self.epoch_astrom = GaiaAstrometry(primary_mass=None, gaia_release="DR4")

        self.epoch_astrom.retrieve_gaia_bh3(exclude_outliers=True, combine_ccds=True)

        self.least_squares = LeastSquares(epoch_astrometry=self.epoch_astrom)

    def test_leastsq_aen(self) -> None:
        self.least_squares.calc_excess_noise()

    def test_leastsq_5param(self) -> None:
        self.least_squares.singl_5param(plot_file=None)

    def test_leastsq_7param(self) -> None:
        self.least_squares.accel_7param(plot_file=None)

    def test_leastsq_9param(self) -> None:
        self.least_squares.accel_9param(plot_file=None)

    def test_orbit_grid(self) -> None:
        self.least_squares.orbit_grid(plot_file=None, n_points=30)

    def test_orbit_fit(self) -> None:
        self.least_squares.orbit_fit(inc_jitter=False, plot_file=None)

    def test_fit_jitter(self) -> None:
        self.least_squares.orbit_fit(inc_jitter=True, plot_file=None)

        assert len(self.least_squares.best_param) == 12
        assert self.least_squares.param_cov.shape == (12, 12)

    def test_mcmc_sampler(self) -> None:
        sampler = MCMCSampler(
            epoch_astrometry=self.epoch_astrom, least_squares=self.least_squares
        )

        sampler.run_ptmcmc(
            pickle_file=self.test_dir + "exogaia.pkl",
            n_temps=10,
            n_walkers=50,
            n_steps=300,
            n_sweeps=1,
            progress=True,
        )

    def test_sampling_walkers(self) -> None:
        results = SamplingResults(burnin=100, pickle_file=self.test_dir + "exogaia.pkl")
        results.plot_walkers(
            n_walkers=50, thin=None, plot_file=self.test_dir + "plot.png"
        )

    def test_sampling_posterior(self) -> None:
        results = SamplingResults(burnin=100, pickle_file=self.test_dir + "exogaia.pkl")
        results.plot_posterior(truths=None, plot_file=self.test_dir + "plot.png")

    def test_sampling_residuals(self) -> None:
        results = SamplingResults(burnin=100, pickle_file=self.test_dir + "exogaia.pkl")
        results.plot_residuals(plot_file=self.test_dir + "plot.png")

    def test_sampling_orbit(self) -> None:
        results = SamplingResults(burnin=100, pickle_file=self.test_dir + "exogaia.pkl")
        results.plot_orbit(plot_file=self.test_dir + "plot.png")
