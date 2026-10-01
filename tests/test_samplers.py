from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy.time import Time

from exogaia.data import GaiaAstrometry
from exogaia.leastsq import LeastSquares
from exogaia.results import SamplingResults
from exogaia.samplers import MCMCSampler, NestedSampler


def test_run_mcmc_sampler(least_squares: LeastSquares, test_dir: Path) -> None:
    sampler = MCMCSampler(
        epoch_astrometry=least_squares.epoch_astrometry,
        least_squares=least_squares,
    )

    sampler.run_ptmcmc(
        pickle_file=str(test_dir / "exogaia.pkl"),
        n_temps=10,
        n_walkers=50,
        n_steps=300,
        n_sweeps=1,
        progress=True,
    )


def test_plot_sampling_walkers(test_dir: Path) -> None:
    results = SamplingResults(burnin=100, pickle_file=str(test_dir / "exogaia.pkl"))
    results.plot_walkers(n_walkers=50, thin=None, plot_file=str(test_dir / "plot.png"))


def test_plot_sampling_posterior(test_dir: Path) -> None:
    results = SamplingResults(burnin=100, pickle_file=str(test_dir / "exogaia.pkl"))
    results.plot_posterior(truths=None, plot_file=str(test_dir / "plot.png"))


def test_plot_sampling_residuals(test_dir: Path) -> None:
    results = SamplingResults(burnin=100, pickle_file=str(test_dir / "exogaia.pkl"))
    results.plot_residuals(plot_file=str(test_dir / "plot.png"))


def test_plot_sampling_orbit(test_dir: Path) -> None:
    results = SamplingResults(burnin=100, pickle_file=str(test_dir / "exogaia.pkl"))
    results.plot_orbit(plot_file=str(test_dir / "plot.png"))


def test_nested_sampler_transforms_prior_cube() -> None:
    epoch_astrometry = GaiaAstrometry(primary_mass=(1.0, 0.1), verbose=False)
    sampler = NestedSampler(epoch_astrometry, restrict_node=True)

    transformed = sampler.prior_transform(np.full(12, 0.5))

    assert transformed.shape == (12,)
    assert transformed[5] == pytest.approx(np.sqrt(10.0 * 5.0e3))
    assert 0.0 <= transformed[11] <= np.pi
