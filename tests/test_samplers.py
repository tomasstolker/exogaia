import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from exogaia.data import GaiaAstrometry
from exogaia.leastsq import LeastSquares
from exogaia.results import SamplingResults
from exogaia.samplers import MCMCSampler, NestedSampler
from exogaia.utils import param_dict_to_list
from tests.test_leastsq import ORBIT_PARAMETERS, least_squares


def test_run_mcmc_sampler(least_squares: LeastSquares, tmp_path: Path) -> None:
    least_squares.best_param = np.array(param_dict_to_list(ORBIT_PARAMETERS))
    least_squares.param_cov = np.diag(
        np.array([0.1, 0.1, 0.1, 0.1, 0.1, 10.0, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1]) ** 2
    )
    sampler = MCMCSampler(
        epoch_astrometry=least_squares.epoch_astrometry,
        least_squares=least_squares,
    )

    sampler.run_ptmcmc(
        pickle_file=str(tmp_path / "exogaia.pkl"),
        n_temps=2,
        n_walkers=30,
        n_steps=20,
        n_sweeps=1,
        progress=False,
    )

    results = SamplingResults(pickle_file=str(tmp_path / "exogaia.pkl"))
    assert results.orig_samples.shape == (20, 30, 12)


@pytest.fixture
def sampling_results(least_squares: LeastSquares, tmp_path: Path) -> SamplingResults:
    rng = np.random.default_rng(43)
    center = np.array(param_dict_to_list(ORBIT_PARAMETERS))
    scale = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 10.0, 0.03, 0.03, 0.1, 0.1, 0.1, 0.1])
    samples = rng.normal(center, scale, size=(20, 30, 12))
    pickle_file = tmp_path / "exogaia.pkl"
    with pickle_file.open("wb") as file:
        pickle.dump(
            {
                "samples": samples,
                "ln_like": rng.normal(size=(20, 30)),
                "ln_z": None,
                "data_table": least_squares.data_table,
                "epoch_astrometry": least_squares.epoch_astrometry,
                "param_indices": MCMCSampler(
                    least_squares.epoch_astrometry
                ).param_indices,
            },
            file,
        )
    return SamplingResults(burnin=5, pickle_file=str(pickle_file))


@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("plot_walkers", {"n_walkers": 5}),
        ("plot_posterior", {}),
        ("plot_residuals", {"n_samples": 5, "seed": 42}),
        ("plot_orbit", {"n_samples": 5}),
    ],
)
def test_plot_sampling_results(
    sampling_results: SamplingResults, tmp_path: Path, method: str, kwargs: dict
) -> None:
    plot_file = tmp_path / f"{method}.png"
    figure = getattr(sampling_results, method)(plot_file=str(plot_file), **kwargs)

    assert figure is not None
    assert plot_file.is_file()
    plt.close(figure)


def test_nested_sampler_transforms_prior_cube() -> None:
    epoch_astrometry = GaiaAstrometry(primary_mass=(1.0, 0.1), verbose=False)
    sampler = NestedSampler(epoch_astrometry, restrict_node=True)

    transformed = sampler.prior_transform(np.full(12, 0.5))

    assert transformed.shape == (12,)
    assert transformed[5] == pytest.approx(np.sqrt(10.0 * 5.0e3))
    assert 0.0 <= transformed[11] <= np.pi
