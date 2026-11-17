import numpy as np
import pandas as pd
import pytest

from exogaia.data import GaiaAstrometry
from exogaia.leastsq import LeastSquares
from exogaia.models import KeplerModel
from exogaia.utils import param_dict_to_list

ORBIT_PARAMETERS = {
    "ra_offset": 1.0,
    "dec_offset": -1.0,
    "parallax": 10.0,
    "pm_ra": 3.0,
    "pm_dec": -2.0,
    "per": 400.0,
    "ecc": 0.2,
    "tau": 0.3,
    "sma": 2.0,
    "inc": 1.0,
    "aop": 1.2,
    "pan": 0.7,
}


@pytest.fixture
def least_squares() -> LeastSquares:
    rng = np.random.default_rng(42)
    astrometry = GaiaAstrometry(primary_mass=(1.0, 0.1), verbose=False)
    astrometry.ra_ref = 10.0
    astrometry.dec_ref = 20.0
    astrometry.sim_data = True
    astrometry.u0_norm = 1.0

    relative_time = np.sort(rng.uniform(-2.5, 2.5, 80))
    scan_angle = rng.uniform(0.0, 2.0 * np.pi, relative_time.size)
    astrometry.data_table = pd.DataFrame(
        {
            "obs_time_tcb": astrometry.ref_epoch.tcb.jyear + relative_time,
            "relative_time_year": relative_time,
            "relative_time_day": relative_time * 365.25,
            "sin_scan_ang": np.sin(scan_angle),
            "cos_scan_ang": np.cos(scan_angle),
            "parallax_factor_al": rng.uniform(-1.0, 1.0, relative_time.size),
            "centroid_pos_error_al": np.full(relative_time.size, 0.2),
        }
    )
    model = KeplerModel(astrometry, verbose=False)
    astrometry.data_table["centroid_pos_al"] = model.calc_1d_model(
        ORBIT_PARAMETERS
    ) + rng.normal(0.0, 0.2, relative_time.size)
    return LeastSquares(astrometry)


def test_calculate_excess_noise(least_squares: LeastSquares) -> None:
    excess_noise, uncertainty = least_squares.calc_excess_noise(verbose=False)
    assert excess_noise is not None and excess_noise > 0.0
    assert uncertainty is not None and uncertainty >= 0.0


def test_fit_single_star(least_squares: LeastSquares) -> None:
    least_squares.singl_5param(plot_file=None)
    assert least_squares.best_param.shape == (5,)


def test_fit_acceleration(least_squares: LeastSquares) -> None:
    least_squares.accel_7param(plot_file=None)
    assert least_squares.best_param.shape == (7,)


def test_fit_acceleration_and_jerk(least_squares: LeastSquares) -> None:
    least_squares.accel_9param(plot_file=None)
    assert least_squares.best_param.shape == (9,)


def test_generate_orbit_grid(least_squares: LeastSquares) -> None:
    least_squares.orbit_grid(plot_file=None, n_points=5, verbose=False)
    assert least_squares.best_param.shape == (12,)


def test_fit_orbit(least_squares: LeastSquares) -> None:
    least_squares.best_param = np.array(param_dict_to_list(ORBIT_PARAMETERS))
    least_squares.orbit_fit(inc_jitter=False, plot_file=None)
    assert least_squares.fit_success
    assert least_squares.best_param.shape == (12,)


def test_fit_orbit_with_jitter(least_squares: LeastSquares) -> None:
    least_squares.best_param = np.array(param_dict_to_list(ORBIT_PARAMETERS))
    least_squares.orbit_fit(inc_jitter=True, plot_file=None)

    assert least_squares.fit_success
    assert len(least_squares.best_param) == 12
    assert least_squares.param_cov.shape == (12, 12)
