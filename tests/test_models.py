from exogaia.leastsq import LeastSquares
from types import SimpleNamespace

import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import CartesianRepresentation

from exogaia.models import KeplerModel, StarModel
import pytest


class GaiaAstrometry:
    def __init__(self, data_table: pd.DataFrame) -> None:
        self.data_table = data_table
        self.ref_epoch = Time(2016.0, format="jyear", scale="tcb")
        self.ra_ref = 0.0
        self.dec_ref = 0.0
        self.sim_data = True
        self.u0_norm = 1.0


def stellar_parameters() -> dict[str, float]:
    return {
        "ra_offset": 1.0,
        "dec_offset": 2.0,
        "parallax": 5.0,
        "pm_ra": 3.0,
        "pm_dec": 4.0,
        "pm_dot_ra": 2.0,
        "pm_dot_dec": 4.0,
        "pm_dotdot_ra": 6.0,
        "pm_dotdot_dec": 12.0,
    }


def test_calculate_excess_noise(least_squares: LeastSquares) -> None:
    least_squares.calc_excess_noise()


def test_star_model_stores_astrometry() -> None:
    epoch_astrometry = SimpleNamespace(
        data_table=pd.DataFrame(),
        ref_epoch=Time(2016.0, format="jyear", scale="tcb"),
        ra_ref=10.0,
        dec_ref=-5.0,
    )

    model = StarModel(epoch_astrometry, verbose=False)

    assert model.epoch_astrometry is epoch_astrometry
    assert model.ra_ref == 10.0
    assert model.dec_ref == -5.0


def test_kepler_model_solves_circular_orbit() -> None:
    epoch_astrometry = SimpleNamespace(
        data_table=pd.DataFrame({"relative_time_day": [0.0, 2.5]}),
        ref_epoch=Time(2016.0, format="jyear", scale="tcb"),
    )
    model = KeplerModel(epoch_astrometry, verbose=False)

    x_coord, y_coord = model.solve_kepler(per=10.0, ecc=0.0, tau=0.0)

    np.testing.assert_allclose(x_coord, [1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(y_coord, [0.0, 1.0], atol=1e-8)


@pytest.mark.parametrize(
    "per, ecc",
    [(0.0, 0.0), (10.0, -0.1), (10.0, 1.0)],
)
def test_kepler_model_rejects_invalid_orbit(per: float, ecc: float) -> None:
    epoch_astrometry = SimpleNamespace(
        data_table=pd.DataFrame({"relative_time_day": [0.0]}),
        ref_epoch=Time(2016.0, format="jyear", scale="tcb"),
    )

    with pytest.raises(ValueError):
        KeplerModel(epoch_astrometry, verbose=False).solve_kepler(per, ecc, 0.0)


def test_star_model_calculates_accelerated_2d_track(monkeypatch) -> None:
    epoch_astrometry = GaiaAstrometry(pd.DataFrame())
    model = StarModel(epoch_astrometry, verbose=False)
    monkeypatch.setattr(
        model,
        "barycentric_position",
        lambda sat_loc, obs_time: CartesianRepresentation(
            np.zeros((3, len(obs_time))) * u.au
        ),
    )

    delta_ra, delta_dec, delta_pos = model.calc_2d_model(
        stellar_parameters(), obs_time=np.array([2016.0, 2017.0, 2018.0])
    )

    np.testing.assert_allclose(delta_ra, [1.0, 6.0, 19.0])
    np.testing.assert_allclose(delta_dec, [2.0, 10.0, 34.0])
    assert delta_pos is None


def test_star_model_calculates_1d_track_with_acceleration() -> None:
    data_table = pd.DataFrame(
        {
            "relative_time_year": [0.0, 1.0],
            "sin_scan_ang": [1.0, 0.0],
            "cos_scan_ang": [0.0, 1.0],
            "parallax_factor_al": [0.5, -0.5],
        }
    )
    model = StarModel(GaiaAstrometry(data_table), verbose=False)

    delta_pos = model.calc_1d_model(stellar_parameters())

    np.testing.assert_allclose(delta_pos, [3.5, 7.5])


def test_kepler_model_calculates_orbit_and_residuals() -> None:
    relative_time_day = np.linspace(0.0, 10.0, 13)
    data_table = pd.DataFrame(
        {
            "relative_time_year": relative_time_day / 365.25,
            "relative_time_day": relative_time_day,
            "sin_scan_ang": np.ones(13),
            "cos_scan_ang": np.zeros(13),
            "parallax_factor_al": np.zeros(13),
            "centroid_pos_al": np.zeros(13),
            "centroid_pos_error_al": np.ones(13),
        }
    )
    model = KeplerModel(GaiaAstrometry(data_table), verbose=False)
    parameters = {
        "ra_offset": 0.0,
        "dec_offset": 0.0,
        "parallax": 0.0,
        "pm_ra": 0.0,
        "pm_dec": 0.0,
        "per": 10.0,
        "ecc": 0.0,
        "tau": 0.0,
        "sma": 2.0,
        "inc": 0.0,
        "aop": 0.0,
        "pan": 0.0,
    }

    delta_ra, delta_dec = model.calc_orbit(
        parameters, obs_time=np.array([2016.0, 2016.0 + 10.0 / 365.25])
    )
    residuals = model.calc_residuals(parameters)

    np.testing.assert_allclose(delta_ra, [0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(delta_dec, [2.0, 2.0], atol=1e-10)
    assert residuals.shape == (13,)
    assert np.all(np.isfinite(residuals))
    assert residuals[0] == pytest.approx(0.0, abs=1e-10)
