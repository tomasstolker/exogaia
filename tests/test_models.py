from exogaia.leastsq import LeastSquares
from types import SimpleNamespace

import numpy as np
import pandas as pd
from astropy.time import Time

from exogaia.models import KeplerModel, StarModel
import pytest


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
