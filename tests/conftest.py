from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def least_squares():
    from exogaia.data import GaiaAstrometry
    from exogaia.leastsq import LeastSquares

    epoch_astrometry = GaiaAstrometry(primary_mass=None, gaia_release="DR4")
    epoch_astrometry.retrieve_gaia_bh3(exclude_outliers=True, combine_ccds=True)
    return LeastSquares(epoch_astrometry=epoch_astrometry)


@pytest.fixture(scope="module")
def test_dir() -> Path:
    return Path(__file__).parent
