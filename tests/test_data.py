from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from beartype.roar import BeartypeCallHintParamViolation

from exogaia.data import GaiaAstrometry, HipparcosAstrometry


def test_gaia_astrometry_without_data() -> None:
    astrometry = GaiaAstrometry(verbose=False)

    assert repr(astrometry) == "Data table is empty"
    assert astrometry.data_table is None


def test_gaia_astrometry_unknown_release() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        GaiaAstrometry(gaia_release="DR0", verbose=False)


def test_gaia_astrometry_read_file(tmp_path) -> None:
    data_file = tmp_path / "epoch.csv"
    pd.DataFrame({"obs_time_tcb": [2016.0, 2016.5]}).to_csv(data_file, index=False)

    astrometry = GaiaAstrometry(verbose=False)
    astrometry.read_file(str(data_file))

    assert "relative_time_year" in astrometry.data_table
    assert "relative_time_day" in astrometry.data_table
    assert astrometry.data_table["relative_time_year"].tolist() == [-1.5, -1.0]


@pytest.mark.parametrize("gaia_release", ["DR1", "DR2", "DR3", "DR4", "DR5"])
def test_gaia_astrometry_releases(gaia_release: str) -> None:
    astrometry = GaiaAstrometry(gaia_release=gaia_release, verbose=False)

    assert astrometry.gaia_release == gaia_release
    assert astrometry.time_end > astrometry.time_start
    assert astrometry.ref_epoch > astrometry.time_start
    assert astrometry.ref_epoch < astrometry.time_end


def test_gaia_astrometry_interpolates_u0() -> None:
    data_file = Path(__file__).resolve().parent.parent / "data" / "table_u0_g_c_p5.txt"

    norm_g_mag, norm_nu_eff, norm_u0 = np.loadtxt(
        data_file, skiprows=1, delimiter=",", unpack=True
    )

    result = GaiaAstrometry._interpolate_u0(norm_g_mag[0], norm_nu_eff[0])

    assert result == pytest.approx(norm_u0[0])


@pytest.mark.parametrize("gaia_release", ["DR3"])
def test_gaia_astrometry_query_source(gaia_release: str) -> None:
    astrometry = GaiaAstrometry(verbose=False)
    astrometry.query_source(source_id=6199395656645840384, gaia_release=gaia_release)


def test_hipparcos_astrometry_initializes() -> None:
    astrometry = HipparcosAstrometry(
        hip_id=27321,
        primary_mass=(1.0, 0.1),
        verbose=False,
    )

    assert astrometry.hip_id == 27321
    assert astrometry.u0_norm == 1.0
