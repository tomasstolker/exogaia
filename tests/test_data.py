import pytest
import numpy as np
import pandas as pd
from beartype.roar import BeartypeCallHintParamViolation

from exogaia.data import GaiaAstrometry, HipparcosAstrometry


def test_gaia_astrometry_starts_without_data() -> None:
    astrometry = GaiaAstrometry(verbose=False)

    assert repr(astrometry) == "Data table is empty"
    assert astrometry.data_table is None


def test_gaia_astrometry_rejects_unknown_release() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        GaiaAstrometry(gaia_release="DR0", verbose=False)


def test_gaia_astrometry_read_file_adds_relative_times(tmp_path) -> None:
    data_file = tmp_path / "epoch.csv"
    pd.DataFrame({"obs_time_tcb": [2016.0, 2016.5]}).to_csv(data_file, index=False)

    astrometry = GaiaAstrometry(verbose=False)
    astrometry.read_file(str(data_file))

    assert "relative_time_year" in astrometry.data_table
    assert "relative_time_day" in astrometry.data_table
    assert astrometry.data_table["relative_time_year"].tolist() == [
        -1.5,
        -1.0,
    ]


@pytest.mark.parametrize("gaia_release", ["DR1", "DR2", "DR3", "DR4", "DR5"])
def test_gaia_astrometry_supports_releases(gaia_release: str) -> None:
    astrometry = GaiaAstrometry(gaia_release=gaia_release, verbose=False)

    assert astrometry.gaia_release == gaia_release
    assert astrometry.ref_epoch.tcb.jyear > 2014.0
    assert astrometry.time_end > astrometry.time_start


def test_gaia_astrometry_interpolates_u0() -> None:
    data_file = "data/table_u0_g_c_p5.txt"
    norm_g_mag, norm_nu_eff, norm_u0 = np.loadtxt(
        data_file, skiprows=1, delimiter=",", unpack=True
    )

    result = GaiaAstrometry._interpolate_u0(norm_g_mag[0], norm_nu_eff[0])

    assert result == pytest.approx(norm_u0[0])


def test_gaia_astrometry_read_file_preserves_existing_relative_times(tmp_path) -> None:
    data_file = tmp_path / "epoch.csv"
    pd.DataFrame(
        {
            "obs_time_tcb": [2016.0],
            "relative_time_year": [0.0],
            "relative_time_day": [0.0],
        }
    ).to_csv(data_file, index=False)

    astrometry = GaiaAstrometry(verbose=False)
    astrometry.read_file(str(data_file))

    assert astrometry.data_table.loc[0, "relative_time_year"] == 0.0
    assert astrometry.data_table.loc[0, "relative_time_day"] == 0.0


@pytest.mark.parametrize("gaia_release", ["DR2", "DR4"])
def test_gaia_astrometry_rejects_unsupported_query_release(gaia_release: str) -> None:
    astrometry = GaiaAstrometry(verbose=False)

    with pytest.raises(ValueError, match="only.*DR3"):
        astrometry.query_source(source_id=123, gaia_release=gaia_release)


def test_gaia_astrometry_rejects_unsupported_nss_release() -> None:
    astrometry = GaiaAstrometry(verbose=False)

    with pytest.raises(ValueError, match="only supports Gaia DR3"):
        astrometry.get_nss_tables(gaia_release="DR4")


def test_gaia_astrometry_retrieve_data_returns_empty_result(monkeypatch) -> None:
    astrometry = GaiaAstrometry(gaia_release="DR4", verbose=False)
    monkeypatch.setattr(
        astrometry,
        "query_source",
        lambda source_id, gaia_release: {},
    )

    with pytest.warns(UserWarning, match="not yet implemented"):
        result = astrometry.retrieve_data(source_id=123)

    assert result == {}
    assert astrometry.source_id == 123


def test_hipparcos_astrometry_initializes() -> None:
    astrometry = HipparcosAstrometry(
        hip_id=123,
        primary_mass=(1.0, 0.1),
        verbose=False,
    )

    assert astrometry.hip_id == 123
    assert astrometry.u0_norm == 1.0


@pytest.mark.parametrize(
    "model_param",
    [
        {},
        {"ra_ref": 10.0},
        {"ra_ref": 10.0, "dec_ref": -5.0},
        {"ra_ref": 10.0, "dec_ref": -5.0, "parallax": 1.0},
        {
            "ra_ref": 10.0,
            "dec_ref": -5.0,
            "parallax": 1.0,
            "pm_ra": 2.0,
        },
    ],
)
def test_simulate_data_requires_stellar_parameters(model_param) -> None:
    astrometry = GaiaAstrometry(primary_mass=(1.0, 0.1), verbose=False)

    with pytest.raises(ValueError):
        astrometry.simulate_data(model_param=model_param)
