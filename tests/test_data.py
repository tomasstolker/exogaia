import pytest
import pandas as pd
from beartype.roar import BeartypeCallHintParamViolation

from exogaia.data import GaiaAstrometry


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
