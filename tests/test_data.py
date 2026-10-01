import pytest
from beartype.roar import BeartypeCallHintParamViolation

from exogaia.data import GaiaAstrometry


def test_gaia_astrometry_starts_without_data() -> None:
    astrometry = GaiaAstrometry(verbose=False)

    assert repr(astrometry) == "Data table is empty"
    assert astrometry.data_table is None


def test_gaia_astrometry_rejects_unknown_release() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        GaiaAstrometry(gaia_release="DR0", verbose=False)
