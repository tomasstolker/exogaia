import os
import pytest

from exogaia import EpochAstrometry


class TestExoGaia:
    def setup_class(self) -> None:

        self.limit = 1e-8
        self.test_dir = os.path.dirname(__file__) + "/"

    def test_epoch_astrometry(self) -> None:

        self.epoch_astrom = EpochAstrometry(primary_mass=(1.0, 0.01), gaia_release='DR4')
