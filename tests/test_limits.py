from exogaia import limits


class FakeGaiaAstrometry:
    def __init__(self, primary_mass, gaia_release, verbose):
        self.primary_mass = primary_mass
        self.gaia_release = gaia_release
        self.verbose = verbose

    def query_source(self, source_id, gaia_release):
        self.source_id = source_id
        self.query_release = gaia_release


def test_completeness_map_initializes_from_source(monkeypatch) -> None:
    monkeypatch.setattr(limits, "GaiaAstrometry", FakeGaiaAstrometry)

    completeness = limits.CompletenessMap(
        source_id=123,
        primary_mass=(1.0, 0.1),
        gaia_release="DR4",
    )

    assert completeness.source_id == 123
    assert completeness.epoch_astrom.source_id == 123
    assert completeness.epoch_astrom.query_release == "DR3"
